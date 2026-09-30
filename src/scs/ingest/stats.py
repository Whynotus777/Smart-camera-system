"""Per-camera ingest statistics and the bounded analytics frame queue.

The health of a camera is not "did read() raise": Reolinks stall silently with the
RTSP session up. So we track *fresh-frame age* (monotonic seconds since the last decoded
frame) and sample it, and report decoded vs. dropped frames separately. A dropped frame
here means the analytics consumer fell behind; it never means lost evidence (the
packet tap runs before this queue).

Clocks, named explicitly because latency numbers are meaningless otherwise:
- `fresh_frame_age_s`: host monotonic clock, now − last decode. Stall detection uses it.
- `capture_to_decode_ms`: `FrameRef.ts − FrameRef.source_ts` = decode wall clock minus
  RTP/NTP sender clock. Crosses hosts, so it includes camera↔host clock offset; only
  meaningful with NTP-synced cameras.
- decode→result latency is measured downstream (T03+) from `FrameRef.ts_mono`.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from scs.ingest.urls import redact_text

T = TypeVar("T")


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[k]


@dataclass
class CameraStats:
    """Counters for one camera. Written by its decode worker, read by anyone."""

    camera_id: str
    decoded: int = 0  # frames out of the decoder
    delivered: int = 0  # frames handed to the analytics consumer
    dropped: int = 0  # frames evicted from the analytics queue (consumer too slow)
    packets: int = 0  # encoded access units passed to the packet tap
    skipped_prekey: int = 0  # access units before a session's first keyframe (not tapped)
    reordered: int = 0  # frames out of decode order (B-frames): packet/frame seq may differ
    unmatched: int = 0  # frames whose PTS matched no tapped packet
    errors: int = 0  # decode/pipeline/connection errors
    reconnects: int = 0
    stalls: int = 0
    epoch: int = 0
    backend: str = ""
    decoder: str = ""  # e.g. "nvh265dec" or "hevc_cuvid"; proves NVDEC was used
    last_error: str = ""
    started_mono: float = field(default_factory=time.monotonic)
    last_frame_mono: float | None = None
    _fps_win: deque[float] = field(default_factory=lambda: deque(maxlen=64), repr=False)
    _age_samples: deque[float] = field(default_factory=lambda: deque(maxlen=100_000), repr=False)
    _c2d_ms: deque[float] = field(default_factory=lambda: deque(maxlen=10_000), repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def on_frame(self, ts_mono: float, capture_to_decode_ms: float | None = None) -> None:
        with self._lock:
            self.decoded += 1
            self.last_frame_mono = ts_mono
            self._fps_win.append(ts_mono)
            if capture_to_decode_ms is not None:
                self._c2d_ms.append(capture_to_decode_ms)

    def on_error(self, msg: str) -> None:
        with self._lock:
            self.errors += 1
            self.last_error = redact_text(msg)[:300]

    def fresh_frame_age(self, now_mono: float | None = None) -> float:
        now = time.monotonic() if now_mono is None else now_mono
        last = self.last_frame_mono if self.last_frame_mono is not None else self.started_mono
        return max(0.0, now - last)

    def sample_age(self, now_mono: float | None = None) -> float:
        """Record one fresh-frame-age sample (the manager calls this on a fixed tick)."""
        age = self.fresh_frame_age(now_mono)
        with self._lock:
            self._age_samples.append(age)
        return age

    def reset_samples(self) -> None:
        """Forget fresh-frame-age and latency samples (e.g. to exclude a warm-up period)."""
        with self._lock:
            self._age_samples.clear()
            self._c2d_ms.clear()

    def fps(self) -> float:
        with self._lock:
            w = list(self._fps_win)
        if len(w) < 2 or w[-1] <= w[0]:
            return 0.0
        return (len(w) - 1) / (w[-1] - w[0])

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            ages = list(self._age_samples)
            c2d = list(self._c2d_ms)
        return {
            "camera_id": self.camera_id,
            "backend": self.backend,
            "decoder": self.decoder,
            "epoch": self.epoch,
            "decoded": self.decoded,
            "delivered": self.delivered,
            "dropped": self.dropped,
            "packets": self.packets,
            "skipped_prekey": self.skipped_prekey,
            "reordered": self.reordered,
            "unmatched": self.unmatched,
            "errors": self.errors,
            "reconnects": self.reconnects,
            "stalls": self.stalls,
            "fps": round(self.fps(), 2),
            "fresh_frame_age_s": round(self.fresh_frame_age(), 3),
            "fresh_frame_age_p50_s": _r(_pct(ages, 50)),
            "fresh_frame_age_p95_s": _r(_pct(ages, 95)),
            "fresh_frame_age_max_s": _r(max(ages) if ages else None),
            "fresh_frame_age_samples": len(ages),
            "capture_to_decode_ms_p50": _r(_pct(c2d, 50)),
            "capture_to_decode_ms_p95": _r(_pct(c2d, 95)),
            "last_error": self.last_error,
        }


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


class DropOldestQueue(Generic[T]):
    """Bounded queue for the analytics path: when full, the oldest item is evicted.

    Analytics wants the freshest frame, not a growing backlog; `on_drop` lets the owner
    count evictions. `close()` wakes blocked readers so workers can shut down.
    """

    def __init__(self, maxsize: int, on_drop: Any = None) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._q: deque[T] = deque()
        self._max = maxsize
        self._cv = threading.Condition()
        self._closed = False
        self._on_drop = on_drop

    def put(self, item: T) -> None:
        with self._cv:
            if len(self._q) >= self._max:
                old = self._q.popleft()
                if self._on_drop is not None:
                    self._on_drop(old)
            self._q.append(item)
            self._cv.notify()

    def get(self, timeout: float | None = None) -> T | None:
        """Oldest item; None on timeout or once closed and empty."""
        with self._cv:
            ok = self._cv.wait_for(lambda: bool(self._q) or self._closed, timeout)
            if not ok or not self._q:
                return None
            return self._q.popleft()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed

    def __len__(self) -> int:
        return len(self._q)
