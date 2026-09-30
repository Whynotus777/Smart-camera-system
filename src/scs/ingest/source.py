"""Decode worker shared by every stream source: identity, packet tap, queue, stall, reconnect.

Why one worker thread per camera, pushing into a bounded queue, instead of decoding
inside `frames()`: a pull-based generator stops decoding whenever the consumer is
slow, so the RTSP socket backs up, the camera drops the session, and evidence packets
stop too. Here decoding never waits on analytics. Per packet, in order:

1. the backend demuxes/depayloads an access unit → `on_packet`: assign `seq`, emit to
   the `PacketTap` (evidence; never dropped here);
2. the decoder outputs a picture → `on_frame`: build the `FrameRef` (same `seq`, found
   by PTS), update stats, `put` into the drop-oldest analytics queue;
3. `frames()` pops from that queue on the consumer's thread and converts the image
   there, so frames that get dropped never pay for color conversion or GPU upload.

A *session* is one connection (one pipeline). `epoch` increases by 1 for each session
that delivers data, so identity `(camera_id, epoch, seq)` never repeats across
reconnects (ADR 0003). A watchdog aborts a session with no fresh frame for
`stall_timeout_s` (Reolinks stall silently with RTSP still up), then the worker
reconnects with exponential backoff.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np

from scs.bus import Bus
from scs.contracts import FrameRef
from scs.ingest.backoff import Backoff
from scs.ingest.convert import Output, RawImage, convert
from scs.ingest.health import HealthReporter
from scs.ingest.packets import EncodedPacket, PacketCodec, PacketTap
from scs.ingest.stats import CameraStats, DropOldestQueue

if TYPE_CHECKING:
    import torch

log = logging.getLogger("scs.ingest")

EndReason = Literal["eos", "error", "stall", "stopped"]


class SessionSink(Protocol):
    """What a backend session calls back into (implemented by `StreamSource`)."""

    def on_packet(self, data: bytes, pts_ns: int | None, keyframe: bool, codec: PacketCodec,
                  source_ts: float | None = None) -> None: ...

    def on_frame(self, image: RawImage, pts_ns: int | None, source_ts: float | None = None) -> None: ...

    def reserve_seq(self, pts_ns: int | None, source_ts: float | None = None) -> int: ...

    def on_loop(self) -> None: ...

    def should_stop(self) -> bool: ...


class Session(Protocol):
    """One connection to one stream. `run` blocks until the session ends."""

    backend: str
    decoder: str  # decoder element/codec actually in use, e.g. "nvh264dec"

    def run(self, sink: SessionSink) -> tuple[EndReason, str]: ...


SessionFactory = Callable[[], Session]


@dataclass
class _Pending:
    seq: int
    source_ts: float | None


class StreamSource:
    """Base `FrameSource` for anything decoded by a backend session (RTSP, files)."""

    def __init__(
        self,
        camera_id: str,
        session_factory: SessionFactory,
        *,
        output: Output = "numpy",
        queue_size: int = 4,
        reconnect: bool = True,
        end_on_eos: bool = False,
        stall_timeout_s: float = 3.0,
        connect_timeout_s: float = 10.0,
        backoff: Backoff | None = None,
        bus: Bus | None = None,
        tap: PacketTap | None = None,
        stream: Literal["main", "sub"] = "main",
        main_size: tuple[int, int] | None = None,
        media_origin: float | Literal["open"] | None = None,
    ) -> None:
        if stream == "sub" and main_size is None:
            raise ValueError("stream='sub' needs main_size: FrameRef.width/height are main-stream dims")
        self.camera_id = camera_id
        self._factory = session_factory
        self.output: Output = output
        self.reconnect = reconnect  # retry after errors/stalls (with backoff)
        self.end_on_eos = end_on_eos  # finite sources (files) end at EOS; cameras reconnect
        self.stall_timeout_s = stall_timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.backoff = backoff or Backoff()
        self.tap = tap or PacketTap()
        self.stats = CameraStats(camera_id)
        self.health = HealthReporter(camera_id, bus)
        self.stream: Literal["main", "sub"] = stream
        self.main_size = main_size
        # Recorded sources: source_ts = origin + media time (continuous across loops/reopens),
        # so time-based rules stay correct in unpaced replay. "open" = wall clock at first packet.
        self.media_origin = media_origin
        self._origin: float | None = media_origin if isinstance(media_origin, float | int) else None
        self._media_base_ns = 0  # media time already played in earlier passes/sessions
        self._pass_max_pts: int | None = None
        self._pass_delta_ns = 0
        self._queue: DropOldestQueue[tuple[FrameRef, RawImage]] = DropOldestQueue(
            queue_size, on_drop=self._on_drop)
        self._stop = threading.Event()
        self._abort = threading.Event()  # set by the watchdog: end the current session
        self._thread: threading.Thread | None = None
        self._watchdog: threading.Thread | None = None
        # per-session state, touched only by the worker/streaming threads
        self._epoch = -1
        self._session_live = False
        self._pkt_seq = 0
        self._frame_idx = 0
        self._last_seq = -1
        self._pending: OrderedDict[int, _Pending] = OrderedDict()  # PTS → seq, awaiting its frame
        self._reserved: OrderedDict[int, int] = OrderedDict()  # PTS → seq, awaiting its tap delivery
        self._session_start_mono = 0.0
        self._session_last_mono = time.monotonic()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public API

    def start(self) -> StreamSource:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name=f"ingest-{self.camera_id}", daemon=True)
            self._watchdog = threading.Thread(target=self._watch, name=f"ingest-wd-{self.camera_id}",
                                              daemon=True)
            self._thread.start()
            self._watchdog.start()
        return self

    def frames(self) -> Iterator[tuple[FrameRef, np.ndarray | torch.Tensor]]:
        """Yield `(ref, image)` until the source ends or `close()` is called."""
        self.start()
        while True:
            item = self._queue.get(timeout=0.5)
            if item is None:
                if self._queue.closed and len(self._queue) == 0:
                    return
                continue
            ref, raw = item
            self.stats.delivered += 1
            yield ref, convert(raw, self.output)

    def get(self, timeout: float | None = None) -> tuple[FrameRef, RawImage] | None:
        """Pop one raw (unconverted) item; for managers that batch conversion themselves."""
        item = self._queue.get(timeout=timeout)
        if item is not None:
            self.stats.delivered += 1
        return item

    def close(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._abort.set()
        for t in (self._thread, self._watchdog):
            if t is not None and t is not threading.current_thread():
                t.join(timeout)
        self._queue.close()

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def epoch(self) -> int:
        return max(self._epoch, 0)

    def __enter__(self) -> StreamSource:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ SessionSink

    def should_stop(self) -> bool:
        return self._stop.is_set() or self._abort.is_set()

    def on_packet(self, data: bytes, pts_ns: int | None, keyframe: bool, codec: PacketCodec,
                  source_ts: float | None = None) -> None:
        if not self._seen_key:
            if not keyframe:  # undecodable until the first keyframe, and useless for clips
                self.stats.skipped_prekey += 1
                return
            self._seen_key = True
        with self._lock:
            reserved = self._reserved.pop(pts_ns, None) if pts_ns is not None else None
        seq = reserved if reserved is not None else self.reserve_seq(pts_ns, source_ts)
        now_mono, now = time.monotonic(), time.time()
        self.stats.packets += 1
        self.tap.emit(EncodedPacket(
            camera_id=self.camera_id, epoch=self._epoch, seq=seq, pts_ns=pts_ns, ts=now,
            ts_mono=now_mono, source_ts=source_ts, keyframe=keyframe, codec=codec, data=data))

    def reserve_seq(self, pts_ns: int | None, source_ts: float | None = None) -> int:
        """Assign the next seq to an access unit and remember PTS→seq for its frame.

        Backends whose tap delivery and decoder output race (GStreamer: two appsinks) call
        this upstream of both, so the mapping always exists before the picture does.
        `on_packet` then reuses the reserved seq for the same PTS.
        """
        self._seen_key = True  # callers only reserve from the first keyframe on
        self._begin_epoch_if_needed()
        if source_ts is None:
            source_ts = self._media_ts(pts_ns)
        with self._lock:
            seq = self._pkt_seq
            self._pkt_seq += 1
            if pts_ns is not None:
                self._pending[pts_ns] = _Pending(seq, source_ts)
                self._reserved[pts_ns] = seq
                for d in (self._pending, self._reserved):
                    while len(d) > 512:  # packets that never produced a picture / never tapped
                        d.popitem(last=False)
        return seq

    def on_frame(self, image: RawImage, pts_ns: int | None, source_ts: float | None = None) -> None:
        self._begin_epoch_if_needed()
        now_mono, now = time.monotonic(), time.time()
        with self._lock:
            pend = self._pending.pop(pts_ns, None) if pts_ns is not None else None
            if pend is not None:
                seq = pend.seq
                source_ts = source_ts if source_ts is not None else pend.source_ts
            else:  # no packet stage (image sources) or a PTS the parser rewrote
                seq = self._last_seq + 1
                source_ts = source_ts if source_ts is not None else self._media_ts(pts_ns, track=False)
                if pts_ns is not None:
                    self.stats.unmatched += 1  # this frame's seq may not match its packet's
            if seq <= self._last_seq:  # B-frame reordering; keep identity unique + ordered
                seq = self._last_seq + 1
                self.stats.reordered += 1  # packet↔frame seq no longer aligned for this frame
            self._last_seq = seq
            idx = self._frame_idx
            self._frame_idx += 1
        w, h = self.main_size if self.main_size else (image.width, image.height)
        ref = FrameRef(camera_id=self.camera_id, epoch=self._epoch, seq=seq, frame_idx=idx, ts=now,
                       ts_mono=now_mono, source_ts=source_ts, transform=None, width=w, height=h,
                       stream=self.stream)
        c2d = (now - source_ts) * 1000 if source_ts is not None else None
        self.stats.on_frame(now_mono, c2d)
        self._session_last_mono = now_mono
        if not self._session_live:
            self._session_live = True
            self.backoff.reset()
            if self._session is not None:
                self.stats.decoder = self._session.decoder
            self.health.emit("connected", self._epoch, backend=self.stats.backend,
                             decoder=self.stats.decoder, width=w, height=h)
        self._queue.put((ref, image))

    def on_loop(self) -> None:
        """A looping file restarted: PTS restart, so forget pending PTS→seq entries."""
        with self._lock:
            self._pending.clear()
            self._reserved.clear()
        self._close_media_pass()

    def _media_ts(self, pts_ns: int | None, track: bool = True) -> float | None:
        if self.media_origin is None or pts_ns is None:
            return None
        if self._origin is None:  # "open": anchor at the first packet ever seen
            self._origin = time.time() - pts_ns / 1e9
        if track:
            if self._pass_max_pts is not None and pts_ns > self._pass_max_pts:
                self._pass_delta_ns = pts_ns - self._pass_max_pts
            self._pass_max_pts = pts_ns if self._pass_max_pts is None else max(self._pass_max_pts, pts_ns)
        return self._origin + (self._media_base_ns + pts_ns) / 1e9

    def _close_media_pass(self) -> None:
        """Media time restarts (loop or reopen): continue the timeline after the last frame."""
        if self._pass_max_pts is not None:
            self._media_base_ns += self._pass_max_pts + self._pass_delta_ns
        self._pass_max_pts = None

    # ------------------------------------------------------------------ internals

    def _on_drop(self, _item: object) -> None:
        self.stats.dropped += 1

    def _begin_epoch_if_needed(self) -> None:
        if self._session_new:
            with self._lock:
                if self._session_new:
                    self._epoch += 1
                    self.stats.epoch = self._epoch
                    self._session_new = False

    def _reset_session_state(self) -> None:
        with self._lock:
            self._session_new = True
            self._session_live = False
            self._seen_key = False
            self._pkt_seq = 0
            self._last_seq = -1
            self._pending.clear()
            self._reserved.clear()
            self._session_start_mono = time.monotonic()
        self._session_last_mono = self._session_start_mono
        self._close_media_pass()  # a reopened file starts its media clock at 0 again

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._reset_session_state()
                self._abort.clear()
                self._stall_flag = False
                reason, detail = self._run_session()
                if self._stop.is_set():
                    break
                if reason == "error":
                    self.stats.on_error(detail)
                if (reason == "eos" and self.end_on_eos) or not self.reconnect:
                    state = "eos" if reason == "eos" else "disconnected"
                    self.health.emit(state, self.epoch, f"{reason}: {detail}", stats=self.stats.snapshot())
                    break
                self.health.emit("disconnected", self.epoch, f"{reason}: {detail}",
                                 stats=self.stats.snapshot())
                delay = self.backoff.next_delay()
                self.health.emit("reconnecting", self.epoch, f"in {delay:.1f}s",
                                 backoff_s=round(delay, 2), attempt=self.backoff.attempt)
                if self._stop.wait(delay):
                    break
                self.stats.reconnects += 1
        finally:
            self.health.emit("stopped", self.epoch, stats=self.stats.snapshot())
            self._queue.close()

    def _run_session(self) -> tuple[EndReason, str]:
        try:
            session = self._session = self._factory()
            self.stats.backend = session.backend
            self.stats.decoder = session.decoder
            self._session_last_mono = time.monotonic()
            self._in_session = True
            reason, detail = session.run(self)
        except Exception as e:  # noqa: BLE001 - any backend failure → reconnect path
            log.debug("camera %s session failed", self.camera_id, exc_info=True)
            reason, detail = "error", f"{type(e).__name__}: {e}"
        finally:
            self._in_session = False
        if self._stall_flag and not self._stop.is_set():
            live = self._session_live
            return "stall", (f"no fresh frame for {self.stall_timeout_s:.1f}s" if live
                             else f"no first frame within {self.connect_timeout_s:.1f}s")
        return reason, detail

    _stall_flag = False
    _session_new = True
    _seen_key = False
    _session: Session | None = None
    _in_session = False

    def _watch(self) -> None:
        """Sample fresh-frame age at 10 Hz; abort sessions with no frame for too long.

        Stall timing uses the *session* clock (time since this session's last frame or its
        start), so a fresh connection gets `connect_timeout_s` to produce its first frame.
        The sampled `fresh_frame_age` is time since the last frame of any session.
        """
        while not self._stop.wait(0.1):
            now = time.monotonic()
            if self._thread is None or not self._thread.is_alive():
                continue
            self.stats.sample_age(now)
            if self._abort.is_set() or not self._in_session:  # nothing to judge during backoff
                continue
            live = self._session_live
            limit = self.stall_timeout_s if live else self.connect_timeout_s
            if now - self._session_last_mono > limit:
                self._stall_flag = True
                self._abort.set()
                # Reported at detection time: a backend blocked in a read may take a moment to exit.
                self.stats.stalls += 1
                self.health.emit("stalled", self.epoch,
                                 f"no fresh frame for {limit:.1f}s" if live else
                                 f"no first frame within {limit:.1f}s",
                                 fresh_frame_age_s=round(self.stats.fresh_frame_age(now), 3))
