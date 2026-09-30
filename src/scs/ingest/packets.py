"""Encoded packet tap: every compressed access unit, before any analytics dropping.

Evidence clips (T10) must be byte-exact copies of what the camera sent, and must not
have holes just because the detector fell behind. So the decode worker hands every
encoded access unit to the tap *before* the frame reaches the bounded analytics queue
(ARCHITECTURE §2). Tap delivery is synchronous on the decode thread: a subscriber
that blocks, blocks decoding, which is the right trade (evidence beats analytics).
Subscribers that want to decouple use `PacketQueue`, which never drops silently.

Packets carry the same identity as frames: a packet with `seq = n` in epoch `e`
decodes to the frame whose `FrameRef.identity == (camera_id, e, n)`. That is exact for
streams without B-frames (surveillance cameras, the T14 farm); see `EncodedPacket.seq`.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

PacketCodec = Literal["h264", "h265"]


@dataclass(frozen=True, slots=True)
class EncodedPacket:
    """One encoded access unit (one picture) in Annex-B byte-stream format.

    Keyframes carry their parameter sets (SPS/PPS, plus VPS for H.265) in-band, so a clip
    can be cut starting at any packet with `keyframe=True` without other state.
    """

    camera_id: str
    epoch: int  # connection id, same as FrameRef.epoch
    # Identity of the frame this packet decodes to (FrameRef.seq). Assigned in decode
    # order, so it equals the frame's seq when decode order == display order (no B-frames).
    seq: int
    pts_ns: int | None  # presentation time in the stream's clock, ns; restarts per epoch/loop
    ts: float  # wall clock (epoch s) when the packet left the demuxer/depayloader
    ts_mono: float  # host monotonic clock at the same instant
    source_ts: float | None  # RTP/NTP capture time (epoch s) when the sender provides it
    keyframe: bool
    codec: PacketCodec
    data: bytes

    @property
    def identity(self) -> tuple[str, int, int]:
        return (self.camera_id, self.epoch, self.seq)


PacketCallback = Callable[[EncodedPacket], None]


class PacketTap:
    """Fan-out of encoded packets to subscribers. Thread-safe subscribe/unsubscribe.

    A subscriber that raises is counted in `errors` and kept; one broken consumer must
    not stop evidence for the others.
    """

    def __init__(self) -> None:
        self._subs: tuple[PacketCallback, ...] = ()
        self._lock = threading.Lock()
        self.delivered = 0
        self.errors = 0

    def subscribe(self, cb: PacketCallback) -> Callable[[], None]:
        """Register `cb`; returns a function that unsubscribes it."""
        with self._lock:
            self._subs = (*self._subs, cb)

        def unsubscribe() -> None:
            with self._lock:
                self._subs = tuple(s for s in self._subs if s is not cb)

        return unsubscribe

    def emit(self, pkt: EncodedPacket) -> None:
        for cb in self._subs:  # snapshot: tuple is replaced, never mutated
            try:
                cb(pkt)
            except Exception:  # noqa: BLE001 - isolate subscribers from each other
                self.errors += 1
        self.delivered += 1


class PacketQueue:
    """A tap subscriber that buffers packets for another thread.

    Bounded by bytes, not count, so a bitrate spike can't exhaust memory. On overflow
    the *newest* packet is refused and `overflow` counts it: evidence consumers must see
    that they lost data rather than get a clip with a silent hole. Size it for the
    consumer's worst stall (default 256 MB ≈ 4 min of one 8 Mbit/s camera).
    """

    def __init__(self, max_bytes: int = 256 * 2**20) -> None:
        self._q: deque[EncodedPacket] = deque()
        self._cv = threading.Condition()
        self._bytes = 0
        self.max_bytes = max_bytes
        self.received = 0
        self.overflow = 0

    def __call__(self, pkt: EncodedPacket) -> None:
        with self._cv:
            if self._bytes + len(pkt.data) > self.max_bytes:
                self.overflow += 1
                return
            self._q.append(pkt)
            self._bytes += len(pkt.data)
            self.received += 1
            self._cv.notify()

    def get(self, timeout: float | None = None) -> EncodedPacket | None:
        """Oldest packet, or None after `timeout` seconds with nothing queued."""
        with self._cv:
            if not self._q and not self._cv.wait_for(lambda: bool(self._q), timeout):
                return None
            pkt = self._q.popleft()
            self._bytes -= len(pkt.data)
            return pkt

    def drain(self) -> list[EncodedPacket]:
        with self._cv:
            out = list(self._q)
            self._q.clear()
            self._bytes = 0
            return out

    def __len__(self) -> int:
        return len(self._q)
