"""Helpers for ingest tests: a scripted fake backend session (importable by test modules)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from scs.ingest.convert import RawImage


@dataclass
class FakeSession:
    """Emits `n` packet+frame pairs at `fps`, then ends with `end` (or hangs if end="hang").

    `hang_after` makes it go silent (session up, no data) after that many frames, like a
    Reolink silent stall; `gop` sets the keyframe interval; `lead_delta` emits that many
    non-keyframe packets first (a stream joined mid-GOP).
    """

    n: int = 30
    fps: float = 0.0  # 0 = as fast as possible
    end: str = "eos"
    hang_after: int | None = None
    gop: int = 10
    lead_delta: int = 0
    size: tuple[int, int] = (8, 4)
    backend: str = "fake"
    decoder: str = "fakedec"
    ran: list[float] = field(default_factory=list)

    def run(self, sink):
        self.ran.append(time.monotonic())
        w, h = self.size
        for _ in range(self.lead_delta):
            sink.on_packet(b"\x00\x00\x00\x01\x41", None, False, "h264")
        for i in range(self.n):
            if sink.should_stop():
                return "stopped", ""
            if self.hang_after is not None and i >= self.hang_after:
                while not sink.should_stop():
                    time.sleep(0.01)
                return "stopped", ""
            pts = i * 66_666_667
            sink.on_packet(bytes([0, 0, 0, 1, 0x65 if i % self.gop == 0 else 0x41, i % 256]), pts,
                           i % self.gop == 0, "h264")
            img = np.full((h, w, 3), i % 256, dtype=np.uint8)
            sink.on_frame(RawImage(img, "rgb", w, h), pts)
            if self.fps:
                time.sleep(1 / self.fps)
        if self.end == "hang":
            while not sink.should_stop():
                time.sleep(0.01)
            return "stopped", ""
        return ("error", "boom") if self.end == "error" else ("eos", "")


class SessionScript:
    """Factory returning a different FakeSession per (re)connect; repeats the last one."""

    def __init__(self, *sessions: FakeSession) -> None:
        self.sessions = list(sessions)
        self.calls = 0
        self.lock = threading.Lock()

    def __call__(self) -> FakeSession:
        with self.lock:
            s = self.sessions[min(self.calls, len(self.sessions) - 1)]
            self.calls += 1
            return s


def nal_types(data: bytes, codec: str) -> set[int]:
    """NAL unit types present in an Annex-B access unit."""
    out = set()
    i = 0
    while True:
        i = data.find(b"\x00\x00\x01", i)
        if i < 0 or i + 3 >= len(data):
            return out
        b = data[i + 3]
        out.add(b & 0x1F if codec == "h264" else (b >> 1) & 0x3F)
        i += 3
