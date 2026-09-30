"""Multi-camera ingest: one decode worker per camera, micro-batches with a deadline, stats.

Each camera is a `StreamSource` with its own thread and drop-oldest queue, so a dead or
slow camera never delays a healthy one (ARCHITECTURE D9). `next_batch` collects whatever
frames are ready within a deadline (default 20 ms), at most one per camera, starting
from a rotating camera, so the detector batches without waiting for stragglers.

Each source's watchdog samples its fresh-frame age on a fixed 10 Hz tick; the p95 over
those samples is the "fresh-frame age p95" in T02's acceptance evidence. Sampling on a
clock instead of per frame is the point: a stalled camera produces no frames, so
per-frame sampling would never see its age grow.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

from scs.contracts import FrameRef
from scs.ingest.convert import Output, RawImage, convert
from scs.ingest.source import StreamSource


class CameraManager:
    def __init__(self, sources: Iterable[StreamSource]) -> None:
        self.sources: dict[str, StreamSource] = {}
        for s in sources:
            if s.camera_id in self.sources:
                raise ValueError(f"duplicate camera_id {s.camera_id}")
            self.sources[s.camera_id] = s
        self._rr = 0  # round-robin start so no camera is always first

    def start(self) -> CameraManager:
        for s in self.sources.values():
            s.start()
        return self

    def next_batch(self, deadline_s: float = 0.02, max_batch: int | None = None,
                   output: Output = "raw") -> list[tuple[FrameRef, Any]]:
        """Up to one frame per camera, gathered for at most `deadline_s`. May be empty."""
        cams = list(self.sources.values())
        limit = max_batch or len(cams)
        got: dict[str, tuple[FrameRef, RawImage]] = {}
        end = time.monotonic() + deadline_s
        self._rr = (self._rr + 1) % max(1, len(cams))
        order = cams[self._rr:] + cams[: self._rr]
        while True:
            for s in order:
                if s.camera_id not in got and len(got) < limit:
                    item = s.get(timeout=0)
                    if item is not None:
                        got[s.camera_id] = item
            if len(got) >= limit or time.monotonic() >= end:
                break
            time.sleep(0.001)
        return [(ref, convert(raw, output)) for ref, raw in got.values()]

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {cid: s.stats.snapshot() for cid, s in self.sources.items()}

    @property
    def alive(self) -> bool:
        return any(s.alive for s in self.sources.values())

    def close(self) -> None:
        for s in self.sources.values():
            s.close()

    def __enter__(self) -> CameraManager:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()
