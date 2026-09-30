"""Deadline micro-batching across cameras (ARCHITECTURE D9).

Why not "batch = number of cameras": a dead or slow camera would then hold every other
camera's frames hostage until a timeout. Here a batch closes when it's full *or* when
the oldest frame in it has waited `deadline_s`, whichever comes first, so latency is
bounded by `deadline_s + inference(batch)` no matter which cameras are sending.

Per-camera slots hold at most `max_pending_per_camera` frames; when a camera outruns
the GPU its **oldest** pending frame is dropped (analytics path may drop, the evidence
path never touches this). A batch takes at most one frame per camera, oldest-waiting
cameras first, so one fast camera can't starve the rest.

Results go to `on_result(ref, detections, timing)` on the worker thread; keep it cheap
(e.g. hand off to the per-camera tracker queue).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from scs.contracts import Detection, FrameRef
from scs.perception.base import Detector


@dataclass(frozen=True)
class BatchTiming:
    submitted: float  # time.monotonic() when the frame was submitted
    batch_closed: float  # when its batch was closed and handed to the detector
    done: float  # when detection finished
    batch_size: int

    @property
    def latency(self) -> float:
        return self.done - self.submitted

    @property
    def queue_wait(self) -> float:
        return self.batch_closed - self.submitted


@dataclass
class BatcherStats:
    batches: int = 0
    frames: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    batch_sizes: list[int] = field(default_factory=list)


ResultFn = Callable[[FrameRef, list[Detection], BatchTiming], None]


class MicroBatcher:
    def __init__(
        self,
        detector: Detector,
        on_result: ResultFn,
        max_batch: int = 16,
        deadline_s: float = 0.020,
        max_pending_per_camera: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_batch < 1 or deadline_s < 0 or max_pending_per_camera < 1:
            raise ValueError("max_batch>=1, deadline_s>=0, max_pending_per_camera>=1 required")
        self.detector, self.on_result = detector, on_result
        self.max_batch, self.deadline_s = max_batch, deadline_s
        self.max_pending = max_pending_per_camera
        self.clock = clock
        self.stats = BatcherStats()
        self._pending: dict[str, deque[tuple[float, FrameRef, Any]]] = {}
        self._cv = threading.Condition()
        self._stop = False
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None

    # ------------------------------------------------------------------ producer side
    def submit(self, ref: FrameRef, image: Any) -> None:
        """Non-blocking. Drops this camera's oldest pending frame if its slot is full."""
        with self._cv:
            q = self._pending.setdefault(ref.camera_id, deque())
            if len(q) >= self.max_pending:
                q.popleft()
                self.stats.dropped[ref.camera_id] = self.stats.dropped.get(ref.camera_id, 0) + 1
            q.append((self.clock(), ref, image))
            self._cv.notify()

    # ------------------------------------------------------------------ worker side
    def _ready(self) -> int:
        return sum(1 for q in self._pending.values() if q)

    def _oldest(self) -> float | None:
        heads = [q[0][0] for q in self._pending.values() if q]
        return min(heads) if heads else None

    def _take_batch(self) -> list[tuple[float, FrameRef, Any]] | None:
        """Block until a batch is due (full or deadline hit); None on stop."""
        with self._cv:
            while not self._stop:
                oldest = self._oldest()
                if oldest is None:
                    self._cv.wait()
                    continue
                if self._ready() >= self.max_batch:
                    break
                remaining = oldest + self.deadline_s - self.clock()
                if remaining <= 0:
                    break
                self._cv.wait(timeout=remaining)
            if self._stop:
                return None
            cams = sorted((q[0][0], cam) for cam, q in self._pending.items() if q)[: self.max_batch]
            return [self._pending[cam].popleft() for _, cam in cams]

    def run_once(self) -> int:
        """Form and run one batch (blocking). Returns its size, 0 on stop. Used by tests and `_loop`."""
        items = self._take_batch()
        if not items:
            return 0
        closed = self.clock()
        dets = self.detector.detect([(ref, img) for _, ref, img in items])
        done = self.clock()
        self.stats.batches += 1
        self.stats.frames += len(items)
        self.stats.batch_sizes.append(len(items))
        for (sub, ref, _), d in zip(items, dets, strict=True):
            self.on_result(ref, d, BatchTiming(sub, closed, done, len(items)))
        return len(items)

    def _loop(self) -> None:
        try:
            while self.run_once():
                pass
        except BaseException as e:  # surface to the owner instead of dying silently
            self.error = e
            raise

    def start(self) -> MicroBatcher:
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="scs-microbatcher", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def __enter__(self) -> MicroBatcher:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
