"""D9 micro-batching: deadline, max batch, fairness, drop-oldest, and dead/slow cameras.

The CPU tests use a fake detector whose cost grows with batch size, so they check the
scheduling logic, not GPU speed. The GPU twin in test_detect_gpu.py repeats the dead
camera scenario with a real TRT engine.
"""

import statistics
import threading
import time

import pytest

from scs.contracts import Detection, FrameRef
from scs.perception.detect_batching import MicroBatcher


class FakeDetector:
    def __init__(self, base_s=0.003, per_img_s=0.0005):
        self.base_s, self.per_img_s = base_s, per_img_s
        self.batches = []

    def detect(self, batch):
        self.batches.append([ref.camera_id for ref, _ in batch])
        time.sleep(self.base_s + self.per_img_s * len(batch))
        return [[Detection(frame=ref, bbox=(0, 0, 10, 10), score=0.9)] for ref, _ in batch]


def _ref(cam, seq):
    return FrameRef(camera_id=cam, epoch=0, seq=seq, frame_idx=seq, ts=time.time(), width=64, height=64)


def run_cameras(n_cams=10, fps=10.0, seconds=1.5, dead=(), slow=None, detector=None, deadline_s=0.02):
    """Simulate `n_cams` producers; returns {camera: [latency_s, ...]} and the batcher."""
    lat: dict[str, list[float]] = {}
    lock = threading.Lock()

    def on_result(ref, dets, timing):
        with lock:
            lat.setdefault(ref.camera_id, []).append(timing.latency)

    det = detector or FakeDetector()
    b = MicroBatcher(det, on_result, max_batch=16, deadline_s=deadline_s)
    stop = threading.Event()

    def producer(cam, period, phase):
        time.sleep(phase)
        seq = 0
        while not stop.is_set():
            b.submit(_ref(cam, seq), None)
            seq += 1
            time.sleep(period)

    threads = []
    for i in range(n_cams):
        cam = f"cam{i}"
        if cam in dead:
            continue
        period = 1.0 / (slow[1] if slow and slow[0] == cam else fps)
        threads.append(threading.Thread(target=producer, args=(cam, period, i * 0.007), daemon=True))
    with b:
        for t in threads:
            t.start()
        time.sleep(seconds)
        stop.set()
        for t in threads:
            t.join()
        time.sleep(0.1)
    return lat, b


def _p95(xs):
    return statistics.quantiles(xs, n=20)[-1]


def test_deadline_closes_partial_batch():
    lat, b = run_cameras(n_cams=1, fps=5, seconds=0.8)
    assert b.stats.batch_sizes and max(b.stats.batch_sizes) == 1
    # one camera alone waits for the deadline (20 ms) + fake inference, never more
    assert max(lat["cam0"]) < 0.020 + 0.0035 + 0.015


def test_full_batch_closes_before_deadline():
    det = FakeDetector()
    got = []
    b = MicroBatcher(det, lambda r, d, t: got.append(t), max_batch=4, deadline_s=10.0)
    for i in range(4):
        b.submit(_ref(f"c{i}", 0), None)
    t0 = time.monotonic()
    assert b.run_once() == 4
    assert time.monotonic() - t0 < 0.5  # didn't wait for the 10 s deadline
    assert all(t.batch_size == 4 for t in got)


def test_one_frame_per_camera_and_drop_oldest():
    det = FakeDetector()
    b = MicroBatcher(det, lambda r, d, t: None, max_batch=8, deadline_s=0.0, max_pending_per_camera=2)
    for s in range(5):
        b.submit(_ref("fast", s), None)
    b.submit(_ref("other", 0), None)
    assert b.stats.dropped == {"fast": 3}
    b.run_once()
    assert sorted(det.batches[0]) == ["fast", "other"]  # fairness: one frame per camera per batch
    b.run_once()
    assert det.batches[1] == ["fast"]


def test_batch_order_matches_results():
    seen = []
    b = MicroBatcher(
        FakeDetector(),
        lambda r, d, t: seen.append((r.camera_id, d[0].frame.camera_id)),
        max_batch=16,
        deadline_s=0.0,
    )
    for i in range(6):
        b.submit(_ref(f"c{i}", 0), None)
    b.run_once()
    assert len(seen) == 6 and all(a == c for a, c in seen)


def test_dead_camera_does_not_change_other_latency():
    base, _ = run_cameras()
    dead, _ = run_cameras(dead=("cam3",))
    assert "cam3" not in dead
    healthy = [c for c in base if c != "cam3"]
    base_p95 = _p95([x for c in healthy for x in base[c]])
    dead_p95 = _p95([x for c in healthy for x in dead[c]])
    # bounded by deadline + inference; a dead camera must not add a wait (it would add >= 1 frame period)
    assert dead_p95 <= base_p95 + 0.005
    assert dead_p95 < 0.020 + 0.003 + 0.0005 * 16 + 0.010


def test_slow_camera_does_not_change_other_latency():
    base, _ = run_cameras()
    slow, _ = run_cameras(slow=("cam5", 1.0))  # 1 fps instead of 10
    healthy = [c for c in base if c != "cam5"]
    assert _p95([x for c in healthy for x in slow[c]]) <= _p95([x for c in healthy for x in base[c]]) + 0.005
    assert len(slow["cam5"]) <= 3


def test_invalid_config():
    with pytest.raises(ValueError):
        MicroBatcher(FakeDetector(), lambda *a: None, max_batch=0)
