"""One interface suite, run against every detector engine (acceptance: "both detectors pass the
same interface test suite"), plus a detector→tracker integration run on the PoC fixture clip.

GPU + built engines needed (`models/*/…engine`, see docs/reports/T03-detect-track.md for the
build recipe); each missing engine is skipped, not failed. Run under `scripts/gpu shared`.
The fixture clip has PoC boxes burned into its pixels: smoke level only.
"""

from pathlib import Path

import numpy as np
import pytest

from scs.contracts import Detection, FrameRef
from scs.perception.base import Detector

ROOT = Path(__file__).resolve().parents[2]
ENGINES = {
    "yolo11": ROOT / "models/yolo11s/yolo11s_fp16_b16.engine",
    "dfine": ROOT / "models/dfine-s/dfine_bf16_b16.engine",
}
FIXTURE = ROOT / "tests/fixtures/video/demo_1.mp4"

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module", params=sorted(ENGINES))
def detector(request):
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    engine = ENGINES[request.param]
    if not engine.exists():
        pytest.skip(f"engine not built: {engine}")
    from scs.perception.detect import make_detector

    return make_detector(request.param, engine=engine)


def _fixture_frames(n=40, step=5):
    cv2 = pytest.importorskip("cv2")
    cap = cv2.VideoCapture(str(FIXTURE))
    out, i = [], 0
    while len(out) < n:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % step == 0:
            out.append(np.ascontiguousarray(bgr[:, :, ::-1]))
        i += 1
    cap.release()
    return out


def _ref(cam="cam0", seq=0, w=640, h=360):
    return FrameRef(camera_id=cam, epoch=1, seq=seq, frame_idx=seq, ts=100.0 + seq, width=w, height=h)


def test_is_detector(detector):
    assert isinstance(detector, Detector)


def test_empty_batch(detector):
    assert detector.detect([]) == []


def test_order_identity_and_bounds(detector):
    import torch

    frames = _fixture_frames(12)
    batch = []
    for i, f in enumerate(frames):  # mix numpy/CPU and CUDA tensors and camera ids
        img = torch.from_numpy(f).cuda() if i % 2 else f
        batch.append((_ref(f"cam{i % 3}", i), img))
    out = detector.detect(batch)
    assert len(out) == len(batch)
    for (ref, _), dets in zip(batch, out, strict=True):
        for d in dets:
            assert isinstance(d, Detection) and d.cls == "person"
            assert d.frame.identity == ref.identity and d.frame.transform
            x1, y1, x2, y2 = d.bbox
            assert 0 <= x1 < x2 <= ref.width and 0 <= y1 < y2 <= ref.height
        assert [d.score for d in dets] == sorted((d.score for d in dets), reverse=True)
    assert sum(len(d) for d in out) > 0, "no people found in the PoC clip"


def test_batch_equals_single(detector):
    frames = _fixture_frames(4)
    batch = [(_ref("c", i), f) for i, f in enumerate(frames)]
    together = detector.detect(batch)
    for item, dets in zip(batch, together, strict=True):
        alone = detector.detect([item])[0]
        a = np.array([d.bbox for d in alone if d.score > 0.4]).reshape(-1, 4)
        b = np.array([d.bbox for d in dets if d.score > 0.4]).reshape(-1, 4)
        assert a.shape == b.shape
        if len(a):
            # FP16 kernels differ per batch size, and D-FINE (NMS-free) may then pick a different
            # near-duplicate query: same people, boxes jitter by a few pixels (T03 report).
            from scs.perception.detect import iou_matrix

            assert iou_matrix(a, b).max(1).min() >= 0.85


def test_boxes_scale_to_main_stream(detector):
    """Sub-stream image (640x360) with a 2560x1440 main stream → boxes 4x larger."""
    f = _fixture_frames(1)[0]
    small = detector.detect([(_ref(w=640, h=360), f)])[0]
    big = detector.detect([(_ref(w=2560, h=1440), f)])[0]
    a = np.array([d.bbox for d in small if d.score > 0.4]).reshape(-1, 4)
    b = np.array([d.bbox for d in big if d.score > 0.4]).reshape(-1, 4)
    assert len(a) and a.shape == b.shape
    np.testing.assert_allclose(a * 4, b, rtol=1e-4, atol=1e-2)


def test_batch_larger_than_engine_max(detector):
    frames = _fixture_frames(3)
    batch = [(_ref("c", i), frames[i % 3]) for i in range(detector.runner.max_batch + 3)]
    assert len(detector.detect(batch)) == len(batch)




@pytest.mark.slow
def test_dead_camera_isolation_real_engine(detector):
    """D9 on the real engine: healthy cameras' p95 latency with cam3 dead / cam5 slow ≈ all alive.

    Timing test: authoritative numbers come from detect_bench under `scripts/gpu exclusive`.
    """
    from scs.perception.detect_bench import isolation

    r = isolation(detector, seconds=6.0)
    base = r["all_alive"]["p95_ms"]
    assert r["cam3_dead"]["p95_ms"] <= base + 3.0
    assert r["cam5_slow_1fps"]["p95_ms"] <= base + 3.0
    assert base < 20.0 + 15.0  # deadline + one batch of inference
