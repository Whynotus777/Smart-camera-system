"""T04 GPU/data tests (5090 box only). Engines come from env vars; skipped if absent.

SCS_RTMPOSE_ENGINE=models/rtmpose/rtmpose-m_body7_b32_fp16.engine
SCS_COCO_DIR=<dir with val2017/ and person_keypoints_val2017.json>
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from scs.contracts import FrameRef, Track
from scs.perception.pose import TopDownPoseEstimator, crop_batch, crop_batch_np, crop_geometry

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("no CUDA", allow_module_level=True)

W, H = 2560, 1440


def _engine(var: str) -> str:
    p = os.environ.get(var)
    if not p or not Path(p).exists():
        pytest.skip(f"{var} not set")
    return p


def test_cuda_crop_matches_numpy():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    gs = [
        crop_geometry(b, (192, 256))
        for b in [(10, 20, 300, 700), (2400, 1000, 2560, 1440), (900, 100, 1100, 900)]
    ]
    a = crop_batch_np(img, gs)
    b = crop_batch(torch.from_numpy(img).cuda(), gs).cpu().numpy()
    np.testing.assert_allclose(a, b, atol=0.05)


def test_rtmpose_trt_batches_and_stays_in_frame():
    from scs.perception.pose_rtmpose import RTMPoseTRT

    be = RTMPoseTRT(_engine("SCS_RTMPOSE_ENGINE"), "rtmpose-m-body7@test")
    est = TopDownPoseEstimator(be)
    ref = FrameRef(camera_id="c", epoch=0, seq=0, frame_idx=0, ts=0.0, width=W, height=H)
    tracks = [
        Track(frame=ref, track_id=i, bbox=(50.0 * i, 100, 50.0 * i + 200, 600), score=0.9)
        for i in range(be.max_batch + 5)
    ]
    img = torch.randint(0, 255, (H, W, 3), dtype=torch.uint8, device="cuda")
    poses = est.estimate(ref, img, tracks)
    assert [p.track_id for p in poses] == [t.track_id for t in tracks]
    for p, t in zip(poses, tracks, strict=True):
        g = crop_geometry(t.bbox, be.input_wh)
        kp = np.array(p.keypoints)
        assert (kp[:, 0] >= g.center[0] - g.scale[0] / 2 - 1).all()
        assert (kp[:, 0] <= g.center[0] + g.scale[0] / 2 + 1).all()


@pytest.mark.data
def test_rtmpose_coco_wrist_pck_smoke():
    """200 COCO val images: wrist PCK@0.2 must be well above chance (catches BGR/RGB, decode bugs)."""
    from scs.perception.pose_eval import run_coco
    from scs.perception.pose_rtmpose import RTMPoseTRT

    d = os.environ.get("SCS_COCO_DIR")
    if not d:
        pytest.skip("SCS_COCO_DIR not set")
    est = TopDownPoseEstimator(
        RTMPoseTRT(_engine("SCS_RTMPOSE_ENGINE"), "rtmpose-m-body7@test"), stream_check="off", min_box_px=1.0
    )
    res = run_coco(
        est, Path(d) / "val2017", Path(d) / "person_keypoints_val2017.json", factors=(1,), limit=200
    )
    assert res["results"]["full"]["wrist"]["pck"] > 0.8
