"""T04 CPU tests: crop geometry, frame identity, gating, smoothing, causality, PoseLift adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scs.contracts import FrameRef, Pose, SiteConfig, Track
from scs.perception.base import PoseEstimator
from scs.perception.pose import (
    CropGeometry,
    FrameIdentityError,
    TopDownPoseEstimator,
    crop_batch_np,
    crop_geometry,
)
from scs.perception.pose_export import CropExporter, hand_boxes
from scs.perception.pose_gate import gate_tracks
from scs.perception.pose_live import LivePosePipeline
from scs.perception.pose_poselift import stgnf_normalize, to_poselift
from scs.perception.pose_smooth import KpMask, LivePoseSmoother, offline_interpolate_smooth

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures"
W, H = 2560, 1440


def _read(name: str, cls):
    return [cls.model_validate_json(line) for line in (FIX / name).read_text().splitlines()]


def _zones():
    return SiteConfig.from_yaml(ROOT / "configs" / "site.example.yaml").cameras[0].zones


def _ref(seq: int, fps: float = 10.0, epoch: int = 0, **kw) -> FrameRef:
    return FrameRef(
        camera_id="cam1",
        epoch=epoch,
        seq=seq,
        frame_idx=seq,
        ts=1e9 + seq / fps,
        ts_mono=100 + seq / fps,
        width=W,
        height=H,
        **kw,
    )


class ArgmaxBackend:
    """Fake model: every keypoint = location of the brightest pixel in the crop (conf 0.9)."""

    model_id = "fake-argmax@0"
    input_wh = (192, 256)
    max_batch = 4

    def __init__(self) -> None:
        self.calls: list[int] = []

    def infer(self, crops) -> np.ndarray:
        crops = np.asarray(crops)
        self.calls.append(len(crops))
        out = np.zeros((len(crops), 17, 3))
        for i, c in enumerate(crops):
            y, x = np.unravel_index(np.argmax(c.sum(0)), c.shape[1:])
            out[i, :, 0], out[i, :, 1], out[i, :, 2] = x, y, 0.9
        return out


# ---------------------------------------------------------------- geometry


def test_crop_geometry_keeps_aspect_and_round_trips():
    g = crop_geometry((1000, 400, 1200, 1000), (192, 256))
    assert g.scale[0] / g.scale[1] == pytest.approx(192 / 256)
    assert g.scale[1] == pytest.approx(600 * 1.25)
    pts = np.array([[1000.0, 400.0], [1234.5, 876.5]])
    np.testing.assert_allclose(g.to_frame(g.to_input(pts)), pts)
    np.testing.assert_allclose(g.to_input(np.array([[1100.0, 700.0]])), [[96, 128]])


def test_numpy_crop_samples_the_right_pixels_and_pads_with_zeros():
    img = np.zeros((H, W, 3), np.uint8)
    img[700, 1100] = 255
    g = CropGeometry((1100.0, 700.0), (192.0, 256.0), (192, 256))  # 1:1 scale
    c = crop_batch_np(img, [g])[0]
    assert np.unravel_index(np.argmax(c.sum(0)), c.shape[1:]) == (128, 96)
    edge = crop_batch_np(img, [CropGeometry((0.0, 0.0), (192.0, 256.0), (192, 256))])[0]
    assert edge[:, :100, :80].max() == 0  # outside the frame


def test_torch_crop_matches_numpy():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    gs = [crop_geometry(b, (192, 256)) for b in [(10, 20, 300, 700), (2400, 1000, 2560, 1440)]]
    a = crop_batch_np(img, gs)
    from scs.perception.pose import crop_batch

    b = crop_batch(torch.from_numpy(img), gs).numpy()
    np.testing.assert_allclose(a, b, atol=0.05)  # float32 grid rounding


# ---------------------------------------------------------------- estimator + identity


def test_estimator_is_a_pose_estimator_and_maps_back_to_main_stream_pixels():
    tracks = _read("tracks.jsonl", Track)
    poses = _read("poses.jsonl", Pose)
    est = TopDownPoseEstimator(ArgmaxBackend())
    assert isinstance(est, PoseEstimator)
    errs = []
    for t, p in zip(tracks, poses, strict=True):
        img = np.zeros((H, W, 3), np.uint8)
        wx, wy = p.keypoints[10][:2]
        img[round(wy) - 2 : round(wy) + 3, round(wx) - 2 : round(wx) + 3] = 255
        geom = crop_geometry(t.bbox, (192, 256))
        u, v = geom.to_input(np.array([wx, wy]))
        if not (2 <= u < 190 and 2 <= v < 254):
            continue  # reaching wrist outside the padded box: top-down can't see it (see T04 report)
        (out,) = est.estimate(t.frame, img, [t])
        assert out.frame.identity == t.frame.identity and out.track_id == t.track_id
        assert len(out.keypoints) == 17 and all(0 <= k[2] <= 1 for k in out.keypoints)
        errs.append(np.hypot(out.keypoints[10][0] - wx, out.keypoints[10][1] - wy))
    # crop scale is ~2.5 frame px per input px here, so the argmax lands within ~2 input px
    assert len(errs) > 150 and np.max(errs) < 6.0


def test_estimator_batches_by_max_batch():
    be = ArgmaxBackend()
    tracks = [
        Track(frame=_ref(0), track_id=i, bbox=(100 + 50 * i, 100, 140 + 50 * i, 300), score=0.9)
        for i in range(10)
    ]
    out = TopDownPoseEstimator(be).estimate(_ref(0), np.zeros((H, W, 3), np.uint8), tracks)
    assert be.calls == [4, 4, 2] and [p.track_id for p in out] == list(range(10))


@pytest.mark.parametrize(
    "bad",
    [
        {"seq": 1},
        {"epoch": 1},
        {"camera_id": "cam2"},
    ],
)
def test_frame_identity_mismatch_raises(bad):
    frame = _ref(0)
    t = Track(frame=frame.model_copy(update=bad), track_id=1, bbox=(100, 100, 200, 400), score=0.9)
    with pytest.raises(FrameIdentityError):
        TopDownPoseEstimator(ArgmaxBackend()).estimate(frame, np.zeros((H, W, 3), np.uint8), [t])


def test_sub_stream_or_transformed_or_wrong_size_image_is_rejected():
    est = TopDownPoseEstimator(ArgmaxBackend())
    for frame, shape in [
        (_ref(0, stream="sub"), (H, W, 3)),
        (_ref(0, transform="resize640"), (H, W, 3)),
        (_ref(0), (360, 640, 3)),
    ]:
        t = Track(frame=frame, track_id=1, bbox=(100, 100, 200, 400), score=0.9)
        with pytest.raises(FrameIdentityError):
            est.estimate(frame, np.zeros(shape, np.uint8), [t])


# ---------------------------------------------------------------- gating


def test_gate_uses_expanded_zones_and_skips_lost():
    zones = _zones()
    f = _ref(0)
    near_shelf = Track(frame=f, track_id=1, bbox=(0.46 * W, 0.3 * H, 0.50 * W, 0.6 * H), score=0.9)
    mid_aisle = Track(frame=f, track_id=2, bbox=(0.50 * W, 0.05 * H, 0.54 * W, 0.2 * H), score=0.9)
    lost = near_shelf.model_copy(update={"track_id": 3, "state": "lost"})
    assert [t.track_id for t in gate_tracks([near_shelf, mid_aisle, lost], zones)] == [1]
    assert [t.track_id for t in gate_tracks([near_shelf], zones, expand=0.0)] == []


def test_fixture_tracks_are_gated_most_of_the_time():
    tracks = _read("tracks.jsonl", Track)
    gated = gate_tracks(tracks, _zones())
    assert 0.5 * len(tracks) < len(gated) <= len(tracks)


# ---------------------------------------------------------------- smoothing


def _pose(seq: int, xy, conf=0.9, tid=1, **kw) -> Pose:
    xy = np.broadcast_to(np.asarray(xy, float), (17, 2))
    c = np.broadcast_to(np.asarray(conf, float), (17,))
    return Pose(
        frame=_ref(seq, **kw),
        track_id=tid,
        model_id="m",
        keypoints=[(float(x), float(y), float(cc)) for (x, y), cc in zip(xy, c, strict=True)],
    )


def test_one_euro_reduces_jitter_on_a_still_person():
    rng = np.random.default_rng(1)
    base = rng.uniform(500, 900, (17, 2))
    sm = LivePoseSmoother()
    raw, out = [], []
    for i in range(60):
        r = sm.update(_pose(i, base + rng.normal(0, 3, (17, 2))))
        raw.append(np.array(r.raw.keypoints)[:, :2])
        out.append(np.array(r.smoothed.keypoints)[:, :2])
    assert np.std(np.array(out[10:]) - base) < 0.6 * np.std(np.array(raw[10:]) - base)


def test_gap_fill_is_bounded_by_elapsed_time_not_frames():
    for fps in (5.0, 10.0, 30.0):
        sm = LivePoseSmoother(max_gap_s=0.5)
        sm.update(_pose(0, (100, 100), fps=fps))
        masks, confs = [], []
        for i in range(1, int(fps) + 1):  # 1 s of low-confidence frames
            r = sm.update(_pose(i, (999, 999), conf=0.05, fps=fps))
            masks.append(r.mask[0])
            confs.append(r.smoothed.keypoints[0][2])
            if r.mask[0] is KpMask.IMPUTED:
                assert r.smoothed.keypoints[0][:2] == pytest.approx((100, 100))
        n_imp = sum(m is KpMask.IMPUTED for m in masks)
        assert n_imp == int(0.5 * fps + 1e-9), fps  # imputed only while elapsed <= 0.5 s
        assert confs[0] < 0.9 and all(a >= b for a, b in zip(confs[:n_imp], confs[1:n_imp], strict=False))


def test_smoother_resets_on_reconnect_and_long_gap():
    sm = LivePoseSmoother()
    sm.update(_pose(0, (100, 100)))
    r = sm.update(_pose(0, (500, 500), epoch=1))  # new epoch, seq restarts
    assert r.smoothed.keypoints[0][:2] == pytest.approx((500, 500))
    r = sm.update(_pose(20, (900, 900), epoch=1))  # 2 s later
    assert r.smoothed.keypoints[0][:2] == pytest.approx((900, 900))


def test_offline_variant_is_named_and_interpolates_using_the_future():
    poses = [_pose(0, (0, 0)), _pose(1, (0, 0), conf=0.0), _pose(2, (20, 20))]
    kp, mask = offline_interpolate_smooth(poses, window=1)
    assert kp[1, 0, :2] == pytest.approx((10, 10)) and mask[1, 0] == KpMask.IMPUTED


# ---------------------------------------------------------------- causality (AGENTS.md rule 11)


def _scene(n: int, seed: int):
    """Frames + tracks for one person; frame content is random so every pixel matters."""
    rng = np.random.default_rng(seed)
    frames, tracks = [], []
    for i in range(n):
        img = rng.integers(0, 40, (H, W, 3), dtype=np.uint8)
        cx, cy = 0.40 * W + 3 * i + rng.normal(0, 4), 0.45 * H + rng.normal(0, 4)
        img[int(cy) - 3 : int(cy) + 3, int(cx) - 3 : int(cx) + 3] = 255
        f = _ref(i)
        frames.append((f, img))
        tracks.append([Track(frame=f, track_id=7, bbox=(cx - 100, cy - 250, cx + 100, cy + 250), score=0.9)])
    return frames, tracks


def _run(frames, tracks):
    pipe = LivePosePipeline(TopDownPoseEstimator(ArgmaxBackend()), _zones())
    out = []
    for (f, img), trs in zip(frames, tracks, strict=True):
        out.append([(s.raw.keypoints, s.smoothed.keypoints, s.mask) for s in pipe.step(f, img, trs)])
    return out


def test_live_path_never_reads_future_frames():
    n, k = 30, 17
    frames, tracks = _scene(n, seed=0)
    full = _run(frames, tracks)
    # (1) prefix invariance: outputs for frames <= k don't change when every later frame changes
    alt_frames, alt_tracks = _scene(n, seed=99)
    mixed = _run(frames[: k + 1] + alt_frames[k + 1 :], tracks[: k + 1] + alt_tracks[k + 1 :])
    assert mixed[: k + 1] == full[: k + 1]
    assert mixed[k + 1 :] != full[k + 1 :]  # sanity: the future really was different
    # (2) running on the truncated stream gives the same outputs: nothing waits for later input
    assert _run(frames[: k + 1], tracks[: k + 1]) == full[: k + 1]
    # (3) structurally: one output per step, emitted before the next frame is produced
    seen: list[int] = []

    def feed():
        for i, item in enumerate(zip(frames, tracks, strict=True)):
            seen.append(i)
            yield item

    pipe = LivePosePipeline(TopDownPoseEstimator(ArgmaxBackend()), _zones())
    for (f, img), trs in feed():
        pipe.step(f, img, trs)
        assert seen[-1] == f.seq  # the pipeline returned before frame seq+1 was requested


def test_poselift_live_mode_is_causal():
    rng = np.random.default_rng(3)
    poses = [_pose(i, rng.uniform(100, 1000, (17, 2))) for i in range(40)]
    full = to_poselift(poses, mode="live")
    part = to_poselift(poses[:25], mode="live", t0=100.0)
    assert all(full[k] == part[k] for k in part)


# ---------------------------------------------------------------- PoseLift adapter


def test_to_poselift_rescales_resamples_and_smooths():
    poses = [_pose(i, (1280.0 + 64 * i, 720.0)) for i in range(20)]  # 10 fps, moving right
    out = to_poselift(poses, mode="offline")
    ks = sorted(out)
    assert ks[0] == 0 and ks[-1] == 28  # 1.9 s at 15 fps
    mid = out[14][1]["keypoints"][0]
    # 8-frame centered window spans k-3..k+4, so its mean sits at k+0.5 (15 fps grid, 640 px/s @2560, x0.75)
    assert mid[0] == pytest.approx((1280 + 640 * 14.5 / 15) * 0.75, rel=1e-6)
    assert mid[1] == pytest.approx(540.0)
    assert len(out[0][1]["bbox"]) == 4


def test_to_poselift_does_not_bridge_long_gaps_and_rejects_non_16x9():
    poses = [_pose(i, (100.0, 100.0)) for i in (0, 1, 2, 12, 13)]  # 1 s hole
    out = to_poselift(poses)
    assert not any(4 <= k <= 16 for k in out)
    f = _ref(0).model_copy(update={"width": 2560, "height": 1920})
    with pytest.raises(ValueError):
        to_poselift([Pose(frame=f, track_id=1, model_id="m", keypoints=[(1, 1, 1)] * 17)])


def test_stgnf_normalize_matches_reference_formula():
    rng = np.random.default_rng(0)
    seg = rng.uniform(0, 1000, (24, 17, 3))
    out = stgnf_normalize(seg)
    ref = seg.copy()
    ref[..., :2] = (seg[..., :2] - seg[..., :2].mean(axis=(0, 1))) / seg[..., 1].std()
    np.testing.assert_allclose(out, ref)


# ---------------------------------------------------------------- crop export


def test_crop_export_is_off_by_default_and_guarded(tmp_path):
    assert CropExporter(root=tmp_path)(np.zeros((H, W, 3), np.uint8), [], []) == 0
    with pytest.raises(ValueError):
        CropExporter(root=tmp_path, enabled=True)
    with pytest.raises(PermissionError):
        CropExporter(root=tmp_path, enabled=True, source_kind="store")


def test_crop_export_writes_person_and_hand_crops(tmp_path):
    pytest.importorskip("cv2")
    t = _read("tracks.jsonl", Track)[0]
    p = _read("poses.jsonl", Pose)[0]
    ex = CropExporter(root=tmp_path, enabled=True, source_kind="sim", group_id="fixture")
    n = ex(np.full((H, W, 3), 128, np.uint8), [t], [p])
    ex.close()
    assert n == 1 + len(hand_boxes(p)) == 3
    files = sorted(x.name for x in (tmp_path / "cam1" / "0").iterdir())
    assert files == ["0_1_lwrist.jpg", "0_1_person.jpg", "0_1_rwrist.jpg"]
    row = json.loads((tmp_path / "cam1" / "index.jsonl").read_text())
    assert (row["camera_id"], row["epoch"], row["seq"], row["group_id"]) == ("cam1", 0, 0, "fixture")
