"""public_pose / sim_transfer / smartspaces_track on tiny synthetic converted datasets."""

import json

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from eval.canonical import (  # noqa: E402
    ClipLabels,
    LabelSupport,
    TrackTable,
    converted_dir,
    write_labels,
    write_tracks,
)
from eval.models.dummy import ConstantBehavior, RandomBehavior, WristMotionBehavior  # noqa: E402
from eval.pose_bench import PublicPose, _poses  # noqa: E402
from eval.splits import ClipMeta, Splits, SplitSpec, generate  # noqa: E402
from eval.suites.base import ModelSpec, RunContext  # noqa: E402
from eval.suites.sim_transfer import SimTransfer  # noqa: E402
from eval.suites.smartspaces_track import SmartSpacesTrack  # noqa: E402


class Oracle:
    """Scores 1 when the window's last frame is labeled positive (reads a side table)."""

    model_id, window, lineage = "oracle", 4, {"train": []}

    def __init__(self, pos):
        self.pos = pos

    def score_windows(self, kps):
        # kps[..., 0] carries the frame index in this fixture
        last = kps[:, -1, 0, 0].astype(int)
        return np.isin(last, self.pos).astype(float)


def _pose_ds(root, n_clips=6, n=40):
    out = converted_dir("poselift")
    metas, pos = [], []
    for c in range(n_clips):
        cid = f"{c % 3 + 1}_{c}"
        y = np.zeros(n, np.uint8)
        y[20:30] = 1
        pos = list(range(20, 30))
        kps = np.zeros((n, 17, 3))
        kps[:, :, 0] = np.arange(n)[:, None]
        kps[:, :, 2] = 0.9
        write_tracks(
            out / "tracks" / f"{cid}.parquet",
            TrackTable(
                f"poselift.C{c % 3 + 1}",
                np.arange(n),
                np.arange(n) / 15,
                np.ones(n, int),
                np.tile([0, 0, 10, 10], (n, 1)),
                np.ones(n),
                kps,
            ),
        )
        (out / "frame_labels").mkdir(exist_ok=True)
        np.save(out / "frame_labels" / f"{cid}.npy", y)
        write_labels(
            out / "labels" / f"{cid}.json",
            ClipLabels(
                clip_id=cid,
                fps=15,
                label_source="human",
                dataset="poselift",
                camera_id=f"poselift.C{c % 3 + 1}",
                width=1920,
                height=1080,
                n_frames=n,
                supports=LabelSupport(frame_labels=True, keypoints=True),
            ),
        )
        metas.append(ClipMeta(cid, camera_id=f"C{c % 3 + 1}"))
    (root / "poselift").mkdir(exist_ok=True)
    (root / "poselift/MANIFEST.json").write_text(json.dumps({"use": "R&D"}))
    sd = root / "splits"
    sd.mkdir(exist_ok=True)
    Splits(generate("poselift", metas, SplitSpec(fractions={"test": 1.0}))).save(sd)
    return sd, pos


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("SCS_DATA_ROOT", str(tmp_path))
    return tmp_path


def _ctx(root, sd, models, **kw):
    return RunContext(models=models, data_root=root, split_dir=sd, bootstrap=50, log=lambda *_: None, **kw)


def _spec(role, obj):
    return ModelSpec(role, "test", obj, obj.model_id, getattr(obj, "lineage", None))


def test_public_pose_oracle_and_constant(root):
    sd, pos = _pose_ds(root)
    r = PublicPose().run(_ctx(root, sd, {"behavior": _spec("behavior", Oracle(pos))}))
    # Oracle is exact on every frame >= window-1; frames 0-2 (no window) get the clip min (0)
    assert r.headline["auc_roc"].value == pytest.approx(1.0)
    assert r.headline["auc_roc"].n["clips"] == 6 and r.params["causal"] is True
    c = PublicPose().run(_ctx(root, sd, {"behavior": _spec("behavior", ConstantBehavior())}))
    assert c.headline["auc_roc"].value == pytest.approx(0.5)


def test_public_pose_needs_a_model_and_data(root):
    assert PublicPose().run(_ctx(root, None, {})).status == "unavailable"
    r = PublicPose().run(_ctx(root, None, {"behavior": _spec("behavior", ConstantBehavior())}))
    assert r.status == "unavailable" and "not" in r.notes[0]


@pytest.mark.parametrize("model", [ConstantBehavior(), RandomBehavior(3), WristMotionBehavior()])
def test_batch_path_matches_behavior_model_interface(model):
    rng = np.random.default_rng(0)
    kps = rng.random((24, 17, 3)) * 100
    kps[..., 2] = rng.random((24, 17))
    poses = _poses("cam", 1, np.arange(24), np.arange(24) / 15, kps, 1920, 1080, "m")
    via_protocol = model.score(poses).score
    via_batch = float(model.score_windows(np.array([p.keypoints for p in poses])[None])[0])
    assert via_protocol == pytest.approx(via_batch)
    assert 0 <= via_batch <= 1


def test_sim_transfer_paired_deltas_and_leak_guard(root):
    sd, pos = _pose_ds(root)
    models = {"real": _spec("real", ConstantBehavior()), "real_sim": _spec("real_sim", Oracle(pos))}
    r = SimTransfer().run(_ctx(root, sd, models))
    d = r.metrics["paired_deltas"]["delta_auc[real_sim - real]"]
    assert d.value == pytest.approx(0.5) and d.ci95[0] > 0
    assert r.metrics["test_clips"] == 6 and len(r.metrics["test_clip_digest"]) == 16
    leaky = Oracle(pos)
    leaky.lineage = {"train": [{"dataset": "poselift", "split": "test"}]}
    with pytest.raises(ValueError, match="leak"):
        SimTransfer().run(_ctx(root, sd, {"real": _spec("real", ConstantBehavior()), "x": _spec("x", leaky)}))


def _ss(root, kinds):
    out = converted_dir("smartspaces")
    metas = []
    for i, (scene, kind) in enumerate(kinds):
        cid = f"{scene}.camera_{i}"
        n = 20
        boxes = np.tile([10.0, 10, 50, 90], (n, 1))
        write_tracks(
            out / "tracks" / f"{cid}.parquet",
            TrackTable(
                cid,
                np.arange(n),
                np.arange(n) / 30,
                np.full(n, 3),
                boxes,
                np.ones(n),
                np.full((n, 17, 3), np.nan),
            ),
        )
        write_labels(
            out / "labels" / f"{cid}.json",
            ClipLabels(
                clip_id=cid,
                fps=30,
                label_source="script",
                dataset="smartspaces",
                camera_id=cid,
                view_group=f"smartspaces:{scene}",
                synthetic=True,
                supports=LabelSupport(gt_tracks=True),
                extra={"scene_kind": kind},
            ),
        )
        metas.append(ClipMeta(cid, camera_id=cid, view_group=f"smartspaces:{scene}"))
    (root / "smartspaces").mkdir(exist_ok=True)
    (root / "smartspaces/MANIFEST.json").write_text(json.dumps({"use": "prod"}))
    sd = root / "splits"
    sd.mkdir(exist_ok=True)
    Splits(generate("smartspaces", metas, SplitSpec(fractions={"test": 1.0}))).save(sd)
    return sd


def test_smartspaces_track_gt_as_predictions_is_perfect(root):
    sd = _ss(root, [("scene_073", "retail"), ("scene_900", "non_retail")])
    r = SmartSpacesTrack().run(_ctx(root, sd, {}, predictions=converted_dir("smartspaces")))
    assert r.headline["idf1"].value == 1.0 and r.headline["det_ap50"].value == pytest.approx(1.0)
    assert r.metrics["clips"] == ["scene_073.camera_0"]  # retail only when available
    assert not any("fallback" in lab for lab in r.labels)


def test_smartspaces_track_falls_back_to_non_retail_flagged(root):
    sd = _ss(root, [("scene_900", "non_retail")])
    r = SmartSpacesTrack().run(_ctx(root, sd, {}, predictions=converted_dir("smartspaces")))
    assert r.status == "partial" and r.params["fallback_non_retail"]
    assert any("fallback: non-retail" in lab for lab in r.labels)
