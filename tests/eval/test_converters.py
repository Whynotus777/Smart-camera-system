"""Converters + loader on tiny synthetic sources in a temp data root (no real data needed)."""

import json
import pickle

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from eval.canonical import ClipLabels, LabelEvent, converted_dir, read_labels, read_tracks  # noqa: E402
from eval.converters import (  # noqa: E402  # noqa: E402
    LicenseError,
    check_license,
    own,
    poselift,
    retails,
    smartspaces,
)
from eval.converters import meva as meva_conv  # noqa: E402
from eval.datasets import load_dataset  # noqa: E402
from eval.splits import ClipMeta, Splits, SplitSpec, generate  # noqa: E402

CLIP = "2018-03-05.14-00-00.14-05-00.school.G421"


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("SCS_DATA_ROOT", str(tmp_path))
    return tmp_path


def _meva_fixture(root):
    ann = root / "meva/annotations/meva-data-repo/annotation/DIVA-phase-2/MEVA/kitware/2018-03-05/14"
    ann.mkdir(parents=True)
    (ann / f"{CLIP}.types.yml").write_text(
        "- {'meta': 'x'}\n- {'types': {'cset3': {'person': 1.0}, 'id1': 5}}\n"
        "- {'types': {'cset3': {'other': 1.0}, 'id1': 6}}\n"
    )
    (ann / f"{CLIP}.activities.yml").write_text(
        "- {'act': {'act2': {'person_picks_up_object': 1.0}, 'actors': [{'id1': 6}, {'id1': 5}], 'id2': 1, "
        "'src_status': 'good', 'timespan': [{'tsr0': [30, 45]}]}}\n"
        "- {'act': {'act2': {'person_texts_on_phone': 1.0}, 'actors': [{'id1': 5}], 'id2': 2, "
        "'src_status': 'good', 'timespan': [{'tsr0': [60, 90]}]}}\n"
        "- {'act': {'act2': {'person_puts_down_object': 1.0}, 'actors': [{'id1': 5}], 'id2': 3, "
        "'src_status': 'not_good', 'timespan': [{'tsr0': [100, 110]}]}}\n"
    )
    # row 2: different key order + y2 overruns the frame; row 3: an object actor (dropped)
    (ann / f"{CLIP}.geom.yml").write_text(
        "- {'geom': {'g0': '10 20 110 220', 'id0': 1, 'id1': 5, 'keyframe': True, 'ts0': 30}}\n"
        "- {'geom': {'id1': 5, 'ts0': 31, 'g0': '11 20 111 1075', 'id0': 2, 'keyframe': False}}\n"
        "- {'geom': {'g0': '500 500 520 540', 'id0': 3, 'id1': 6, 'keyframe': True, 'ts0': 30}}\n"
    )
    meta = root / "meva/annotations/meva-data-repo/metadata"
    meta.mkdir(parents=True)
    (meta / "meva-clip-camera-and-time-table.txt").write_text(
        f"{CLIP} 2018-03-05.14-00-00 cam.krtd 3-421 self 0 0\n"
    )
    (root / "meva/MANIFEST.json").write_text(json.dumps({"use": "prod", "files": {}}))


def test_meva_converter(root):
    _meva_fixture(root)
    counts = meva_conv.convert(clips=[CLIP], do_probe=False, log=lambda *_: None)
    assert counts["clips"] == 1 and counts["person_picks_up_object"] == 1
    lab = read_labels(converted_dir("meva") / "labels" / f"{CLIP}.json")
    assert [e.type for e in lab.events] == ["item_pickup", "item_put_down"]
    e = lab.events[0]
    assert (e.t_start, e.t_end, e.track_id, e.source_label) == (1.0, 1.5, 5, "person_picks_up_object")
    assert lab.events[1].extra["src_status"] == "not_good"
    assert lab.extra["other_activities"][0]["name"] == "person_texts_on_phone"
    assert lab.view_group == "meva:3-421:2018-03-05.14-00-00" and lab.site_id == "school"
    assert lab.continuous and lab.supports.events and lab.supports.exhaustive and lab.actor_ids is None
    t = read_tracks(converted_dir("meva") / "tracks" / f"{CLIP}.parquet")
    assert t.track_id.tolist() == [5, 5] and t.frame_idx.tolist() == [30, 31]
    assert t.boxes[1].tolist() == [11, 20, 111, 1072]  # clipped to the 1072-px frame
    assert np.isnan(t.kps).all()


def test_meva_geom_parser_fails_loudly(tmp_path):
    p = tmp_path / "x.geom.yml"
    p.write_text("- {'geom': {'g0': 'garbage', 'id1': 5, 'ts0': 1}}\n")
    with pytest.raises(ValueError, match="unparsed"):
        meva_conv.parse_geom(p)


def test_smartspaces_converter(root):
    s = root / "smartspaces/raw/MTMC_Tracking_2024/test/scene_073"
    (s / "camera_0700").mkdir(parents=True)
    (s / "ground_truth.txt").write_text(
        "700 1 0 10 10 50 100 1.0 2.0\n700 1 1 12 10 50 100 1 2\n701 2 0 5 5 5 5 0 0\n"
    )
    files = {
        "MTMC_Tracking_2024/test/scene_073/camera_0700/video.mp4": {},
        "MTMC_Tracking_2024/test/scene_074/camera_0800/video.mp4": {},
    }
    (root / "smartspaces/MANIFEST.json").write_text(json.dumps({"use": "prod", "files": files, "notes": ""}))
    c = smartspaces.convert(log=lambda *_: None)
    assert c["clips"] == 2 and c["clips_with_gt"] == 1 and c["gt_boxes"] == 2
    t = read_tracks(converted_dir("smartspaces") / "tracks" / "scene_073.camera_0700.parquet")
    assert t.boxes.tolist() == [[10, 10, 60, 110], [12, 10, 62, 110]] and t.frame_idx.tolist() == [0, 1]
    lab = read_labels(converted_dir("smartspaces") / "labels" / "scene_074.camera_0800.json")
    assert not lab.supports.gt_tracks and lab.synthetic and lab.extra["scene_kind"] == "retail"


def _kp(v):
    return [[v, v, 0.9]] * 17


@pytest.mark.parametrize("layout", ["list_of_dicts", "dict_by_pid", "tuples"])
def test_poselift_pickle_layouts(root, layout):
    raw = root / "poselift/raw/Pickle_files"
    raw.mkdir(parents=True)
    frames = {}
    for f in range(4):
        recs = [(1, [100, 100, 50, 200], _kp(120 + f)), (2, [400, 100, 50, 200], _kp(420))]
        if layout == "list_of_dicts":
            frames[f] = [{"Person ID": i, "Bounding Box": b, "Keypoints": k} for i, b, k in recs]
        elif layout == "dict_by_pid":
            frames[f] = {i: {"bbox": b, "keypoints": k} for i, b, k in recs}
        else:
            frames[f] = recs
    with open(raw / "3_7.pkl", "wb") as fh:
        pickle.dump(frames, fh)
    np.save(raw / "3_7.npy", np.array([0, 1, 1, 0, 0]))
    (root / "poselift/MANIFEST.json").write_text(json.dumps({"use": "R&D", "files": {}}))
    c = poselift.convert(log=lambda *_: None)
    assert c == {"clips": 1, "events": 1, "frames": 5, "official_None": 1}
    out = converted_dir("poselift")
    t = read_tracks(out / "tracks" / "3_7.parquet")
    assert len(t) == 8 and t.boxes[0].tolist() == [100, 100, 150, 300] and t.camera_id == "poselift.C3"
    lab = read_labels(out / "labels" / "3_7.json")
    assert [(e.type, e.t_start, e.t_end) for e in lab.events] == [("shoplifting", 1 / 15, 3 / 15)]
    assert np.load(out / "frame_labels" / "3_7.npy").tolist() == [0, 1, 1, 0, 0]


def test_poselift_stgnf_json_and_official_split(root):
    base = root / "poselift/raw/STG-NF/PoseLift/pose"
    for split in ("train", "test"):
        (base / split).mkdir(parents=True)
        (base / split / f"C1_{1 if split == 'train' else 2}_alphapose_tracked_person.json").write_text(
            json.dumps({"1": {"0": {"keypoints": sum(_kp(10), []), "scores": 0.9}}})
        )
    (root / "poselift/MANIFEST.json").write_text(json.dumps({"use": "R&D", "files": {}}))
    found = poselift.discover(root / "poselift/raw")
    assert {c: r["split"] for c, r in found.items()} == {"1_1": "train", "1_2": "test"}


def test_retails_refuses_while_pending_but_parser_works(root, tmp_path):
    with pytest.raises(LicenseError, match="pending"):
        retails.convert()
    rows = list(
        retails.iter_json(
            [[0, 7, sum(_kp(1), [])], {"frame_id": 1, "person_id": 7, "keypoints": sum(_kp(2), [])}]
        )
    )
    assert [(f, p) for f, p, _, _ in rows] == [(0, 7), (1, 7)]
    assert retails.subset_of(tmp_path / "RetailS/test_staged/C1_2.json") == "test_staged"


def test_license_gate_reads_manifest(root):
    (root / "meva").mkdir()
    (root / "meva/MANIFEST.json").write_text(json.dumps({"use": "blocked"}))
    with pytest.raises(LicenseError):
        check_license("meva")
    with pytest.raises(LicenseError, match="not registered"):
        check_license("ucf_crime")


def test_take_log_and_sim_stub(root):
    log = root / "takes.csv"
    log.write_text(
        "take_id,actor_id,subtype,type,t_start,t_end,camera_id,visible,video\n"
        "t1,a1,pocket,item_to_clothing,3,6,cam1,observed,quick_capture/raw/s1_cam1.mp4\n"
        "t1,a1,pocket,item_to_clothing,3,6,cam2,not_observed,quick_capture/raw/s1_cam2.mp4\n"
        "t2,a1,,item_returned,10,12,cam1,observed,quick_capture/raw/s1_cam1.mp4\n"
    )
    c = own.convert_quick_capture(log, log=lambda *_: None)
    assert c == {"clips": 2, "events": 3}
    lab = read_labels(converted_dir("quick_capture") / "labels" / "s1_cam2.json")
    assert (
        lab.events[0].visible == "not_observed" and lab.actor_ids == ["a1"] and lab.label_source == "script"
    )
    with pytest.raises(NotImplementedError):
        own.convert_sim()


def test_unknown_label_type_rejected():
    with pytest.raises(ValueError, match="unknown label type"):
        LabelEvent(type="theft_detected", t_start=0, t_end=1)


def test_loader_splits_derivatives_and_windows(root, tmp_path):
    from eval.canonical import LabelSupport, TrackTable, write_labels, write_tracks

    out = converted_dir("toy")
    for clip, gid in (("a", "a"), ("b", "b"), ("a_emu", "a")):
        write_labels(
            out / "labels" / f"{clip}.json",
            ClipLabels(
                clip_id=clip,
                fps=10,
                label_source="human",
                group_id=gid,
                camera_id="c",
                supports=LabelSupport(frame_labels=True),
            ),
        )
        n = 30
        write_tracks(
            out / "tracks" / f"{clip}.parquet",
            TrackTable(
                "c",
                np.arange(n),
                np.arange(n) / 10,
                np.ones(n, dtype=int),
                np.tile([0, 0, 10, 10], (n, 1)),
                np.ones(n),
                np.random.default_rng(0).random((n, 17, 3)),
            ),
        )
        (out / "frame_labels").mkdir(exist_ok=True)
        np.save(out / "frame_labels" / f"{clip}.npy", (np.arange(n) >= 20).astype(np.uint8))
    sd = tmp_path / "splits"
    sd.mkdir()
    Splits(
        generate(
            "toy",
            [ClipMeta("a", camera_id="A"), ClipMeta("b", camera_id="B")],
            SplitSpec(holdout={"test": {"cameras": ["A"]}}),
        )
    ).save(sd)
    from eval.converters import base

    base.APPROVAL["toy"] = "own"
    try:
        ds = load_dataset("toy", split_dir=sd)
        assert ds.clip_ids("test") == ["a", "a_emu"]  # derivative resolved through its group
        assert ds.clip_ids("train") == ["b"]
        ws = list(ds.windows("test", window=24, stride=3))
        assert len(ws) == 2 * 3 and ws[0].kps.shape == (24, 17, 3)
        assert ws[0].frame_idx[-1] == 23 and ws[0].label == 1
        assert ds.version()["split_digest"] == ds.splits.digest
    finally:
        del base.APPROVAL["toy"]
