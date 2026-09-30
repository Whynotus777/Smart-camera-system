"""Checks on the real converted datasets under the shared data root (`pytest -m data`).

Skipped in CI. Run on the 5090 box after `python -m eval.converters <id>`.
"""

import collections

import numpy as np
import pytest

from eval.canonical import converted_dir, data_root

pytestmark = pytest.mark.data


def _need(dataset):
    if not (converted_dir(dataset) / "labels").exists():
        pytest.skip(f"{dataset} not converted under {data_root()}")
    from eval.datasets import load_dataset

    return load_dataset(dataset)


def test_meva_every_converted_clip_has_a_split_and_sane_geometry():
    ds = _need("meva")
    ids = ds.clip_ids()
    placed = collections.Counter(ds.splits.split_for(c) for c in ids)
    assert None not in placed, f"converted clips missing from eval/splits/meva.json: {placed}"
    for c in ids[:: max(1, len(ids) // 25)]:
        lab = ds.clip(c).labels
        assert lab.fps == pytest.approx(30, abs=0.1) and lab.width == 1920 and lab.height in (1072, 1080)
        assert 250 < lab.duration_s < 330
        t = ds.clip(c).tracks()
        if len(t):
            assert (t.boxes[:, 2] <= lab.width).all() and (t.boxes[:, 3] <= lab.height).all()
        for e in lab.events:
            assert 0 <= e.t_start <= e.t_end <= lab.duration_s + 1


def test_meva_test_split_is_held_out_by_camera_and_site():
    ds = _need("meva")
    cams = {ds.clip(c).labels.camera_id for c in ds.clip_ids("test")}
    train_cams = {ds.clip(c).labels.camera_id for c in ds.clip_ids("train")}
    assert cams <= {"bus.G331", "bus.G508", "school.G421"} and not cams & train_cams
    assert not {ds.clip(c).labels.site_id for c in ds.clip_ids("train")} & {"bus"}


def test_poselift_real_files_parse():
    ds = _need("poselift")
    ids = ds.clip_ids()
    assert ids
    for c in ids[:10]:
        clip = ds.clip(c)
        t, y = clip.tracks(), clip.frame_labels()
        assert t.kps.shape[1:] == (17, 3) and np.isfinite(t.kps[..., :2]).all()
        assert y is None or t.frame_idx.max() < len(y)
