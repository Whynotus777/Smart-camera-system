"""Split generator: groups never span splits, hold-outs are honored, leaks get excluded.

`test_committed_splits_*` checks every frozen file in eval/splits/, so a hand edit that
puts a derivative group in two splits fails CI.
"""

import json
import random

import pytest

from eval.splits import SPLITS_DIR, ClipMeta, Splits, SplitSpec, generate, validate


def _random_clips(seed, n=300):
    rng = random.Random(seed)  # noqa: S311 - test data
    clips = []
    for i in range(n):
        g = f"g{rng.randrange(n // 3)}"  # ~3 derivatives per group
        clips.append(
            ClipMeta(
                clip_id=f"c{i:04d}",
                group_id=g,
                camera_id=f"cam{rng.randrange(8)}",
                site_id=f"s{rng.randrange(3)}",
                view_group=f"v{rng.randrange(n // 2)}" if rng.random() < 0.5 else None,
                actor_ids=tuple(f"a{rng.randrange(40)}" for _ in range(rng.randrange(0, 2))) or None,
            )
        )
    return clips


def _split_of(d):
    return {c: s for s, cs in d["splits"].items() for c in cs}


@pytest.mark.parametrize("seed", range(10))
def test_no_group_or_actor_spans_splits(seed):
    clips = _random_clips(seed)
    spec = SplitSpec(fractions={"train": 0.7, "val": 0.15, "test": 0.15})
    d = generate("toy", clips, spec)
    where = _split_of(d)
    by_group, by_actor = {}, {}
    for m in clips:
        s = where.get(m.clip_id)
        assert s is not None or m.clip_id in d["excluded"]
        if s is None:
            continue
        assert by_group.setdefault(m.gid, s) == s, f"group {m.gid} spans splits"
        for a in m.actor_ids or ():
            assert by_actor.setdefault(a, s) == s, f"actor {a} spans splits"


def test_validate_rejects_group_spanning_splits():
    d = {
        "splits": {"train": ["a"], "val": [], "test": ["b"]},
        "group_of": {"a": "G", "b": "G"},
        "excluded": {},
    }
    with pytest.raises(ValueError, match="spans"):
        validate(d)
    with pytest.raises(ValueError, match="both"):
        validate({"splits": {"train": ["a"], "test": ["a"]}})


def test_holdout_by_camera_and_site_with_leak_exclusion():
    clips = [
        ClipMeta("s1_camA_t0", camera_id="A", site_id="s1", view_group="s1:t0"),
        ClipMeta("s1_camB_t0", camera_id="B", site_id="s1", view_group="s1:t0"),  # same time as held-out A
        ClipMeta("s1_camB_t1", camera_id="B", site_id="s1", view_group="s1:t1"),
        ClipMeta("s2_camC_t0", camera_id="C", site_id="s2", view_group="s2:t0"),
        ClipMeta("s1_camD_t2", camera_id="D", site_id="s1"),
        ClipMeta("s1_camA_t3", group_id="G1", camera_id="A", site_id="s1"),
        ClipMeta("emu_of_t3", group_id="G1", camera_id="A_emulated", site_id="s1"),  # derivative
    ]
    spec = SplitSpec(holdout={"test": {"sites": ["s2"], "cameras": ["A"]}, "val": {"cameras": ["D"]}})
    d = generate("toy", clips, spec)
    where = _split_of(d)
    assert where["s1_camA_t0"] == "test" and where["s2_camC_t0"] == "test" and where["s1_camA_t3"] == "test"
    assert where["s1_camD_t2"] == "val" and where["s1_camB_t1"] == "train"
    # simultaneous view of a held-out camera: never trained on
    assert "s1_camB_t0" in d["excluded"]
    # derivative (same group) of a held-out clip follows it into test
    assert where["emu_of_t3"] == "test"
    s = Splits(d)
    assert s.split_for("unlisted_crop", group_id="G1") == "test"
    assert s.split_for("s1_camB_t0") is None


def test_assignment_is_stable_when_data_grows():
    clips = _random_clips(1, n=200)
    spec = SplitSpec(fractions={"train": 0.8, "val": 0.1, "test": 0.1}, link_actors=False, link_views=False)
    a = _split_of(generate("toy", clips, spec))
    more = clips + [ClipMeta(f"new{i}", group_id=f"newg{i}") for i in range(50)]
    b = _split_of(generate("toy", more, spec))
    assert all(b[c] == s for c, s in a.items())


@pytest.mark.parametrize("path", sorted(SPLITS_DIR.glob("*.json")), ids=lambda p: p.name)
def test_committed_splits_groups_never_span(path):
    d = json.loads(path.read_text())
    validate(d)
    assert d["format"] == "t09-splits/1"
    # IDs only: no absolute paths or file contents in the frozen file
    text = path.read_text()
    assert "/home/" not in text and ".avi" not in text and ".mp4" not in text


def test_committed_splits_exist():
    names = {p.stem for p in SPLITS_DIR.glob("*.json")}
    assert {"meva", "smartspaces"} <= names
