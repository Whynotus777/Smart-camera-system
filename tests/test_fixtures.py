import importlib.util
from pathlib import Path

from scs.contracts import Pose, SiteConfig, Track

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"


def _load_generator():
    spec = importlib.util.spec_from_file_location("make_fixtures", FIX / "make_fixtures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read(path, cls):
    return [cls.model_validate_json(line) for line in path.read_text().splitlines()]


def test_fixtures_load_into_contracts():
    tracks = _read(FIX / "tracks.jsonl", Track)
    poses = _read(FIX / "poses.jsonl", Pose)
    assert len(tracks) == len(poses) == 200
    assert {t.track_id for t in tracks} == {1, 2}
    span = max(t.frame.ts for t in tracks) - min(t.frame.ts for t in tracks)
    assert 9.8 <= span <= 10.0


def test_committed_fixtures_match_generator():
    tracks_txt, poses_txt = _load_generator().render()
    assert (FIX / "tracks.jsonl").read_text() == tracks_txt, "run tests/fixtures/make_fixtures.py"
    assert (FIX / "poses.jsonl").read_text() == poses_txt, "run tests/fixtures/make_fixtures.py"


def _inside(pt, poly):
    x, y = pt
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys)  # site.example zones are rectangles


def test_scenario_matches_docstring():
    site = SiteConfig.from_yaml(ROOT / "configs/site.example.yaml")
    zones = {z.id: z.polygon for z in site.cameras[0].zones}
    tracks = _read(FIX / "tracks.jsonl", Track)
    poses = _read(FIX / "poses.jsonl", Pose)

    def foot(t):
        return ((t.bbox[0] + t.bbox[2]) / 2 / t.frame.width, t.bbox[3] / t.frame.height)

    def wrists(p):
        return [(p.keypoints[i][0] / p.frame.width, p.keypoints[i][1] / p.frame.height) for i in (9, 10)]

    t1 = [t for t in tracks if t.track_id == 1]
    t2 = [t for t in tracks if t.track_id == 2]
    p1 = [p for p in poses if p.track_id == 1]
    # Track 1: reaches into the shelf, never stands in it, ends at the door.
    reach = [any(_inside(w, zones["shelf_a"]) for w in wrists(p)) for p in p1]
    assert any(reach[5:25]) and not any(reach[:5]) and not any(reach[25:])
    assert not _inside(foot(t1[0]), zones["door"])
    assert all(_inside(foot(t), zones["door"]) for t in t1[-10:])
    # Track 2: ends at the counter, never at the door.
    assert all(_inside(foot(t), zones["counter"]) for t in t2[-50:])
    assert not any(_inside(foot(t), zones["door"]) for t in t2)
