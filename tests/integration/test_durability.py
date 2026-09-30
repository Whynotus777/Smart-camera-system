"""M1 durability: kill -9 any process at any point; after restart nothing is lost or duplicated.

`test_chaos_once` is the CI-sized version. `test_chaos_20_in_a_row` is the M1 acceptance
run (slow, ~10 min on CPU); its output goes in docs/reports/T13-m1.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from m1_chaos import run_chaos
from m1_harness import have_ffmpeg, synthetic_video

pytestmark = pytest.mark.skipif(not have_ffmpeg(), reason="M1 integration needs ffmpeg + ffprobe on PATH")


@pytest.fixture(scope="module")
def video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synthetic_video(tmp_path_factory.mktemp("video") / "synthetic.mp4")


def test_chaos_once(tmp_path: Path, video: Path) -> None:
    rep, stats = run_chaos(tmp_path, video, seed=7, loops=3, speed=10)
    assert rep.errors == [] and stats["liveness"] is None, (rep, stats)
    assert stats["kills"] >= 5 and rep.events >= 2 and rep.reviews == rep.clips >= 2


@pytest.mark.slow
def test_chaos_20_in_a_row(tmp_path: Path, video: Path) -> None:
    for i in range(20):
        rep, stats = run_chaos(tmp_path / f"run{i:02d}", video, seed=1000 + i)
        assert rep.errors == [] and stats["liveness"] is None, (i, rep, stats)
