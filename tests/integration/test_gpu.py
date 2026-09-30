"""M1 GPU variant (5090 box only): NVDEC decode + NVENC clip masters, same invariants.

Run through the GPU lock (AGENTS.md):
    scripts/gpu shared -- pytest -m gpu tests/integration/test_gpu.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from m1_chaos import run_chaos
from m1_harness import have_ffmpeg, synthetic_video

GPU_ENV = {"SCS_HWACCEL": "cuda", "SCS_ENCODER": "h264_nvenc"}

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not (have_ffmpeg() and shutil.which("nvidia-smi")), reason="needs ffmpeg + GPU"),
]


def test_chaos_once_on_gpu(tmp_path: Path) -> None:
    video = synthetic_video(tmp_path / "synthetic.mp4")
    rep, stats = run_chaos(tmp_path / "run", video, seed=21, loops=3, speed=10, env=GPU_ENV)
    assert rep.errors == [], (rep, stats)
    assert rep.events >= 2 and rep.reviews == rep.clips >= 2


def test_demo_clip_on_gpu(tmp_path: Path) -> None:
    """The PoC fixture (MPEG-4 Part 2) through NVDEC → NVENC master → remuxed H.264 clip."""
    video = Path(__file__).resolve().parents[1] / "fixtures" / "video" / "demo_1.mp4"
    rep, stats = run_chaos(tmp_path / "run", video, seed=22, loops=3, speed=6, env=GPU_ENV,
                           crashpoints="", kill_gap=(5.0, 8.0))
    # demo_1's first event fires 5 s after the camera "starts", so that clip is exempt from
    # the pre-roll check (no footage exists before the origin); every later one isn't.
    assert rep.errors == [], (rep, stats)
    assert rep.clips >= 2, (rep, stats)
