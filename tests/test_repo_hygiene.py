"""Tracked files obey AGENTS.md rule 4: no weights, and no video except the named fixtures."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_EXT = {".pt", ".pth", ".onnx", ".engine"}
VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov"}
ALLOWED_VIDEO = {"tests/fixtures/video/demo_1.mp4", "tests/fixtures/video/demo2.mp4"}


def _tracked() -> list[str]:
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run([git, "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)  # noqa: S603
    return out.stdout.splitlines()


def test_no_weights_tracked():
    assert [f for f in _tracked() if Path(f).suffix.lower() in WEIGHT_EXT] == []


def test_only_allowed_videos_tracked():
    videos = {f for f in _tracked() if Path(f).suffix.lower() in VIDEO_EXT}
    assert videos == ALLOWED_VIDEO
