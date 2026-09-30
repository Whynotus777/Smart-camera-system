"""Fixtures for ingest tests: small H.264/H.265 clips encoded on the CPU from the PoC fixture."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "video" / "demo_1.mp4"


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        pytest.skip("ffmpeg not installed")
    return exe


@pytest.fixture(scope="session")
def clips(tmp_path_factory) -> dict[str, Path]:
    """4 s H.264 and H.265 clips (no B-frames, 1 s GOP) made from the PoC fixture on the CPU."""
    ff = _ffmpeg()
    out = tmp_path_factory.mktemp("clips")
    enc = {"h264": ["-c:v", "libx264", "-bf", "0", "-preset", "veryfast"],
           "h265": ["-c:v", "libx265", "-preset", "veryfast", "-x265-params", "bframes=0:log-level=error"]}
    paths = {}
    for codec, args in enc.items():
        p = out / f"clip_{codec}.mp4"
        subprocess.run([ff, "-v", "error", "-y", "-i", str(FIXTURE), "-t", "4", "-an", "-r", "15", *args,  # noqa: S603
                        "-g", "15", "-pix_fmt", "yuv420p", str(p)], check=True)
        paths[codec] = p
    return paths
