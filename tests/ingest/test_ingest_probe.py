"""scripts/rtsp_probe.py: reads the URL from an env var, prints specs, never leaks credentials."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROBE = Path(__file__).resolve().parents[2] / "scripts" / "rtsp_probe.py"
FAKE_CRED = "admin" + ":" + "hunter2" + "@"  # built at runtime (secret scanners)


def _probe(env_val: str, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "SCS_PROBE_TEST_URL": env_val}
    return subprocess.run([sys.executable, str(PROBE), "--env", "SCS_PROBE_TEST_URL", *args],  # noqa: S603
                          capture_output=True, text=True, env=env, timeout=60)


def test_probe_reports_file_specs(clips):
    pytest.importorskip("av")
    r = _probe(str(clips["h265"]), "--seconds", "30", "--json")
    assert r.returncode == 0, r.stderr
    spec = json.loads(r.stdout)
    assert (spec["codec"], spec["width"], spec["height"]) == ("h265", 640, 360)
    assert spec["fps_measured"] == pytest.approx(15, abs=0.2)
    assert spec["gop_frames"] == 15 and spec["b_frames"] is False
    assert spec["bitrate_kbps_measured"] > 0


def test_probe_never_prints_credentials():
    pytest.importorskip("av")
    r = _probe("rtsp://" + FAKE_CRED + "127.0.0.1:9/nothing?password=hunter2", "--seconds", "1")
    assert r.returncode == 1
    out = r.stdout + r.stderr
    assert "hunter2" not in out and "admin:" not in out
    assert "rtsp://***@127.0.0.1:9/nothing" in out


def test_probe_requires_env_var():
    env = {k: v for k, v in os.environ.items() if k != "SCS_PROBE_UNSET"}
    r = subprocess.run([sys.executable, str(PROBE), "--env", "SCS_PROBE_UNSET"], capture_output=True,  # noqa: S603
                       text=True, env=env, timeout=30)
    assert r.returncode == 2 and "not set" in r.stderr
