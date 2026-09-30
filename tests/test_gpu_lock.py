"""scripts/gpu: shared jobs overlap, exclusive jobs run alone, and exclusive isn't starved."""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GPU = ROOT / "scripts" / "gpu"

pytestmark = pytest.mark.skipif(shutil.which("flock") is None, reason="needs flock(1)")

# Records wall-clock start/end of the locked section to a file.
_JOB = ("import sys, time; s = time.time(); time.sleep(float(sys.argv[2])); "
        "open(sys.argv[1], 'w').write(f'{s} {time.time()}')")


def _start(mode, out, seconds, lock_dir):
    env = {**os.environ, "SCS_GPU_LOCK_DIR": str(lock_dir)}
    return subprocess.Popen(  # noqa: S603
        [str(GPU), mode, "--", sys.executable, "-c", _JOB, str(out), str(seconds)], env=env)


def _span(out):
    s, e = out.read_text().split()
    return float(s), float(e)


def _wait(*procs):
    for p in procs:
        assert p.wait(timeout=20) == 0


def test_shared_jobs_overlap(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    pa, pb = _start("shared", a, 0.6, tmp_path), _start("shared", b, 0.6, tmp_path)
    _wait(pa, pb)
    (sa, ea), (sb, eb) = _span(a), _span(b)
    assert sb < ea and sa < eb


def test_exclusive_waits_for_shared_and_blocks_new_shared(tmp_path):
    s1, ex, s2 = tmp_path / "s1", tmp_path / "ex", tmp_path / "s2"
    p1 = _start("shared", s1, 0.8, tmp_path)
    time.sleep(0.2)
    pe = _start("exclusive", ex, 0.5, tmp_path)  # queues behind s1
    time.sleep(0.2)
    p2 = _start("shared", s2, 0.1, tmp_path)  # arrives while ex waits: must not jump ahead
    _wait(p1, pe, p2)
    (_, e1), (sx, ex_end), (s2_start, _) = _span(s1), _span(ex), _span(s2)
    assert sx >= e1, "exclusive started while a shared job still held the GPU"
    assert s2_start >= ex_end, "a new shared job overtook a waiting exclusive job (starvation)"


def test_exclusive_jobs_serialize(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    pa = _start("exclusive", a, 0.4, tmp_path)
    time.sleep(0.1)
    pb = _start("exclusive", b, 0.1, tmp_path)
    _wait(pa, pb)
    assert _span(b)[0] >= _span(a)[1]


def test_usage_and_exit_code(tmp_path):
    env = {**os.environ, "SCS_GPU_LOCK_DIR": str(tmp_path)}
    run = lambda *a: subprocess.run([str(GPU), *a], env=env, capture_output=True).returncode  # noqa: E731, S603
    assert run("sometimes", "--", "true") == 2
    assert run("shared", "true") == 2
    assert run("shared", "--", "false") == 1
