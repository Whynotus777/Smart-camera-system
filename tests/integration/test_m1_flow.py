"""M1 walking skeleton, CPU only (runs in CI): video → Event → clip → review → label.

Uses a synthetic clip (no people) generated with ffmpeg, plus the tracked PoC fixture for
the one-command demo. Needs the `ffmpeg`/`ffprobe` binaries; skipped with a loud reason
if they are missing (CI needs them installed: see docs/RELEASE.md, "Handoffs").
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from m1_harness import (
    REPO,
    Stack,
    check_invariants,
    free_port,
    have_ffmpeg,
    http_json,
    reference_events,
    synthetic_video,
)
from scs.app import config as cfgmod
from scs.app.roles import db_path
from scs.app.store import Store

# Generous: on the shared dev box the HDD is often saturated by other agents (docs/RELEASE.md R13).
WAIT_S = 240

pytestmark = pytest.mark.skipif(not have_ffmpeg(), reason="M1 integration needs ffmpeg + ffprobe on PATH")


@pytest.fixture(scope="module")
def video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synthetic_video(tmp_path_factory.mktemp("video") / "synthetic.mp4")


def _wait(pred, timeout: float, what: str):  # noqa: ANN001, ANN202
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if v := pred():
            return v
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for {what}")


def _done_clips(base: str) -> list[dict]:
    try:
        _, items = http_json("GET", base + "api/alerts")
    except OSError:
        return []
    return [it for it in items if it["clip"] and it["clip"]["status"] == "done"]  # type: ignore[union-attr]


def test_one_command_demo_prints_review_url(tmp_path: Path) -> None:
    port = free_port()
    out = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "scs.app",
            "demo",
            "--video",
            "demo_1.mp4",
            "--workdir",
            str(tmp_path / "demo"),
            "--speed",
            "4",
            "--port",
            str(port),
            "--exit-when-ready",
            "--timeout",
            "90",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": __import__("os").environ["PATH"]},
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert f"Review queue: http://127.0.0.1:{port}/" in out.stdout
    st = Store(db_path(tmp_path / "demo"))
    rows = st.alerts()
    assert rows and rows[0][1] is not None and rows[0][1].status == "done"


def test_full_flow_file_source(tmp_path: Path, video: Path) -> None:
    port = free_port()
    base = f"http://127.0.0.1:{port}/"
    cfgmod.save(cfgmod.AppConfig(source=str(video), speed=8, port=port), tmp_path)
    stack = Stack(tmp_path)
    stack.start_all()
    try:
        items = _wait(lambda: (c := _done_clips(base)) and len(c) >= 2 and c, WAIT_S, "two clips")
        a1, a2 = items[0]["alert"]["alert_id"], items[1]["alert"]["alert_id"]

        # the page lists them and the clip is served with Range support (browser seeking)
        with urllib.request.urlopen(base, timeout=5) as r:  # noqa: S310
            page = r.read().decode()
        assert f"/clips/{a1}.mp4" in page and "<video" in page
        req = urllib.request.Request(base + f"clips/{a1}.mp4", headers={"Range": "bytes=0-99"})  # noqa: S310
        with urllib.request.urlopen(req, timeout=5) as r:  # noqa: S310
            assert (
                r.status == 206
                and len(r.read()) == 100
                and r.headers["Content-Range"].startswith("bytes 0-99/")
            )

        # review: confirm one, dismiss one; a retry is idempotent; a contradicting outcome is refused
        code, body = http_json(
            "POST", base + f"api/alerts/{a1}/review", {"decision": "confirmed", "request_id": "r1"}
        )
        assert code == 200 and body["created"] is True  # type: ignore[index]
        code, body = http_json(
            "POST", base + f"api/alerts/{a1}/review", {"decision": "confirmed", "request_id": "r1b"}
        )
        assert code == 200 and body["created"] is False  # type: ignore[index]
        code, _ = http_json("POST", base + f"api/alerts/{a1}/review", {"decision": "dismissed"})
        assert code == 409
        code, _ = http_json(
            "POST", base + f"api/alerts/{a2}/review", {"decision": "dismissed", "reason": "staff"}
        )
        assert code == 200
        assert http_json("POST", base + f"api/alerts/{'0' * 32}/review", {"decision": "confirmed"})[0] == 404
        assert http_json("POST", base + f"api/alerts/{a2}/review", {"decision": "maybe"})[0] == 400
    finally:
        stack.stop_and_drain()

    st = Store(db_path(tmp_path))
    position = st.load_checkpoint("cam01").media_frame  # type: ignore[union-attr]
    st.close()
    ref = reference_events(video, cfgmod.load(tmp_path), position + 1, tmp_path)
    rep = check_invariants(tmp_path, {a1: "confirmed", a2: "dismissed"}, ref, require_all_reviewed=False)
    assert rep.errors == []
    assert rep.labels == 2
    label = (tmp_path / "labels" / "scs_review" / "converted" / "labels" / f"{a1}.json").read_text()
    assert '"label_source": "human"' in label and '"type": "zone_enter"' in label


def test_injected_event_gets_clip(tmp_path: Path, video: Path) -> None:
    """The M1 path must work without any detector: an injected test event."""
    port = free_port()
    cfgmod.save(
        cfgmod.AppConfig(source=str(video), speed=8, port=port, min_dwell_s=1e9), tmp_path
    )  # rule off
    stack = Stack(tmp_path)
    stack.start_all()
    try:
        _wait(
            lambda: (
                (st := Store(db_path(tmp_path))).load_checkpoint("cam01") is not None
                and st.load_checkpoint("cam01").ts - st.load_checkpoint("cam01").state["origin_ts"] > 12
            ),
            60,  # type: ignore[union-attr]
            "12 s of footage",
        )
        out = subprocess.run(
            [sys.executable, "-m", "scs.app", "inject-event", "--workdir", str(tmp_path)],  # noqa: S603
            capture_output=True,
            text=True,
            check=True,
            env=stack.env,
        )
        assert '"new": true' in out.stdout
        items = _wait(lambda: _done_clips(f"http://127.0.0.1:{port}/"), WAIT_S, "injected clip")
        assert items[0]["alert"]["reason_codes"] == ["injected_test_event"]
    finally:
        stack.stop()


def test_demo_supervisor_kill9_leaves_no_orphans_and_resumes(tmp_path: Path, video: Path) -> None:
    import os
    import signal

    port = free_port()
    wd = tmp_path / "demo"
    cmd = [
        sys.executable,
        "-m",
        "scs.app",
        "demo",
        "--video",
        str(video),
        "--workdir",
        str(wd),
        "--speed",
        "8",
        "--port",
        str(port),
    ]
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    demo = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env=env, cwd=REPO)  # noqa: S603
    assert demo.stdout is not None
    _wait(lambda: "Review queue:" in (demo.stdout.readline() or ""), WAIT_S, "review URL")
    roles = [
        int(p)
        for p in subprocess.run(
            ["pgrep", "-f", f"scs.app run .* --workdir {wd}"],  # noqa: S603, S607
            capture_output=True,
            text=True,
        ).stdout.split()
    ]
    assert len(roles) == 3
    os.kill(demo.pid, signal.SIGKILL)
    demo.wait()
    _wait(lambda: not any(Path(f"/proc/{p}").exists() for p in roles), 10, "roles to die with the supervisor")
    st = Store(db_path(wd))
    before = set(st.event_ids())
    st.close()

    # same workdir: resumes the timeline, keeps the port, and adds no duplicates
    out = subprocess.run(
        [*cmd, "--exit-when-ready", "--timeout", "60"],
        capture_output=True,
        text=True,  # noqa: S603
        env=env,
        cwd=REPO,
        timeout=90,
    )
    assert out.returncode == 0 and f"Review queue: http://127.0.0.1:{port}/" in out.stdout, out.stderr[-2000:]
    st = Store(db_path(wd))
    after = st.event_ids()
    st.close()
    assert before <= set(after) and len(after) == len(set(after))
