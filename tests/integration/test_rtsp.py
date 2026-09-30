"""M1 over RTSP: a camera that drops, a server that dies, an ingest that gets kill -9'd.

Runs its own mediamtx container on a free port (not T14's shared farm on :8554, whose
`fault server` / `down` would hit every agent's tests at once) and publishes a looping
clip into it. Set `SCS_M1_RTSP_VIDEO` to publish something else (e.g. a MEVA clip, as in
docs/reports/T13-m1.md); default is the synthetic CI clip.
Marked slow (~2 min, real time) and skipped without Docker or the mediamtx image.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from m1_harness import Reviewer, Stack, check_invariants, free_port, have_ffmpeg, synthetic_video
from scs.app import config as cfgmod
from scs.app.roles import db_path
from scs.app.store import Store

IMAGE = "bluenviron/mediamtx:1.21.1"  # same pin as T14's replay farm


def _docker_ok() -> bool:
    if not shutil.which("docker"):
        return False
    r = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True)
    return r.returncode == 0


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not (have_ffmpeg() and _docker_ok()), reason=f"needs ffmpeg and docker image {IMAGE}"),
]


class Camera:
    """mediamtx + a looping publisher: a fake RTSP camera we can break on purpose."""

    def __init__(self, video: Path) -> None:
        self.video, self.port = video, free_port()
        self.name = f"scs-t13-mtx-{self.port}"
        self.url = f"rtsp://127.0.0.1:{self.port}/cam01"
        self.pub: subprocess.Popen | None = None

    def server_up(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
        subprocess.run(["docker", "run", "-d", "--rm", "--name", self.name, "-p",
                        f"127.0.0.1:{self.port}:8554", IMAGE], check=True, capture_output=True)
        time.sleep(1.0)

    def publish(self) -> None:
        self.pub = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-nostdin", "-re", "-stream_loop", "-1", "-i", str(self.video),
             "-map", "0:v:0", "-c", "copy", "-f", "rtsp", "-rtsp_transport", "tcp", self.url])

    def unpublish(self) -> None:
        if self.pub and self.pub.poll() is None:
            self.pub.kill()
            self.pub.wait()

    def server_down(self) -> None:
        self.unpublish()
        subprocess.run(["docker", "kill", self.name], capture_output=True)


@pytest.fixture
def camera(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Camera]:
    env_video = os.environ.get("SCS_M1_RTSP_VIDEO")
    video = Path(env_video) if env_video else synthetic_video(tmp_path_factory.mktemp("v") / "synthetic.mp4")
    cam = Camera(video)
    cam.server_up()
    cam.publish()
    yield cam
    cam.server_down()


def _epochs(workdir: Path) -> int:
    st = Store(db_path(workdir))
    try:
        return int(st._db.execute("SELECT COUNT(*) FROM epochs").fetchone()[0])
    finally:
        st.close()


def _wait(pred, timeout: float, what: str) -> None:  # noqa: ANN001
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.5)


def test_rtsp_flow_survives_drops_and_kills(tmp_path: Path, camera: Camera) -> None:
    port = free_port()
    cfgmod.save(cfgmod.AppConfig(source="env:SCS_TEST_CAM", port=port), tmp_path)
    stack = Stack(tmp_path, env={"SCS_TEST_CAM": camera.url})
    stack.start_all()
    reviewer = Reviewer(f"http://127.0.0.1:{port}/", seed=3)
    reviewer.start()
    # No footage exists from before the system first connected: that's an outage too.
    outages: list[tuple[float, float]] = [(0.0, time.time() + 3)]
    reconnect_s: dict[str, float] = {}

    def fault(name: str, break_it, fix_it) -> None:  # noqa: ANN001
        """Break the camera path, fix it, and wait for the ingest to reconnect (new epoch)."""
        n = _epochs(tmp_path)
        t = time.time()
        break_it()
        fixed = fix_it()
        _wait(lambda: _epochs(tmp_path) > n, 90, f"reconnect after {name}")
        reconnect_s[name] = round(time.time() - fixed, 1)
        outages.append((t, time.time() + 3))

    def kill_ingest() -> None:
        stack.kill("ingest")
        time.sleep(1)

    def restart_ingest() -> float:
        stack.ensure_running()
        return time.time()

    def drop_back() -> float:
        time.sleep(8)
        camera.publish()
        return time.time()

    def server_back() -> float:
        time.sleep(8)
        camera.server_up()
        camera.publish()
        return time.time()

    try:
        _wait(lambda: len(reviewer.acked) >= 1, 90, "first reviewed clip")
        fault("ingest kill -9", kill_ingest, restart_ingest)
        fault("camera drop 8 s", camera.unpublish, drop_back)
        fault("rtsp server down 8 s", camera.server_down, server_back)
        n_before = len(reviewer.acked)
        _wait(lambda: len(reviewer.acked) >= n_before + 1, 90, "a reviewed clip after the faults")
        print(f"reconnect latency after fix (s): {reconnect_s}")
    finally:
        stack.stop_and_drain()
        reviewer.stop_flag.set()
        # the drain may have produced clips the reviewer hasn't seen: review them now
        stack.start("web")
        _wait(lambda: reviewer.review_ready() >= 0 and _all_reviewed(tmp_path, reviewer), 30, "final reviews")
        stack.stop()
    rep = check_invariants(tmp_path, reviewer.acked, reference=None, outages=outages)
    assert rep.errors == [], rep
    assert rep.events >= 2 and rep.reviews == rep.clips >= 2


def _all_reviewed(workdir: Path, reviewer: Reviewer) -> bool:
    st = Store(db_path(workdir))
    try:
        return all(a.alert_id in reviewer.acked for a, j, _ in st.alerts() if j and j.status == "done")
    finally:
        st.close()
