import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

from data_ops.replay import farm


class FakeProc:
    def __init__(self):
        self.pid, self.dead = 4242, False

    def poll(self):
        return 1 if self.dead else None


@pytest.fixture
def fake(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(farm.Publisher, "start", lambda self: setattr(self, "proc", FakeProc()))
    monkeypatch.setattr(farm.Publisher, "stop", lambda self: setattr(self, "proc", None))
    monkeypatch.setattr(farm.os, "killpg", lambda pid, sig: calls.append(("killpg", sig)))
    monkeypatch.setattr(farm, "server_kill", lambda: calls.append(("server", "kill")))
    monkeypatch.setattr(farm, "server_start", lambda: calls.append(("server", "start")))
    monkeypatch.setattr(farm, "wait_ready", lambda timeout_s=0: True)
    pubs = [farm.Publisher(cam=f"cam0{i}", codec=c, files={"normal": Path("n"), "spike": Path("s")})
            for c in ("h264", "h265") for i in (1, 2)]
    fm = farm.Farm(pubs=pubs, run_dir=tmp_path, t0=0.0, log=lambda *_: None)
    (tmp_path / "requests").mkdir()
    for p in pubs:
        p.start()
    return fm, calls


def test_fault_validation_and_schedule(tmp_path):
    with pytest.raises(ValueError):
        farm.Fault(at=0, kind="explode", seconds=1)
    with pytest.raises(ValueError):
        farm.Fault(at=0, kind="drop", seconds=0)
    f = tmp_path / "s.json"
    f.write_text(json.dumps([{"at": 30, "fault": "stall", "seconds": 5, "target": "cam02"},
                             {"at": 10, "fault": "drop", "seconds": 10}]))
    sched = farm.load_schedule(f)
    assert [(x.at, x.kind, x.target) for x in sched] == [(10, "drop", "*"), (30, "stall", "cam02")]


def test_targets(fake):
    fm, _ = fake
    f = farm.Fault(at=0, kind="drop", seconds=1, target="h265/cam02")
    assert [p.path for p in fm.pubs if f.matches(p)] == ["meva/h265/cam02"]
    f = farm.Fault(at=0, kind="drop", seconds=1, target="cam02")
    assert [p.path for p in fm.pubs if f.matches(p)] == ["meva/h264/cam02", "meva/h265/cam02"]


def test_drop_holds_publisher_down_then_restarts(fake):
    fm, _ = fake
    fm.schedule = [farm.Fault(at=5, kind="drop", seconds=10, target="h264/cam01")]
    fm.tick(4.0)
    assert all(p.alive() for p in fm.pubs)
    fm.tick(5.0)
    cam = fm.pubs[0]
    assert not cam.alive()
    fm.tick(14.9)
    assert not cam.alive(), "dropped publisher must stay down for the whole fault"
    fm.tick(15.0)
    assert cam.alive() and cam.restarts == 1
    state = json.loads((fm.run_dir / "state.json").read_text())
    assert state["streams"]["meva/h264/cam01"]["alive"]


def test_stall_pauses_and_resumes_without_reconnect(fake):
    fm, calls = fake
    fm.apply(farm.Fault(at=0, kind="stall", seconds=3, target="h265/cam02"), now=100.0)
    assert calls == [("killpg", farm.signal.SIGSTOP)]
    fm.tick(102.0)
    assert fm.pubs[3].alive() and fm.pubs[3].resume_at == 103.0
    fm.tick(103.0)
    assert calls[-1] == ("killpg", farm.signal.SIGCONT) and fm.pubs[3].restarts == 0


def test_spike_switches_variant_and_back(fake):
    fm, _ = fake
    fm.apply(farm.Fault(at=0, kind="spike", seconds=2, target="h264/cam02"), now=0.0)
    assert fm.pubs[1].variant == "spike"
    fm.tick(2.0)
    assert fm.pubs[1].variant == "normal"


def test_crashing_publisher_backs_off_exponentially(fake):
    fm, _ = fake
    cam = fm.pubs[0]
    starts = []
    for t in [x * 0.2 for x in range(0, 60)]:  # 12 s of ticks; the publisher dies right after each start
        if cam.proc:
            cam.proc.dead = True
        before = cam.restarts
        fm.tick(t)
        if cam.restarts > before:
            starts.append(round(t, 1))
    assert starts[:3] == [1.0, 3.2, 7.4], f"restart schedule {starts}"  # +1 s, +2 s, +4 s after each death
    assert len(starts) <= 4, "no restart storm"
    assert cam.crashes >= 4


def test_server_fault_and_on_demand_request(fake, monkeypatch):
    fm, calls = fake
    monkeypatch.setattr(farm, "farm_dir", lambda: fm.run_dir.parent)
    fm.run_dir = fm.run_dir.parent / "run"
    farm.request_fault("server", 4)
    fm.tick(10.0)
    assert ("server", "kill") in calls and fm.server_down_until == 14.0
    assert not any(p.alive() for p in fm.pubs), "a server fault stops every publisher"
    fm.tick(12.0)
    assert not any(p.alive() for p in fm.pubs)
    fm.tick(14.0)
    assert calls[-1] == ("server", "start")
    fm.tick(14.2)
    assert all(p.alive() for p in fm.pubs)


# ---------------------------------------------------------------- integration (docker + ffmpeg)


def _paths_ready():
    with urllib.request.urlopen(f"http://127.0.0.1:{farm.API_PORT}/v3/paths/list", timeout=3) as r:  # noqa: S310
        return {i["name"]: i["ready"] for i in json.load(r)["items"]}


def _run_until(fm, deadline):
    """Tick the supervisor until `deadline`, and always at least once (slow readers can overshoot)."""
    while True:
        fm.tick(time.monotonic())
        if time.monotonic() >= deadline:
            return
        time.sleep(0.2)


def _frames(u, seconds=3.0):
    """Frames an RTSP reader receives in `seconds` (0 if it can't connect or nothing arrives)."""
    try:
        out = subprocess.run(  # noqa: S603
            [shutil.which("ffprobe"), "-v", "error", "-rtsp_transport", "tcp", "-rw_timeout", "2000000",
             "-read_intervals", f"%+{seconds}", "-select_streams", "v", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", u],
            capture_output=True, text=True, timeout=seconds + 5)
    except subprocess.TimeoutExpired:
        return 0  # a stalled stream can block the reader; that counts as no frames
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return 0


@pytest.mark.slow
@pytest.mark.skipif(not (shutil.which("docker") and shutil.which("ffmpeg")), reason="needs docker + ffmpeg")
def test_farm_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setattr(farm, "farm_dir", lambda: tmp_path)
    clips = tmp_path / "clips"
    clips.mkdir()
    for cam in ("cam01", "cam02"):
        subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",  # noqa: S603
                        "testsrc2=size=640x360:rate=30:duration=10", "-c:v", "libx264",
                        "-g", "60", "-bf", "0",
                        str(clips / f"{cam}_h264.mp4")], check=True)
    (tmp_path / "clips.json").write_text(json.dumps([{"cam": "cam01"}, {"cam": "cam02"}]))
    farm.server_up()
    fm = farm.Farm(pubs=farm.build_publishers(("h264",)), log=lambda *_: None)
    for p in fm.pubs:
        p.files.pop("spike")
    (fm.run_dir / "requests").mkdir(parents=True, exist_ok=True)
    try:
        for p in fm.pubs:
            p.start()
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            fm.tick(time.monotonic())
            try:
                if sum(_paths_ready().values()) == 2:
                    break
            except OSError:
                pass
            time.sleep(0.3)
        u1, u2 = farm.url("h264", "cam01"), farm.url("h264", "cam02")
        assert _frames(u1) > 30 and _frames(u2) > 30

        now = time.monotonic()
        fm.apply(farm.Fault(at=0, kind="stall", seconds=8, target="cam01"), now)
        time.sleep(1.0)
        assert _paths_ready()["meva/h264/cam01"], "a stall keeps the path up (silent stall)"
        assert _frames(u1, 2) < 10, "stalled stream should deliver (almost) no frames"
        assert _frames(u2) > 30, "a fault on one camera must not affect the others"
        _run_until(fm, now + 8.5)
        assert _frames(u1) > 30, "stream recovers after the stall"

        now = time.monotonic()
        fm.apply(farm.Fault(at=0, kind="drop", seconds=4, target="cam02"), now)
        time.sleep(1.0)
        assert "meva/h264/cam02" not in {k for k, v in _paths_ready().items() if v}
        _run_until(fm, now + 8)
        assert _frames(u2) > 30, "stream is back after the drop"

        now = time.monotonic()
        fm.apply(farm.Fault(at=0, kind="server", seconds=3), now)
        time.sleep(1.0)
        assert _frames(u1, 1) == 0
        _run_until(fm, now + 5)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:  # wait for both paths to come back, then read
            _run_until(fm, time.monotonic() + 0.5)
            try:
                if sum(_paths_ready().values()) == 2:
                    break
            except OSError:
                pass
        assert _frames(u1) > 30 and _frames(u2) > 30, "both streams recover after a server restart"
        assert all(p.crashes == 0 for p in fm.pubs), "server-fault stops must not count as crashes"
    finally:
        for p in fm.pubs:
            p.stop()
        farm.server_down()
