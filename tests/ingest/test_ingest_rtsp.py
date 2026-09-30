"""RTSP against a real mediamtx: server outage, silent publisher stall, thread hygiene.

Runs a private mediamtx (release binary via SCS_MEDIAMTX_BIN, else the Docker image) on a
free localhost port (never T14's farm port) and an ffmpeg publisher looping a test clip,
restarted by a supervisor thread whenever it dies (as T14's farm does). Faults:
- server kill: mediamtx killed for 10 s, then restarted on the same port (acceptance);
- silent stall: publisher SIGSTOPped, so the RTSP session stays up but no frames flow
  (the Reolink failure mode; mediamtx only drops the publisher after readTimeout=30 s).
"""

from __future__ import annotations

import os
import signal
import threading
import time

import pytest
from rtsp_farm import Mediamtx, Publisher

from scs.ingest import gst as gst_backend
from scs.ingest import pyav as pyav_backend
from scs.ingest.backoff import Backoff
from scs.ingest.packets import PacketQueue
from scs.ingest.sources import RtspSource

pytestmark = pytest.mark.slow
@pytest.fixture
def farm(clips, tmp_path):
    mtx = Mediamtx.start_or_skip(tmp_path)
    url = mtx.url("cam")
    pub = Publisher(clips["h264"], url).start()
    time.sleep(1.5)
    yield {"url": url, "mtx": mtx, "pub": pub}
    pub.stop()
    mtx.remove()


def _need(backend: str) -> None:
    if backend == "gstreamer" and not gst_backend.available():
        pytest.skip("GStreamer not available")
    if backend == "pyav" and not pyav_backend.available():
        pytest.skip("PyAV not installed")


def _os_threads() -> int:
    return len(os.listdir("/proc/self/task"))


def _wait_frames(src: RtspSource, n: int, timeout: float, epoch_at_least: int = 0):
    got = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and len(got) < n:
        item = src.get(timeout=0.5)
        if item is not None and item[0].epoch >= epoch_at_least:
            got.append(item[0])
    return got


def _states(src: RtspSource) -> list[str]:
    return [e.data["state"] for e in src.health.events]


@pytest.mark.parametrize("backend", ["gstreamer", "pyav"])
def test_server_killed_10s_recovers_with_health_events_and_no_thread_leak(farm, backend):
    _need(backend)
    py_before, os_before = threading.active_count(), _os_threads()
    q = PacketQueue()
    src = RtspSource("rtsp1", url=farm["url"], backend=backend, decode="software", queue_size=8,
                     backoff=Backoff(initial_s=0.5, cap_s=4.0))
    src.tap.subscribe(q)
    src.start()
    first = _wait_frames(src, 20, 20)
    assert len(first) == 20 and first[0].epoch == 0 and first[0].seq == 0

    t_kill = time.monotonic()
    farm["mtx"].kill()
    time.sleep(10)
    farm["mtx"].restart()
    after = _wait_frames(src, 20, 40, epoch_at_least=1)
    t_recovered = time.monotonic()
    assert len(after) == 20, _states(src)
    assert after[0].seq == 0 and after[0].epoch >= 1
    assert all(b.seq == a.seq + 1 for a, b in zip(after, after[1:], strict=False))
    st = _states(src)
    assert st[0] == "connected" and "disconnected" in st and "reconnecting" in st
    assert st.count("connected") >= 2
    assert 10 <= t_recovered - t_kill < 30
    assert src.stats.snapshot()["reconnects"] >= 1
    # packets keep identity across epochs, one entry per (epoch, seq), and every frame's
    # identity names a tapped packet (streams are joined mid-GOP after each reconnect)
    pk = q.drain()
    assert len({p.identity for p in pk}) == len(pk)
    tapped = {p.identity for p in pk}
    assert all(r.identity in tapped for r in first + after)
    assert all(p.keyframe for p in pk if p.seq == 0)
    snap = src.stats.snapshot()
    assert snap["reordered"] == 0 and snap["unmatched"] == 0, snap

    src.close()
    time.sleep(0.5)
    assert not src.alive
    assert not [t for t in threading.enumerate() if t.name.startswith("ingest-")]
    assert threading.active_count() <= py_before + 2  # GStreamer streaming threads seen by Python
    assert _os_threads() <= os_before + 12, (os_before, _os_threads())  # GLib pools, not per-epoch


@pytest.mark.parametrize("backend", ["gstreamer", "pyav"])
def test_silent_stall_detected_within_3s_then_recovers(farm, backend):
    _need(backend)
    src = RtspSource("rtsp1", url=farm["url"], backend=backend, decode="software", queue_size=8,
                     stall_timeout_s=3.0, backoff=Backoff(initial_s=0.5, cap_s=2.0))
    src.start()
    assert len(_wait_frames(src, 15, 20)) == 15
    farm["pub"].signal(signal.SIGSTOP)
    t_stop = time.monotonic()
    deadline = t_stop + 10
    while "stalled" not in _states(src) and time.monotonic() < deadline:
        time.sleep(0.05)
    t_detect = time.monotonic() - t_stop
    assert "stalled" in _states(src)
    assert 2.9 <= t_detect <= 4.5, t_detect
    time.sleep(3)
    farm["pub"].signal(signal.SIGCONT)
    after = _wait_frames(src, 15, 30, epoch_at_least=1)
    src.close()
    assert len(after) == 15, _states(src)
    ev = next(e for e in src.health.events if e.data["state"] == "stalled")
    assert ev.camera_id == "rtsp1" and ev.data["epoch"] == 0
