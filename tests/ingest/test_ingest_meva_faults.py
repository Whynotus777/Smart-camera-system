"""Reconnect evidence on MEVA 1440p streams through mediamtx with scripted faults (T02 acceptance).

Stand-in for T14's replay farm until it lands on main: same shape (mediamtx + one ffmpeg
publisher per stream, looping pre-encoded MEVA files, faults drop/stall/server), on a
private port. Inputs are `python -m scs.ingest.bench prepare` outputs (data marker).

Timeline (s from start): 12 drop cam01 for 10 s · 34 stall cam02 for 6 s (SIGSTOP; RTSP
session stays up) · 56 kill mediamtx for 10 s · end at 100. Checks that each fault is
detected on the affected camera(s) only, every camera recovers with a new epoch, and no
threads leak. Writes a JSON timeline to runs/T02/meva_faults.json.

Runs NVDEC decode of 4 × 1440p for ~100 s: `scripts/gpu shared -- pytest -m slow ...`.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path

import pytest
from rtsp_farm import Mediamtx, Publisher

from scs.bus import InMemoryBus
from scs.contracts import Event, Streams
from scs.ingest import gst as gst_backend
from scs.ingest.backoff import Backoff
from scs.ingest.bench import bench_dir
from scs.ingest.manager import CameraManager
from scs.ingest.packets import PacketQueue
from scs.ingest.sources import RtspSource

pytestmark = [pytest.mark.slow, pytest.mark.data, pytest.mark.gpu]
REPO = Path(__file__).resolve().parents[2]
CAMS = {"cam01": "h264", "cam02": "h265", "cam03": "h264", "cam04": "h265"}


def test_meva_farm_faults_recover_per_camera(tmp_path):
    if not gst_backend.available():
        pytest.skip("GStreamer not available")
    files = {c: bench_dir() / f"{c}_{codec}_1440p15.mp4" for c, codec in CAMS.items()}
    if not all(f.exists() for f in files.values()):
        pytest.skip("run `python -m scs.ingest.bench prepare` first")
    mtx = Mediamtx.start_or_skip(tmp_path)
    pubs = {c: Publisher(f, mtx.url(f"meva/{CAMS[c]}/{c}")).start() for c, f in files.items()}
    time.sleep(2.0)
    bus = InMemoryBus()
    taps = {c: PacketQueue(max_bytes=2**30) for c in CAMS}
    srcs = [RtspSource(c, url=mtx.url(f"meva/{CAMS[c]}/{c}"), decode="nvdec", bus=bus, queue_size=4,
                       backoff=Backoff(initial_s=0.5, cap_s=30.0)) for c in CAMS]
    for s in srcs:
        s.tap.subscribe(taps[s.camera_id])
    os_threads0 = len(os.listdir("/proc/self/task"))
    mgr = CameraManager(srcs).start()
    t0 = time.monotonic()
    frames: dict[str, list[tuple[float, int, int]]] = {c: [] for c in CAMS}
    stop = threading.Event()

    def consume() -> None:
        while not stop.is_set():
            for ref, _ in mgr.next_batch(output="raw"):
                frames[ref.camera_id].append((time.monotonic() - t0, ref.epoch, ref.seq))

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    faults: list[dict] = []

    def at(t: float) -> None:
        time.sleep(max(0.0, t0 + t - time.monotonic()))

    at(12)
    faults.append({"t": round(time.monotonic() - t0, 2), "fault": "drop", "target": "cam01", "s": 10})
    pubs["cam01"].drop(10)
    at(34)
    faults.append({"t": round(time.monotonic() - t0, 2), "fault": "stall", "target": "cam02", "s": 6})
    pubs["cam02"].signal(signal.SIGSTOP)
    time.sleep(6)
    pubs["cam02"].signal(signal.SIGCONT)
    at(56)
    faults.append({"t": round(time.monotonic() - t0, 2), "fault": "server", "target": "*", "s": 10})
    mtx.kill()
    time.sleep(10)
    mtx.restart()
    at(100)
    stop.set()
    consumer.join(2)
    snap = mgr.snapshot()
    mgr.close()
    for p in pubs.values():
        p.stop()
    mtx.remove()
    time.sleep(0.5)

    events = [ev for _, ev in bus.consume(Streams.HEALTH, "t02", "t02", Event, count=10_000)]
    wall_t0 = time.time() - (time.monotonic() - t0)
    tl = [{"t": round(ev.ts - wall_t0, 2), "camera": ev.camera_id, "state": ev.data["state"],
           "epoch": ev.data["epoch"], "reason": ev.data["reason"]} for ev in events]

    def states(cam: str, t_from: float, t_to: float) -> list[str]:
        return [e["state"] for e in tl if e["camera"] == cam and t_from <= e["t"] < t_to]

    def recovery(cam: str, t_fault: float) -> float | None:
        """Seconds from fault start to the first frame of a later epoch."""
        before = max((ep for t, ep, _ in frames[cam] if t < t_fault), default=0)
        after = [t for t, ep, _ in frames[cam] if t >= t_fault and ep > before]
        return round(after[0] - t_fault, 2) if after else None

    rec = {"cam01_drop": recovery("cam01", 12), "cam02_stall": recovery("cam02", 34),
           **{f"{c}_server": recovery(c, 56) for c in CAMS}}
    report = {"faults": faults, "recovery_s": rec, "health_timeline": tl,
              "stats": snap, "tap_packets": {c: q.received for c, q in taps.items()},
              "tap_overflow": {c: q.overflow for c, q in taps.items()},
              "os_threads_before": os_threads0, "os_threads_after": len(os.listdir("/proc/self/task")),
              "note": "local mediamtx stand-in for T14 farm; MEVA upscaled 1080p->1440p15"}
    out = REPO / "runs" / "T02" / "meva_faults.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))

    # drop on cam01: detected + recovered; other cameras untouched during it
    assert {"disconnected", "reconnecting", "connected"} <= set(states("cam01", 12, 34))
    assert rec["cam01_drop"] is not None and 10 <= rec["cam01_drop"] < 20
    for c in ("cam02", "cam03", "cam04"):
        assert "disconnected" not in states(c, 12, 34), c
    # silent stall on cam02: flagged as a stall within ~3 s, others untouched
    st2 = [e for e in tl if e["camera"] == "cam02" and 34 <= e["t"] < 56]
    assert st2 and st2[0]["state"] == "stalled" and st2[0]["t"] - 34 < 4.5
    assert rec["cam02_stall"] is not None and rec["cam02_stall"] < 20
    for c in ("cam01", "cam03", "cam04"):
        assert "stalled" not in states(c, 34, 56), c
    # server outage: every camera reconnects after the restart
    for c in CAMS:
        assert rec[f"{c}_server"] is not None and 10 <= rec[f"{c}_server"] < 40, (c, rec)
        assert frames[c][-1][0] > 95  # still delivering at the end
    for c in ("cam01", "cam03", "cam04"):  # only cam02 had a silent stall; none during backoff waits
        assert "stalled" not in states(c, 0, 101), c
    for c, st in snap.items():  # every frame identity names its tapped packet
        assert st["reordered"] == 0 and st["unmatched"] == 0, (c, st)
    assert all(q.overflow == 0 for q in taps.values())
    assert not [t for t in threading.enumerate() if t.name.startswith("ingest-")]
    assert report["os_threads_after"] <= os_threads0 + 16
