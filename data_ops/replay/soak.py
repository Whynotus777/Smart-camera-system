"""Soak test for the fake-camera farm: N minutes of all streams plus scheduled faults, with evidence.

Starts `python -m data_ops.replay up` as a child process with a fault schedule, then:
- every `poll_s`: reads the mediamtx API and records, per path, ready + bytes received
  (a healthy stream's bytes must grow between polls, except while a fault is active);
- every `probe_every_s` and ~20 s after each fault ends: pulls 3 s from every stream with
  ffprobe (a real RTSP reader) and records frames received;
- records supervisor RSS and publisher restart counts from state.json.
Writes <data_root>/replay/soak/<stamp>/{schedule.json, samples.jsonl, summary.json}.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from data_ops.replay import farm

DEFAULT_SCHEDULE = [
    {"at": 600, "fault": "stall", "seconds": 8, "target": "h264/cam03"},
    {"at": 1200, "fault": "drop", "seconds": 20, "target": "h265/cam05"},
    {"at": 1800, "fault": "spike", "seconds": 60, "target": "h264/cam01"},
    {"at": 2400, "fault": "server", "seconds": 15},
    {"at": 3000, "fault": "stall", "seconds": 5},
]


def _paths() -> dict[str, dict]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{farm.API_PORT}/v3/paths/list", timeout=3) as r:  # noqa: S310
            return {i["name"]: {"ready": i["ready"], "bytes": i.get("bytesReceived", 0)}
                    for i in json.load(r)["items"]}
    except OSError:
        return {}


def _frames(u: str, seconds: float = 3.0) -> int:
    try:
        out = subprocess.run(  # noqa: S603
            [shutil.which("ffprobe"), "-v", "error", "-rtsp_transport", "tcp", "-rw_timeout", "3000000",
             "-read_intervals", f"%+{seconds}", "-select_streams", "v", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", u],
            capture_output=True, text=True, timeout=seconds + 10)
        return int(out.stdout.strip().splitlines()[0])
    except (subprocess.TimeoutExpired, IndexError, ValueError):
        return 0


def _rss_mb(pid: int) -> float:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return 0.0


def soak(minutes: float, schedule: list[dict] | None = None, poll_s: float = 60,
         probe_every_s: float = 600, log=print) -> dict:
    schedule = DEFAULT_SCHEDULE if schedule is None else schedule
    out = farm.farm_dir() / "soak" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    (out / "schedule.json").write_text(json.dumps(schedule, indent=1))
    dur = minutes * 60
    cmd = [sys.executable, "-m", "data_ops.replay", "up", "--schedule", str(out / "schedule.json"),
           "--duration", str(dur)]
    farm_log = open(out / "farm.log", "w")  # noqa: SIM115 - lives as long as the child process
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=farm_log)  # noqa: S603
    t0 = time.monotonic()
    fault_ends = sorted(f["at"] + f["seconds"] + 20 for f in schedule if f["at"] < dur)
    next_probe, prev, samples = 60.0, {}, []
    stalled_polls: dict[str, int] = {}
    with (out / "samples.jsonl").open("w") as f:
        while (t := time.monotonic() - t0) < dur - 5 and proc.poll() is None:
            active = [x for x in schedule if x["at"] <= t < x["at"] + x["seconds"] + 5]
            paths = _paths()
            growth = {p: v["bytes"] - prev.get(p, {}).get("bytes", 0) for p, v in paths.items()}
            if prev and not active:
                for p in paths:
                    if growth.get(p, 0) <= 0:
                        stalled_polls[p] = stalled_polls.get(p, 0) + 1
            rec = {"t": round(t), "ready": sum(v["ready"] for v in paths.values()), "paths": len(paths),
                   "fault_active": [x["fault"] for x in active],
                   "supervisor_rss_mb": round(_rss_mb(proc.pid), 1)}
            if t >= next_probe or (fault_ends and t >= fault_ends[0]):
                while fault_ends and t >= fault_ends[0]:
                    fault_ends.pop(0)
                cams = [f"cam{i:02d}" for i in range(1, 11)]
                rec["frames_3s"] = {f"{c}/{cam}": _frames(farm.url(c, cam))
                                    for c in ("h264", "h265") for cam in cams}
                next_probe = t + probe_every_s
            state = farm.farm_dir() / "run" / "state.json"
            if state.exists():
                st = json.loads(state.read_text())["streams"]
                rec["restarts"] = sum(s["restarts"] for s in st.values())
            f.write(json.dumps(rec) + "\n")
            f.flush()
            samples.append(rec)
            prev = paths
            log(f"[soak {rec['t']:>5}s] ready {rec['ready']}/{rec['paths']} faults {rec['fault_active']}"
                + (f" min frames {min(rec['frames_3s'].values())}" if "frames_3s" in rec else ""))
            time.sleep(max(1.0, poll_s - (time.monotonic() - t0 - t)))
    proc.wait(timeout=60)
    probes = [s for s in samples if "frames_3s" in s]
    summary = {
        "minutes": minutes, "streams": 20, "schedule": schedule, "farm_exit_code": proc.returncode,
        "polls": len(samples),
        "polls_all_ready_outside_faults": sum(1 for s in samples
                                              if not s["fault_active"] and s["ready"] == 20),
        "polls_outside_faults": sum(1 for s in samples if not s["fault_active"]),
        "unexpected_zero_growth_polls": stalled_polls,
        "probes": len(probes),
        "probe_min_frames": min((min(s["frames_3s"].values()) for s in probes), default=None),
        "probe_streams_below_60_frames": sorted({k for s in probes for k, v in s["frames_3s"].items()
                                                 if v < 60}),
        "supervisor_rss_mb_first_last": [samples[0]["supervisor_rss_mb"], samples[-1]["supervisor_rss_mb"]]
        if samples else None,
        "publisher_restarts_total": samples[-1].get("restarts") if samples else None,
        "out_dir": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
