"""`scs` command line: the one-command demo and the individual M1 roles.

    scs demo --video demo_1.mp4            # whole flow, headless; prints the review URL
    scs demo --rtsp-env SCS_DEMO_RTSP      # same, from an RTSP camera / T14 replay farm
    scs run {ingest,clipper,web} --workdir W
    scs inject-event --workdir W [--at TS] # a test event without the detector
    scs export-labels --workdir W
    scs status --workdir W

`demo` is a tiny supervisor (T12 replaces it): it starts each role as its own process,
restarts any that die, and exits children with it (PR_SET_PDEATHSIG). RTSP URLs are
passed by environment variable name (contracts.CameraInstall convention) so credentials
never land in shell history or logs.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from scs.app import config as cfgmod
from scs.app.source import die_with_parent

ROLES = ("ingest", "clipper", "web")
REPO = Path(__file__).resolve().parents[3]


def resolve_video(name: str) -> Path:
    p = Path(name)
    for cand in (p, REPO / "tests" / "fixtures" / "video" / p.name):
        if cand.is_file():
            return cand.resolve()
    raise SystemExit(f"video not found: {name} (also looked in tests/fixtures/video/)")


def spawn_role(role: str, workdir: Path, env: dict[str, str] | None = None) -> subprocess.Popen:
    return subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "scs.app", "run", role, "--workdir", str(workdir)],
        preexec_fn=die_with_parent,
        env={**os.environ, **(env or {})},
    )


def _get_json(url: str, timeout: float = 2.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (local URL we built)
        return json.load(r)


def cmd_demo(a: argparse.Namespace) -> int:
    workdir = Path(a.workdir or f"runs/demo/{time.strftime('%Y%m%d-%H%M%S')}").resolve()
    if a.rtsp_env:
        source = f"env:{a.rtsp_env}"
    else:
        source = str(resolve_video(a.video))
    if cfgmod.config_path(workdir).exists():
        cfg = cfgmod.load(workdir)  # resuming an existing demo: keep its timeline and IDs
    else:
        cfg = cfgmod.AppConfig(source=source, speed=a.speed, port=a.port, host=a.host)
        cfgmod.save(cfg, workdir)
    if cfg.is_live:
        cfg.resolved_source()  # fail fast if the env var is missing
    env = {"SCS_ENCODER": "h264_nvenc"} if a.gpu else {}
    url = f"http://{cfg.host}:{cfg.port}/"
    print(f"workdir: {workdir}", flush=True)
    procs: dict[str, subprocess.Popen] = {r: spawn_role(r, workdir, env) for r in ROLES}
    stopping = False

    def _stop(*_: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    announced = False
    deadline = time.monotonic() + a.timeout if a.timeout else None
    try:
        while not stopping:
            for role, p in procs.items():
                if p.poll() is not None:
                    print(f"[demo] {role} exited ({p.returncode}); restarting", file=sys.stderr, flush=True)
                    procs[role] = spawn_role(role, workdir, env)
            if not announced:
                try:
                    items = _get_json(url + "api/alerts")
                    if any(it["clip"] and it["clip"]["status"] == "done" for it in items):
                        print(f"Review queue: {url}", flush=True)
                        announced = True
                        if a.exit_when_ready:
                            return 0
                except OSError:
                    pass
            if deadline and time.monotonic() > deadline:
                print("[demo] timed out before the first review item was ready", file=sys.stderr)
                return 1
            time.sleep(0.5)
    finally:
        for p in procs.values():
            p.terminate()
        for p in procs.values():
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    workdir = Path(a.workdir).resolve()
    if a.role == "ingest":
        from scs.app.roles import run_ingest

        run_ingest(workdir)
    elif a.role == "clipper":
        from scs.app.roles import run_clipper

        run_clipper(workdir)
    else:
        from scs.app.web import run_web

        run_web(workdir)
    return 0


def cmd_inject(a: argparse.Namespace) -> int:
    from scs.app.roles import db_path
    from scs.app.store import Store
    from scs.app.stubs import stable_id
    from scs.contracts import Alert, Event, EventType

    workdir = Path(a.workdir).resolve()
    cfg = cfgmod.load(workdir)
    store = Store(db_path(workdir))
    cp = store.load_checkpoint(cfg.camera_id)
    ts = a.at if a.at is not None else (cp.ts if cp else time.time())
    ev = Event(
        event_id=stable_id(cfg.camera_id, "inject", ts),
        type=EventType.ZONE_ENTER,
        camera_id=cfg.camera_id,
        ts=ts,
        zone_id=cfg.zone.id,
        source="scs.app.inject@m1",
        data={"rule": "injected", "dwell_start_ts": ts},
    )
    al = Alert(
        alert_id=stable_id("alert", ev.event_id),
        site_id=cfg.site_id,
        ts_open=ts,
        camera_ids=[cfg.camera_id],
        score=1.0,
        reason_codes=["injected_test_event"],
        event_ids=[ev.event_id],
    )
    new = store.inject(ev, al, (ts - cfg.pre_roll_s, ts + cfg.post_roll_s))
    print(json.dumps({"event_id": ev.event_id, "alert_id": al.alert_id, "new": new}))
    return 0


def cmd_export(a: argparse.Namespace) -> int:
    from scs.app.web import App

    app = App(Path(a.workdir).resolve())
    for p in app.export():
        print(p)
    return 0


def cmd_status(a: argparse.Namespace) -> int:
    from scs.app.roles import dump_status

    print(dump_status(Path(a.workdir).resolve()))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scs", description="Smart Camera System (M1 walking skeleton)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run the whole flow headless and print the review URL")
    src = d.add_mutually_exclusive_group()
    src.add_argument("--video", default="demo_1.mp4", help="file (looped like a camera)")
    src.add_argument("--rtsp-env", help="name of an env var holding an rtsp:// URL")
    d.add_argument("--workdir", help="default: runs/demo/<timestamp>; reuse one to resume")
    d.add_argument("--speed", type=float, default=1.0, help="file pacing (1 = real time)")
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--port", type=int, default=8765)
    d.add_argument("--gpu", action="store_true", help="NVENC for clip masters; run under scripts/gpu shared")
    d.add_argument(
        "--exit-when-ready", action="store_true", help="exit once the first review item has a clip"
    )
    d.add_argument("--timeout", type=float, default=None, help="give up after N s without a review item")
    d.set_defaults(fn=cmd_demo)
    r = sub.add_parser("run", help="run one role in the foreground")
    r.add_argument("role", choices=ROLES)
    r.add_argument("--workdir", required=True)
    r.set_defaults(fn=cmd_run)
    i = sub.add_parser("inject-event", help="insert a test event/alert (bypasses the detector)")
    i.add_argument("--workdir", required=True)
    i.add_argument("--at", type=float, default=None, help="event ts (default: the camera's current position)")
    i.set_defaults(fn=cmd_inject)
    e = sub.add_parser("export-labels", help="write reviewed alerts as canonical labels")
    e.add_argument("--workdir", required=True)
    e.set_defaults(fn=cmd_export)
    s = sub.add_parser("status")
    s.add_argument("--workdir", required=True)
    s.set_defaults(fn=cmd_status)
    a = ap.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    sys.exit(main())
