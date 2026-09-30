"""Durability test driver: kill -9 random M1 processes mid-flow, restart, check invariants.

    python tests/integration/m1_chaos.py --runs 20             # CPU, synthetic video
    python tests/integration/m1_chaos.py --runs 20 --video tests/fixtures/video/demo_1.mp4

Each run: start ingest + clipper + web as separate processes on a looping video at N×
speed, with a reviewer client posting decisions as clips become ready. Every
0.2–1.2 s one target is SIGKILLed: a role, or the ingest role's ffmpeg decoder. It is
restarted after 0–0.5 s (sometimes while the corpse is still being reaped, which the
per-role lock has to handle). When the ingest has processed `loops` loops the chaos
stops, the stack drains, and `harness.check_invariants` compares the result to an
uninterrupted reference run over exactly the same frames.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

from m1_harness import (
    Report,
    Reviewer,
    Stack,
    check_invariants,
    free_port,
    have_ffmpeg,
    reference_events,
    synthetic_video,
)
from scs.app import config as cfgmod
from scs.app.roles import db_path, file_info
from scs.app.store import Store

TARGETS = ("ingest", "clipper", "web", "ingest-ffmpeg")
# Self-kill probabilities at the durable boundaries (scs.app.crash); first match wins.
# Ingest commits ~16x/s at 8x speed but only ~3 per run carry events: those get p=0.5.
# Drain budget after the chaos stops. On this shared HDD box (load avg ~30 while other agents
# ran) 60 s was too short for a web role that kept crashing at its crash points.
DRAIN_S = 180
CRASHPOINTS = "ingest.*.event=0.5,ingest.*=0.02,clipper.*=0.25,web.*=0.25"


def _position(workdir: Path, camera_id: str) -> int:
    st = Store(db_path(workdir))
    try:
        cp = st.load_checkpoint(camera_id)
        return -1 if cp is None else cp.media_frame
    finally:
        st.close()


def _drained(workdir: Path, reviewer: Reviewer) -> bool:
    st = Store(db_path(workdir))
    try:
        cp = st.load_checkpoint(cfgmod.load(workdir).camera_id)
        rows = st.alerts()
    finally:
        st.close()
    live = cp.ts if cp else 0.0
    for a, job, _ in rows:
        if job and job.t1 <= live and job.status == "pending":
            return False
        if job and job.status == "done" and a.alert_id not in reviewer.acked:
            return False
    return True


def run_chaos(
    workdir: Path,
    video: Path,
    seed: int,
    loops: int = 3,
    speed: float = 8.0,
    kill_gap: tuple[float, float] = (0.2, 1.2),
    timeout_s: float = 240.0,
    env: dict[str, str] | None = None,
    crashpoints: str = CRASHPOINTS,
) -> tuple[Report, dict]:
    rng = random.Random(seed)
    port = free_port()
    cfg = cfgmod.AppConfig(source=str(video), speed=speed, port=port)
    cfgmod.save(cfg, workdir)
    n_frames = file_info(workdir, str(video)).n_frames
    target_pos = loops * n_frames
    stack = Stack(workdir, {**(env or {}), "SCS_CRASHPOINTS": crashpoints})
    stack.start_all()
    reviewer = Reviewer(f"http://127.0.0.1:{port}/", seed)
    reviewer.start()
    t0 = time.monotonic()
    liveness: str | None = None  # "didn't finish in time": reported separately from integrity
    try:
        # chaos phase
        while _position(workdir, cfg.camera_id) < target_pos:
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(f"no progress to frame {target_pos} after {timeout_s}s")
            time.sleep(rng.uniform(*kill_gap))
            stack.kill(rng.choice(TARGETS))
            time.sleep(rng.uniform(0, 0.5))
            stack.ensure_running()
        # stop the camera: last word from ingest is its last committed checkpoint
        stack.kill("ingest")
        stack.procs.pop("ingest")
        # one more clipper kill/restart, so its startup cleanup runs after the last kill
        stack.kill("clipper")
        stack.start("clipper")
        if stack.procs["web"].poll() is not None:
            stack.start("web")
        deadline = time.monotonic() + DRAIN_S
        while not _drained(workdir, reviewer):
            if time.monotonic() > deadline:
                raise TimeoutError(f"stack did not drain within {DRAIN_S} s")
            for r in ("clipper", "web"):
                if stack.procs[r].poll() is not None:
                    stack.start(r)
            time.sleep(0.2)
        time.sleep(0.5)  # let a straggling clipper finish its temp-file cleanup
    except TimeoutError as e:
        liveness = str(e)
    finally:
        reviewer.stop_flag.set()
        reviewer.join(5)
        stack.stop()
    final = _position(workdir, cfg.camera_id)
    ref = reference_events(video, cfg, final + 1, workdir)
    if liveness is None:
        rep = check_invariants(workdir, reviewer.acked, ref)
    else:
        # Cut short: a review may be stored with its response still in flight, so only require
        # acknowledged ⊆ stored (same decision) on top of every other invariant.
        rep = check_invariants(workdir, None, ref, require_all_reviewed=False)
        st = Store(db_path(workdir))
        stored = {r.alert_id: r.decision for r in st.reviews()}
        st.close()
        if any(stored.get(k) != v for k, v in reviewer.acked.items()):
            rep.errors.append(f"acknowledged reviews missing or changed: {reviewer.acked} vs {stored}")
    stats = {
        "liveness": liveness,
        "seed": seed,
        "kills": len(stack.kills),
        "self_crashes": stack.self_crashes,
        "by_target": {t: stack.kills.count(t) for t in TARGETS},
        "review_posts": reviewer.attempts,
        "wall_s": round(time.monotonic() - t0, 1),
        "frames": final + 1,
    }
    return rep, stats


def _git_sha() -> str:
    import subprocess

    r = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent
    )
    return r.stdout.strip() or "unknown (not a git checkout)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--speed", type=float, default=8.0)
    ap.add_argument("--video", type=Path, default=None, help="default: synthetic CI clip")
    ap.add_argument("--out", type=Path, default=None, help="write per-run JSON results here")
    a = ap.parse_args(argv)
    if not have_ffmpeg():
        print("ffmpeg/ffprobe not found", file=sys.stderr)
        return 2
    results, failed = [], 0
    tmp = Path(tempfile.mkdtemp(prefix="scs-chaos-"))
    video = a.video.resolve() if a.video else synthetic_video(tmp / "synthetic.mp4")
    for i in range(a.runs):
        wd = tmp / f"run{i:02d}"
        try:
            rep, stats = run_chaos(wd, video, a.seed + i, a.loops, a.speed)
        except Exception as e:  # noqa: BLE001 (a hung/crashed run is a failed run, with its logs kept)
            rep, stats = Report(errors=[f"harness: {e!r}"]), {"seed": a.seed + i}
        ok = not rep.errors and not stats.get("liveness")
        failed += not ok
        row = {"run": i, "ok": ok, "integrity_ok": not rep.errors, **stats, **rep.__dict__}
        results.append(row)
        print(json.dumps(row), flush=True)
        if not ok:
            print(f"run {i} FAILED; workdir and logs kept in {wd}", file=sys.stderr)
            break
        shutil.rmtree(wd)
    summary = {
        "commit": _git_sha(),
        "runs": len(results),
        "passed": sum(r["ok"] for r in results),
        "consecutive_pass": failed == 0,
        "kills": sum(r.get("kills", 0) for r in results),
        "self_crashes": sum(r.get("self_crashes", 0) for r in results),
        "integrity_failures": sum(not r["integrity_ok"] for r in results),
        "liveness_failures": sum(bool(r.get("liveness")) for r in results),
        "events": sum(r["events"] for r in results),
        "clips": sum(r["clips"] for r in results),
        "reviews": sum(r["reviews"] for r in results),
    }
    print(json.dumps({"summary": summary}))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"summary": summary, "runs": results}, indent=2))
    return 0 if failed == 0 and len(results) == a.runs else 1


if __name__ == "__main__":
    sys.exit(main())
