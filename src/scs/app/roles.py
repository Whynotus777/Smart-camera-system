"""The walking skeleton's role processes: ingest and clipper (web lives in `web.py`).

Each role is a separate OS process so the durability test can kill -9 any one of them.
Each takes an exclusive `flock` on `<workdir>/locks/<role>.lock` for its lifetime, so a
restarted role can never run alongside a not-yet-dead predecessor.

ingest   frames → detector → tracker → dwell engine → one transaction per checkpoint:
         (events, alerts, pending clip jobs, checkpoint + tracker/engine state).
clipper  pending clip job whose post-roll has been recorded → atomic MP4 → mark done.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from scs.app import config as cfgmod
from scs.app.crash import crashpoint
from scs.app.evidence import EvidenceStore, FileLoopEvidence, SegmentEvidence, build_master, clean_temp
from scs.app.pipeline import M1Pipeline
from scs.app.source import (
    FileLoopSource,
    LiveSource,
    Segment,
    VideoInfo,
    analytics_size,
    median_background,
    probe,
)
from scs.app.store import Checkpoint, Store
from scs.app.stubs import BackgroundDiffDetector, DwellEngine
from scs.contracts import Alert, Event, FrameRef


def db_path(workdir: Path) -> Path:
    return workdir / "scs.db"


def clip_dir(workdir: Path) -> Path:
    return workdir / "clips"


def log(role: str, msg: str) -> None:
    print(f"[{role} {os.getpid()}] {msg}", file=sys.stderr, flush=True)


@contextmanager
def role_lock(workdir: Path, role: str, wait_s: float = 30.0) -> Iterator[None]:
    d = workdir / "locks"
    d.mkdir(parents=True, exist_ok=True)
    f = open(d / f"{role}.lock", "w")  # noqa: SIM115 (held for the process lifetime)
    deadline = time.monotonic() + wait_s
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() > deadline:
                raise SystemExit(f"{role}: another {role} process holds the lock") from None
            time.sleep(0.1)
    try:
        yield
    finally:
        f.close()


def file_info(workdir: Path, source: str) -> VideoInfo:
    return probe(source, cache=workdir / "probe.json")


# --------------------------------------------------------------------------- ingest


def run_ingest(workdir: Path, stop_after_frames: int | None = None) -> None:
    cfg = cfgmod.load(workdir)
    with role_lock(workdir, "ingest"):
        store = Store(db_path(workdir))
        try:
            if cfg.is_live:
                _ingest_live(cfg, store, workdir, stop_after_frames)
            else:
                _ingest_file(cfg, store, workdir, stop_after_frames)
        finally:
            store.close()


def _background(workdir: Path, cfg: cfgmod.AppConfig, info: VideoInfo) -> np.ndarray:
    """Median background, cached atomically (deterministic, so recomputing is also fine)."""
    p = workdir / "background.npy"
    if p.exists():
        return np.load(p)
    w, h = analytics_size(info.width, info.height, cfg.analytics_width)
    bg = median_background(cfg.source, info, w, h)
    tmp = workdir / f".background.{os.getpid()}.npy"
    np.save(tmp, bg)
    os.replace(tmp, p)
    return bg


def _ingest_file(cfg: cfgmod.AppConfig, store: Store, workdir: Path, stop_after: int | None) -> None:
    info = file_info(workdir, cfg.source)
    engine = DwellEngine(cfg.site_id, cfg.zone, cfg.min_dwell_s, cfg.cooldown_s)
    cp = store.load_checkpoint(cfg.camera_id)
    pipe = M1Pipeline(
        BackgroundDiffDetector(_background(workdir, cfg, info)),
        engine,
        last_epoch=None if cp is None else cp.epoch,
    )
    if cp is None:
        origin, start = time.time(), 0
    else:
        origin, start = cp.state["origin_ts"], cp.media_frame + 1
        pipe.restore(cp.state)
    log("ingest", f"file source {cfg.source} from media frame {start} ({info.n_frames} frames/loop)")
    src = FileLoopSource(
        cfg.camera_id, cfg.source, info, cfg.analytics_width, origin, start, cfg.speed, cfg.loop
    )
    _run(
        cfg,
        store,
        src.frames(),
        pipe,
        max(1, round(info.fps / 2)),
        stop_after,  # checkpoint 2x per media s
        extra_state={"origin_ts": origin},
        position=lambda r: r.epoch * info.n_frames + r.seq,
    )


def _ingest_live(cfg: cfgmod.AppConfig, store: Store, workdir: Path, stop_after: int | None) -> None:
    engine = DwellEngine(cfg.site_id, cfg.zone, cfg.min_dwell_s, cfg.cooldown_s)
    cp = store.load_checkpoint(cfg.camera_id)
    if cp is not None:
        engine.restore(cp.state["engine"])
        engine.reset_transient()  # the process was down: whatever dwell was running is broken
    pipe = M1Pipeline(BackgroundDiffDetector(None), engine, learn_background=True)
    seg_root = workdir / "segments"
    recover_segments(store, seg_root, cfg.camera_id)

    def on_segments(segs: list[Segment]) -> None:
        store.add_segments([(cfg.camera_id, s.epoch, s.idx, s.t0, s.t1, str(s.path)) for s in segs])

    src = LiveSource(
        cfg.camera_id,
        cfg.resolved_source(),
        cfg.analytics_width,
        seg_root,
        cfg.segment_s,
        new_epoch=lambda: store.new_epoch(cfg.camera_id),
        on_segments=on_segments,
    )
    log("ingest", f"live source {cfg.camera_id} from ${cfg.source[4:]} (url not logged)")
    _run(
        cfg,
        store,
        src.frames(),
        pipe,
        commit_every=10,
        stop_after=stop_after,
        extra_state={},
        position=lambda r: r.seq,
    )


def _run(
    cfg: cfgmod.AppConfig,
    store: Store,
    frames: Iterator[tuple[FrameRef, np.ndarray]],
    pipe: M1Pipeline,
    commit_every: int,
    stop_after: int | None,
    extra_state: dict,
    position: Callable[[FrameRef], int],
) -> None:
    """Feed frames to the streaming pipeline; commit its output per checkpoint (module docstring).

    `position(ref)` is what the checkpoint stores (file sources resume at position + 1).
    """
    events: list[Event] = []
    alerts: list[Alert] = []
    windows: dict[str, tuple[float, float]] = {}
    n = 0
    for ref, img in frames:
        for item in pipe.process(ref, img):
            if isinstance(item, Alert):
                alerts.append(item)
                windows[item.alert_id] = (item.ts_open - cfg.pre_roll_s, item.ts_open + cfg.post_roll_s)
            else:
                events.append(item)
        n += 1
        if events or alerts or n % commit_every == 0:
            cp = Checkpoint(cfg.camera_id, ref.epoch, position(ref), ref.ts, {**extra_state, **pipe.state()})
            kind = ".event" if events else ""  # commits that carry events are the rare, risky ones
            crashpoint(f"ingest.before_commit{kind}")
            new = store.commit_frames(cp, events, alerts, windows)
            crashpoint(f"ingest.after_commit{kind}")
            for e in events:
                log(
                    "ingest",
                    f"event {e.event_id[:8]} ts={e.ts:.2f} seq={ref.seq} {'new' if new else 'replayed'}",
                )
            events, alerts, windows = [], [], {}
        if stop_after is not None and n >= stop_after:
            return


def recover_segments(store: Store, seg_root: Path, camera_id: str) -> None:
    """Register segments that ffmpeg finished but a killed ingest never recorded."""
    base = seg_root / camera_id
    if not base.exists():
        return
    rows = []
    for d in sorted(base.iterdir()):
        w0, lst = d / "wall0", d / "list.csv"
        if not (w0.exists() and lst.exists()):
            continue
        wall0 = float(w0.read_text())
        epoch = int(d.name[1:])
        for idx, line in enumerate(lst.read_text().splitlines()):
            p = line.split(",")
            if len(p) >= 3:
                rows.append((camera_id, epoch, idx, wall0 + float(p[1]), wall0 + float(p[2]), str(d / p[0])))
    if rows:
        store.add_segments(rows)


# -------------------------------------------------------------------------- clipper


def make_evidence(cfg: cfgmod.AppConfig, store: Store, workdir: Path, encoder: str) -> EvidenceStore:
    if cfg.is_live:
        return SegmentEvidence(store)
    info = file_info(workdir, cfg.source)
    master = workdir / "master.mp4"
    build_master(cfg.source, info, master, encoder)

    def origin() -> float | None:
        cp = store.load_checkpoint(cfg.camera_id)
        return None if cp is None else cp.state["origin_ts"]

    def live() -> float | None:
        cp = store.load_checkpoint(cfg.camera_id)
        return None if cp is None else cp.ts

    return FileLoopEvidence(master, info, origin, live, cfg.loop)


def run_clipper(workdir: Path, once: bool = False, max_attempts: int = 5) -> None:
    cfg = cfgmod.load(workdir)
    with role_lock(workdir, "clipper"):
        store = Store(db_path(workdir))
        encoder = os.environ.get("SCS_ENCODER", "libx264")
        evidence = make_evidence(cfg, store, workdir, encoder)
        clean_temp(clip_dir(workdir))
        while True:
            for job in store.clip_jobs("pending"):
                if not evidence.ready(job.camera_id, job.t1):
                    continue
                store.clip_attempt(job.alert_id)
                crashpoint("clipper.after_attempt")
                out = clip_dir(workdir) / f"{job.alert_id}.mp4"
                try:
                    start, end = evidence.export(job.camera_id, job.t0, job.t1, out)
                except Exception as e:  # noqa: BLE001 (one bad clip must not stop the queue)
                    log("clipper", f"clip {job.alert_id[:8]} failed: {e}")
                    if job.attempts + 1 >= max_attempts:
                        store.clip_failed(job.alert_id, repr(e))
                    continue
                crashpoint("clipper.after_export")  # file in place, row still pending
                store.clip_done(job.alert_id, str(out), start, end)
                crashpoint("clipper.after_done")
                log("clipper", f"clip {job.alert_id[:8]} [{start:.1f}, {end:.1f}] done")
            if once:
                return
            time.sleep(0.2)


def status(workdir: Path) -> dict:
    store = Store(db_path(workdir))
    rows = store.alerts()
    cp = store.load_checkpoint(cfgmod.load(workdir).camera_id)
    return {
        "events": len(store.event_ids()),
        "alerts": len(rows),
        "clips_done": sum(1 for _, j, _ in rows if j and j.status == "done"),
        "clips_pending": sum(1 for _, j, _ in rows if j and j.status == "pending"),
        "clips_failed": sum(1 for _, j, _ in rows if j and j.status == "failed"),
        "reviews": sum(1 for _, _, r in rows if r),
        "checkpoint": None if cp is None else {"epoch": cp.epoch, "media_frame": cp.media_frame, "ts": cp.ts},
    }


def dump_status(workdir: Path) -> str:
    return json.dumps(status(workdir), indent=2)
