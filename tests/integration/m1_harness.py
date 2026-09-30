"""Shared harness for the M1 integration tests (and `python -m tests.integration.chaos`).

The invariants in `check_invariants` are the M1 contract. They hold for whatever
implementation sits behind the flow, so when T02/T03/T05/T10 swap their pieces in, these
checks must stay green unchanged:

1. every event the pipeline derives from the frames it processed is persisted exactly once
   (compared with an uninterrupted reference run over the same frames);
2. one alert and one clip job per event;
3. every clip whose post-roll has been recorded exists, is browser-playable H.264 MP4,
   and covers ≥ 10 s before and ≥ 5 s after its event;
4. every review outcome the API acknowledged (HTTP 200) is stored exactly once, with that
   decision, and exported as exactly one canonical label file;
5. no orphan or half-written clip files.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from scs.app import config as cfgmod
from scs.app.labels import label_dir
from scs.app.roles import db_path, run_ingest
from scs.app.store import Store

ROLES = ("ingest", "clipper", "web")
MIN_PRE_S, MIN_POST_S = 10.0, 5.0  # M1 acceptance
REPO = Path(__file__).resolve().parents[2]


def have_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def synthetic_video(path: Path, loop_s: int = 20, fps: int = 20) -> Path:
    """A 'person' (bright box) walks into the zone at 6 s, stays until 14 s, leaves.

    No real people, so it's safe for CI; generated on the fly because video can't be
    committed (AGENTS.md rule 4). With the default zone the dwell event fires at ~11 s,
    so even the first event has ≥ 10 s of pre-roll.
    """
    if path.exists():
        return path
    # (x, t_from, t_to): walk in from the left, stand in the zone (x ≥ 0.45 W), walk out.
    steps = [(20, 2, 4), (180, 4, 6), (440, 6, 14), (180, 14, 16), (20, 16, 18)]
    vf = ",".join(
        ["drawbox=x=0:y=300:w=640:h=60:color=0x303030:t=fill"]
        + [
            f"drawbox=x={x}:y=110:w=90:h=220:color=0xE0E0E0:t=fill:enable='between(t,{a},{b - 0.001})'"
            for x, a, b in steps
        ]
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",  # noqa: S603, S607
            f"color=c=0x707070:s=640x360:r={fps}:d={loop_s}",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(fps),
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def http_json(method: str, url: str, body: dict | None = None, timeout: float = 30.0) -> tuple[int, object]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method=method,  # noqa: S310
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",  # noqa: S603, S607
            "stream=codec_name,pix_fmt:format=duration,format_name",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def children(pid: int) -> list[int]:
    out = []
    for task in Path(f"/proc/{pid}/task").glob("*"):
        try:
            out += [int(c) for c in (task / "children").read_text().split()]
        except OSError:
            pass
    return out


# ------------------------------------------------------------------------ the stack


class Stack:
    """The three M1 roles as separate processes; the test plays supervisor."""

    def __init__(self, workdir: Path, env: dict[str, str] | None = None) -> None:
        self.workdir, self.env = workdir, {**os.environ, **(env or {})}
        self.env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + self.env.get("PYTHONPATH", "")
        self.procs: dict[str, subprocess.Popen] = {}
        self.logs = workdir / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.kills: list[str] = []
        self.self_crashes = 0
        self._killed_by_us: dict[str, bool] = {}

    def start(self, role: str) -> None:
        log = open(self.logs / f"{role}.log", "a")  # noqa: SIM115
        self.procs[role] = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", "scs.app", "run", role, "--workdir", str(self.workdir)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=self.env,
        )

    def start_all(self) -> None:
        for r in ROLES:
            self.start(r)

    def ensure_running(self) -> None:
        for r in ROLES:
            p = self.procs.get(r)
            if p is None or p.poll() is not None:
                if p is not None and p.returncode == -9 and not self._killed_by_us.pop(r, False):
                    self.self_crashes += 1  # a crashpoint fired (scs.app.crash)
                self.start(r)

    def kill(self, target: str) -> None:
        """SIGKILL a role, or (`ingest-ffmpeg`) the decoder child of the ingest role."""
        if target == "ingest-ffmpeg":
            p = self.procs.get("ingest")
            kids = children(p.pid) if p and p.poll() is None else []
            if not kids:
                return
            os.kill(kids[0], signal.SIGKILL)
        else:
            p = self.procs.get(target)
            if p is None or p.poll() is not None:
                return
            p.kill()
            p.wait()
            self._killed_by_us[target] = True
        self.kills.append(target)

    def stop_and_drain(self, timeout: float = 180.0) -> None:
        """Stop the camera (ingest), let the clipper finish every due clip, then stop all."""
        self.kill("ingest")
        st = Store(db_path(self.workdir))
        try:
            cam = cfgmod.load(self.workdir).camera_id
            deadline = time.monotonic() + timeout
            while True:
                cp = st.load_checkpoint(cam)
                live = cp.ts if cp else 0.0
                if not any(j.t1 <= live for j in st.clip_jobs("pending")):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("clipper did not drain")
                time.sleep(0.2)
        finally:
            st.close()
            self.stop()

    def stop(self) -> None:
        for p in self.procs.values():
            if p.poll() is None:
                p.kill()
        for p in self.procs.values():
            p.wait()


class Reviewer(threading.Thread):
    """Reviews every alert whose clip is ready, retrying until the API acknowledges it."""

    def __init__(self, base_url: str, seed: int) -> None:
        super().__init__(daemon=True)
        self.base, self.rng = base_url, random.Random(seed)
        self.acked: dict[str, str] = {}
        self.stop_flag = threading.Event()
        self.attempts = 0

    def decision(self, alert_id: str) -> str:
        return "confirmed" if int(alert_id[:8], 16) % 2 else "dismissed"

    def run(self) -> None:
        while not self.stop_flag.is_set():
            self.review_ready()
            time.sleep(0.1)

    def review_ready(self) -> int:
        try:
            _, items = http_json("GET", self.base + "api/alerts")
        except OSError:
            return 0
        n = 0
        for it in items:  # type: ignore[union-attr]
            aid = it["alert"]["alert_id"]
            if aid in self.acked or not it["clip"] or it["clip"]["status"] != "done":
                continue
            body = {
                "decision": self.decision(aid),
                "reason": "m1-test",
                "reviewer": "chaos",
                "request_id": f"{aid}-{self.rng.random()}",
            }
            self.attempts += 1
            try:
                code, resp = http_json("POST", self.base + f"api/alerts/{aid}/review", body)
            except OSError:
                continue  # web was killed mid-request; retry next round
            if code == 200:
                self.acked[aid] = body["decision"]
                n += 1
        return n


# -------------------------------------------------------------------- invariants


@dataclass
class Report:
    events: int = 0
    alerts: int = 0
    clips: int = 0
    reviews: int = 0
    labels: int = 0
    final_position: int = 0
    errors: list[str] = field(default_factory=list)


def reference_events(video: Path, cfg: cfgmod.AppConfig, frames: int, workdir: Path) -> dict[str, dict]:
    """Uninterrupted in-process run over the same frames: what the pipeline *should* persist."""
    ref_dir = workdir / "reference"
    cfgmod.save(cfg.model_copy(update={"speed": 1e6}), ref_dir)
    run_ingest(ref_dir, stop_after_frames=frames)
    st = Store(db_path(ref_dir))
    try:
        return {eid: _event_key(st, eid) for eid in st.event_ids()}
    finally:
        st.close()


def _event_key(st: Store, eid: str) -> dict:
    e = st.event(eid)
    return {
        "type": e.type.value,
        "zone": e.zone_id,
        "track": e.track_id,
        "fire": e.data.get("fire"),
        "start": e.data.get("dwell_start"),
    }


DATA_MD_CLIP_KEYS = {"clip_id", "camera_profile", "fps", "label_source", "events"}
DATA_MD_EVENT_KEYS = {"type", "track_id", "t_start", "t_end", "actor_id", "visible", "label_source"}


def label_schema_errors(lab: dict) -> list[str]:
    """Validate one label file: T09's `eval.canonical.ClipLabels` when it's importable
    (i.e. once T09 has merged: then this is the contract test), else DATA.md's minimum."""
    try:
        from eval.canonical import ClipLabels  # type: ignore[import-not-found]
    except ImportError:
        ClipLabels = None  # noqa: N806
    if ClipLabels is not None:
        try:
            ClipLabels.model_validate(lab)
        except Exception as e:  # noqa: BLE001
            return [f"rejected by eval.canonical.ClipLabels: {e}"]
        return []
    errs = []
    if not DATA_MD_CLIP_KEYS <= set(lab):
        errs.append(f"missing DATA.md keys {DATA_MD_CLIP_KEYS - set(lab)}")
    if not (isinstance(lab.get("fps"), int | float) and lab["fps"] > 0):
        errs.append(f"fps {lab.get('fps')!r}")
    for ev in lab.get("events", []):
        if not DATA_MD_EVENT_KEYS <= set(ev):
            errs.append(f"event missing {DATA_MD_EVENT_KEYS - set(ev)}")
        if ev.get("visible") not in ("observed", "partially_observed", "not_observed"):
            errs.append(f"visible must be a scalar for this camera, got {ev.get('visible')!r}")
    return errs


def check_invariants(
    workdir: Path,
    acked: dict[str, str] | None,
    reference: dict[str, dict] | None,
    require_all_reviewed: bool = True,
    outages: list[tuple[float, float]] = (),  # type: ignore[assignment]
) -> Report:
    """Check the M1 invariants (module docstring).

    `outages`: wall-clock windows when a live camera was unreachable. Footage from them
    doesn't exist, so clips overlapping one are exempt from the coverage checks (they
    must still exist and be playable).
    """
    cfg = cfgmod.load(workdir)
    st = Store(db_path(workdir))
    rep = Report()
    err = rep.errors.append
    try:
        cp = st.load_checkpoint(cfg.camera_id)
        rep.final_position = -1 if cp is None else cp.media_frame
        if cp is not None and "origin_ts" in cp.state:  # file camera: nothing before it started
            outages = [*outages, (0.0, cp.state["origin_ts"])]
        ids = st.event_ids()
        rep.events = len(ids)
        if len(ids) != len(set(ids)):
            err("duplicate event ids")
        fires = [json.dumps(_event_key(st, eid)["fire"]) for eid in ids]
        if len(fires) != len(set(fires)):
            err("the same detection was persisted as more than one event (non-deterministic ids?)")
        if reference is not None:
            if not reference:
                err("reference run produced no events: the test would pass vacuously")
            got = {eid: _event_key(st, eid) for eid in ids}
            if set(got) != set(reference):
                err(
                    f"events differ from reference: missing={sorted(set(reference) - set(got))} "
                    f"extra={sorted(set(got) - set(reference))}"
                )
            for eid in set(got) & set(reference):
                if got[eid] != reference[eid]:
                    err(f"event {eid} differs: {got[eid]} vs reference {reference[eid]}")
        rows = st.alerts()
        rep.alerts = len(rows)
        if sorted(a.event_ids[0] for a, _, _ in rows) != sorted(ids):
            err("alerts are not 1:1 with events")
        live_ts = cp.ts if cp else 0.0
        done_paths = set()
        for alert, job, _rev in rows:
            if job is None:
                err(f"alert {alert.alert_id} has no clip job")
                continue
            if job.t1 <= live_ts and job.status != "done":
                err(f"clip {alert.alert_id} is {job.status} though its post-roll was recorded")
            if job.status != "done":
                continue
            rep.clips += 1
            p = Path(job.path or "")
            done_paths.add(p.name)
            if not p.exists():
                err(f"clip file missing: {p}")
                continue
            info = ffprobe(p)
            s = info["streams"][0]
            if (s["codec_name"], s["pix_fmt"]) != ("h264", "yuv420p") or "mp4" not in info["format"][
                "format_name"
            ]:
                err(f"clip {p.name} not browser-playable: {s}")
            assert job.clip_start is not None and job.clip_end is not None
            pre, post = alert.ts_open - job.clip_start, job.clip_end - alert.ts_open
            dur = float(info["format"]["duration"])
            if any(job.t0 < b and a < job.t1 for a, b in outages):
                continue
            if pre < MIN_PRE_S or post < MIN_POST_S:
                err(f"clip {p.name} covers {pre:.1f}s before / {post:.1f}s after (need ≥10/≥5)")
            if dur < (job.clip_end - job.clip_start) - 0.5:
                err(f"clip {p.name} is {dur:.1f}s, shorter than its recorded window")
        clip_files = {p.name for p in (workdir / "clips").glob("*.mp4") if not p.name.startswith(".")}
        if clip_files - done_paths:
            err(f"orphan clip files: {sorted(clip_files - done_paths)}")
        reviews = {r.alert_id: r.decision for r in st.reviews()}
        rep.reviews = len(reviews)
        if acked is not None:
            if reviews != acked:
                err(
                    f"stored reviews != acknowledged: stored-only={set(reviews) - set(acked)} "
                    f"acked-only={set(acked) - set(reviews)} "
                    f"changed={[k for k in set(reviews) & set(acked) if reviews[k] != acked[k]]}"
                )
            if require_all_reviewed and rep.clips != len(acked):
                err(f"{rep.clips} clips but {len(acked)} acknowledged reviews")
        ldir = label_dir(workdir / "labels", cfg.label_dataset_id)
        labels = {p.stem: json.loads(p.read_text()) for p in ldir.glob("*.json")} if ldir.exists() else {}
        rep.labels = len(labels)
        if set(labels) != set(reviews):
            err(f"labels != reviews: {set(labels) ^ set(reviews)}")
        for aid, lab in labels.items():
            for e in label_schema_errors(lab):
                err(f"label {aid}: {e}")
            if lab["label_source"] != "human" or lab["clip_id"] != aid:
                err(f"label {aid} header wrong")
            if bool(lab["events"]) != (reviews.get(aid) == "confirmed"):
                err(f"label {aid} events don't match decision {reviews.get(aid)}")
            if lab.get("extra", {}).get("review", {}).get("decision") != reviews.get(aid):
                err(f"label {aid} extra.review.decision != stored decision")
            for ev in lab["events"]:
                if not 0 <= ev["t_start"] <= ev["t_end"]:
                    err(f"label {aid} event times {ev['t_start']}..{ev['t_end']}")
    finally:
        st.close()
    return rep
