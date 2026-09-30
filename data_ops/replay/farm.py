"""Fake-camera farm: MEVA indoor clips served as looping RTSP streams, with scripted faults.

No real cameras for weeks, but T02 (ingest), T12 (soak/perf) and T13 (durability) need
RTSP sources that behave like cameras, including the bad parts. The farm is:

- one mediamtx server (Docker, MIT, bound to 127.0.0.1 only);
- one host ffmpeg publisher per stream, copying a pre-encoded loop file (cheap: no
  live transcoding), under a supervisor that restarts dead publishers;
- scripted faults, per stream or server-wide, from a schedule file or on demand:
    drop <s>      publisher killed for <s> seconds: path goes offline, readers disconnect
    stall <s>     publisher SIGSTOPped: RTSP session stays up but no frames arrive
                  (Reolink's silent stall; readers must detect staleness). Silent for
                  up to ~30 s (mediamtx readTimeout); longer stalls end in a disconnect.
    spike <s>     publisher switched to a ~4x bitrate variant for <s> seconds
    server <s>    mediamtx container killed for <s> seconds, then restarted

Streams: rtsp://127.0.0.1:8554/meva/<codec>/<cam>, codec in {h264, h265}, cam01..cam10.
Each loop file is a 5-minute MEVA clip re-encoded with NVENC at a fixed 2 s GOP.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from data_ops.budget import require
from data_ops.fetch.common import run, tool
from data_ops.manifest import Manifest, file_hashes
from data_ops.paths import data_root, dataset_dir

IMAGE = "bluenviron/mediamtx:1.21.1"
CONTAINER = "scs-replay-mediamtx"
RTSP_PORT = 8554
API_PORT = 9997
FAULTS = ("drop", "stall", "spike", "server")
VARIANTS = {  # name -> ffmpeg video args; fps is MEVA's 30, GOP 2 s like Reolink's default
    "h264": ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "3M", "-maxrate", "4M", "-bufsize", "6M"],
    "h265": ["-c:v", "hevc_nvenc", "-preset", "p4", "-b:v", "2M", "-maxrate", "3M", "-bufsize", "4M"],
    "h264_spike": ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "12M", "-maxrate", "16M",
                   "-bufsize", "24M"],
}
MTX_CONFIG = """\
logLevel: info
# A publisher silent for longer than this is disconnected. 30 s keeps "stall" faults silent
# (connection up, no frames) up to ~30 s; longer stalls turn into a disconnect, like real cameras.
readTimeout: 30s
api: yes
apiAddress: :9997
rtsp: yes
rtspAddress: :8554
rtmp: no
hls: no
webrtc: no
srt: no
# Ports are published on 127.0.0.1 only (see server_up), so no auth inside the container.
# Requests arrive from the Docker bridge, not localhost, so the API grant must not be IP-limited.
authInternalUsers:
- user: any
  pass:
  ips: []
  permissions:
  - action: publish
  - action: read
  - action: playback
  - action: api
paths:
  all_others:
"""


def farm_dir() -> Path:
    return data_root() / "replay"


def url(codec: str, cam: str, host: str = "127.0.0.1") -> str:
    return f"rtsp://{host}:{RTSP_PORT}/meva/{codec}/{cam}"


# --------------------------------------------------------------------------- clips


def pick_clips(n: int = 10) -> list[dict]:
    """One annotated clip per indoor camera (most interaction events first), only if downloaded."""
    root = dataset_dir("meva")
    man = json.loads((root / "MANIFEST.json").read_text())["files"]
    rows = [json.loads(line) for line in (root / "index" / "clips.jsonl").read_text().splitlines()]
    best: dict[str, dict] = {}
    for r in rows:
        rel = r["key"].split("/", 1)[1]
        if not r["annotated"] or rel not in man:
            continue
        r = {**r, "path": str(root / "raw" / rel), "score": sum(r["events"].values())}
        if r["camera"] not in best or r["score"] > best[r["camera"]]["score"]:
            best[r["camera"]] = r
    picked = sorted(best.values(), key=lambda r: r["camera"])[:n]
    return [{**r, "cam": f"cam{i + 1:02d}"} for i, r in enumerate(picked)]


def prepare(n: int = 10, gpu_wrapper: str | None = None, log=print) -> list[dict]:
    """Encode loop files for each picked clip (idempotent) and record them in a manifest."""
    out = farm_dir() / "clips"
    out.mkdir(parents=True, exist_ok=True)
    clips = pick_clips(n)
    if len(clips) < n:
        log(f"replay: only {len(clips)} indoor cameras have a downloaded annotated clip yet")
    require(len(clips) * 3 * 500 * 10**6, out, what="replay loop files")
    man_path = farm_dir() / "MANIFEST.json"
    man = Manifest.load_or_new(
        man_path, dataset_id="replay", license="CC-BY-4.0 (derived from MEVA)",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        attribution="Derived from MEVA (Kitware Inc. and IARPA), CC BY 4.0. Re-encoded for RTSP replay.",
        use="prod", notes="Loop files for the fake-camera farm (data_ops/replay).")
    ffmpeg = tool("ffmpeg")
    wrap = [gpu_wrapper, "shared", "--"] if gpu_wrapper else []
    for c in clips:
        for var, args in VARIANTS.items():
            dest = out / f"{c['cam']}_{var}.mp4"
            if not dest.exists():
                tmp = dest.with_suffix(".part.mp4")
                run([*wrap, ffmpeg, "-v", "error", "-y", "-fflags", "+genpts", "-i", c["path"], "-an", *args,
                     "-g", "60", "-bf", "0", "-movflags", "+faststart", str(tmp)])
                tmp.replace(dest)
            man.add(dest.name, bytes=dest.stat().st_size, sha256=file_hashes(dest)["sha256"],
                    source=c["path"], source_clip=c["clip"], source_camera=c["camera"], variant=var,
                    verified="encoded locally")
        log(f"  {c['cam']} <- {c['clip']} ({c['score']} target events)")
    man.save(man_path)
    (farm_dir() / "clips.json").write_text(json.dumps(clips, indent=1))
    return clips


# --------------------------------------------------------------------------- server


def server_up() -> None:
    cfg = farm_dir() / "mediamtx.yml"
    cfg.write_text(MTX_CONFIG)
    docker = tool("docker")
    subprocess.run([docker, "rm", "-f", CONTAINER], capture_output=True)  # noqa: S603
    run([docker, "run", "-d", "--name", CONTAINER, "-p", f"127.0.0.1:{RTSP_PORT}:8554",
         "-p", f"127.0.0.1:{API_PORT}:9997", "-v", f"{cfg}:/mediamtx.yml:ro", IMAGE])


def server_kill() -> None:
    subprocess.run([tool("docker"), "kill", CONTAINER], capture_output=True)  # noqa: S603


def server_start() -> None:
    subprocess.run([tool("docker"), "start", CONTAINER], capture_output=True)  # noqa: S603


def server_down() -> None:
    subprocess.run([tool("docker"), "rm", "-f", CONTAINER], capture_output=True)  # noqa: S603


# --------------------------------------------------------------------------- publishers + faults


@dataclass
class Publisher:
    cam: str
    codec: str
    files: dict[str, Path]  # "normal" / "spike" -> loop file
    proc: subprocess.Popen | None = None
    variant: str = "normal"
    down_until: float = 0.0  # drop fault: don't restart before this
    resume_at: float = 0.0  # stall fault: SIGCONT at this time
    unspike_at: float = 0.0
    restarts: int = 0

    @property
    def path(self) -> str:
        return f"meva/{self.codec}/{self.cam}"

    def start(self) -> None:
        f = self.files.get(self.variant, self.files["normal"])
        self.proc = subprocess.Popen(  # noqa: S603
            [tool("ffmpeg"), "-v", "error", "-re", "-stream_loop", "-1", "-i", str(f), "-c", "copy",
             "-f", "rtsp", "-rtsp_transport", "tcp", url(self.codec, self.cam)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGCONT)
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


@dataclass
class Fault:
    at: float  # seconds after farm start (schedule) or 0 for "now"
    kind: str
    seconds: float
    target: str = "*"  # "cam03", "h265/cam03", "*" (all), ignored for "server"

    def __post_init__(self) -> None:
        if self.kind not in FAULTS:
            raise ValueError(f"unknown fault {self.kind!r}; expected one of {FAULTS}")
        if self.seconds <= 0:
            raise ValueError("fault duration must be > 0")

    def matches(self, p: Publisher) -> bool:
        return self.target in ("*", p.cam, f"{p.codec}/{p.cam}")


def load_schedule(path: Path) -> list[Fault]:
    """JSON list of {"at": s, "fault": kind, "seconds": s, "target": "cam03"|"h265/cam03"|"*"}."""
    items = json.loads(Path(path).read_text())
    return sorted((Fault(at=float(i["at"]), kind=i["fault"], seconds=float(i["seconds"]),
                         target=i.get("target", "*")) for i in items), key=lambda f: f.at)


@dataclass
class Farm:
    pubs: list[Publisher]
    schedule: list[Fault] = field(default_factory=list)
    run_dir: Path = field(default_factory=lambda: farm_dir() / "run")
    t0: float = field(default_factory=time.monotonic)
    server_down_until: float = 0.0
    log: object = print

    def apply(self, f: Fault, now: float) -> None:
        self.log(f"[fault] {f.kind} {f.target} {f.seconds:g}s")
        if f.kind == "server":
            server_kill()
            self.server_down_until = now + f.seconds
            return
        for p in (p for p in self.pubs if f.matches(p)):
            if f.kind == "drop":
                p.stop()
                p.down_until = now + f.seconds
            elif f.kind == "stall" and p.alive():
                os.killpg(p.proc.pid, signal.SIGSTOP)
                p.resume_at = now + f.seconds
            elif f.kind == "spike" and "spike" in p.files:
                p.stop()
                p.variant, p.unspike_at = "spike", now + f.seconds
                p.start()

    def tick(self, now: float) -> None:
        while self.schedule and self.schedule[0].at <= now - self.t0:
            self.apply(self.schedule.pop(0), now)
        for req in sorted((self.run_dir / "requests").glob("*.json")):  # on-demand faults
            d = json.loads(req.read_text())
            req.unlink()
            fault = Fault(at=0, kind=d["fault"], seconds=float(d["seconds"]), target=d.get("target", "*"))
            self.apply(fault, now)
        if self.server_down_until and now >= self.server_down_until:
            server_start()
            self.server_down_until = 0.0
        for p in self.pubs:
            if p.resume_at and now >= p.resume_at and p.proc:
                os.killpg(p.proc.pid, signal.SIGCONT)
                p.resume_at = 0.0
            if p.unspike_at and now >= p.unspike_at:
                p.stop()
                p.variant, p.unspike_at = "normal", 0.0
                p.start()
            if not p.alive() and now >= p.down_until and not self.server_down_until:
                p.start()  # also restarts publishers that died (e.g. during a server fault)
                p.restarts += 1
        self.write_state(now)

    def write_state(self, now: float) -> None:
        state = {"uptime_s": round(now - self.t0, 1), "server_down": bool(self.server_down_until),
                 "streams": {p.path: {"url": url(p.codec, p.cam), "alive": p.alive(), "variant": p.variant,
                                      "stalled": bool(p.resume_at), "dropped": now < p.down_until,
                                      "restarts": p.restarts} for p in self.pubs}}
        tmp = self.run_dir / "state.json.tmp"
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(self.run_dir / "state.json")

    def serve(self, stop_after: float | None = None) -> None:
        (self.run_dir / "requests").mkdir(parents=True, exist_ok=True)
        for p in self.pubs:
            p.start()
        try:
            while stop_after is None or time.monotonic() - self.t0 < stop_after:
                self.tick(time.monotonic())
                time.sleep(0.2)
        finally:
            for p in self.pubs:
                p.stop()


def build_publishers(codecs: tuple[str, ...] = ("h264", "h265")) -> list[Publisher]:
    clips = json.loads((farm_dir() / "clips.json").read_text())
    d = farm_dir() / "clips"
    pubs = []
    for codec in codecs:
        for c in clips:
            files = {"normal": d / f"{c['cam']}_{codec}.mp4"}
            if codec == "h264":
                files["spike"] = d / f"{c['cam']}_h264_spike.mp4"
            pubs.append(Publisher(cam=c["cam"], codec=codec, files=files))
    return pubs


def request_fault(kind: str, seconds: float, target: str = "*") -> Path:
    Fault(at=0, kind=kind, seconds=seconds, target=target)  # validate
    req = farm_dir() / "run" / "requests"
    req.mkdir(parents=True, exist_ok=True)
    p = req / f"{time.time_ns()}.json"
    p.write_text(json.dumps({"fault": kind, "seconds": seconds, "target": target}))
    return p
