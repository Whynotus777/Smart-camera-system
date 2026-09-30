"""Evidence clip stand-ins behind the `EvidenceStore` interface that T10 will implement.

T10's real design (ARCHITECTURE §2) keeps the last 60 s of *encoded* packets per camera
and remuxes clips out of it. M1 needs the same contract, so it has two thin versions:

- `FileLoopEvidence`: for looping file cameras. The source is transcoded once to an
  H.264 "master" with a keyframe every second (browsers can't play the PoC's MPEG-4
  Part 2, and 1 s GOPs make remux cut points exact). Clips are remuxed (`-c copy`) from
  the master through ffmpeg's concat demuxer, spanning loop boundaries when needed.
- `SegmentEvidence`: for live RTSP. The ingest role's ffmpeg copies the encoded stream
  into 2 s MPEG-TS segments; a clip is the remuxed concatenation of the overlapping ones.

Both write clips atomically (temp file, fsync, rename), so a crash never leaves a
half-written clip where the review page would serve it.
"""

from __future__ import annotations

import math
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from scs.app.crash import crashpoint
from scs.app.source import FFMPEG, VideoInfo, die_with_parent
from scs.app.store import Store


class EvidenceStore(Protocol):
    def ready(self, camera_id: str, t1: float) -> bool:
        """True once footage up to `t1` exists, or no more footage will come for it."""
        ...

    def export(self, camera_id: str, t0: float, t1: float, out: Path) -> tuple[float, float]:
        """Write a browser-playable MP4 covering [t0, t1] as far as footage allows.

        Returns the (start, end) timestamps the clip actually covers.
        """
        ...


def atomic_ffmpeg(args: list[str], out: Path, timeout: float = 120) -> None:
    """Run ffmpeg into a temp file next to `out`, fsync it, then rename into place."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.{os.getpid()}.tmp.mp4")
    try:
        subprocess.run(  # noqa: S603
            [FFMPEG, "-v", "error", "-nostdin", "-y", *args, str(tmp)],
            check=True,
            capture_output=True,
            timeout=timeout,
            preexec_fn=die_with_parent,
        )
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        crashpoint("clipper.before_rename")
        os.replace(tmp, out)
        dfd = os.open(out.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        tmp.unlink(missing_ok=True)


def clean_temp(clip_dir: Path) -> None:
    """Remove temp files left by a killed exporter (their jobs are still pending)."""
    if clip_dir.exists():
        for p in clip_dir.glob(".*.tmp.mp4"):
            p.unlink(missing_ok=True)


def build_master(source: str, info: VideoInfo, out: Path, encoder: str = "libx264") -> None:
    """Transcode the source once into an H.264 master with a keyframe every second."""
    if out.exists():
        return
    for stale in out.parent.glob(f".{out.stem}.*.tmp.mp4"):  # left by a killed build
        stale.unlink(missing_ok=True)
    fps = info.fps
    venc = (
        ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
        if encoder == "h264_nvenc"
        else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
    )
    atomic_ffmpeg(
        [
            "-i",
            source,
            "-map",
            "0:v:0",
            "-an",
            *venc,
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(round(fps)),
            "-keyint_min",
            str(round(fps)),
            "-force_key_frames",
            "expr:gte(t,n_forced*1)",
            "-fps_mode",
            "passthrough",
            "-movflags",
            "+faststart",
        ],
        out,
        timeout=3600,
    )


class FileLoopEvidence:
    def __init__(
        self,
        master: Path,
        info: VideoInfo,
        origin_ts: Callable[[], float | None],
        live_ts: Callable[[], float | None],
        loop: bool = True,
    ) -> None:
        self.master, self.info, self.loop = master, info, loop
        self.origin_ts, self.live_ts = origin_ts, live_ts

    def ready(self, camera_id: str, t1: float) -> bool:
        origin, live = self.origin_ts(), self.live_ts()
        if origin is None or live is None:
            return False
        if not self.loop and live >= origin + self.info.duration - 1 / self.info.fps:
            return True  # the file has ended; nothing more will come
        return live >= t1

    def export(self, camera_id: str, t0: float, t1: float, out: Path) -> tuple[float, float]:
        out.parent.mkdir(parents=True, exist_ok=True)
        origin = self.origin_ts()
        assert origin is not None
        dur = self.info.duration
        a = max(0.0, t0 - origin)
        b = t1 - origin if self.loop else min(t1 - origin, dur)
        a = math.floor(a)  # keyframes are on whole seconds of each loop
        lines = []
        for k in range(int(a // dur), int(math.ceil(b / dur))):
            inp = max(a - k * dur, 0.0)
            outp = min(b - k * dur, dur)
            if outp <= inp:
                continue
            inp = math.floor(inp)
            lines += [f"file '{self.master.resolve()}'", f"inpoint {inp:.3f}"]
            if outp < dur:
                lines.append(f"outpoint {min(math.ceil(outp), dur):.3f}")
        lst = out.with_name(f".{out.stem}.{os.getpid()}.txt")
        lst.write_text("\n".join(lines) + "\n")
        try:
            atomic_ffmpeg(
                ["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart"], out
            )
        finally:
            lst.unlink(missing_ok=True)
        return origin + a, origin + b


class SegmentEvidence:
    def __init__(self, store: Store, grace_s: float = 60.0, retain_s: float = 120.0) -> None:
        self.store, self.grace_s, self.retain_s = store, grace_s, retain_s

    def prune(self) -> int:
        """Bound disk use like a ring buffer: drop footage older than `retain_s` that no
        pending clip needs (exported clips are separate files). DB row first, then file:
        a crash in between leaves an orphan file, never a row pointing at nothing."""
        paths = self.store.prune_segments(time.time() - self.retain_s)
        for p in paths:
            Path(p).unlink(missing_ok=True)
        return len(paths)

    def ready(self, camera_id: str, t1: float) -> bool:
        segs = self.store.segments(camera_id, t1 - 1e-6, float("inf"))
        # Camera offline past the post-roll for longer than `grace_s`: export what exists.
        return bool(segs) or time.time() > t1 + self.grace_s

    def export(self, camera_id: str, t0: float, t1: float, out: Path) -> tuple[float, float]:
        segs = [s for s in self.store.segments(camera_id, t0, t1) if Path(s["path"]).exists()]
        if not segs:
            raise FileNotFoundError(f"no recorded footage for {camera_id} in [{t0:.1f}, {t1:.1f}]")
        out.parent.mkdir(parents=True, exist_ok=True)
        lst = out.with_name(f".{out.stem}.{os.getpid()}.txt")
        lst.write_text("".join(f"file '{Path(s['path']).resolve()}'\n" for s in segs))
        try:
            atomic_ffmpeg(
                ["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart"], out
            )
        finally:
            lst.unlink(missing_ok=True)
        return segs[0]["t0"], segs[-1]["t1"]
