"""`FrameSource` stand-in built on the ffmpeg CLI, until T02's sources land.

Two modes, both yielding small grayscale analytics frames (the detector stand-in needs
nothing more):

- **File** (`FileLoopSource`): the file loops forever like a camera. Every frame has a
  global *media frame* index `m = loop * n_frames + i`; its identity is `epoch = loop`,
  `seq = i` (the jump back to frame 0 is a discontinuity, like a reconnect, so journey
  state resets there as it would after a camera drop), and `FrameRef.ts` is a virtual
  camera clock `origin + m / fps`. That makes the whole pipeline replayable: after a crash
  the ingest role resumes at `checkpoint + 1` and re-derives exactly the same frames, IDs
  and timestamps. (Consequence: when `speed > 1`, `ts` runs ahead of the wall clock. The
  contract says wall clock at decode; the file stub trades that for determinism.)
- **Live** (`LiveSource`, rtsp://): reconnects with backoff; each connection is a new
  `epoch` and `seq` restarts at 0 (ADR 0003). The same ffmpeg process also copies the
  encoded stream into short segments: the evidence "packet tap" that T10's ring buffer
  replaces. Frames missed while the process is down are gone, as with a real camera.
"""

from __future__ import annotations

import ctypes
import json
import signal
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from scs.contracts import FrameRef

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


def die_with_parent() -> None:
    """preexec_fn: SIGKILL this child when the parent dies (so kill -9 leaves no orphans)."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    n_frames: int
    codec: str

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps


def probe(path: str, cache: Path | None = None) -> VideoInfo:
    """Stream specs + exact decoded frame count (cached: counting decodes the whole file)."""
    if cache is not None and cache.exists():
        return VideoInfo(**json.loads(cache.read_text()))
    out = subprocess.run(  # noqa: S603
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_frames,codec_name",
            "-of",
            "json",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    s = json.loads(out.stdout)["streams"][0]
    rate = s["avg_frame_rate"] if s.get("avg_frame_rate", "0/0") != "0/0" else s["r_frame_rate"]
    info = VideoInfo(
        int(s["width"]), int(s["height"]), float(Fraction(rate)), int(s["nb_read_frames"]), s["codec_name"]
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(info.__dict__))
    return info


def analytics_size(info_w: int, info_h: int, width: int) -> tuple[int, int]:
    h = round(info_h * width / info_w / 2) * 2
    return width, max(h, 2)


def _read_exact(pipe, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = pipe.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def decode_gray(
    path: str,
    width: int,
    height: int,
    start_frame: int = 0,
    fps: float = 0,
    extra_in: list[str] | None = None,
) -> Iterator[np.ndarray]:
    """Decode a file (from `start_frame`, frame-accurately) to HxW uint8 grayscale frames."""
    ss = [] if start_frame == 0 else ["-ss", f"{(start_frame - 0.5) / fps:.6f}"]
    cmd = [
        FFMPEG,
        "-v",
        "error",
        "-nostdin",
        *(extra_in or []),
        *ss,
        "-i",
        path,
        "-map",
        "0:v:0",
        "-vf",
        f"scale={width}:{height}:flags=area,format=gray",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, preexec_fn=die_with_parent)  # noqa: S603
    try:
        assert proc.stdout is not None
        while (buf := _read_exact(proc.stdout, width * height)) is not None:
            yield np.frombuffer(buf, np.uint8).reshape(height, width)
        # EOF must mean end of file, never "decoder died": skipping the rest of a loop would
        # silently drop frames and make re-derived events differ after a restart.
        if (rc := proc.wait()) != 0:
            raise RuntimeError(f"ffmpeg decoder exited with {rc} on {path}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def median_background(path: str, info: VideoInfo, width: int, height: int, samples: int = 25) -> np.ndarray:
    """Median of frames sampled evenly across the file: people who move get voted out."""
    frames = []
    for k in range(samples):
        i = int(k * info.n_frames / samples)
        f = next(decode_gray(path, width, height, start_frame=i, fps=info.fps), None)
        if f is not None:
            frames.append(f)
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


class FileLoopSource:
    """Looping file camera with exact resume (see module docstring)."""

    def __init__(
        self,
        camera_id: str,
        path: str,
        info: VideoInfo,
        width: int,
        origin_ts: float,
        start_media_frame: int = 0,
        speed: float = 1.0,
        loop: bool = True,
    ) -> None:
        self.camera_id, self.path, self.info = camera_id, path, info
        self.w, self.h = analytics_size(info.width, info.height, width)
        self.origin_ts, self.start, self.speed, self.loop = origin_ts, start_media_frame, speed, loop

    def media_ts(self, m: int) -> float:
        return self.origin_ts + m / self.info.fps

    def frames(self) -> Iterator[tuple[FrameRef, np.ndarray]]:
        n = self.info.n_frames
        m = self.start
        wall0, m0 = time.monotonic(), m
        while True:
            loop_idx, i = divmod(m, n)
            if loop_idx > 0 and not self.loop:
                return
            for img in decode_gray(self.path, self.w, self.h, start_frame=i, fps=self.info.fps):
                due = wall0 + (m - m0) / self.info.fps / self.speed
                if (delay := due - time.monotonic()) > 0:
                    time.sleep(delay)
                # Each loop is a discontinuity in the scene, like a reconnect: epoch = loop index.
                ref = FrameRef(
                    camera_id=self.camera_id,
                    epoch=m // n,
                    seq=m % n,
                    frame_idx=m % n,
                    ts=self.media_ts(m),
                    ts_mono=time.monotonic(),
                    width=self.info.width,
                    height=self.info.height,
                    transform=f"gray:scale{self.w}x{self.h}",
                )
                yield ref, img
                m += 1
                if m % n == 0:
                    break  # next loop iteration re-opens the file at frame 0
            else:
                # File ended early (fewer frames than probed): move on to the next loop.
                m = (m // n + 1) * n


@dataclass(frozen=True)
class Segment:
    epoch: int
    idx: int
    t0: float
    t1: float
    path: Path


class LiveSource:
    """RTSP camera with reconnect + encoded segment tap (see module docstring)."""

    def __init__(
        self,
        camera_id: str,
        url: str,
        width: int,
        segment_root: Path,
        segment_s: float,
        new_epoch: Callable[[], int],
        on_segments: Callable[[list[Segment]], None],
        io_timeout_s: float = 5.0,
        main_size: tuple[int, int] | None = None,
    ) -> None:
        self.camera_id, self.url, self.width = camera_id, url, width
        self.segment_root, self.segment_s = segment_root, segment_s
        self.new_epoch, self.on_segments = new_epoch, on_segments
        self.io_timeout_s = io_timeout_s
        self.main_size = main_size
        self.reconnects = 0

    def _probe_live(self) -> tuple[int, int, float]:
        out = subprocess.run(  # noqa: S603
            [
                FFPROBE,
                "-v",
                "error",
                "-rtsp_transport",
                "tcp",
                "-timeout",
                str(int(self.io_timeout_s * 1e6)),
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-of",
                "json",
                self.url,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=self.io_timeout_s * 3,
        )
        s = json.loads(out.stdout)["streams"][0]
        return int(s["width"]), int(s["height"]), float(Fraction(s["r_frame_rate"]))

    def frames(self) -> Iterator[tuple[FrameRef, np.ndarray]]:
        backoff = 0.5
        while True:
            try:
                yield from self._one_connection()
                backoff = 0.5
            except (subprocess.SubprocessError, OSError, KeyError, IndexError, ValueError):
                pass
            self.reconnects += 1
            time.sleep(backoff)
            backoff = min(backoff * 2, 5.0)

    def _one_connection(self) -> Iterator[tuple[FrameRef, np.ndarray]]:
        mw, mh, fps = self._probe_live()
        w, h = analytics_size(mw, mh, self.width)
        epoch = self.new_epoch()
        segdir = self.segment_root / self.camera_id / f"e{epoch:05d}"
        segdir.mkdir(parents=True, exist_ok=True)
        seglist = segdir / "list.csv"
        cmd = [
            FFMPEG,
            "-v",
            "error",
            "-nostdin",
            "-rtsp_transport",
            "tcp",
            "-timeout",
            str(int(self.io_timeout_s * 1e6)),
            "-i",
            self.url,
            # evidence tap: encoded packets, no re-encode, cut at keyframes
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            str(self.segment_s),
            "-segment_format",
            "mpegts",
            "-segment_list",
            str(seglist),
            "-segment_list_type",
            "csv",
            str(segdir / "%06d.ts"),
            # analytics: small gray frames at the stream's nominal rate
            "-map",
            "0:v:0",
            "-vf",
            f"scale={w}:{h}:flags=area,format=gray",
            "-fps_mode",
            "cfr",
            "-r",
            f"{fps}",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, preexec_fn=die_with_parent)  # noqa: S603
        seen = 0
        wall0: float | None = None
        try:
            assert proc.stdout is not None
            seq = 0
            while (buf := _read_exact(proc.stdout, w * h)) is not None:
                now = time.time()
                if wall0 is None:
                    wall0 = now
                    tmp = segdir / "wall0.tmp"  # lets a restarted ingest recover this epoch's segments
                    tmp.write_text(repr(wall0))
                    tmp.replace(segdir / "wall0")
                seen = self._poll_segments(seglist, epoch, wall0, seen)
                ref = FrameRef(
                    camera_id=self.camera_id,
                    epoch=epoch,
                    seq=seq,
                    frame_idx=seq,
                    ts=now,
                    ts_mono=time.monotonic(),
                    width=mw,
                    height=mh,
                    transform=f"gray:scale{w}x{h}",
                )
                yield ref, np.frombuffer(buf, np.uint8).reshape(h, w)
                seq += 1
        finally:
            proc.kill()
            proc.wait()
            if wall0 is not None:
                self._poll_segments(seglist, epoch, wall0, seen)

    def _poll_segments(self, seglist: Path, epoch: int, wall0: float, seen: int) -> int:
        if not seglist.exists():
            return seen
        lines = seglist.read_text().splitlines()
        new = []
        for idx, line in enumerate(lines[seen:], start=seen):
            parts = line.split(",")
            if len(parts) < 3:
                return idx  # partially written line; pick it up next time
            new.append(
                Segment(
                    epoch, idx, wall0 + float(parts[1]), wall0 + float(parts[2]), seglist.parent / parts[0]
                )
            )
        if new:
            self.on_segments(new)
        return len(lines)
