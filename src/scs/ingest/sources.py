"""Public frame sources: `RtspSource`, `FileSource`, `DirectorySource`.

All three implement `FrameSource` (base.py) and yield `(FrameRef, image)` with full
frame identity. RTSP and file sources share `StreamSource` (worker thread, packet tap,
drop-oldest queue, stall watchdog, reconnect) and differ only in the session they build.

Backend choice (`backend="auto"`): GStreamer if PyGObject and the parser plugins are
present (NVDEC via nvh26Xdec when `decode` allows), otherwise PyAV (NVDEC via cuvid).
`stats.backend`/`stats.decoder` record what actually ran, so a benchmark can prove it
decoded on NVDEC rather than silently falling back to the CPU.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from scs.bus import Bus
from scs.contracts import FrameRef
from scs.ingest import gst as gst_backend
from scs.ingest import pyav as pyav_backend
from scs.ingest.backoff import Backoff
from scs.ingest.convert import Output, RawImage, convert
from scs.ingest.health import HealthReporter
from scs.ingest.packets import PacketTap
from scs.ingest.source import Session, StreamSource
from scs.ingest.stats import CameraStats
from scs.ingest.urls import redact_url, url_from_env

if TYPE_CHECKING:
    import torch

log = logging.getLogger("scs.ingest")

Backend = Literal["auto", "gstreamer", "pyav"]
Decode = Literal["auto", "nvdec", "software"]


def resolve_backend(backend: Backend) -> Literal["gstreamer", "pyav"]:
    if backend == "gstreamer":
        if not gst_backend.available():
            raise RuntimeError("GStreamer backend requested but PyGObject/GStreamer parsers are unavailable")
        return "gstreamer"
    if backend == "pyav":
        return "pyav"
    return "gstreamer" if gst_backend.available() else "pyav"


def _session(location: str, *, rtsp: bool, backend: Backend, decode: Decode, realtime: bool,
             loop: bool, tcp: bool = True, read_timeout_s: float = 5.0) -> Session:
    if resolve_backend(backend) == "gstreamer":
        return gst_backend.GstSession(location, rtsp=rtsp, decode=decode, realtime=realtime, loop=loop,
                                      tcp=tcp)
    # PyAV blocks inside demux; its read timeout (set just past the stall timeout) unblocks it.
    return pyav_backend.PyAvSession(location, rtsp=rtsp, decode=decode, realtime=realtime, loop=loop,
                                    tcp=tcp, read_timeout_s=read_timeout_s)


class RtspSource(StreamSource):
    """A live camera. Main stream by default (ARCHITECTURE D1); reconnects forever.

    The URL comes from an environment variable (`url_env`, as in `CameraInstall`) so it
    never passes through configs or logs; `url=` exists for tests with credential-free
    local URLs. Pass `stream="sub"` + `main_size` only if you really decode the sub-stream.
    """

    def __init__(self, camera_id: str, *, url_env: str | None = None, url: str | None = None,
                 backend: Backend = "auto", decode: Decode = "auto", tcp: bool = True,
                 output: Output = "numpy", queue_size: int = 4, stall_timeout_s: float = 3.0,
                 connect_timeout_s: float = 10.0, backoff: Backoff | None = None, bus: Bus | None = None,
                 tap: PacketTap | None = None, stream: Literal["main", "sub"] = "main",
                 main_size: tuple[int, int] | None = None) -> None:
        if (url_env is None) == (url is None):
            raise ValueError("pass exactly one of url_env or url")
        self._url = url_from_env(url_env) if url_env else str(url)
        self.url_redacted = redact_url(self._url)
        super().__init__(
            camera_id,
            lambda: _session(self._url, rtsp=True, backend=backend, decode=decode, realtime=False, loop=False,
                             tcp=tcp, read_timeout_s=stall_timeout_s + 1.0),
            output=output, queue_size=queue_size, reconnect=True, stall_timeout_s=stall_timeout_s,
            connect_timeout_s=connect_timeout_s, backoff=backoff, bus=bus, tap=tap, stream=stream,
            main_size=main_size)

    def __repr__(self) -> str:
        return f"RtspSource({self.camera_id!r}, {self.url_redacted})"


class FileSource(StreamSource):
    """A video file (H.264/H.265 in MP4/MKV/TS). `realtime=True` paces to the file's timestamps.

    `loop=True` restarts at EOF within the same epoch (frame identity keeps counting);
    without loop the source ends after one pass. Errors and stalls (e.g. a slow decoder
    start) reopen the file as a new epoch, like a camera reconnect.

    `source_ts` is the recording's own timeline: `origin_ts` (default: wall clock when the
    first packet is read) + media time, continuous across loops and reopens. Durations
    measured on `source_ts` are right even when replay runs faster than realtime (T13 B1).
    Give several files the same `origin_ts` to replay them on one shared timeline.
    """

    def __init__(self, path: str | Path, camera_id: str | None = None, *, loop: bool = False,
                 realtime: bool = False, backend: Backend = "auto", decode: Decode = "auto",
                 output: Output = "numpy", queue_size: int = 4, stall_timeout_s: float = 3.0,
                 bus: Bus | None = None, tap: PacketTap | None = None,
                 origin_ts: float | None = None) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        super().__init__(
            camera_id or self.path.stem,
            lambda: _session(str(self.path), rtsp=False, backend=backend, decode=decode, realtime=realtime,
                             loop=loop),
            output=output, queue_size=queue_size, reconnect=True, end_on_eos=True,
            stall_timeout_s=stall_timeout_s, bus=bus, tap=tap,
            media_origin=origin_ts if origin_ts is not None else "open")


_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".npy")


class DirectorySource:
    """Frames from a directory of images (sim output, T08), sorted by file name.

    No encoded stream exists, so there is no packet tap. `fps` paces playback when
    `realtime=True` and defines media time: `source_ts = origin_ts + seq / fps` (origin
    defaults to the wall clock at start). Epoch 0; `seq == frame_idx`, continuing across
    loops. `.npy` files are HxWx3 RGB uint8.
    """

    def __init__(self, path: str | Path, camera_id: str | None = None, *, fps: float = 15.0,
                 realtime: bool = False, loop: bool = False, output: Output = "numpy",
                 bus: Bus | None = None, origin_ts: float | None = None) -> None:
        self.path = Path(path)
        self.files = sorted(p for p in self.path.iterdir() if p.suffix.lower() in _IMAGE_EXT)
        if not self.files:
            raise FileNotFoundError(f"no images in {self.path}")
        self.camera_id = camera_id or self.path.name
        self.fps = fps
        self.realtime = realtime
        self.loop = loop
        self.output: Output = output
        self.origin_ts = origin_ts
        self.stats = CameraStats(self.camera_id, backend="directory", decoder="imread")
        self.health = HealthReporter(self.camera_id, bus)
        self._closed = threading.Event()

    def _read(self, p: Path) -> np.ndarray:
        if p.suffix.lower() == ".npy":
            return np.load(p)
        import cv2

        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"unreadable image {p.name}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def frames(self) -> Iterator[tuple[FrameRef, np.ndarray | torch.Tensor]]:
        t0 = time.monotonic()
        origin = self.origin_ts if self.origin_ts is not None else time.time()
        seq = 0
        self.health.emit("connected", 0, backend="directory", n_files=len(self.files))
        while not self._closed.is_set():
            for p in self.files:
                if self._closed.is_set():
                    break
                if self.realtime:
                    delay = t0 + seq / self.fps - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                try:
                    rgb = self._read(p)
                except (ValueError, OSError) as e:
                    self.stats.on_error(str(e))
                    continue
                now_mono = time.monotonic()
                h, w = rgb.shape[:2]
                ref = FrameRef(camera_id=self.camera_id, epoch=0, seq=seq, frame_idx=seq, ts=time.time(),
                               ts_mono=now_mono, source_ts=origin + seq / self.fps, width=w, height=h)
                self.stats.on_frame(now_mono)
                self.stats.delivered += 1
                seq += 1
                yield ref, convert(RawImage(rgb, "rgb", w, h), self.output)
            if not self.loop:
                break
        self.health.emit("eos", 0, stats=self.stats.snapshot())

    def close(self) -> None:
        self._closed.set()


def source_for(uri: str, camera_id: str, **kw: Any) -> StreamSource | DirectorySource:
    """Small factory for tools: `rtsp://…` / `env:VAR` → RtspSource, dir → DirectorySource, else file."""
    if uri.startswith("env:"):
        return RtspSource(camera_id, url_env=uri[4:], **kw)
    if uri.startswith(("rtsp://", "rtsps://")):
        return RtspSource(camera_id, url=uri, **kw)
    p = Path(uri)
    if p.is_dir():
        return DirectorySource(p, camera_id, **kw)
    return FileSource(p, camera_id, **kw)
