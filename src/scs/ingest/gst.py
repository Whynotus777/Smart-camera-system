"""GStreamer backend: NVDEC decode (nvh264dec/nvh265dec), packet tap, RTP capture time.

Primary backend (T02 brief). One pipeline per session:

    rtspsrc | filesrc ! parsebin          (codec found from caps on the dynamic pad)
      → rtph26Xdepay (RTSP only) → h26Xparse config-interval=-1
      → byte-stream/AU caps → tee ─┬─ queue → appsink "pkt"    (packet tap, never leaky)
                                   └─ queue → nvh26Xdec → NV12 → appsink "frame"

The tee sits *after* the parser and *before* the decoder, so the tap sees every access
unit, with SPS/PPS(/VPS) re-inserted before each keyframe, whatever happens downstream.
`realtime=True` sets `sync` on both appsinks, which paces files to their timestamps
(camera-like arrival for packets and frames alike).

Capture time: `rtspsrc add-reference-timestamp-meta=true` attaches the sender's NTP time
(from RTCP sender reports) to each buffer; we read it as `source_ts`. It is absent until
the first SR arrives, and for files. With mediamtx it is the server's clock at ingest, not
the camera's shutter; on a real camera it's the camera's (NTP-synced, hopefully) clock.

The environment must expose the `nvcodec` and `videoparsersbad` plugins
(gstreamer1.0-plugins-bad; see docs/reports/T02-ingest.md for a no-root dev setup).
"""

from __future__ import annotations

import ctypes
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

import numpy as np

from scs.ingest.convert import PixFmt, pack_planes
from scs.ingest.packets import PacketCodec
from scs.ingest.source import EndReason, SessionSink
from scs.ingest.urls import redact_text

log = logging.getLogger("scs.ingest.gst")

_NTP_UNIX_OFFSET_S = 2_208_988_800
_Gst: Any = None
_GstVideo: Any = None
_init_lock = threading.Lock()

Decode = Literal["auto", "nvdec", "software"]
_DECODERS: dict[str, dict[str, list[str]]] = {
    "h264": {"nvdec": ["nvh264dec", "nvv4l2decoder"], "software": ["avdec_h264"]},
    "h265": {"nvdec": ["nvh265dec", "nvv4l2decoder"], "software": ["avdec_h265"]},
}


def gst() -> tuple[Any, Any]:
    """Import and init GStreamer once. Raises ImportError if PyGObject/GStreamer are missing."""
    global _Gst, _GstVideo
    with _init_lock:
        if _Gst is None:
            import gi  # type: ignore[import-untyped]

            gi.require_version("Gst", "1.0")
            gi.require_version("GstVideo", "1.0")
            from gi.repository import Gst, GstVideo  # type: ignore[import-untyped]

            Gst.init(None)
            _Gst, _GstVideo = Gst, GstVideo
    return _Gst, _GstVideo


class _GstMapInfo(ctypes.Structure):
    _fields_ = [("memory", ctypes.c_void_p), ("flags", ctypes.c_int), ("data", ctypes.c_void_p),
                ("size", ctypes.c_size_t), ("maxsize", ctypes.c_size_t),
                ("user_data", ctypes.c_void_p * 4), ("_gst_reserved", ctypes.c_void_p * 4)]


_libgst: Any = None


def _ctypes_map_ok() -> bool:
    """Map buffers through ctypes (GIL released while copying) if the ABI checks out.

    PyGObject's `MapInfo.data` copies the whole frame into `bytes` with the GIL held
    (~2 ms per 1440p NV12 frame); with ten decode threads that serializes ingest at
    ~100 fps total. `hash(Gst.Buffer)` is the `GstBuffer*`, so we call gst_buffer_map
    directly and let numpy do one GIL-free copy. Verified once; falls back if it fails.
    """
    global _libgst
    if _libgst is not None:
        return bool(_libgst)
    try:
        Gst, _ = gst()
        lib = ctypes.CDLL("libgstreamer-1.0.so.0")
        lib.gst_buffer_map.argtypes = [ctypes.c_void_p, ctypes.POINTER(_GstMapInfo), ctypes.c_int]
        lib.gst_buffer_map.restype = ctypes.c_int
        lib.gst_buffer_unmap.argtypes = [ctypes.c_void_p, ctypes.POINTER(_GstMapInfo)]
        lib.gst_buffer_unmap.restype = None
        probe = Gst.Buffer.new_wrapped(b"scs-map-probe")
        mi = _GstMapInfo()
        ok = lib.gst_buffer_map(hash(probe), ctypes.byref(mi), 1)
        good = bool(ok) and mi.size == 13 and ctypes.string_at(mi.data, mi.size) == b"scs-map-probe"
        if ok:
            lib.gst_buffer_unmap(hash(probe), ctypes.byref(mi))
        _libgst = lib if good else False
    except (OSError, AttributeError, ImportError):
        _libgst = False
    return bool(_libgst)


@contextmanager
def _mapped(buf: Any) -> Iterator[np.ndarray | bytes]:
    """Read-only view of a buffer's bytes: numpy view via ctypes, or a GI `bytes` copy."""
    Gst, _ = gst()
    if _ctypes_map_ok():
        mi = _GstMapInfo()
        if not _libgst.gst_buffer_map(hash(buf), ctypes.byref(mi), 1):  # GST_MAP_READ
            raise RuntimeError("gst_buffer_map failed")
        try:
            yield np.ctypeslib.as_array((ctypes.c_uint8 * mi.size).from_address(mi.data))
        finally:
            _libgst.gst_buffer_unmap(hash(buf), ctypes.byref(mi))
        return
    ok, info = buf.map(Gst.MapFlags.READ)
    if not ok:
        raise RuntimeError("buffer map failed")
    try:
        yield info.data
    finally:
        buf.unmap(info)


def available() -> bool:
    try:
        Gst, _ = gst()
    except (ImportError, ValueError):
        return False
    return all(Gst.ElementFactory.find(e) for e in ("h264parse", "h265parse", "appsink"))


def pick_decoder(codec: PacketCodec, decode: Decode) -> str:
    Gst, _ = gst()
    order = {"auto": ["nvdec", "software"], "nvdec": ["nvdec"], "software": ["software"]}[decode]
    for kind in order:
        for name in _DECODERS[codec][kind]:
            if Gst.ElementFactory.find(name):
                return name
    raise RuntimeError(f"no GStreamer {decode} decoder for {codec} (is gstreamer1.0-plugins-bad installed?)")


class GstSession:
    """One GStreamer pipeline = one connection/epoch. See module docstring."""

    backend = "gstreamer"

    def __init__(self, location: str, *, rtsp: bool, decode: Decode = "auto", realtime: bool = False,
                 loop: bool = False, latency_ms: int = 200, tcp: bool = True) -> None:
        self.location = location
        self.rtsp = rtsp
        self.decode = decode
        self.realtime = realtime
        self.loop = loop
        self.latency_ms = latency_ms
        self.tcp = tcp
        self.decoder = "pending"  # set once the codec is known (dynamic pad)
        self.codec: PacketCodec | None = None
        self._sink: SessionSink | None = None
        self._pipeline: Any = None
        self._linked = False
        self._link_error: str | None = None
        self._ntp_caps: Any = None
        self._frame_prerolled = threading.Event()
        self._key_seen = False

    # ------------------------------------------------------------------ pipeline

    def _build(self) -> None:
        Gst, _ = gst()
        self._ntp_caps = Gst.Caps.from_string("timestamp/x-ntp")
        p = Gst.Pipeline.new("scs-ingest")
        if self.rtsp:
            src = _make("rtspsrc", location=self.location, latency=self.latency_ms,
                        protocols=4 if self.tcp else 7)  # 4 = TCP; 7 = UDP|UDP_MCAST|TCP
            if src.find_property("add-reference-timestamp-meta"):
                src.set_property("add-reference-timestamp-meta", True)
            p.add(src)
        else:
            src = _make("filesrc", location=self.location)
            demux = _make("parsebin")
            p.add(src)
            p.add(demux)
            src.link(demux)
            src = demux
        src.connect("pad-added", self._on_pad)
        self._pipeline = p

    def _on_pad(self, _el: Any, pad: Any) -> None:
        Gst, _ = gst()
        caps = pad.get_current_caps() or pad.query_caps(None)
        s = caps.get_structure(0)
        name = s.get_name()
        codec: PacketCodec | None = None
        depay = None
        if name == "application/x-rtp":
            if s.get_string("media") != "video":
                return self._fakesink(pad)
            enc = (s.get_string("encoding-name") or "").upper()
            codec = {"H264": "h264", "H265": "h265"}.get(enc)  # type: ignore[assignment]
            if codec:
                depay = _make(f"rtp{codec}depay")
        elif name in ("video/x-h264", "video/x-h265"):
            codec = "h264" if name.endswith("264") else "h265"
        if codec is None:
            if name.startswith("video/") or (name == "application/x-rtp"):
                self._link_error = f"unsupported video stream caps {name}"
            return self._fakesink(pad)
        if self._linked:  # second video stream: ignore
            return self._fakesink(pad)
        try:
            self._link_chain(pad, codec, depay)
        except Exception as e:  # noqa: BLE001 - reported via bus loop
            self._link_error = f"{type(e).__name__}: {e}"

    def _fakesink(self, pad: Any) -> None:
        fs = _make("fakesink", sync=False, async_=False)
        self._pipeline.add(fs)
        fs.sync_state_with_parent()
        pad.link(fs.get_static_pad("sink"))

    def _link_chain(self, pad: Any, codec: PacketCodec, depay: Any) -> None:
        Gst, _ = gst()
        self.codec = codec
        self.decoder = pick_decoder(codec, self.decode)
        caps = f"video/x-{codec},stream-format=byte-stream,alignment=au"
        out_fmt = "NV12" if not self.decoder.startswith("avdec") else "{ NV12, I420 }"
        els = [
            *( [depay] if depay is not None else [] ),
            _make(f"{codec}parse", config_interval=-1),
            _make("capsfilter", caps=Gst.Caps.from_string(caps)),
            tee := _make("tee"),
            q_pkt := _make("queue", max_size_buffers=0, max_size_bytes=64 * 2**20, max_size_time=0),
            pkt := _make("appsink", name="pkt", sync=self.realtime, emit_signals=True, max_buffers=0),
            q_dec := _make("queue", max_size_buffers=0, max_size_bytes=64 * 2**20, max_size_time=0),
            dec := _make(self.decoder),
            conv := _make("capsfilter", caps=Gst.Caps.from_string(f"video/x-raw,format={out_fmt}")),
            frm := _make("appsink", name="frame", sync=self.realtime, emit_signals=True, max_buffers=4,
                         drop=False),
        ]
        if dec.find_property("max-display-delay"):
            # Hand each picture over as soon as it's decoded: lowest latency, and no bursts
            # that overflow the small analytics queue (surveillance streams have no B-frames).
            dec.set_property("max-display-delay", 0)
        for e in els:
            self._pipeline.add(e)
        chain = els[: els.index(tee) + 1]
        for a, b in zip(chain, chain[1:], strict=False):
            _link(a, b)
        _link(tee, q_pkt)
        _link(q_pkt, pkt)
        _link(tee, q_dec)
        _link(q_dec, dec)
        _link(dec, conv)
        _link(conv, frm)
        pkt.connect("new-sample", self._on_packet)
        frm.connect("new-sample", self._on_frame)
        frm.connect("new-preroll", self._on_preroll)
        for e in els:
            e.sync_state_with_parent()
        # Identity is fixed here, upstream of both branches: access units before the first
        # keyframe are dropped (a mid-GOP RTSP join), and every other AU gets its seq before
        # the decoder can see it, so a frame never races ahead of its packet's PTS→seq entry.
        tee.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, self._on_access_unit)
        if pad.link(chain[0].get_static_pad("sink")) != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"could not link {codec} stream")
        self._linked = True

    # ------------------------------------------------------------------ callbacks (streaming threads)

    def _on_access_unit(self, _pad: Any, info: Any) -> Any:
        Gst, _ = gst()
        buf = info.get_buffer()
        if not self._key_seen:
            if buf.has_flags(Gst.BufferFlags.DELTA_UNIT):
                return Gst.PadProbeReturn.DROP
            self._key_seen = True
        assert self._sink is not None
        pts = None if buf.pts == Gst.CLOCK_TIME_NONE else int(buf.pts)
        self._sink.reserve_seq(pts, self._ntp(buf))
        return Gst.PadProbeReturn.OK

    def _on_preroll(self, _appsink: Any) -> Any:
        Gst, _ = gst()
        self._frame_prerolled.set()
        return Gst.FlowReturn.OK

    def _ntp(self, buf: Any) -> float | None:
        meta = buf.get_reference_timestamp_meta(self._ntp_caps)
        if meta is None:
            return None
        return meta.timestamp / 1e9 - _NTP_UNIX_OFFSET_S

    def _on_packet(self, appsink: Any) -> Any:
        Gst, _ = gst()
        sample = appsink.emit("pull-sample")
        buf = sample.get_buffer()
        with _mapped(buf) as view:
            data = bytes(view)
        pts = None if buf.pts == Gst.CLOCK_TIME_NONE else int(buf.pts)
        key = not buf.has_flags(Gst.BufferFlags.DELTA_UNIT)
        assert self._sink is not None and self.codec is not None
        self._sink.on_packet(data, pts, key, self.codec, self._ntp(buf))
        return Gst.FlowReturn.OK

    def _on_frame(self, appsink: Any) -> Any:
        Gst, GstVideo = gst()
        sample = appsink.emit("pull-sample")
        buf = sample.get_buffer()
        vi = GstVideo.VideoInfo.new_from_caps(sample.get_caps())
        fmt: PixFmt = "nv12" if vi.finfo.format == GstVideo.VideoFormat.NV12 else "i420"
        nplanes = 2 if fmt == "nv12" else 3
        strides, offsets = tuple(vi.stride[:nplanes]), tuple(vi.offset[:nplanes])
        vmeta = GstVideo.buffer_get_video_meta(buf)
        if vmeta is not None:  # decoder-chosen padding overrides caps defaults
            strides, offsets = tuple(vmeta.stride[:nplanes]), tuple(vmeta.offset[:nplanes])
        with _mapped(buf) as view:  # the only copy of the frame: planes → packed numpy
            img = pack_planes(view, fmt, vi.width, vi.height, strides, offsets)
        pts = None if buf.pts == Gst.CLOCK_TIME_NONE else int(buf.pts)
        assert self._sink is not None
        self._sink.on_frame(img, pts, self._ntp(buf))
        return Gst.FlowReturn.OK

    # ------------------------------------------------------------------ run

    def run(self, sink: SessionSink) -> tuple[EndReason, str]:
        Gst, _ = gst()
        self._sink = sink
        self._build()
        p = self._pipeline
        bus = p.get_bus()
        try:
            if not self.rtsp:
                # Start the clock only once the *decoder* branch has a frame. The packet branch
                # prerolls at once, while NVDEC needs ~1-2 s to initialize; starting earlier
                # makes realtime playback open with a burst of "late" frames.
                if p.set_state(Gst.State.PAUSED) == Gst.StateChangeReturn.FAILURE:
                    return "error", self._drain_error(bus) or "pipeline failed to preroll"
                deadline = 15.0
                while not self._frame_prerolled.wait(0.1) and deadline > 0:
                    deadline -= 0.1
                    if sink.should_stop() or self._link_error:
                        break
            if p.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                return "error", self._drain_error(bus) or "pipeline failed to start"
            mask = Gst.MessageType.ERROR | Gst.MessageType.EOS
            while True:
                if sink.should_stop():
                    return "stopped", ""
                if self._link_error:
                    return "error", self._link_error
                msg = bus.timed_pop_filtered(100 * Gst.MSECOND, mask)
                if msg is None:
                    continue
                if msg.type == Gst.MessageType.ERROR:
                    err, dbg = msg.parse_error()
                    return "error", redact_text(f"{msg.src.get_name()}: {err.message}")
                if msg.type == Gst.MessageType.EOS:
                    if self.loop and not self.rtsp:
                        sink.on_loop()
                        if p.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, 0):
                            continue
                        return "error", "loop seek failed"
                    return "eos", ""
        finally:
            p.set_state(Gst.State.NULL)
            p.get_state(5 * Gst.SECOND)
            self._pipeline = None

    def _drain_error(self, bus: Any) -> str | None:
        Gst, _ = gst()
        msg = bus.timed_pop_filtered(0, Gst.MessageType.ERROR)
        if msg is None:
            return None
        err, _dbg = msg.parse_error()
        return redact_text(err.message)


def _make(factory: str, **props: Any) -> Any:
    Gst, _ = gst()
    el = Gst.ElementFactory.make(factory, None)
    if el is None:
        raise RuntimeError(f"GStreamer element {factory!r} not available")
    for k, v in props.items():
        el.set_property(k.rstrip("_").replace("_", "-"), v)
    return el


def _link(a: Any, b: Any) -> None:
    if not a.link(b):
        raise RuntimeError(f"could not link {a.get_name()} → {b.get_name()}")
