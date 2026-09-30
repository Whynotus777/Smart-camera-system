"""PyAV (FFmpeg) fallback backend: NVDEC via h264_cuvid/hevc_cuvid, software otherwise.

Used when GStreamer or its NVDEC plugin isn't installed (the pip `av` wheel ships its
own FFmpeg with cuvid). Packets are converted to Annex-B with parameter sets repeated on
keyframes (`*_mp4toannexb,dump_extra`), the same format the GStreamer tap emits, and
the decoder is fed exactly those bytes, so evidence and analytics see identical data.

Limitations vs. GStreamer: no RTP/NTP capture time (`source_ts` stays None: PyAV doesn't
expose RTCP sender reports), and realtime pacing is done in Python from packet PTS.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

from scs.ingest.convert import PixFmt, RawImage
from scs.ingest.packets import PacketCodec
from scs.ingest.source import EndReason, SessionSink
from scs.ingest.urls import redact_text

log = logging.getLogger("scs.ingest.pyav")

Decode = Literal["auto", "nvdec", "software"]
_FF = {"h264": "h264", "hevc": "h265"}


def available() -> bool:
    try:
        import av  # noqa: F401
    except ImportError:
        return False
    return True


def _open_decoder(ff_codec: str, decode: Decode) -> tuple[Any, str]:
    import av

    names = {"auto": [f"{ff_codec}_cuvid", ff_codec], "nvdec": [f"{ff_codec}_cuvid"],
             "software": [ff_codec]}[decode]
    last: Exception | None = None
    for name in names:
        try:
            cc = av.CodecContext.create(name, "r")
            cc.open()
            return cc, name
        except Exception as e:  # noqa: BLE001 - try the next decoder
            last = e
    raise RuntimeError(f"no {decode} decoder for {ff_codec}: {last}")


class PyAvSession:
    backend = "pyav"

    def __init__(self, location: str, *, rtsp: bool, decode: Decode = "auto", realtime: bool = False,
                 loop: bool = False, open_timeout_s: float = 10.0, read_timeout_s: float = 5.0,
                 tcp: bool = True) -> None:
        self.location = location
        self.rtsp = rtsp
        self.decode = decode
        self.realtime = realtime
        self.loop = loop
        self.open_timeout_s = open_timeout_s
        self.read_timeout_s = read_timeout_s
        self.tcp = tcp
        self.decoder = "pending"
        self.corrupt = 0  # packets the decoder rejected

    def run(self, sink: SessionSink) -> tuple[EndReason, str]:
        import av

        opts = {"rtsp_transport": "tcp" if self.tcp else "udp"} if self.rtsp else {}
        try:
            timeout = (self.open_timeout_s, self.read_timeout_s)
            container = av.open(self.location, options=opts, timeout=timeout)
        except av.error.FFmpegError as e:
            return "error", redact_text(f"open: {e}")
        try:
            st = container.streams.video[0]
            ff = st.codec_context.name
            if ff not in _FF:
                return "error", f"unsupported codec {ff}"
            codec: PacketCodec = _FF[ff]  # type: ignore[assignment]
            tb = float(st.time_base) if st.time_base else 1 / 90000
            return self._loop(container, st, ff, codec, tb, sink)
        except av.error.FFmpegError as e:
            return "error", redact_text(str(e))
        finally:
            container.close()

    def _loop(self, container: Any, st: Any, ff: str, codec: PacketCodec, tb: float,
              sink: SessionSink) -> tuple[EndReason, str]:
        import av
        import av.bitstream

        t0: float | None = None
        pts0 = 0
        while True:
            # Fresh decoder + filter per pass: both are in EOF state after draining.
            dec, self.decoder = _open_decoder(ff, self.decode)
            bsf = av.bitstream.BitStreamFilterContext(f"{ff}_mp4toannexb,dump_extra=freq=keyframe", st)
            seen_key = False
            for pkt in container.demux(st):
                if sink.should_stop():
                    return "stopped", ""
                for p in bsf.filter(pkt if pkt.size else None):
                    seen_key = seen_key or bool(p.is_keyframe)
                    pts_ns = int(p.pts * tb * 1e9) if p.pts is not None else None
                    if self.realtime and pts_ns is not None:
                        if t0 is None:
                            t0, pts0 = time.monotonic(), pts_ns
                        delay = t0 + (pts_ns - pts0) / 1e9 - time.monotonic()
                        if delay > 0:
                            time.sleep(delay)
                    sink.on_packet(bytes(p), pts_ns, bool(p.is_keyframe), codec)
                    if not seen_key:  # joined mid-GOP: nothing decodable yet
                        continue
                    try:
                        frames = dec.decode(p)
                    except av.error.InvalidDataError as e:  # one corrupt AU must not end the session
                        self.corrupt += 1
                        log.debug("corrupt packet: %s", e)
                        continue
                    for fr in frames:
                        self._emit(fr, tb, sink)
            for fr in dec.decode(None):  # drain frames still inside the decoder
                self._emit(fr, tb, sink)
            if not (self.loop and not self.rtsp):
                return "eos", ""
            container.seek(0)
            sink.on_loop()
            t0 = None

    def _emit(self, fr: Any, tb: float, sink: SessionSink) -> None:
        fmt = fr.format.name
        if fmt not in ("nv12", "yuv420p", "yuvj420p"):
            fr = fr.reformat(format="nv12")
            fmt = "nv12"
        pix: PixFmt = "nv12" if fmt == "nv12" else "i420"
        arr = fr.to_ndarray()  # nv12/yuv420p → (H*3/2, W), tightly packed
        pts_ns = int(fr.pts * tb * 1e9) if fr.pts is not None else None
        sink.on_frame(RawImage(arr, pix, fr.width, fr.height), pts_ns)
