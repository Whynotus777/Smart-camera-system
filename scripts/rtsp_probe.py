#!/usr/bin/env python3
"""Print a camera stream's real specs, for filling `configs/cameras/*.yaml` (T02).

The URL comes from an environment variable only and is never printed unredacted
(docs/SECURITY.md §1):

    export SCS_CAM1_RTSP_MAIN='rtsp://<user>:<password>@192.168.1.20:554/h264Preview_01_main'
    python scripts/rtsp_probe.py --env SCS_CAM1_RTSP_MAIN --seconds 10 [--json]

It reads packets for `--seconds` (no decoding) and reports codec/profile, size, nominal
and measured fps, measured bitrate, GOP length, and whether B-frames occur, plus a
`StreamSpec` snippet. Results from a physical camera justify
`verification: measured` in a CameraProfile (ADR 0003); cite the probe log in `sources`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scs.ingest.urls import UrlEnvError, redact_text, redact_url, url_from_env  # noqa: E402

CODEC = {"h264": "h264", "hevc": "h265", "mjpeg": "mjpeg"}


def probe(url: str, seconds: float, tcp: bool = True) -> dict:
    import av

    opts = {"rtsp_transport": "tcp" if tcp else "udp"} if url.startswith("rtsp") else {}
    t_open = time.monotonic()
    with av.open(url, options=opts, timeout=(10.0, 5.0)) as c:
        open_s = time.monotonic() - t_open
        v = c.streams.video[0]
        cc = v.codec_context
        tb = float(v.time_base) if v.time_base else 1 / 90000
        n = nbytes = 0
        keys: list[int] = []
        pts: list[float] = []
        t0 = time.monotonic()
        for pkt in c.demux(v):
            if pkt.size == 0:
                continue
            n += 1
            nbytes += pkt.size
            if pkt.is_keyframe:
                keys.append(n)
            if pkt.pts is not None:
                pts.append(pkt.pts * tb)
            if time.monotonic() - t0 >= seconds:
                break
        wall = time.monotonic() - t0
        audio = [f"{a.codec_context.name} {a.codec_context.sample_rate} Hz" for a in c.streams.audio]
    gops = [b - a for a, b in zip(keys, keys[1:], strict=False)]
    reordered = sum(1 for a, b in zip(pts, pts[1:], strict=False) if b < a)
    span = (max(pts) - min(pts)) if len(pts) > 1 else 0.0
    fps_measured = (len(pts) - 1) / span if span > 0 else None
    return {
        "codec": CODEC.get(cc.name, cc.name),
        "profile": cc.profile,
        "width": cc.width,
        "height": cc.height,
        "pix_fmt": cc.pix_fmt,
        "fps_nominal": float(v.average_rate) if v.average_rate else None,
        "fps_measured": round(fps_measured, 3) if fps_measured else None,
        "bitrate_kbps_measured": round(nbytes * 8 / 1000 / span) if span > 0 else None,
        "gop_frames": statistics.median(gops) if gops else None,
        "gop_seconds": round(statistics.median(gops) / fps_measured, 2) if gops and fps_measured else None,
        "b_frames": reordered > 0,
        "packets": n,
        "keyframes": len(keys),
        "sample_wall_s": round(wall, 2),
        "open_s": round(open_s, 2),
        "audio": audio,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="SCS_RTSP_URL", help="env var holding the stream URL")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--udp", action="store_true", help="RTP over UDP instead of TCP interleaved")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        url = url_from_env(a.env)
    except UrlEnvError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    target = redact_url(url)
    try:
        spec = probe(url, a.seconds, tcp=not a.udp)
    except Exception as e:  # noqa: BLE001 - report any failure without leaking the URL
        print(f"error probing {target}: {redact_text(str(e))}", file=sys.stderr)
        return 1
    spec = {"url": target, "env": a.env, "probed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **spec}
    if a.json:
        print(json.dumps(spec, indent=1))
        return 0
    w = max(len(k) for k in spec)
    for k, val in spec.items():
        print(f"{k:<{w}}  {val}")
    fps = spec["fps_measured"] or spec["fps_nominal"]
    print("\n# StreamSpec for a CameraProfile (verification: measured, cite this probe in sources)")
    print(f"width: {spec['width']}\nheight: {spec['height']}\nfps: {round(fps, 2) if fps else 'null'}\n"
          f"codec: {spec['codec']}\nbitrate_kbps: {spec['bitrate_kbps_measured']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
