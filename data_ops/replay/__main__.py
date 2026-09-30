"""python -m data_ops.replay {prepare|up|fault|status|down}. See data_ops/replay/farm.py."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from data_ops.replay import farm

REPO = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m data_ops.replay")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="encode loop files (NVENC, under scripts/gpu shared)")
    p.add_argument("-n", type=int, default=10)
    u = sub.add_parser("up", help="start mediamtx + publishers and supervise (foreground)")
    u.add_argument("--codecs", default="h264,h265")
    u.add_argument("--schedule", type=Path, help="JSON fault schedule")
    u.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    f = sub.add_parser("fault", help="inject a fault into a running farm")
    f.add_argument("kind", choices=farm.FAULTS)
    f.add_argument("seconds", type=float)
    f.add_argument("--target", default="*", help='"cam03", "h265/cam03", or "*"')
    sub.add_parser("status", help="print stream URLs and state")
    sub.add_parser("down", help="stop the mediamtx container")
    a = ap.parse_args(argv)

    if a.cmd == "prepare":
        for c in farm.prepare(a.n, gpu_wrapper=str(REPO / "scripts" / "gpu")):
            print(f"{c['cam']}: {c['clip']}")
        return 0
    if a.cmd == "up":
        farm.server_up()
        fm = farm.Farm(pubs=farm.build_publishers(tuple(a.codecs.split(","))),
                       schedule=farm.load_schedule(a.schedule) if a.schedule else [])
        for pub in fm.pubs:
            print(farm.url(pub.codec, pub.cam))
        sys.stdout.flush()
        try:
            fm.serve(stop_after=a.duration)
        except KeyboardInterrupt:
            pass
        return 0
    if a.cmd == "fault":
        print(farm.request_fault(a.kind, a.seconds, a.target))
        return 0
    if a.cmd == "status":
        st = farm.farm_dir() / "run" / "state.json"
        print(st.read_text() if st.exists() else "farm not running (no state.json)")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{farm.API_PORT}/v3/paths/list", timeout=3) as r:  # noqa: S310
                items = json.load(r)["items"]
            print(f"mediamtx: {sum(i['ready'] for i in items)}/{len(items)} paths ready")
        except OSError as e:
            print(f"mediamtx API unreachable: {e}")
        return 0
    if a.cmd == "down":
        farm.server_down()
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
