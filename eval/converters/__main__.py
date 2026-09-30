"""CLI: python -m eval.converters {meva|smartspaces|poselift|retails} [--limit N] [--no-tracks]."""

from __future__ import annotations

import argparse
import sys

from eval.converters import CONVERTERS, LicenseError


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.converters")
    ap.add_argument("dataset", choices=sorted(CONVERTERS))
    ap.add_argument("--limit", type=int, default=None, help="meva: convert only the first N clips")
    ap.add_argument("--no-tracks", action="store_true", help="meva: labels only (skip GT boxes)")
    ap.add_argument("--take-log", help="quick_capture / lab_mock_aisle: take log CSV")
    a = ap.parse_args(argv)
    info = CONVERTERS[a.dataset]
    kw: dict = {}
    if a.dataset == "meva":
        kw = {"limit": a.limit, "with_tracks": not a.no_tracks}
    elif a.dataset in ("quick_capture", "lab_mock_aisle"):
        if not a.take_log:
            ap.error("--take-log is required")
        kw = {"take_log": a.take_log}
    try:
        info.fn(**kw)
    except (LicenseError, NotImplementedError, FileNotFoundError) as e:
        print(f"{a.dataset}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
