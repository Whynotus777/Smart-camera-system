"""python -m data_ops.fetch <dataset> [options]. Every fetcher checks the disk budget first."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m data_ops.fetch")
    sub = ap.add_subparsers(dest="dataset", required=True)
    m = sub.add_parser("meva", help="MEVA indoor cameras + Kitware activity annotations (CC-BY-4.0)")
    m.add_argument("--indoor", action="store_true", required=True,
                   help="indoor cameras only (required: outdoor MEVA isn't in the plan)")
    m.add_argument("--max-gb", type=float, default=150.0)
    m.add_argument("--jobs", type=int, default=8)
    m.add_argument("--dry-run", action="store_true", help="select + index only, no video download")
    s = sub.add_parser("smartspaces", help="NVIDIA PhysicalAI-SmartSpaces retail scenes 071-080 (CC-BY-4.0)")
    s.add_argument("--jobs", type=int, default=4)
    s.add_argument("--dry-run", action="store_true")
    c = sub.add_parser("coco_kp", help="COCO 2017 val keypoints + images")
    c.add_argument("--jobs", type=int, default=2)
    c.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("poselift", help="PoseLift pose data from its public Google Drive folder")
    p.add_argument("--from-dir", type=Path, default=None, help="ingest an existing download instead")
    sub.add_parser("simuletic_sample", help="BLOCKED: CC BY-NC-SA 4.0 + needs a Kaggle token (see DATA.md)")
    a = ap.parse_args(argv)
    if a.dataset == "meva":
        from data_ops.fetch import meva
        return meva.fetch(a.max_gb, jobs=a.jobs, dry_run=a.dry_run)
    if a.dataset == "smartspaces":
        from data_ops.fetch import smartspaces
        return smartspaces.fetch(jobs=a.jobs, dry_run=a.dry_run)
    if a.dataset == "coco_kp":
        from data_ops.fetch import coco_kp
        return coco_kp.fetch(jobs=a.jobs, dry_run=a.dry_run)
    if a.dataset == "poselift":
        from data_ops.fetch import poselift
        return poselift.fetch(src=a.from_dir)
    if a.dataset == "simuletic_sample":
        print("simuletic_sample is blocked: license is CC BY-NC-SA 4.0 (non-commercial) and download needs a "
              "Kaggle API token. Needs the owner's OK in docs/DATA.md first.", file=sys.stderr)
        return 3
    return 2


if __name__ == "__main__":
    sys.exit(main())
