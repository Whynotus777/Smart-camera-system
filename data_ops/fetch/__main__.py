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
    m.add_argument("--reverify", action="store_true", help="re-check size-only files against S3 ETags")
    s = sub.add_parser("smartspaces", help="NVIDIA PhysicalAI-SmartSpaces retail scenes 071-080 (CC-BY-4.0)")
    s.add_argument("--jobs", type=int, default=4)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--areas", action="store_true", help="only (re)build index/camera_areas.json")
    c = sub.add_parser("coco_kp", help="COCO 2017 val keypoints + images")
    c.add_argument("--jobs", type=int, default=2)
    c.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("poselift", help="PoseLift pose data from its public Google Drive folder")
    p.add_argument("--from-dir", type=Path, default=None, help="ingest an existing download instead")
    p.add_argument("--retry-every-h", type=float, default=None, help="keep retrying until complete")
    sub.add_parser("simuletic_sample", help="DROPPED (owner decision): CC BY-NC-SA 4.0, lineage risk")
    pr = sub.add_parser("probe", help="backfill per-file video properties into a dataset manifest")
    pr.add_argument("id")
    a = ap.parse_args(argv)
    if a.dataset == "meva":
        from data_ops.fetch import meva
        if a.reverify:
            return 1 if meva.reverify(jobs=a.jobs)["mismatch"] else 0
        return meva.fetch(a.max_gb, jobs=a.jobs, dry_run=a.dry_run)
    if a.dataset == "smartspaces":
        from data_ops.fetch import smartspaces
        if a.areas:
            smartspaces.camera_areas()
            return 0
        rc = smartspaces.fetch(jobs=a.jobs, dry_run=a.dry_run)
        if rc == 0 and not a.dry_run:
            smartspaces.camera_areas()
        return rc
    if a.dataset == "coco_kp":
        from data_ops.fetch import coco_kp
        return coco_kp.fetch(jobs=a.jobs, dry_run=a.dry_run)
    if a.dataset == "poselift":
        from data_ops.fetch import poselift
        return poselift.fetch(src=a.from_dir, retry_every_h=a.retry_every_h)
    if a.dataset == "simuletic_sample":
        print("simuletic_sample is dropped (owner decision 2026-09-29): CC BY-NC-SA 4.0 = non-commercial + "
              "share-alike, lineage risk for anything trained on it. See docs/DATA.md.", file=sys.stderr)
        return 3
    if a.dataset == "probe":
        from data_ops.fetch.common import backfill_video
        from data_ops.manifest import Manifest
        from data_ops.paths import dataset_dir
        mp = dataset_dir(a.id, "MANIFEST.json")
        d = __import__("json").loads(mp.read_text())
        man = Manifest.load_or_new(mp, **{k: d[k] for k in ("dataset_id", "license", "license_url",
                                                            "attribution", "use")})
        backfill_video(man, mp, dataset_dir(a.id, "raw"))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
