"""COCO 2017 keypoints, val split only (for T04's wrist-PCK substitute on COCO val).

Annotations are CC-BY-4.0 (COCO Consortium). Images are Flickr photos under their own
per-image licenses (listed in the annotation JSON's `licenses` + each image's `license`
id); that's why docs/DATA.md allows COCO for R&D/eval and for weights, not for shipping
images. COCO publishes no checksums: files are verified by size (HTTP Content-Length)
and their sha256 is recorded so later copies can be checked.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

from data_ops.fetch.common import RemoteFile, fetch_all, run, tool
from data_ops.manifest import Manifest
from data_ops.paths import dataset_dir

URLS = {
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
}
META = {
    "dataset_id": "coco_kp",
    "license": "Annotations CC-BY-4.0; images: per-image Flickr licenses (see annotation JSON)",
    "license_url": "https://cocodataset.org/#termsofuse",
    "attribution": "COCO Consortium, Microsoft COCO: Common Objects in Context (Lin et al., 2014).",
    "use": "R&D/eval (images); annotations prod",
    "notes": "val2017 only: person_keypoints_val2017.json + 5k images.",
}


def content_length(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")  # noqa: S310 - fixed http(s) URLs above
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
        return int(r.headers["Content-Length"])


def fetch(jobs: int = 2, dry_run: bool = False, log=print) -> int:
    root = dataset_dir("coco_kp")
    files = [RemoteFile(relpath=name, source=url, nbytes=content_length(url)) for name, url in URLS.items()]
    log(f"coco_kp: {len(files)} archives, {sum(f.nbytes for f in files) / 1e9:.2f} GB")
    if dry_run:
        return 0
    man_path = root / "MANIFEST.json"
    man = Manifest.load_or_new(man_path, **META)
    curl = tool("curl")

    def download(rf: RemoteFile, dest: Path) -> None:
        run([curl, "-sSfL", "--retry", "3", "-o", str(dest), rf.source])

    stats = fetch_all(files, root / "raw", man, man_path, download, jobs=jobs, log=log)
    unpack(root)
    log(f"coco_kp: done {stats}")
    return 1 if stats["failed"] else 0


def unpack(root: Path) -> None:
    """Extract only what T04 needs: keypoint annotations + val images."""
    out = root / "extracted"
    ann = out / "annotations" / "person_keypoints_val2017.json"
    if not ann.exists():
        with zipfile.ZipFile(root / "raw" / "annotations_trainval2017.zip") as z:
            z.extract("annotations/person_keypoints_val2017.json", out)
    if not (out / "val2017").exists():
        with zipfile.ZipFile(root / "raw" / "val2017.zip") as z:
            z.extractall(out)
