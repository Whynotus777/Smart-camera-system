"""PoseLift (UNC Charlotte, WACV 2025): pose-only shoplifting benchmark.

The data (per-video .pkl pose files + .npy frame labels) is a public Google Drive
folder linked from https://github.com/TeCSAR-UNCC/PoseLift. That GitHub repo is
Apache-2.0, but the data isn't in it, so whether Apache-2.0 covers the data is
unconfirmed (docs/DATA.md: R&D until confirmed). Drive publishes no checksums; we
record sha256 of what we got. Uses `gdown` (MIT) via `uvx`, so no project dependency.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from data_ops.budget import require
from data_ops.fetch.common import run, tool
from data_ops.manifest import Manifest, file_hashes
from data_ops.paths import dataset_dir

DRIVE_FOLDER = "https://drive.google.com/drive/folders/1aEkENZlVE4ZvF_BZXV1VJOwuiXQq6trn"
REPO = "https://github.com/TeCSAR-UNCC/PoseLift"
META = {
    "dataset_id": "poselift",
    "license": "Apache-2.0 (GitHub repo); coverage of the Drive-hosted data unconfirmed",
    "license_url": f"{REPO}/blob/main/LICENSE",
    "attribution": ("Rashvand et al., Exploring Pose-Based Anomaly Detection for Retail Security: "
                    "A Real-World Shoplifting Dataset and Benchmark (PoseLift), WACV 2025. " + REPO),
    "use": "R&D",
    "notes": "Pose-only (COCO17 via HRNet, interpolated + smoothed), 15 fps, 1080p source.",
}


def fetch(src: Path | None = None, log=print) -> int:
    """Download the Drive folder (or ingest an already-downloaded copy at `src`) and record it."""
    root = dataset_dir("poselift")
    raw = root / "raw"
    require(2 * 10**9, root, what="poselift")  # generous upper bound; the pose data is small
    if src is None:
        stage = root / ".staging"
        stage.mkdir(parents=True, exist_ok=True)
        run([tool("uvx"), "--quiet", "gdown", "--folder", DRIVE_FOLDER, "-O", str(stage)])
        src = stage
    man_path = root / "MANIFEST.json"
    man = Manifest.load_or_new(man_path, **META)
    n = 0
    for f in sorted(p for p in Path(src).rglob("*") if p.is_file()):
        rel = f.relative_to(src).as_posix()
        dest = raw / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if f.resolve() != dest.resolve():
            shutil.copy2(f, dest)
        h = file_hashes(dest)
        man.add(rel, bytes=dest.stat().st_size, sha256=h["sha256"], source=DRIVE_FOLDER,
                source_checksum=None, verified="size-only (Drive publishes no checksum)")
        n += 1
    man.save(man_path)
    log(f"poselift: {n} files recorded, {man.total_bytes() / 1e6:.1f} MB")
    return 0
