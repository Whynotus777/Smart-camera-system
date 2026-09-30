"""PoseLift (UNC Charlotte, WACV 2025): pose-only shoplifting benchmark.

The data (per-video .pkl pose files + .npy frame labels) is a public Google Drive
folder linked from https://github.com/TeCSAR-UNCC/PoseLift. That GitHub repo is
Apache-2.0, but the data isn't in it, so whether Apache-2.0 covers the data is
unconfirmed (docs/DATA.md: R&D until confirmed). Drive publishes no checksums; we
record sha256 of what we got. Uses `gdown` (MIT) via `uvx`, so no project dependency.

Google Drive throttles folders with many small files ("Cannot retrieve the public link ...
many accesses"), typically for ~24 h. So each attempt is capped, resumes with
`gdown --continue` into a persistent staging dir, and records what it has; the manifest
notes say `INCOMPLETE n/329` until every file is in. Retry with `--retry-every-h`.

Security: the Drive folder holds `.pkl` Python pickles. Never `pickle.load` them unless
through a restricted unpickler; prefer the `.json` pose files with the same content.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from data_ops.budget import require
from data_ops.fetch.common import tool
from data_ops.manifest import Manifest, file_hashes
from data_ops.paths import dataset_dir

DRIVE_FOLDER = "https://drive.google.com/drive/folders/1aEkENZlVE4ZvF_BZXV1VJOwuiXQq6trn"
EXPECTED_FILES = 329  # Drive listing on 2026-09-29: 151 .json, 94 .npy, 83 .pkl, 1 other
ATTEMPT_TIMEOUT_S = 600
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


def fetch(src: Path | None = None, retry_every_h: float | None = None, log=print) -> int:
    """Download the Drive folder (or ingest an already-downloaded copy at `src`) and record it.

    Returns 0 when complete, 4 when Drive is still throttling (partial copy recorded).
    With `retry_every_h`, keeps making one capped attempt per interval until complete.
    """
    while True:
        rc = _attempt(src, log)
        if rc == 0 or retry_every_h is None or src is not None:
            return rc
        log(f"poselift: incomplete, next attempt in {retry_every_h:g} h")
        time.sleep(retry_every_h * 3600)


def _attempt(src: Path | None, log) -> int:
    root = dataset_dir("poselift")
    raw = root / "raw"
    require(2 * 10**9, root, what="poselift")  # generous upper bound; the pose data is small
    if src is None:
        stage = root / ".staging"
        stage.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run([tool("uvx"), "--quiet", "gdown", "--continue", "--retries", "2",  # noqa: S603
                            DRIVE_FOLDER, "-O", str(stage)], capture_output=True, text=True,
                           timeout=ATTEMPT_TIMEOUT_S, check=False)
        except subprocess.TimeoutExpired:
            log(f"poselift: attempt hit the {ATTEMPT_TIMEOUT_S}s cap (Drive throttling?)")
        src = stage
    man_path = root / "MANIFEST.json"
    man = Manifest.load_or_new(man_path, **META)
    n = 0
    done = (p for p in Path(src).rglob("*")
            if p.is_file() and not p.name.startswith(".") and not p.name.endswith((".part", ".tmp")))
    for f in sorted(done):
        rel = f.relative_to(src).as_posix()
        dest = raw / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if f.resolve() != dest.resolve():
            shutil.copy2(f, dest)
        h = file_hashes(dest)
        man.add(rel, bytes=dest.stat().st_size, sha256=h["sha256"], source=DRIVE_FOLDER,
                source_checksum=None, verified="size-only (Drive publishes no checksum)")
        n += 1
    complete = n >= EXPECTED_FILES
    status = "" if complete else f" INCOMPLETE {n}/{EXPECTED_FILES} files (Drive throttling)."
    man.notes = META["notes"] + status
    man.save(man_path)
    log(f"poselift: {n}/{EXPECTED_FILES} files recorded, {man.total_bytes() / 1e6:.1f} MB")
    return 0 if complete else 4
