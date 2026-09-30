"""NVIDIA PhysicalAI-SmartSpaces (CC-BY-4.0): retail scenes only.

The dataset is 6 TB of mostly warehouse/hospital scenes. Only `MTMC_Tracking_2024`
scenes 071-080 are retail: the README says so, and a one-frame-per-scene visual check
(2026-09-29) confirmed 001-070 are warehouse and 081-090 hospital. All 10 are in the 2024
*test* split, whose ground truth is published (verified: GT boxes align with video).

Each retail scene ALSO contains a back storage room (README: "a storage room distinct from
the primary retail space"), seen by 1-5 of its 16 cameras. `camera_areas()` labels every
camera `retail` or `storage` from its floor (storage = blue epoxy floor; the split is
bimodal: storage >= 0.5 blue, retail <= 0.06) and writes index/camera_areas.json, so
retail-only consumers (T09 smartspaces_track, T14 keyframes) can filter.
Depth maps are skipped (huge). Scene 071 camera_0649 is corrupt per the README (ffprobe
confirms: no moov atom); it's downloaded but flagged.

Every file is verified against the Hugging Face LFS sha256 (or git blob size for
small non-LFS files). Needs `huggingface_hub` (data_ops/requirements.txt).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from data_ops.budget import GB
from data_ops.fetch.common import RemoteFile, fetch_all
from data_ops.manifest import Manifest
from data_ops.paths import dataset_dir

REPO = "nvidia/PhysicalAI-SmartSpaces"
RETAIL_SCENES = tuple(f"MTMC_Tracking_2024/test/scene_{n:03d}" for n in range(71, 81))
SKIP_SUFFIXES = (".h5",)  # depth maps
KNOWN_BAD = {"MTMC_Tracking_2024/test/scene_071/camera_0649/video.mp4": "corrupt per dataset README"}

META = {
    "dataset_id": "smartspaces",
    "license": "CC-BY-4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "attribution": ("NVIDIA PhysicalAI-SmartSpaces (AI City Challenge MTMC), CC BY 4.0. "
                    "https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces"),
    "use": "prod",
    "notes": "Synthetic (Isaac Sim) retail scenes 071-080 of MTMC_Tracking_2024 only; no depth maps.",
}


def list_files(revision: str | None = None) -> tuple[str, list[RemoteFile]]:
    from huggingface_hub import HfApi

    api = HfApi()
    sha = api.dataset_info(REPO, revision=revision).sha
    out: list[RemoteFile] = []
    for scene in RETAIL_SCENES:
        for f in api.list_repo_tree(REPO, path_in_repo=scene, repo_type="dataset", recursive=True,
                                    revision=sha, expand=True):
            size = getattr(f, "size", None)
            if size is None or f.path.endswith(SKIP_SUFFIXES):
                continue
            lfs = getattr(f, "lfs", None)
            checksum = f"sha256:{lfs.sha256}" if lfs else None
            out.append(RemoteFile(relpath=f.path, source=f"hf://datasets/{REPO}@{sha}/{f.path}",
                                  nbytes=size, source_checksum=checksum))
    return sha, out


def fetch(jobs: int = 4, dry_run: bool = False, log=print) -> int:
    root = dataset_dir("smartspaces")
    sha, files = list_files()
    log(f"smartspaces: {REPO}@{sha[:12]}: {len(files)} files, {sum(f.nbytes for f in files) / GB:,.1f} GB "
        f"(retail scenes 071-080)")
    if dry_run:
        return 0
    from huggingface_hub import hf_hub_download

    man_path = root / "MANIFEST.json"
    man = Manifest.load_or_new(man_path, **META)
    man.notes = f"{META['notes']} Revision {sha}. Known bad: {KNOWN_BAD}."

    def download(rf: RemoteFile, dest: Path) -> None:
        with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
            p = hf_hub_download(REPO, rf.relpath, repo_type="dataset", revision=sha, local_dir=tmp)
            shutil.move(p, dest)

    stats = fetch_all(files, root / "raw", man, man_path, download, jobs=jobs, log=log)
    log(f"smartspaces: done {stats}")
    return 1 if stats["failed"] else 0


BLUE_FLOOR_STORAGE = 0.3  # fraction of the lower half that is blue floor; bimodal, see docstring


def camera_areas(log=print) -> dict[str, dict]:
    """Label each downloaded retail-scene camera `retail` / `storage` / `unreadable`."""
    import json

    import cv2

    root = dataset_dir("smartspaces")
    out: dict[str, dict] = {}
    for video in sorted((root / "raw").glob("MTMC_Tracking_2024/test/scene_0*/camera_*/video.mp4")):
        key = f"{video.parent.parent.name}/{video.parent.name}"
        fracs = []
        cap = cv2.VideoCapture(str(video))
        for frame in (3000, 12000):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, img = cap.read()
            if ok:
                hsv = cv2.cvtColor(img[img.shape[0] // 2:], cv2.COLOR_BGR2HSV)
                h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
                fracs.append(float(((h > 95) & (h < 130) & (sat > 80) & (val > 40)).mean()))
        cap.release()
        if not fracs:
            out[key] = {"area": "unreadable", "blue_floor_frac": None}
            continue
        f = round(max(fracs), 3)
        out[key] = {"area": "storage" if f >= BLUE_FLOOR_STORAGE else "retail", "blue_floor_frac": f}
    idx = root / "index" / "camera_areas.json"
    idx.parent.mkdir(parents=True, exist_ok=True)
    idx.write_text(json.dumps({"method": "blue-floor fraction of lower half, frames 3000/12000, "
                                         f"storage if >= {BLUE_FLOOR_STORAGE}", "cameras": out}, indent=1))
    n = {a: sum(v["area"] == a for v in out.values()) for a in ("retail", "storage", "unreadable")}
    log(f"smartspaces: camera areas {n} -> {idx}")
    return out
