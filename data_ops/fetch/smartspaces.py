"""NVIDIA PhysicalAI-SmartSpaces (CC-BY-4.0): retail scenes only.

The dataset is 6 TB of mostly warehouse/hospital scenes. Only `MTMC_Tracking_2024`
scenes 071-080 are retail: the README says so (071-080 share a retail space plus a
storage room; 072 is that storage room), and a one-frame-per-scene visual check on
2026-09-29 confirmed 001-070 are warehouse and 081-090 hospital. All 10 are in the
2024 *test* split, whose ground truth is published. Depth maps are skipped (huge; RGB
is enough for detection/tracking). Scene 071 camera_0649 is corrupt per the README;
it's downloaded but flagged in the manifest.

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
