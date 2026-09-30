"""Shared helpers for fetchers: run external CLIs, download-verify-record loops."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from data_ops.budget import GB, require
from data_ops.manifest import Manifest, file_hashes


def tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"required tool {name!r} not found on PATH")
    return path


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)  # noqa: S603


@dataclass(frozen=True)
class RemoteFile:
    relpath: str  # path under the dataset's raw/ dir
    source: str  # URL / s3:// URI
    nbytes: int
    source_checksum: str | None = None  # e.g. "md5:<hex>", "sha256:<hex>"


def verify(local: Path, rf: RemoteFile) -> tuple[bool, str, dict[str, str]]:
    """Check size and, when the origin publishes one, the checksum."""
    size = local.stat().st_size
    if size != rf.nbytes:
        return False, f"size {size} != {rf.nbytes}", {}
    hashes = file_hashes(local)
    if rf.source_checksum:
        algo, _, want = rf.source_checksum.partition(":")
        if hashes.get(algo) != want:
            return False, f"{algo} mismatch", hashes
        return True, f"size+{algo}", hashes
    return True, "size-only (origin publishes no checksum)", hashes


def fetch_all(files: Iterable[RemoteFile], raw_dir: Path, manifest: Manifest, manifest_path: Path,
              download: Callable[[RemoteFile, Path], None], jobs: int = 8, log=print) -> dict[str, int]:
    """Download missing files in parallel; verify, hash, and record each in the manifest."""
    todo = [f for f in files if not manifest.has_verified(f.relpath, f.nbytes)]
    need = sum(f.nbytes for f in todo)
    require(need, raw_dir, what=f"{manifest.dataset_id} fetch")
    log(f"{manifest.dataset_id}: {len(todo)} files to fetch, {need / GB:,.1f} GB "
        f"({len(manifest.files)} already in manifest)")
    lock = Lock()
    stats = {"ok": 0, "failed": 0, "bytes": 0}

    def one(rf: RemoteFile) -> None:
        dest = raw_dir / rf.relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        require(rf.nbytes, raw_dir, what=rf.relpath)
        download(rf, part)
        ok, how, hashes = verify(part, rf)
        if not ok:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"{rf.relpath}: verification failed ({how})")
        part.replace(dest)
        with lock:
            manifest.add(rf.relpath, bytes=rf.nbytes, sha256=hashes["sha256"], source=rf.source,
                         source_checksum=rf.source_checksum, verified=how)
            manifest.save(manifest_path)
            stats["ok"] += 1
            stats["bytes"] += rf.nbytes
            if stats["ok"] % 25 == 0:
                log(f"  {stats['ok']}/{len(todo)} files, {stats['bytes'] / GB:,.1f} GB")

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(one, rf): rf for rf in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:  # keep going; report at the end
                stats["failed"] += 1
                log(f"  FAILED {futs[fut].relpath}: {e}")
    manifest.save(manifest_path)
    return stats
