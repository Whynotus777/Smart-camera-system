"""Shared helpers for fetchers: run external CLIs, download-verify-record loops."""

from __future__ import annotations

import fcntl
import json
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
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


VIDEO_EXT = (".avi", ".mp4", ".mkv", ".mov")


def probe_video(path: Path) -> dict | None:
    """codec / width / height / fps / frames / duration of a video file (None if not a video)."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or path.suffix.lower() not in VIDEO_EXT:
        return None
    try:
        out = json.loads(run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration",
                              "-of", "json", str(path)]).stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None  # unreadable as video: record the file anyway, just without properties
    if not out.get("streams"):
        return None
    st = out["streams"][0]
    num, _, den = st.get("r_frame_rate", "0/1").partition("/")
    return {"codec": st.get("codec_name"), "width": st.get("width"), "height": st.get("height"),
            "fps": round(float(num) / float(den or 1), 3),
            "frames": int(st["nb_frames"]) if str(st.get("nb_frames", "")).isdigit() else None,
            "duration_s": round(float(out.get("format", {}).get("duration", 0) or 0), 3)}


def video_summary(manifest: Manifest) -> str:
    """e.g. '1920x1072@30fps h264: 1180 files; 1920x1080@30fps h264: 216 files'."""
    c = Counter(f"{v['width']}x{v['height']}@{v['fps']:g}fps {v['codec']}"
                for f in manifest.files.values() if (v := f.get("video")))
    return "; ".join(f"{k}: {n} files" for k, n in c.most_common()) or "no video probed"


@contextmanager
def dataset_lock(root: Path) -> Iterator[None]:
    """One writer per dataset manifest at a time (fetch vs. backfill), across processes."""
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".fetch.lock", "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another fetch/backfill holds {root / '.fetch.lock'}") from None
        yield


def backfill_video(manifest: Manifest, manifest_path: Path, raw_dir: Path, log=print) -> int:
    """Probe files recorded without `video` properties (e.g. fetched by an older version)."""
    n = 0
    with dataset_lock(manifest_path.parent):
        for rel, entry in manifest.files.items():
            if "video" not in entry and (v := probe_video(raw_dir / rel)):
                entry["video"] = v
                n += 1
        manifest.save(manifest_path)
    log(f"{manifest.dataset_id}: probed {n} files; {video_summary(manifest)}")
    return n


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
        extra = {"video": v} if (v := probe_video(dest)) else {}
        with lock:
            manifest.add(rf.relpath, bytes=rf.nbytes, sha256=hashes["sha256"], source=rf.source,
                         source_checksum=rf.source_checksum, verified=how, **extra)
            manifest.save(manifest_path)
            stats["ok"] += 1
            stats["bytes"] += rf.nbytes
            if stats["ok"] % 25 == 0:
                log(f"  {stats['ok']}/{len(todo)} files, {stats['bytes'] / GB:,.1f} GB")

    with dataset_lock(manifest_path.parent), ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(one, rf): rf for rf in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:  # keep going; report at the end
                stats["failed"] += 1
                log(f"  FAILED {futs[fut].relpath}: {e}")
    manifest.save(manifest_path)
    return stats
