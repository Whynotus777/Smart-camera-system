import hashlib
import json
import shutil
import subprocess

import pytest

from data_ops import paths
from data_ops.manifest import Manifest, file_hashes


def test_file_hashes(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    h = file_hashes(f)
    assert h["sha256"] == hashlib.sha256(b"hello").hexdigest()
    assert h["md5"] == "5d41402abc4b2a76b9719d911017c592"


def test_manifest_roundtrip_and_resume(tmp_path):
    p = tmp_path / "ds" / "MANIFEST.json"
    meta = {"dataset_id": "ds", "license": "CC-BY-4.0", "license_url": "u", "use": "prod"}
    m = Manifest.load_or_new(p, **meta, attribution="a")
    m.add("a/b.avi", bytes=5, sha256="00", source="s3://x", verified="size+md5")
    m.save(p)
    assert not list(p.parent.glob(".MANIFEST.*.tmp")), "atomic save left a temp file"
    m2 = Manifest.load_or_new(p, **meta, attribution="a2")
    assert m2.has_verified("a/b.avi", 5) and not m2.has_verified("a/b.avi", 6)
    assert m2.attribution == "a2", "license facts come from code, not the stale file"
    assert m2.total_bytes() == 5
    assert json.loads(p.read_text())["files"]["a/b.avi"]["verified"] == "size+md5"


def test_data_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SCS_DATA_ROOT", str(tmp_path))
    assert paths.data_root() == tmp_path.resolve()
    assert paths.dataset_dir("meva", "raw") == tmp_path.resolve() / "meva" / "raw"


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_data_root_is_shared_across_worktrees(monkeypatch, tmp_path):
    monkeypatch.delenv("SCS_DATA_ROOT", raising=False)
    main, wt = tmp_path / "main", tmp_path / "wt"
    git = shutil.which("git")
    run = lambda *a, cwd=main: subprocess.run([git, *a], cwd=cwd, check=True, capture_output=True)  # noqa: E731, S603
    main.mkdir()
    run("init", "-q")
    run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x")
    run("worktree", "add", "-q", str(wt))
    assert paths.data_root(main) == main.resolve() / "data"
    assert paths.data_root(wt) == main.resolve() / "data", "worktrees must share the main checkout's data/"
