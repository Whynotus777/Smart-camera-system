import hashlib

import pytest

from data_ops.budget import BudgetError
from data_ops.fetch import common
from data_ops.fetch.common import RemoteFile, fetch_all
from data_ops.manifest import Manifest

META = {"dataset_id": "t", "license": "CC-BY-4.0", "license_url": "u", "attribution": "a", "use": "prod"}


def _payload(name: str) -> bytes:
    return (name * 100).encode()


def _fake_download(rf, dest):
    dest.write_bytes(_payload(rf.relpath))


def _rf(name, checksum=True, corrupt=False):
    data = _payload(name)
    md5 = hashlib.md5(b"x" if corrupt else data, usedforsecurity=False).hexdigest()
    return RemoteFile(relpath=name, source=f"s3://b/{name}", nbytes=len(data),
                      source_checksum=f"md5:{md5}" if checksum else None)


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "require", lambda n, p, what="": None)
    mp = tmp_path / "MANIFEST.json"
    return tmp_path / "raw", Manifest.load_or_new(mp, **META), mp


def test_fetch_verifies_records_and_resumes(env):
    raw, man, mp = env
    files = [_rf("a/1.avi"), _rf("a/2.avi", checksum=False), _rf("b/3.avi", corrupt=True)]
    stats = fetch_all(files, raw, man, mp, _fake_download, jobs=2, log=lambda *_: None)
    assert stats == {"ok": 2, "failed": 1, "bytes": files[0].nbytes + files[1].nbytes}
    assert man.files["a/1.avi"]["verified"] == "size+md5"
    assert man.files["a/2.avi"]["verified"].startswith("size-only")
    assert "b/3.avi" not in man.files and not (raw / "b" / "3.avi").exists()
    assert not list(raw.rglob("*.part")), "failed/partial downloads must be cleaned up"
    calls = []
    fetch_all(files[:2], raw, Manifest.load_or_new(mp, **META), mp,
              lambda rf, d: calls.append(rf), log=lambda *_: None)
    assert calls == [], "verified files must not be downloaded again"


def test_fetch_refuses_over_budget(monkeypatch, env):
    raw, man, mp = env

    def refuse(n, p, what=""):
        raise BudgetError("no")

    monkeypatch.setattr(common, "require", refuse)
    with pytest.raises(BudgetError):
        fetch_all([_rf("x.avi")], raw, man, mp, _fake_download, log=lambda *_: None)


def test_probe_video_and_summary(tmp_path):
    import shutil
    import subprocess

    assert common.probe_video(tmp_path / "x.json") is None  # not a video extension
    bogus = tmp_path / "bogus.avi"
    bogus.write_bytes(b"not a video")
    assert common.probe_video(bogus) is None  # unreadable: no crash
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        v = tmp_path / "v.mp4"
        subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",  # noqa: S603
                        "testsrc2=size=1920x1072:rate=30:duration=1", "-c:v", "libx264", str(v)], check=True)
        p = common.probe_video(v)
        assert (p["codec"], p["width"], p["height"], p["fps"], p["frames"]) == ("h264", 1920, 1072, 30.0, 30)
        m = Manifest.load_or_new(tmp_path / "M.json", **META)
        m.add("v.mp4", bytes=1, video=p)
        m.add("w.mp4", bytes=1, video={**p, "height": 1080})
        m.add("x.mp4", bytes=1, video=p)
        assert common.video_summary(m) == "1920x1072@30fps h264: 2 files; 1920x1080@30fps h264: 1 files"


def test_dataset_lock_is_exclusive(tmp_path):
    with common.dataset_lock(tmp_path), pytest.raises(RuntimeError, match="holds"):
        with common.dataset_lock(tmp_path):
            pass
    with common.dataset_lock(tmp_path):  # released after the first block
        pass


def test_s3_multipart_etag(tmp_path):
    f = tmp_path / "obj.bin"
    data = bytes(range(256)) * (4096 * 50)  # 50 MiB
    f.write_bytes(data)
    ps = 8 * 1024 * 1024
    parts = [hashlib.md5(data[i:i + ps], usedforsecurity=False).digest() for i in range(0, len(data), ps)]
    etag = f'"{hashlib.md5(b"".join(parts), usedforsecurity=False).hexdigest()}-{len(parts)}"'
    assert common.s3_etag_matches(f, etag) is True
    assert common.s3_etag_matches(f, etag.replace(etag[1], "0" if etag[1] != "0" else "1")) is False
    single = f'"{hashlib.md5(data, usedforsecurity=False).hexdigest()}"'
    assert common.s3_etag_matches(f, single) is True
    rf = RemoteFile("obj.bin", "s3://b/obj.bin", len(data), f"s3etag:{etag.strip(chr(34))}")
    ok, how, _ = common.verify(f, rf)
    assert ok and how == "size+s3etag(multipart md5)"
