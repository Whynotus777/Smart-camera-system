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
