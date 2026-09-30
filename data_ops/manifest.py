"""Per-dataset MANIFEST.json: what we fetched, from where, under which license, and checksums.

Rule 5 (license gate) and reproducibility both hinge on being able to answer "where
did this file come from and is it intact?" months later. Each dataset dir gets
`<data_root>/<id>/MANIFEST.json`:

    {"dataset_id", "license", "license_url", "attribution", "use",
     "fetched_by", "created", "updated", "notes",
     "files": {"<relpath>": {"bytes", "sha256", "source", "source_checksum", "verified", "fetched"}}}

`source_checksum` is whatever the origin publishes (S3 single-part ETag = MD5, Hugging
Face LFS sha256, ...). `verified` says how the file was checked against it.
The manifest is rewritten atomically after every file, so an interrupted fetch resumes
cleanly and never leaves a half-written manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CHUNK = 8 * 1024 * 1024


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def file_hashes(path: Path, algos: tuple[str, ...] = ("sha256", "md5")) -> dict[str, str]:
    hs = {a: hashlib.new(a, usedforsecurity=False) if a == "md5" else hashlib.new(a) for a in algos}
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            for h in hs.values():
                h.update(chunk)
    return {a: h.hexdigest() for a, h in hs.items()}


@dataclass
class Manifest:
    dataset_id: str
    license: str
    license_url: str
    attribution: str
    use: str  # prod | R&D | pending | blocked, mirrors docs/DATA.md
    fetched_by: str = "data_ops (T14)"
    notes: str = ""
    created: str = field(default_factory=now)
    updated: str = field(default_factory=now)
    files: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load_or_new(cls, path: Path, **defaults: Any) -> Manifest:
        if path.exists():
            d = json.loads(path.read_text())
            m = cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})
            for k, v in defaults.items():  # license facts come from code, not from stale files
                if k in ("license", "license_url", "attribution", "use", "notes") and v:
                    setattr(m, k, v)
            return m
        return cls(**defaults)

    def add(self, relpath: str, **info: Any) -> None:
        self.files[relpath] = {**info, "fetched": info.get("fetched", now())}

    def has_verified(self, relpath: str, nbytes: int | None = None) -> bool:
        e = self.files.get(relpath)
        return bool(e and e.get("verified") and (nbytes is None or e.get("bytes") == nbytes))

    def total_bytes(self) -> int:
        return sum(int(f.get("bytes", 0)) for f in self.files.values())

    def save(self, path: Path) -> None:
        self.updated = now()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self.__dict__, indent=1, sort_keys=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".MANIFEST.", suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.replace(tmp, path)
