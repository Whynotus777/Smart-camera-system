"""DirectorySource: sim frames from a directory of images."""

from __future__ import annotations

import numpy as np
import pytest

from scs.ingest.base import FrameSource
from scs.ingest.sources import DirectorySource


def test_directory_source_reads_sorted_frames_with_media_time(tmp_path):
    for i in (2, 0, 1):
        np.save(tmp_path / f"frame_{i:04d}.npy", np.full((4, 6, 3), i, np.uint8))
    (tmp_path / "notes.txt").write_text("ignored")
    src = DirectorySource(tmp_path, "sim1", fps=10.0, origin_ts=50.0)
    assert isinstance(src, FrameSource)
    out = list(src.frames())
    assert [int(img[0, 0, 0]) for _, img in out] == [0, 1, 2]
    refs = [r for r, _ in out]
    assert [r.identity for r in refs] == [("sim1", 0, i) for i in range(3)]
    assert [r.source_ts for r in refs] == pytest.approx([50.0, 50.1, 50.2])
    assert (refs[0].width, refs[0].height) == (6, 4) and out[0][1].shape == (4, 6, 3)
    assert [e.data["state"] for e in src.health.events] == ["connected", "eos"]


def test_directory_source_empty_dir_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        DirectorySource(tmp_path)
