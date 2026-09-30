"""Loader API over converted datasets: the one way models, suites and training code read data.

    from eval.datasets import load_dataset
    ds = load_dataset("poselift")                 # license gate + frozen split
    for clip in ds.clips("train"):                # split from eval/splits/poselift.json
        tracks = clip.tracks()                    # TrackTable (boxes, COCO17 kps, ids)
        y = clip.frame_labels()                   # np.ndarray | None
        for seq in clip.pose_sequences(min_len=24):
            ...                                   # PoseSequence: one track, frame-ordered
    for w in ds.windows("train", window=24, stride=6):   # fixed-length causal windows
        w.kps, w.label                            # (24, 17, 3); 1 if the window's last frame is positive

Going through this API (not raw files) is what keeps splits, derivative groups and
the license gate enforced for everyone.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from eval.canonical import ClipLabels, TrackTable, converted_dir, data_root, read_labels, read_tracks
from eval.converters.base import check_license
from eval.splits import Splits


@dataclass(frozen=True)
class PoseSequence:
    clip_id: str
    camera_id: str
    track_id: int
    frame_idx: np.ndarray  # (T,)
    ts: np.ndarray
    kps: np.ndarray  # (T, 17, 3)
    boxes: np.ndarray  # (T, 4)


@dataclass(frozen=True)
class Window:
    clip_id: str
    group_id: str
    track_id: int
    frame_idx: np.ndarray  # (W,) — the window ends at frame_idx[-1] (causal)
    kps: np.ndarray  # (W, 17, 3)
    label: int | None  # frame label at the last frame; None if the clip has none


@dataclass
class Clip:
    dataset: str
    root: Path
    labels: ClipLabels

    @property
    def clip_id(self) -> str:
        return self.labels.clip_id

    @property
    def group_id(self) -> str:
        return self.labels.gid()

    def video_path(self) -> Path | None:
        return data_root() / self.labels.video if self.labels.video else None

    def has_tracks(self) -> bool:
        return (self.root / "tracks" / f"{self.clip_id}.parquet").exists()

    def tracks(self) -> TrackTable:
        p = self.root / "tracks" / f"{self.clip_id}.parquet"
        if not p.exists():
            return TrackTable.empty(self.labels.camera_id or self.clip_id)
        return read_tracks(p)

    def frame_labels(self) -> np.ndarray | None:
        p = self.root / "frame_labels" / f"{self.clip_id}.npy"
        return np.load(p) if p.exists() else None

    def pose_sequences(self, min_len: int = 1) -> Iterator[PoseSequence]:
        t = self.tracks().sorted()
        for tid in np.unique(t.track_id):
            s = t.track(int(tid))
            if len(s) >= min_len:
                yield PoseSequence(self.clip_id, t.camera_id, int(tid), s.frame_idx, s.ts, s.kps, s.boxes)


class Dataset:
    def __init__(
        self, dataset_id: str, root: Path | None = None, split_dir: Path | None = None, check: bool = True
    ) -> None:
        self.id = dataset_id
        self.data_root = root or data_root()
        if check:
            check_license(dataset_id, self.data_root)
        self.dir = converted_dir(dataset_id, self.data_root)
        if not (self.dir / "labels").exists():
            raise FileNotFoundError(f"{dataset_id} not converted: run python -m eval.converters {dataset_id}")
        self._split_dir = split_dir

    @cached_property
    def info(self) -> dict:
        p = self.dir / "INFO.json"
        return json.loads(p.read_text()) if p.exists() else {}

    @cached_property
    def splits(self) -> Splits:
        return Splits.load(self.id, self._split_dir) if self._split_dir else Splits.load(self.id)

    @cached_property
    def _all(self) -> dict[str, ClipLabels]:
        return {p.stem: read_labels(p) for p in sorted((self.dir / "labels").glob("*.json"))}

    def clip_ids(self, split: str | None = None) -> list[str]:
        """Converted clips in `split` (None = all). Unlisted derivatives resolve via group_id;
        clips the split file excludes or doesn't know are never returned for a split."""
        if split is None:
            return list(self._all)
        return [c for c, lab in self._all.items() if self.splits.split_for(c, lab.gid()) == split]

    def clip(self, clip_id: str) -> Clip:
        return Clip(self.id, self.dir, self._all[clip_id])

    def clips(self, split: str | None = None) -> Iterator[Clip]:
        for c in self.clip_ids(split):
            yield self.clip(c)

    def windows(self, split: str, window: int, stride: int = 1) -> Iterator[Window]:
        """Causal fixed-length windows over each track (consecutive rows of one track)."""
        for clip in self.clips(split):
            y = clip.frame_labels()
            for s in clip.pose_sequences(min_len=window):
                for end in range(window - 1, len(s.frame_idx), stride):
                    fi = s.frame_idx[end - window + 1 : end + 1]
                    lab = int(y[fi[-1]]) if y is not None and fi[-1] < len(y) else None
                    yield Window(
                        clip.clip_id, clip.group_id, s.track_id, fi, s.kps[end - window + 1 : end + 1], lab
                    )

    def version(self) -> dict:
        """What a report records as this dataset's version."""
        return {
            "dataset": self.id,
            "converter": self.info.get("converter_version"),
            "format": self.info.get("format_version"),
            "source": self.info.get("source"),
            "split_version": self.splits.data.get("version"),
            "split_digest": self.splits.digest,
        }


def load_dataset(dataset_id: str, root: Path | None = None, split_dir: Path | None = None) -> Dataset:
    return Dataset(dataset_id, root, split_dir)
