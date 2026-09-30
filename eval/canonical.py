"""The canonical dataset format (docs/DATA.md "Canonical format") and its reader/writer.

Every converter writes this, every suite reads it, so a model never sees a
dataset-specific quirk. Layout under `<data_root>/<id>/converted/`:

    tracks/<clip_id>.parquet        one row per (frame, track): camera_id, frame_idx, ts,
                                    track_id, x1, y1, x2, y2, score, kp_0_x ... kp_16_c
                                    (keypoints NaN when the dataset has none)
    labels/<clip_id>.json           ClipLabels (below)
    frame_labels/<clip_id>.npy      optional per-frame 0/1 labels (PoseLift-style benchmarks)
    INFO.json                       converter id/version, source manifest digest, counts

Additions to the DATA.md format (all optional, all backward compatible) are marked
"T09" below and listed in docs/EVAL.md. Labels only claim what the source supports:
`supports` says which metrics the labels can back; everything else is "unavailable".
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from scs.contracts import EventType

FORMAT_VERSION = "1"

# docs/DATA.md interaction labels
INTERACTION_LABELS = frozenset(
    {
        "item_pickup",
        "item_returned",
        "item_to_basket",
        "item_to_bag",
        "item_to_clothing",
        "obscured_interaction",
        "exit_no_checkout",
        "grab_run",
    }
)
# T09 additions, so converters never have to force a source label into a wrong bucket:
#  item_put_down   object put down on any surface (MEVA puts_down; not "returned to shelf")
#  item_transfer   object handed person to person (MEVA transfers)
#  shoplifting     dataset-level theft label with no subtype (PoseLift/RetailS frame labels)
T09_LABELS = frozenset({"item_put_down", "item_transfer", "shoplifting"})
EVENT_TYPE_NAMES = frozenset(e.value for e in EventType)
LABEL_VOCAB = INTERACTION_LABELS | T09_LABELS | EVENT_TYPE_NAMES

Visible = Literal["observed", "partially_observed", "not_observed"]
LabelSource = Literal["human", "script", "model"]

KP_COLS = [f"kp_{i}_{c}" for i in range(17) for c in ("x", "y", "c")]
TRACK_COLS = ["camera_id", "frame_idx", "ts", "track_id", "x1", "y1", "x2", "y2", "score", *KP_COLS]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LabelEvent(_M):
    type: str
    t_start: float  # seconds from clip start
    t_end: float
    track_id: int | None = None
    actor_id: str | None = None
    visible: Visible = "observed"  # for THIS camera
    subtype: str | None = None  # pocket | waistband | jacket | bag | ...
    label_source: LabelSource | None = None  # None = the clip's label_source
    source_label: str | None = None  # T09: the dataset's own label, verbatim
    event_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)  # T09: dataset-specific, never used by metrics

    @field_validator("type")
    @classmethod
    def _vocab(cls, v: str) -> str:
        if v not in LABEL_VOCAB:
            raise ValueError(f"unknown label type {v!r}; add it to eval.canonical first")
        return v

    @model_validator(mode="after")
    def _span(self) -> LabelEvent:
        if self.t_end < self.t_start:
            raise ValueError("t_end < t_start")
        return self


class LabelSupport(_M):
    """T09: which metrics this clip's labels can back. False -> metric 'unavailable'."""

    events: bool = False  # interaction/theft event spans (recall, FA)
    subtypes: bool = False
    actors: bool = False  # actor ids stable across clips
    journeys: bool = False  # checkout_visit / store_exit labels
    gt_tracks: bool = False  # tracks parquet is human/sim GT (IDF1, mAP)
    keypoints: bool = False
    frame_labels: bool = False
    exhaustive: bool = False  # every instance of the labeled types is labeled (FA countable)


class ClipLabels(_M):
    clip_id: str
    camera_profile: str | None = None
    fps: float = Field(gt=0)
    label_source: LabelSource
    events: list[LabelEvent] = Field(default_factory=list)
    # --- T09 additions (optional) ---
    dataset: str | None = None
    camera_id: str | None = None
    site_id: str | None = None
    group_id: str | None = None  # derivative group: all derivatives share it and a split
    view_group: str | None = None  # simultaneous overlapping views (same people, same time)
    actor_ids: list[str] | None = None  # None = unknown (not "no actors")
    width: int | None = None
    height: int | None = None
    n_frames: int | None = None
    duration_s: float | None = None
    t0: float | None = None  # wall-clock epoch seconds of frame 0, if known
    continuous: bool = False  # uncut footage: false alerts/hour may be computed on it
    synthetic: bool = False
    video: str | None = None  # path relative to the data root, if video exists
    supports: LabelSupport = Field(default_factory=LabelSupport)
    extra: dict[str, Any] = Field(default_factory=dict)

    def gid(self) -> str:
        return self.group_id or self.clip_id


# ---------------------------------------------------------------------------
# Data root and paths
# ---------------------------------------------------------------------------


def data_root() -> Path:
    """Shared data dir: $SCS_DATA_ROOT, else <main checkout>/data (same rule as data_ops/T14)."""
    env = os.environ.get("SCS_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    git = shutil.which("git")
    if git:
        try:
            common = subprocess.run(  # noqa: S603
                [git, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                cwd=Path(__file__).resolve().parent,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            return Path(common).parent / "data"
        except subprocess.CalledProcessError:
            pass
    return Path.cwd() / "data"


def converted_dir(dataset_id: str, root: Path | None = None) -> Path:
    return (root or data_root()) / dataset_id / "converted"


# ---------------------------------------------------------------------------
# Track tables
# ---------------------------------------------------------------------------


@dataclass
class TrackTable:
    """Column arrays of one clip's tracks. `kps` is (N, 17, 3), NaN when absent."""

    camera_id: str
    frame_idx: np.ndarray
    ts: np.ndarray
    track_id: np.ndarray
    boxes: np.ndarray  # (N, 4) x1 y1 x2 y2, main-stream pixels
    score: np.ndarray
    kps: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.frame_idx)
        shapes = [len(self.ts), len(self.track_id), len(self.score), len(self.boxes), len(self.kps)]
        if any(s != n for s in shapes) or self.boxes.shape[1:] != (4,) or self.kps.shape[1:] != (17, 3):
            raise ValueError("TrackTable columns have inconsistent shapes")

    def __len__(self) -> int:
        return len(self.frame_idx)

    @classmethod
    def empty(cls, camera_id: str) -> TrackTable:
        z = np.zeros(0)
        return cls(
            camera_id, z.astype(np.int64), z, z.astype(np.int64), np.zeros((0, 4)), z, np.zeros((0, 17, 3))
        )

    def sorted(self) -> TrackTable:
        o = np.lexsort((self.track_id, self.frame_idx))
        return TrackTable(
            self.camera_id,
            self.frame_idx[o],
            self.ts[o],
            self.track_id[o],
            self.boxes[o],
            self.score[o],
            self.kps[o],
        )

    def track(self, tid: int) -> TrackTable:
        m = self.track_id == tid
        return TrackTable(
            self.camera_id,
            self.frame_idx[m],
            self.ts[m],
            self.track_id[m],
            self.boxes[m],
            self.score[m],
            self.kps[m],
        )


def _pa():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise ImportError("canonical tracks are parquet: pip install -r eval/requirements.txt") from e
    return pa, pq


def write_tracks(path: Path, t: TrackTable) -> None:
    pa, pq = _pa()
    t = t.sorted()
    cols: dict[str, Any] = {
        "camera_id": pa.array([t.camera_id] * len(t), pa.string()),
        "frame_idx": pa.array(t.frame_idx.astype(np.int64)),
        "ts": pa.array(t.ts.astype(np.float64)),
        "track_id": pa.array(t.track_id.astype(np.int64)),
    }
    for i, c in enumerate(("x1", "y1", "x2", "y2")):
        cols[c] = pa.array(t.boxes[:, i].astype(np.float32))
    cols["score"] = pa.array(t.score.astype(np.float32))
    flat = t.kps.reshape(len(t), 51).astype(np.float32)
    for i, c in enumerate(KP_COLS):
        cols[c] = pa.array(flat[:, i])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(pa.table(cols), tmp, compression="zstd")
    os.replace(tmp, path)


def read_tracks(path: Path) -> TrackTable:
    _, pq = _pa()
    tb = pq.read_table(path)
    missing = [c for c in TRACK_COLS if c not in tb.column_names]
    if missing:
        raise ValueError(f"{path}: missing canonical columns {missing[:5]}")
    col = {c: tb.column(c).to_numpy(zero_copy_only=False) for c in TRACK_COLS}
    n = tb.num_rows
    cam = str(col["camera_id"][0]) if n else path.stem
    kps = (
        np.stack([col[c] for c in KP_COLS], axis=1).reshape(n, 17, 3).astype(np.float64)
        if n
        else np.zeros((0, 17, 3))
    )
    return TrackTable(
        cam,
        col["frame_idx"].astype(np.int64),
        col["ts"].astype(np.float64),
        col["track_id"].astype(np.int64),
        np.stack([col[c] for c in ("x1", "y1", "x2", "y2")], axis=1).astype(np.float64).reshape(n, 4),
        col["score"].astype(np.float64),
        kps,
    )


# ---------------------------------------------------------------------------
# Labels + dataset info
# ---------------------------------------------------------------------------


def write_labels(path: Path, labels: ClipLabels) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(labels.model_dump_json(indent=1, exclude_defaults=False))
    os.replace(tmp, path)


def read_labels(path: Path) -> ClipLabels:
    return ClipLabels.model_validate_json(path.read_text())


def manifest_digest(dataset_id: str, root: Path | None = None) -> dict[str, Any]:
    """Version facts of the raw data from T14's MANIFEST.json (license, updated, digest)."""
    p = (root or data_root()) / dataset_id / "MANIFEST.json"
    if not p.exists():
        return {"manifest": None}
    m = json.loads(p.read_text())
    files = m.get("files", {})
    h = hashlib.sha256(
        json.dumps({k: v.get("sha256") for k, v in sorted(files.items())}).encode()
    ).hexdigest()
    return {
        "manifest": str(p.name),
        "license": m.get("license"),
        "use": m.get("use"),
        "updated": m.get("updated"),
        "n_files": len(files),
        "files_digest": h[:16],
    }


def write_info(
    dataset_id: str,
    converter: str,
    converter_version: str,
    counts: dict[str, Any],
    license_: str,
    root: Path | None = None,
    notes: str = "",
) -> dict[str, Any]:
    info = {
        "dataset_id": dataset_id,
        "format_version": FORMAT_VERSION,
        "converter": converter,
        "converter_version": converter_version,
        "license": license_,
        "source": manifest_digest(dataset_id, root),
        "counts": counts,
        "notes": notes,
    }
    d = converted_dir(dataset_id, root)
    d.mkdir(parents=True, exist_ok=True)
    (d / "INFO.json").write_text(json.dumps(info, indent=1, sort_keys=True))
    return info
