"""PoseLift (Rashvand et al., WACV 2025) pose tracks + frame labels -> canonical format.

Why: the one public real-store theft benchmark (pose only), used by `public_pose` for
comparability with published STG-NF numbers. Not a system-level validation.

Documented format (github.com/TeCSAR-UNCC/PoseLift README; Drive folder has
`Pickle_files/`, `Json_files/`, `STG-NF/`, `Instructions.pdf`):
- `<camera>_<video>.pkl`: dict keyed by frame number; per person: id, bbox XYWH,
  keypoints XYC (COCO17 via HRNet, 1920x1080, 15 fps; interpolated + smoothed upstream).
- `<camera>_<video>.npy`: per-frame 0/1 shoplifting labels, length = frame count.
- `STG-NF/`: the data as STG-NF consumes it (ShanghaiTech-style `{person: {frame:
  {"keypoints": [51], "scores": s}}}` JSON under `.../train|test/`, test frame masks
  as `.npy`). Its train/test directories are the official split.

The exact nesting inside the pickles is NOT verified yet (data not downloaded when this
was written), so the parser accepts the plausible layouts and fails loudly on anything
else. TODO(T09) when T14 fetches the data: run `pytest -m data tests/eval/test_poselift_real.py`.

Canonical output: tracks are *model outputs* upstream (YOLOv8+ByteTrack+HRNet), so
`supports.gt_tracks` is False; frame labels -> `frame_labels/<clip>.npy` and contiguous
runs of 1s -> events of type `shoplifting` (no subtype, no track: the labels don't say).
"""

from __future__ import annotations

import json
import pickle  # noqa: S403 - PoseLift ships pickles; only load files listed in T14's manifest
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from eval.canonical import (
    ClipLabels,
    LabelEvent,
    LabelSupport,
    TrackTable,
    converted_dir,
    data_root,
    write_info,
    write_labels,
    write_tracks,
)
from eval.converters.base import LICENSES, check_license, register

VERSION = "1"
FPS = 15.0
WIDTH, HEIGHT = 1920, 1080
_NAME = re.compile(r"(?:^|/)C?(\d+)_(\d+)")


def clip_name(path: Path) -> str:
    """'.../C1_12_alphapose_tracked_person.json' or '1_12.pkl' -> '1_12'."""
    m = _NAME.search(path.name)
    if not m:
        raise ValueError(f"can't parse PoseLift clip name from {path.name}")
    return f"{int(m.group(1))}_{int(m.group(2))}"


def _kps(v: Any) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.size != 51:
        raise ValueError(f"expected 17x3 keypoints, got {a.size} values")
    return a.reshape(17, 3)


def _box_xywh(v: Any) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.size != 4:
        raise ValueError(f"expected XYWH box, got {a.size} values")
    return np.array([a[0], a[1], a[0] + a[2], a[1] + a[3]])


def _pick(d: dict, *names: str) -> Any:
    for k in d:
        if str(k).lower().replace(" ", "_") in names:
            return d[k]
    return None


def _person(rec: Any, pid: Any = None) -> tuple[Any, np.ndarray | None, np.ndarray]:
    if isinstance(rec, dict):
        pid = _pick(rec, "person_id", "id", "track_id", "pid", "idx") if pid is None else pid
        box = _pick(rec, "bbox", "box", "bounding_box", "bboxes")
        kp = _pick(rec, "keypoints", "kps", "pose", "keypoint")
        if kp is None:
            raise ValueError(f"no keypoints in record keys {list(rec)}")
        return pid, (_box_xywh(box) if box is not None else None), _kps(kp)
    if isinstance(rec, list | tuple) and len(rec) == 3:  # (id, bbox, keypoints)
        return rec[0], _box_xywh(rec[1]), _kps(rec[2])
    raise ValueError(f"unrecognized person record: {type(rec).__name__}")


def iter_pickle(obj: Any) -> Iterator[tuple[int, Any, np.ndarray | None, np.ndarray]]:
    """Yield (frame, person_id, box_xyxy|None, kps17x3) from `{frame: people}`."""
    if not isinstance(obj, dict):
        raise ValueError(f"PoseLift pickle: expected dict keyed by frame, got {type(obj).__name__}")
    for f, people in obj.items():
        frame = int(f)
        if isinstance(people, dict) and _pick(people, "keypoints", "kps", "pose") is None:
            items = [(_person(r, pid)) for pid, r in people.items()]  # {pid: record}
        elif isinstance(people, dict):
            items = [_person(people)]  # single record
        else:
            items = [_person(r) for r in people]  # [record, ...]
        for pid, box, kp in items:
            yield frame, pid, box, kp


def iter_stgnf_json(obj: dict) -> Iterator[tuple[int, Any, np.ndarray | None, np.ndarray]]:
    """ShanghaiTech/STG-NF layout: {person_id: {frame: {"keypoints": [51], "scores": s}}}."""
    for pid, frames in obj.items():
        for f, rec in frames.items():
            yield int(f), pid, None, _kps(rec["keypoints"])


def _table(camera_id: str, rows: list[tuple[int, Any, np.ndarray | None, np.ndarray]]) -> TrackTable:
    ids: dict[Any, int] = {}
    for _, pid, _, _ in rows:
        if pid not in ids:
            ids[pid] = int(pid) if str(pid).lstrip("-").isdigit() else len(ids)
    if len(set(ids.values())) != len(ids):
        raise ValueError("person ids collide after int conversion")
    n = len(rows)
    frame = np.array([r[0] for r in rows], dtype=np.int64)
    kps = np.stack([r[3] for r in rows]) if n else np.zeros((0, 17, 3))
    boxes = np.zeros((n, 4))
    for i, (_, _, box, kp) in enumerate(rows):
        if box is None:  # derive from confident keypoints (STG-NF json has no boxes)
            ok = kp[:, 2] > 0
            pts = kp[ok, :2] if ok.any() else kp[:, :2]
            box = np.r_[pts.min(0), pts.max(0) + 1e-3]
        boxes[i] = box
    return TrackTable(
        camera_id,
        frame,
        frame / FPS,
        np.array([ids[r[1]] for r in rows], dtype=np.int64),
        boxes,
        np.nanmean(kps[:, :, 2], axis=1) if n else np.zeros(0),
        kps,
    )


def label_events(y: np.ndarray, fps: float = FPS) -> list[LabelEvent]:
    y = np.asarray(y).astype(int).reshape(-1)
    d = np.diff(np.r_[0, y, 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)  # [start, end) in frames
    return [
        LabelEvent(
            type="shoplifting",
            t_start=s / fps,
            t_end=e / fps,
            visible="observed",
            source_label="shoplifting (frame label)",
        )
        for s, e in zip(starts, ends, strict=True)
    ]


def discover(raw: Path) -> dict[str, dict[str, Path]]:
    """clip -> {"pose": path, "labels": path|None, "split": train|test|None}. Pickles preferred."""
    out: dict[str, dict[str, Any]] = defaultdict(dict)
    for p in sorted(raw.rglob("*")):
        if not p.is_file():
            continue
        parts = {x.lower() for x in p.relative_to(raw).parts}
        split = "test" if "test" in parts else ("train" if "train" in parts else None)
        try:
            clip = clip_name(p)
        except ValueError:
            continue
        rec = out[clip]
        if split:
            rec.setdefault("split", split)
        if p.suffix == ".pkl" and ("pose" not in rec or rec["pose"].suffix != ".pkl"):
            rec["pose"] = p
        elif p.suffix == ".json" and "pose" not in rec:
            rec["pose"] = p
        elif p.suffix == ".npy" and "labels" not in rec:
            rec["labels"] = p
    return {c: r for c, r in out.items() if "pose" in r}


def convert_clip(clip: str, rec: dict[str, Any], out: Path) -> ClipLabels:
    p: Path = rec["pose"]
    if p.suffix == ".pkl":
        with open(p, "rb") as f:
            rows = list(iter_pickle(pickle.load(f)))  # noqa: S301 - manifest-listed dataset file
    else:
        rows = list(iter_stgnf_json(json.loads(p.read_text())))
    cam = clip.split("_")[0]
    table = _table(f"poselift.C{cam}", rows)
    write_tracks(out / "tracks" / f"{clip}.parquet", table)
    y = np.load(rec["labels"]) if rec.get("labels") else None
    if y is not None:
        (out / "frame_labels").mkdir(parents=True, exist_ok=True)
        np.save(out / "frame_labels" / f"{clip}.npy", np.asarray(y).astype(np.uint8).reshape(-1))
        if len(table) and table.frame_idx.max() >= len(y):
            raise ValueError(f"{clip}: pose frame {table.frame_idx.max()} beyond {len(y)} labeled frames")
    n_frames = len(y) if y is not None else (int(table.frame_idx.max()) + 1 if len(table) else 0)
    labels = ClipLabels(
        clip_id=clip,
        fps=FPS,
        label_source="human",
        events=label_events(y) if y is not None else [],
        dataset="poselift",
        camera_id=f"poselift.C{cam}",
        site_id="poselift_store",
        group_id=clip,
        actor_ids=None,
        width=WIDTH,
        height=HEIGHT,
        n_frames=n_frames,
        duration_s=n_frames / FPS,
        continuous=False,
        synthetic=False,
        supports=LabelSupport(events=y is not None, frame_labels=y is not None, keypoints=True),
        extra={
            "official_split": rec.get("split"),
            "pose_source": p.name,
            "pose_pipeline": "YOLOv8 + ByteTrack + HRNet, interpolated + 8-frame smoothed (upstream)",
        },
    )
    write_labels(out / "labels" / f"{clip}.json", labels)
    return labels


@register("poselift", VERSION)
def convert(root: Path | None = None, raw: Path | None = None, log=print) -> dict[str, Any]:
    root = root or data_root()
    check_license("poselift", root)
    raw = raw or root / "poselift" / "raw"
    found = discover(raw)
    if not found:
        raise FileNotFoundError(
            f"no PoseLift pose files under {raw} (T14 fetches: python -m data_ops.fetch poselift)"
        )
    out = converted_dir("poselift", root)
    counts: dict[str, Any] = defaultdict(int)
    for clip, rec in sorted(found.items()):
        lab = convert_clip(clip, rec, out)
        counts["clips"] += 1
        counts["events"] += len(lab.events)
        counts["frames"] += lab.n_frames or 0
        counts[f"official_{rec.get('split')}"] += 1
    counts = dict(counts)
    write_info(
        "poselift",
        "eval.converters.poselift",
        VERSION,
        counts,
        LICENSES["poselift"],
        root,
        notes="pose-only; tracks are upstream model outputs",
    )
    log(f"poselift: {counts}")
    return counts
