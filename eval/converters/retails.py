"""RetailS (Rashvand et al.) -> canonical format. LICENSE STATUS: `pending` — NOT downloaded.

docs/DATA.md: no license stated = not licensed; the authors have been asked. This
converter is written against the *documented* format only and `convert()` refuses to run
until docs/DATA.md (and `eval.converters.base.APPROVAL`) say approved. Tests exercise the
parser on synthetic files; nothing here fetches data.

Documented format (github.com/TeCSAR-UNCC/RetailS README + arXiv 2603.04723):
- JSON per video, "named according to the camera and video ID", with Person ID, Frame
  ID and COCO17 keypoints in XYC (YOLOv8 + ByteTrack + HRNet, interpolated, 8-frame
  smoothed) — i.e. tuples `(frame_id, person_id, kpts)`.
- `.npy` per video: 0/1 frame labels, length = frame count.
- Subsets: train (normal only), staged test (898 events), real test (53 events); 6 cameras,
  15 fps. Resolution is given as "1080x720" in the paper (unverified; recorded as-is).
The staged/real distinction is kept in `extra.subset`: real-test results must be
reported separately (STG-NF: 87.2 staged vs 63.2 real AUC).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from eval.canonical import (
    ClipLabels,
    LabelSupport,
    converted_dir,
    data_root,
    write_info,
    write_labels,
    write_tracks,
)
from eval.converters.base import LICENSES, check_license, register
from eval.converters.poselift import FPS, _kps, _table, clip_name, iter_stgnf_json, label_events

VERSION = "1"
WIDTH, HEIGHT = 1080, 720  # as documented; verify on first real file


def subset_of(path: Path) -> str | None:
    parts = " ".join(path.parts).lower()
    if "staged" in parts:
        return "test_staged"
    if "real" in parts:
        return "test_real"
    if "train" in parts:
        return "train"
    return None


def iter_json(obj: Any) -> Iterator[tuple[int, Any, np.ndarray | None, np.ndarray]]:
    """Accepts `[(frame_id, person_id, kpts), ...]`, `[{"frame_id","person_id","keypoints"}, ...]`
    or the STG-NF `{person: {frame: {"keypoints"}}}` dict. Anything else raises."""
    if isinstance(obj, dict):
        yield from iter_stgnf_json(obj)
        return
    for rec in obj:
        if isinstance(rec, list | tuple) and len(rec) == 3:
            yield int(rec[0]), rec[1], None, _kps(rec[2])
        elif isinstance(rec, dict):
            f = rec.get("frame_id", rec.get("frame"))
            pid = rec.get("person_id", rec.get("id"))
            kp = rec.get("keypoints", rec.get("kpts"))
            if f is None or pid is None or kp is None:
                raise ValueError(f"RetailS record missing frame/person/keypoints: {list(rec)}")
            yield int(f), pid, None, _kps(kp)
        else:
            raise ValueError(f"unrecognized RetailS record {type(rec).__name__}")


def convert_file(pose: Path, labels: Path | None, out: Path) -> ClipLabels:
    clip = clip_name(pose)
    subset = subset_of(pose)
    clip_id = f"{subset or 'unknown'}.{clip}"
    cam = clip.split("_")[0]
    table = _table(f"retails.C{cam}", list(iter_json(json.loads(pose.read_text()))))
    write_tracks(out / "tracks" / f"{clip_id}.parquet", table)
    y = np.load(labels).astype(np.uint8).reshape(-1) if labels else None
    if y is not None:
        (out / "frame_labels").mkdir(parents=True, exist_ok=True)
        np.save(out / "frame_labels" / f"{clip_id}.npy", y)
    n_frames = len(y) if y is not None else (int(table.frame_idx.max()) + 1 if len(table) else 0)
    lab = ClipLabels(
        clip_id=clip_id,
        fps=FPS,
        label_source="human",
        events=label_events(y) if y is not None else [],
        dataset="retails",
        camera_id=f"retails.C{cam}",
        site_id="retails_store",
        group_id=clip_id,
        actor_ids=None,
        width=WIDTH,
        height=HEIGHT,
        n_frames=n_frames,
        duration_s=n_frames / FPS,
        continuous=False,
        supports=LabelSupport(events=y is not None, frame_labels=y is not None, keypoints=True),
        extra={"subset": subset, "staged": subset == "test_staged"},
    )
    write_labels(out / "labels" / f"{clip_id}.json", lab)
    return lab


@register("retails", VERSION)
def convert(root: Path | None = None, log=print) -> dict[str, Any]:
    root = root or data_root()
    check_license("retails", root)  # raises while status is pending
    raw = root / "retails" / "raw"
    npys = {(subset_of(p), clip_name(p)): p for p in raw.rglob("*.npy")}
    out = converted_dir("retails", root)
    counts: dict[str, Any] = defaultdict(int)
    for pose in sorted(raw.rglob("*.json")):
        lab = convert_file(pose, npys.get((subset_of(pose), clip_name(pose))), out)
        counts[f"clips_{lab.extra['subset']}"] += 1
        counts["events"] += len(lab.events)
    counts = dict(counts)
    write_info("retails", "eval.converters.retails", VERSION, counts, LICENSES["retails"], root)
    log(f"retails: {counts}")
    return counts
