"""Own footage (`quick_capture`, `lab_mock_aisle`) from the take log, and a `sim` stub.

Why a take log: docs/DATA.md makes the tablet log written during staging *the* label
file (`label_source: script`). Scripted truth (what happened) and camera evidence
(what each camera could see) are separate: `visible` is per camera, and only
`observed` counts as a positive for that camera.

Take log CSV (one row per take x camera; header required):
    take_id,actor_id,subtype,type,t_start,t_end,camera_id,visible,video,group_id
- `type`: canonical label (item_to_clothing, item_to_bag, item_returned, ...); `subtype`
  pocket/waistband/jacket/bag or empty. Benign matched twins use their benign type.
- `t_start/t_end`: seconds from the start of `video` (relative path under data root).
- `visible`: observed | partially_observed | not_observed (reviewed per camera).
- `group_id`: optional; defaults to the video (all takes in one recording share a group).
Rows with the same `video` form one clip. Tracks come later from the pipeline or the
pre-labeler (never test truth).

STATUS: quick_capture/lab_mock_aisle are not recorded yet, so this runs only on test
fixtures. `sim`: stub until T08 publishes its output format (see HANDOFF in the PR).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.canonical import (
    ClipLabels,
    LabelEvent,
    LabelSupport,
    converted_dir,
    data_root,
    write_info,
    write_labels,
)
from eval.converters.base import LICENSES, check_license, register

VERSION = "0"
REQUIRED = ("take_id", "actor_id", "type", "t_start", "t_end", "camera_id", "visible", "video")


def convert_take_log(
    dataset_id: str,
    take_log: Path,
    fps: float,
    root: Path | None = None,
    camera_profile: str | None = None,
    continuous_videos: set[str] | None = None,
    log=print,
) -> dict[str, Any]:
    root = root or data_root()
    check_license(dataset_id, root)
    with open(take_log, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{take_log}: empty take log")
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        raise ValueError(f"{take_log}: missing columns {missing}")
    by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_video[r["video"]].append(r)
    out = converted_dir(dataset_id, root)
    counts: dict[str, int] = defaultdict(int)
    for video, rs in sorted(by_video.items()):
        cams = {r["camera_id"] for r in rs}
        if len(cams) != 1:
            raise ValueError(f"{video}: one video must be one camera, got {cams}")
        clip_id = Path(video).stem
        events = [
            LabelEvent(
                type=r["type"],
                t_start=float(r["t_start"]),
                t_end=float(r["t_end"]),
                actor_id=r["actor_id"] or None,
                visible=r["visible"],  # type: ignore[arg-type]
                subtype=r.get("subtype") or None,
                event_id=f"{r['take_id']}@{r['camera_id']}",
                label_source="script",
            )
            for r in rs
        ]
        actors = sorted({r["actor_id"] for r in rs if r["actor_id"]})
        write_labels(
            out / "labels" / f"{clip_id}.json",
            ClipLabels(
                clip_id=clip_id,
                camera_profile=camera_profile,
                fps=fps,
                label_source="script",
                events=events,
                dataset=dataset_id,
                camera_id=cams.pop(),
                site_id=dataset_id,
                group_id=rs[0].get("group_id") or clip_id,
                actor_ids=actors or None,
                continuous=video in (continuous_videos or set()),
                video=video,
                supports=LabelSupport(
                    events=True, subtypes=any(e.subtype for e in events), actors=bool(actors), exhaustive=True
                ),
            ),
        )
        counts["clips"] += 1
        counts["events"] += len(events)
    write_info(dataset_id, "eval.converters.own", VERSION, dict(counts), LICENSES[dataset_id], root)
    log(f"{dataset_id}: {dict(counts)}")
    return dict(counts)


@register("quick_capture", VERSION)
def convert_quick_capture(
    take_log: Path, fps: float = 15.0, root: Path | None = None, **kw: Any
) -> dict[str, Any]:
    return convert_take_log("quick_capture", take_log, fps, root, **kw)


@register("lab_mock_aisle", VERSION)
def convert_lab_mock_aisle(
    take_log: Path, fps: float = 15.0, root: Path | None = None, **kw: Any
) -> dict[str, Any]:
    return convert_take_log("lab_mock_aisle", take_log, fps, root, **kw)


@register("sim_store", VERSION, stub=True)
def convert_sim(*_: Any, **__: Any) -> dict[str, Any]:
    """Stub. Expected from T08 per clip: video (or frame dir), per-camera GT tracks + COCO17
    keypoints in main-stream pixels, scripted events with per-camera visibility (occlusion
    ray-cast), actor ids, camera profile id, mount height, and generator seed/lineage.
    Output will be canonical with `synthetic=True`, `label_source="script"`."""
    raise NotImplementedError("sim converter waits for T08's output format (HANDOFF: T08)")
