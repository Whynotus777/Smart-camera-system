"""Export review outcomes as labels in the canonical format (docs/DATA.md, eval.canonical).

Reviewer decisions are the product's training data (ARCHITECTURE D5), so they leave the
system in the same format every dataset converter produces:

    <out>/<dataset_id>/converted/labels/<clip_id>.json      (one file per reviewed clip)

- `clip_id` is the alert id; the clip is `<workdir>/clips/<clip_id>.mp4` (`video`).
- A **confirmed** alert yields one event per alert event: the rule's event type,
  `t_start`/`t_end` in seconds from the clip's first frame (dwell start → rule fire).
- A **dismissed** alert yields `events: []`: a reviewed clip with nothing in it (a hard
  negative). `supports.events` is true either way, because a human looked.
- `visible` is a scalar for *this* camera, per T09's `eval.canonical.LabelEvent`
  (DATA.md's "per camera" wording read as one label file per camera clip).
- Fields beyond DATA.md's minimum (`camera_id`, `site_id`, `dataset`, `t0`, `duration_s`,
  `video`, `supports`, `event_id`, `extra.review`) are T09's optional additions, so a
  reader of either schema accepts the file. The decision, reason and reviewer live in
  `extra.review`, which metrics never read.

Export is a pure function of the database, rewritten atomically, so running it again
(e.g. after a crash between recording a review and exporting it) never duplicates.
"""

from __future__ import annotations

import json
import os
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from scs.app.crash import crashpoint
from scs.app.source import FFPROBE
from scs.app.store import Store
from scs.contracts import CONTRACTS_VERSION


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    crashpoint("web.label_before_rename")
    os.replace(tmp, path)


def clip_fps(path: Path) -> float:
    out = subprocess.run(  # noqa: S603
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return round(float(Fraction(out.stdout.strip())), 3)


def label_dir(out: Path, dataset_id: str) -> Path:
    return out / dataset_id / "converted" / "labels"


def export_labels(
    store: Store,
    out: Path,
    dataset_id: str,
    camera_profile: str,
    only: str | None = None,
    missing_only: bool = False,
) -> list[Path]:
    """Write one label file per reviewed alert (or just alert `only`). Returns the paths written.

    Each file costs an fsync, so the review API exports only the alert it just recorded, and
    web startup (crash repair) only writes labels that are missing: a review never changes
    once stored, so an existing label file is final. `scs export-labels` rewrites them all.
    """
    labels = label_dir(out, dataset_id)
    written = []
    for alert, job, rev in store.alerts():
        if only is not None and alert.alert_id != only:
            continue
        if missing_only and (labels / f"{alert.alert_id}.json").exists():
            continue
        if rev is None or job is None or job.status != "done" or job.clip_start is None:
            continue  # only reviewed alerts with a clip become labels
        events = []
        if rev.decision == "confirmed":
            for eid in alert.event_ids:
                ev = store.event(eid)
                t_start = float(ev.data.get("dwell_start_ts", ev.ts)) - job.clip_start
                events.append(
                    {
                        "type": ev.type.value,
                        "t_start": round(max(t_start, 0.0), 3),
                        "t_end": round(ev.ts - job.clip_start, 3),
                        "track_id": ev.track_id,
                        "actor_id": None,
                        "visible": "observed",
                        "label_source": "human",
                        "event_id": ev.event_id,
                    }
                )
        assert job.path is not None and job.clip_end is not None
        label = {
            "clip_id": alert.alert_id,
            "camera_profile": camera_profile,
            "fps": clip_fps(Path(job.path)),
            "label_source": "human",
            "events": events,
            "dataset": dataset_id,
            "camera_id": alert.camera_ids[0],
            "site_id": alert.site_id,
            "t0": job.clip_start,
            "duration_s": round(job.clip_end - job.clip_start, 3),
            "video": job.path,
            "supports": {"events": True},
            "extra": {
                "review": {
                    "alert_id": alert.alert_id,
                    "decision": rev.decision,
                    "reason": rev.reason,
                    "reviewer": rev.reviewer,
                    "request_id": rev.request_id,
                    "reviewed_ts": rev.ts,
                    "reason_codes": alert.reason_codes,
                    "contracts_version": CONTRACTS_VERSION,
                }
            },
        }
        _atomic_json(labels / f"{alert.alert_id}.json", label)
        written.append(labels / f"{alert.alert_id}.json")
    return written
