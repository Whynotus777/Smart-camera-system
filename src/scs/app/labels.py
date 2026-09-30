"""Export review outcomes as labels in the canonical format (docs/DATA.md).

Reviewer decisions are the product's training data (ARCHITECTURE D5), so they leave the
system in the same format every dataset converter produces:

    <out>/<dataset_id>/converted/labels/<clip_id>.json
        {clip_id, camera_profile, fps, label_source: "human", events: [...]}

- `clip_id` is the alert id; the clip is `<workdir>/clips/<clip_id>.mp4`.
- A **confirmed** alert yields one event: the rule's event type, `t_start`/`t_end` in
  seconds from the clip's first frame (dwell start → rule fire), `visible` per camera.
- A **dismissed** alert yields `events: []`, i.e. a reviewed clip with nothing in it
  (a hard negative). DATA.md has no field for "reviewed and rejected"; see the review
  sidecar below and the open question in docs/RELEASE.md.
- Review metadata that the canonical format has no field for (decision, reason, reviewer,
  event ids, absolute clip times) goes in `<out>/<dataset_id>/converted/reviews/<clip_id>.json`.

Export is a pure function of the database, rewritten atomically, so running it again
(e.g. after a crash between recording a review and exporting it) never duplicates.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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
    os.replace(tmp, path)


def label_dir(out: Path, dataset_id: str) -> Path:
    return out / dataset_id / "converted" / "labels"


def export_labels(
    store: Store,
    out: Path,
    dataset_id: str,
    camera_profile: str,
    fps: float | None,
    only: str | None = None,
) -> list[Path]:
    """Write one label file per reviewed alert (or just alert `only`). Returns the paths written.

    Each file costs an fsync, so the review API exports only the alert it just recorded;
    the full export runs at web startup (crash repair) and from `scs export-labels`.
    """
    labels, reviews = label_dir(out, dataset_id), out / dataset_id / "converted" / "reviews"
    written = []
    for alert, job, rev in store.alerts():
        if only is not None and alert.alert_id != only:
            continue
        if rev is None or job is None or job.status != "done" or job.clip_start is None:
            continue  # only reviewed alerts with a clip become labels
        events = []
        if rev.decision == "confirmed":
            for eid in alert.event_ids:
                ev = store.event(eid)
                t_fire = ev.ts - job.clip_start
                t_start = float(ev.data.get("dwell_start_ts", ev.ts)) - job.clip_start
                events.append(
                    {
                        "type": ev.type.value,
                        "track_id": ev.track_id,
                        "t_start": round(max(t_start, 0.0), 3),
                        "t_end": round(t_fire, 3),
                        "actor_id": None,
                        "visible": {ev.camera_id: "observed"},
                        "label_source": "human",
                    }
                )
        label = {
            "clip_id": alert.alert_id,
            "camera_profile": camera_profile,
            "fps": fps,
            "label_source": "human",
            "events": events,
        }
        _atomic_json(labels / f"{alert.alert_id}.json", label)
        _atomic_json(
            reviews / f"{alert.alert_id}.json",
            {
                "clip_id": alert.alert_id,
                "alert": json.loads(alert.model_dump_json()),
                "decision": rev.decision,
                "reason": rev.reason,
                "reviewer": rev.reviewer,
                "request_id": rev.request_id,
                "reviewed_ts": rev.ts,
                "clip_path": job.path,
                "clip_start_ts": job.clip_start,
                "clip_end_ts": job.clip_end,
                "contracts_version": CONTRACTS_VERSION,
            },
        )
        written.append(labels / f"{alert.alert_id}.json")
    return written
