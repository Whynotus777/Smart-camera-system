"""MEVA (Kitware/IARPA, CC-BY-4.0) indoor clips -> canonical tracks + interaction labels.

Why this mapping: MEVA is our only free *real* footage with object-interaction
events, but it isn't retail, so labels must not claim more than MEVA says:

| MEVA activity             | canonical type   | note                                         |
|---------------------------|------------------|----------------------------------------------|
| person_picks_up_object    | item_pickup      |                                              |
| person_puts_down_object   | item_put_down    | any surface; NOT "returned to shelf"         |
| person_transfers_object   | item_transfer    | person to person                             |
| person_steals_object      | item_pickup      | `source_label` keeps "steals"; only 5 exist  |

Every other MEVA activity (phone use, carrying, entering, ...) is kept verbatim in
`extra.other_activities` as hard-negative context, never as a canonical label.

Inputs (all fetched by T14; this module never downloads):
- `raw/<date>/<hour>/<clip>.r13.avi` videos listed in `MANIFEST.json`
- KPF annotations `annotations/meva-data-repo/annotation/DIVA-phase-2/MEVA/{kitware,kitware-meva-training}/`
  (`*.activities.yml`, `*.geom.yml` boxes, `*.types.yml` actor classes)
- clip table `.../metadata/meva-clip-camera-and-time-table.txt` for camera sets
  (overlapping fields of view) -> `view_group`

Geometry: frames are mostly 1920x1072 but not all (bus.G331 is 1920x1080), so every
clip is probed with ffprobe; boxes are "x1 y1 x2 y2" pixels, clipped to the frame; 30 fps.
Actor ids in KPF are per clip, so `actor_ids` is None (actor-disjoint splits are
impossible on MEVA; reports say so). Activities in annotated clips are exhaustive for
MEVA's 37 types, so annotated clips support false-alert counting.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

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
FPS = 30.0
WIDTH, HEIGHT = 1920, 1072
INDOOR_CAMERAS = (
    "admin.G326",
    "admin.G329",
    "bus.G331",
    "bus.G508",
    "school.G299",
    "school.G330",
    "school.G419",
    "school.G420",
    "school.G421",
    "school.G423",
)
ACTIVITY_MAP = {
    "person_picks_up_object": "item_pickup",
    "person_puts_down_object": "item_put_down",
    "person_transfers_object": "item_transfer",
    "person_steals_object": "item_pickup",
}
ANN_SETS = ("kitware", "kitware-meva-training")  # first wins if a clip were in both

_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
_CLIP_RE = re.compile(r"^(\d{4}-\d\d-\d\d)\.(\d\d-\d\d-\d\d)\.(\d\d-\d\d-\d\d)\.([a-z]+)\.(G\d+)$")
_G0 = re.compile(r"'g0':\s*'(-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?)'")
_ID1 = re.compile(r"'id1':\s*(\d+)")
_TS0 = re.compile(r"'ts0':\s*(\d+)")


def parse_clip_name(clip: str) -> dict[str, Any]:
    m = _CLIP_RE.match(clip)
    if not m:
        raise ValueError(f"not a MEVA clip name: {clip}")
    date, start, end, site, cam = m.groups()

    def sec(t: str) -> int:
        h, mi, s = (int(x) for x in t.split("-"))
        return h * 3600 + mi * 60 + s

    return {
        "date": date,
        "start": start,
        "end": end,
        "site": site,
        "camera_id": f"{site}.{cam}",
        "duration_s": float(sec(end) - sec(start)),
    }


def parse_types(path: Path) -> dict[int, str]:
    out = {}
    for row in yaml.load(path.read_text(), Loader=_LOADER) or []:  # noqa: S506 - (C)SafeLoader
        t = row.get("types")
        if t:
            cls = max(t["cset3"].items(), key=lambda kv: kv[1])[0]
            out[int(t["id1"])] = cls
    return out


def parse_activities(path: Path) -> list[dict[str, Any]]:
    out = []
    for row in yaml.load(path.read_text(), Loader=_LOADER) or []:  # noqa: S506 - (C)SafeLoader
        a = row.get("act")
        if not a:
            continue
        name = max(a["act2"].items(), key=lambda kv: kv[1])[0]
        spans = [ts["tsr0"] for ts in a.get("timespan", []) if "tsr0" in ts]
        if not spans:
            continue
        f0 = min(s[0] for s in spans)
        f1 = max(s[1] for s in spans)
        actors = [int(x["id1"]) for x in a.get("actors", [])]
        out.append(
            {
                "name": name,
                "f0": int(f0),
                "f1": int(f1),
                "actors": actors,
                "id2": a.get("id2"),
                "src_status": a.get("src_status", ""),
            }
        )
    return out


def parse_geom(path: Path) -> np.ndarray:
    """(N, 6) int array: id1, frame, x1, y1, x2, y2. Regex per key (90x faster than YAML);
    raises if any geom row fails to parse, so a format change can't drop boxes silently."""
    rows, n_geom = [], 0
    with open(path) as f:
        for line in f:
            if "'geom'" not in line:
                continue
            n_geom += 1
            g, i, t = _G0.search(line), _ID1.search(line), _TS0.search(line)
            if not (g and i and t):
                raise ValueError(f"{path}: unparsed geom row: {line[:120]}")
            rows.append((int(i.group(1)), int(t.group(1)), *(float(x) for x in g.groups())))
    if len(rows) != n_geom:
        raise ValueError(f"{path}: parsed {len(rows)} of {n_geom} geom rows")
    return np.array(rows, dtype=float).reshape(-1, 6)


def load_clip_table(path: Path) -> dict[str, tuple[str, str]]:
    """clip -> (reference time slot, camera set)."""
    out: dict[str, tuple[str, str]] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) >= 4:
            out[f[0]] = (f[1], f[3])
    return out


def view_group(clip: str, table: dict[str, tuple[str, str]]) -> str | None:
    """Simultaneous overlapping views = same camera set (shared FOV) in the same 5-min slot.

    Finer than T14's slot+site `view_group`: cameras of one site that don't share a field
    of view don't see the same event, so they needn't be linked. No camera set -> None.
    """
    if clip not in table:
        return None
    slot, cset = table[clip]
    if not re.match(r"^\d+-\d+$", cset):
        return None
    return f"meva:{cset}:{slot}"


def annotation_index(ann_root: Path) -> dict[str, Path]:
    """clip -> annotation path prefix (without '.activities.yml')."""
    idx: dict[str, Path] = {}
    for s in ANN_SETS:
        for p in sorted((ann_root / s).rglob("*.activities.yml")):
            idx.setdefault(
                p.name.removesuffix(".activities.yml"), p.with_name(p.name.removesuffix(".activities.yml"))
            )
    return idx


def t14_groups(root: Path) -> dict[str, str]:
    """T14 dedup handoff: `meva:<clip>` -> group_id (derivatives + near-duplicates)."""
    p = root / "dedup" / "groups.jsonl"
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r.get("item", "").startswith("meva:"):
                out[r["item"][5:]] = str(r["group_id"])
    return out


def probe(video: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {}
    r = subprocess.run(  # noqa: S603
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames:format=duration",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode:
        return {}
    d = json.loads(r.stdout)
    st = (d.get("streams") or [{}])[0]
    num, _, den = st.get("r_frame_rate", "0/1").partition("/")
    return {
        "width": st.get("width"),
        "height": st.get("height"),
        "fps": float(num) / float(den or 1) if float(den or 1) else None,
        "n_frames": int(st["nb_frames"]) if str(st.get("nb_frames", "")).isdigit() else None,
        "duration_s": float(d.get("format", {}).get("duration", 0) or 0) or None,
    }


def convert_clip(
    clip: str,
    video_rel: str | None,
    ann_prefix: Path | None,
    table: dict[str, tuple[str, str]],
    groups: dict[str, str],
    out: Path,
    root: Path,
    with_tracks: bool = True,
    do_probe: bool = True,
) -> ClipLabels:
    meta = parse_clip_name(clip)
    pr = probe(root / video_rel) if (video_rel and do_probe) else {}
    fps = pr.get("fps") or FPS
    events: list[LabelEvent] = []
    other: list[dict[str, Any]] = []
    types: dict[int, str] = {}
    dropped = 0
    if ann_prefix is not None:
        types = parse_types(Path(f"{ann_prefix}.types.yml"))
        for a in parse_activities(Path(f"{ann_prefix}.activities.yml")):
            persons = [x for x in a["actors"] if types.get(x) == "person"]
            tid = persons[0] if persons else None
            rec = {"t_start": a["f0"] / fps, "t_end": a["f1"] / fps, "track_id": tid}
            if a["name"] in ACTIVITY_MAP:
                events.append(
                    LabelEvent(
                        type=ACTIVITY_MAP[a["name"]],
                        source_label=a["name"],
                        visible="observed",
                        event_id=f"{clip}:{a['id2']}",
                        extra={"src_status": a["src_status"], "actors": a["actors"]},
                        **rec,
                    )
                )
            else:
                other.append({"name": a["name"], **rec, "src_status": a["src_status"]})
        if with_tracks:
            g = parse_geom(Path(f"{ann_prefix}.geom.yml"))
            keep = (
                np.array([types.get(int(i)) == "person" for i in g[:, 0]], dtype=bool)
                if len(g)
                else np.zeros(0, dtype=bool)
            )
            g = g[keep]
            # KPF boxes overrun the frame by a pixel at the borders: clip to the probed frame,
            # drop boxes left with no area (counted in extra.dropped_boxes)
            w, h = pr.get("width") or WIDTH, pr.get("height") or HEIGHT
            g[:, [2, 4]] = np.clip(g[:, [2, 4]], 0, w)
            g[:, [3, 5]] = np.clip(g[:, [3, 5]], 0, h)
            ok = (g[:, 4] > g[:, 2]) & (g[:, 5] > g[:, 3])
            dropped = int((~ok).sum())
            g = g[ok]
            n = len(g)
            write_tracks(
                out / "tracks" / f"{clip}.parquet",
                TrackTable(
                    meta["camera_id"],
                    g[:, 1].astype(np.int64),
                    g[:, 1] / fps,
                    g[:, 0].astype(np.int64),
                    g[:, 2:6],
                    np.ones(n),
                    np.full((n, 17, 3), np.nan),
                ),
            )
    annotated = ann_prefix is not None
    labels = ClipLabels(
        clip_id=clip,
        camera_profile=None,
        fps=fps,
        label_source="human",
        events=events,
        dataset="meva",
        camera_id=meta["camera_id"],
        site_id=meta["site"],
        group_id=groups.get(clip, clip),
        view_group=view_group(clip, table),
        actor_ids=None,
        width=pr.get("width") or WIDTH,
        height=pr.get("height") or HEIGHT,
        n_frames=pr.get("n_frames"),
        duration_s=pr.get("duration_s") or meta["duration_s"],
        continuous=True,
        synthetic=False,
        video=video_rel,
        supports=LabelSupport(events=annotated, exhaustive=annotated, gt_tracks=annotated and with_tracks),
        extra={
            "annotated": annotated,
            "date": meta["date"],
            "start_local": meta["start"],
            "other_activities": other,
            "dropped_boxes": dropped,
            "proxy": "proxy, not retail",
            "n_frames_source": "ffprobe" if pr.get("n_frames") else None,
        },
    )
    write_labels(out / "labels" / f"{clip}.json", labels)
    return labels


@register("meva", VERSION)
def convert(
    root: Path | None = None,
    clips: list[str] | None = None,
    limit: int | None = None,
    with_tracks: bool = True,
    do_probe: bool = True,
    log=print,
) -> dict[str, Any]:
    root = root or data_root()
    check_license("meva", root)
    d = root / "meva"
    ann_root = d / "annotations" / "meva-data-repo" / "annotation" / "DIVA-phase-2" / "MEVA"
    table = load_clip_table(
        d / "annotations" / "meva-data-repo" / "metadata" / "meva-clip-camera-and-time-table.txt"
    )
    ann = annotation_index(ann_root) if ann_root.exists() else {}
    groups = t14_groups(root)
    videos: dict[str, str] = {}
    man = d / "MANIFEST.json"
    for rel in json.loads(man.read_text())["files"] if man.exists() else {}:
        if rel.endswith(".avi"):
            clip = Path(rel).name.removesuffix(".avi").removesuffix(".r13")
            if parse_clip_name(clip)["camera_id"] in INDOOR_CAMERAS and (d / "raw" / rel).exists():
                videos[clip] = f"meva/raw/{rel}"
    todo = sorted(set(clips) if clips else set(videos))
    if limit:
        todo = todo[:limit]
    out = converted_dir("meva", root)
    counts: dict[str, Any] = defaultdict(int)
    for i, clip in enumerate(todo):
        lab = convert_clip(
            clip, videos.get(clip), ann.get(clip), table, groups, out, root, with_tracks, do_probe
        )
        counts["clips"] += 1
        counts["annotated"] += bool(lab.extra["annotated"])
        counts["hours"] += (lab.duration_s or 0) / 3600
        for e in lab.events:
            counts[e.source_label or e.type] += 1
        if (i + 1) % 50 == 0:
            log(f"meva: {i + 1}/{len(todo)} clips")
    counts = dict(counts)
    counts["hours"] = round(counts.get("hours", 0.0), 3)
    write_info(
        "meva",
        "eval.converters.meva",
        VERSION,
        counts,
        LICENSES["meva"],
        root,
        notes="proxy, not retail; indoor cameras only",
    )
    log(f"meva: converted {counts}")
    return counts
