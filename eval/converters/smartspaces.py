"""NVIDIA PhysicalAI-SmartSpaces (CC-BY-4.0) MTMC_Tracking_2024 -> canonical GT tracks.

Why: the only free overhead *retail-like* person tracking GT (synthetic, Isaac Sim), so
`smartspaces_track` can report detection AP, IDF1 and ID switches on overhead views.
It is synthetic: reports label it "synthetic, not production validation".

Source format (dataset README): `ground_truth.txt` per scene, one row per box:
`<camera_id> <obj_id> <frame_id> <xmin> <ymin> <width> <height> <xworld> <yworld>`,
frame_id 0-based (checked: GT ends at frame 23993 of a 23994-frame video), 1080p30.
Camera `635` -> folder `camera_0635`. obj_id is global within a scene.

Scene kinds: T14 fetched scenes 071-080, which the README describes as the retail
space (plus a storage room some cameras see). Scenes without `ground_truth.txt` are
converted with `gt_tracks=False` so the suite can skip them. Known-bad videos listed
in T14's manifest notes (scene_071/camera_0649, corrupt) are skipped.
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

from eval.canonical import (
    ClipLabels,
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
RETAIL_SCENES = {f"scene_{i:03d}" for i in range(71, 81)}  # MTMC 2024 test: one retail space
KNOWN_BAD = {"MTMC_Tracking_2024/test/scene_071/camera_0649/video.mp4"}


def scene_kind(scene: str) -> str:
    return "retail" if scene in RETAIL_SCENES else "non_retail"


def _probe(video: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not video.exists():
        return {}
    r = subprocess.run(  # noqa: S603
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames",
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
    st = (json.loads(r.stdout).get("streams") or [{}])[0]
    return {
        "width": st.get("width"),
        "height": st.get("height"),
        "n_frames": int(st["nb_frames"]) if str(st.get("nb_frames", "")).isdigit() else None,
    }


def read_gt(path: Path) -> np.ndarray:
    """(N, 7) float: camera, obj, frame, x1, y1, x2, y2."""
    a = np.loadtxt(path, dtype=float, ndmin=2)
    if a.shape[1] < 7:
        raise ValueError(f"{path}: expected >= 7 columns, got {a.shape[1]}")
    out = a[:, :7].copy()
    out[:, 5] = a[:, 3] + a[:, 5]  # x2 = x + w
    out[:, 6] = a[:, 4] + a[:, 6]  # y2 = y + h
    return out


@register("smartspaces", VERSION)
def convert(root: Path | None = None, scenes: list[str] | None = None, log=print) -> dict[str, Any]:
    root = root or data_root()
    check_license("smartspaces", root)
    man = json.loads((root / "smartspaces" / "MANIFEST.json").read_text())
    bad = set(KNOWN_BAD)
    bad |= set(re.findall(r"'([^']+video\.mp4)': 'corrupt", man.get("notes", "")))
    cams: dict[str, list[str]] = defaultdict(list)
    for rel in man["files"]:
        if rel.endswith("/video.mp4") and rel not in bad:
            parts = rel.split("/")
            cams[parts[2]].append(parts[3])
    out = converted_dir("smartspaces", root)
    counts: dict[str, Any] = defaultdict(int)
    for scene in sorted(scenes or cams):
        sdir = root / "smartspaces" / "raw" / "MTMC_Tracking_2024" / "test" / scene
        gt_path = sdir / "ground_truth.txt"
        gt = read_gt(gt_path) if gt_path.exists() else None
        for cam in sorted(cams.get(scene, [])):
            cid = int(cam.split("_")[1])
            clip = f"{scene}.{cam}"
            video_rel = f"smartspaces/raw/MTMC_Tracking_2024/test/{scene}/{cam}/video.mp4"
            pr = _probe(root / video_rel)
            w, h = pr.get("width") or 1920, pr.get("height") or 1080
            if gt is not None:
                g = gt[gt[:, 0] == cid]
                g[:, [3, 5]] = np.clip(g[:, [3, 5]], 0, w)
                g[:, [4, 6]] = np.clip(g[:, [4, 6]], 0, h)
                g = g[(g[:, 5] > g[:, 3]) & (g[:, 6] > g[:, 4])]
                n = len(g)
                write_tracks(
                    out / "tracks" / f"{clip}.parquet",
                    TrackTable(
                        f"smartspaces.{clip}",
                        g[:, 2].astype(np.int64),
                        g[:, 2] / FPS,
                        g[:, 1].astype(np.int64),
                        g[:, 3:7],
                        np.ones(n),
                        np.full((n, 17, 3), np.nan),
                    ),
                )
                counts["gt_boxes"] += n
            n_frames = pr.get("n_frames")
            write_labels(
                out / "labels" / f"{clip}.json",
                ClipLabels(
                    clip_id=clip,
                    fps=FPS,
                    label_source="script",
                    events=[],
                    dataset="smartspaces",
                    camera_id=f"smartspaces.{clip}",
                    site_id=f"smartspaces_{scene_kind(scene)}",
                    group_id=clip,
                    view_group=f"smartspaces:{scene}",
                    actor_ids=None,
                    width=w,
                    height=h,
                    n_frames=n_frames,
                    duration_s=n_frames / FPS if n_frames else None,
                    continuous=True,
                    synthetic=True,
                    video=video_rel,
                    supports=LabelSupport(gt_tracks=gt is not None),
                    extra={
                        "scene": scene,
                        "scene_kind": scene_kind(scene),
                        "has_gt": gt is not None,
                        "note": "synthetic (Isaac Sim); scenes 071-080 share one retail space + characters",
                    },
                ),
            )
            counts["clips"] += 1
            counts["clips_with_gt"] += gt is not None
        counts["scenes"] += 1
        log(f"smartspaces: {scene} gt={'yes' if gt is not None else 'no'} cams={len(cams.get(scene, []))}")
    counts = dict(counts)
    write_info(
        "smartspaces",
        "eval.converters.smartspaces",
        VERSION,
        counts,
        LICENSES["smartspaces"],
        root,
        notes="synthetic retail scenes; tracks are GT",
    )
    return counts
