"""Ground-truth readers for T03's detection/tracking evaluation: SmartSpaces retail and MEVA indoor.

Why here and not in eval/: T09 owns the canonical converters and the `smartspaces_track`
suite; until they land, T03 needs *some* reader to produce its comparison numbers. These
readers are deliberately thin (raw files → per-frame boxes/ids) so T09 can replace them
without changing any T03 model code. See HANDOFF note in the T03 PR.

Label caveats that shape the metrics (reported in docs/reports/T03-detect-track.md):
- SmartSpaces (synthetic, CC-BY-4.0): every person is labelled → mAP, IDF1, ID switches valid.
- MEVA (real CCTV, CC-BY-4.0): only people taking part in an annotated activity get boxes
  (`partial=True`). Unlabelled bystanders make precision, mAP and IDF1 **lower bounds**;
  ID switches per labelled person-minute and recall are unaffected, so those are the
  MEVA headline numbers.

Splits are by scene/camera, never by frame (docs/EVAL.md). Data roots default to the
shared `data/` of the main checkout (`SCS_DATA` overrides).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def data_root() -> Path:
    env = os.environ.get("SCS_DATA")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for p in here.parents:  # worktree → sibling main checkout's data/
        cand = p / "data"
        if (cand / "smartspaces").exists() or (cand / "meva").exists():
            return cand
    return Path.home() / "Smart-camera-system" / "data"


@dataclass
class GtClip:
    """Per-frame GT for one camera clip. `frames[idx] = (ids (N,), boxes (N,4) xyxy)` for GT frames."""

    dataset: str
    clip_id: str
    camera_id: str
    video: Path
    fps: float
    width: int
    height: int
    frames: dict[int, tuple[np.ndarray, np.ndarray]]
    start: int = 0  # first video frame index to evaluate
    end: int = 0  # exclusive
    partial: bool = False  # True if not every person in view is labelled
    notes: list[str] = field(default_factory=list)

    def eval_frames(self, stride: int) -> list[int]:
        return list(range(self.start, self.end, stride))

    def gt(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.frames.get(idx, (np.zeros(0, int), np.zeros((0, 4))))


# --------------------------------------------------------------------------- SmartSpaces

SMARTSPACES_SPLITS = {
    # Mirrors T09's canonical split (eval/make_splits.py on agent/T09-suites, PR #9): held out
    # by scene, because all cameras of a scene are simultaneous views of the same shoppers.
    # Scenes 071-080 share one synthetic store and character set, so appearance still overlaps.
    "test": ["scene_073"],
    "val": ["scene_072"],
    "train": [
        "scene_071",
        "scene_074",
        "scene_075",
        "scene_076",
        "scene_077",
        "scene_078",
        "scene_079",
        "scene_080",
    ],
}
_SS_BAD = {("scene_071", "camera_0649")}  # corrupt video per dataset README


def _ss_scene_dir(scene: str) -> Path:
    return data_root() / "smartspaces" / "raw" / "MTMC_Tracking_2024" / "test" / scene


def _ss_gt(scene: str) -> np.ndarray:
    """(N,9) cam, id, frame, x, y, w, h, xw, yw — cached as .npy next to data (not in git)."""
    src = _ss_scene_dir(scene) / "ground_truth.txt"
    cache = data_root() / "smartspaces" / "converted" / "t03_cache" / f"{scene}_gt.npy"
    if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        return np.load(cache)
    arr = np.loadtxt(src, dtype=np.float64).reshape(-1, 9)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


def smartspaces_cameras(scene: str) -> list[str]:
    d = _ss_scene_dir(scene)
    return sorted(
        p.name for p in d.glob("camera_*") if (p / "video.mp4").exists() and (scene, p.name) not in _SS_BAD
    )


def load_smartspaces(
    scene: str, camera: str, start_s: float = 0.0, dur_s: float = 120.0, frame_offset: int = 0
) -> GtClip:
    """One camera of one scene. Video is 1920x1080 @ 30 fps. `frame_offset` = video_idx - gt_frame_id."""
    gt = _ss_gt(scene)
    cam_num = int(camera.split("_")[1])
    g = gt[gt[:, 0] == cam_num]
    fps = 30.0
    start = int(start_s * fps)
    end = start + int(dur_s * fps)
    frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    vid_idx = g[:, 2].astype(int) + frame_offset
    sel = (vid_idx >= start) & (vid_idx < end)
    for idx in np.unique(vid_idx[sel]):
        rows = g[vid_idx == idx]
        boxes = np.c_[rows[:, 3], rows[:, 4], rows[:, 3] + rows[:, 5], rows[:, 4] + rows[:, 6]]
        frames[int(idx)] = (rows[:, 1].astype(int), boxes)
    return GtClip(
        "smartspaces",
        f"{scene}/{camera}",
        camera,
        _ss_scene_dir(scene) / camera / "video.mp4",
        fps,
        1920,
        1080,
        frames,
        start,
        end,
        partial=False,
    )


def smartspaces_clips(split: str, dur_s: float = 120.0, max_cams: int | None = None) -> Iterator[GtClip]:
    for scene in SMARTSPACES_SPLITS[split]:
        if not (_ss_scene_dir(scene) / "ground_truth.txt").exists():
            continue
        cams = smartspaces_cameras(scene)
        for cam in cams[: max_cams or len(cams)]:
            yield load_smartspaces(scene, cam, 0.0, dur_s)


# --------------------------------------------------------------------------- MEVA

MEVA_SPLITS = {
    # Mirrors T09 (PR #9): unseen site `bus` + unseen camera school.G421; val school.G423.
    # Held out by site, not just camera: cameras at one site film the same actors at once.
    "test": ["school.G421", "bus.G331", "bus.G508"],
    "val": ["school.G423"],
    "train": ["school.G299", "school.G330", "school.G419", "school.G420", "admin.G326", "admin.G329"],
}
_ANN = Path("annotations/meva-data-repo/annotation/DIVA-phase-2/MEVA")
_R_ID1 = re.compile(r"'?id1'?: (\d+)")
_R_TS0 = re.compile(r"'?ts0'?: (\d+)")
_R_G0 = re.compile(r"'?g0'?: '?(\d+) (\d+) (\d+) (\d+)'?")


def _meva_ann(clip: str) -> Path | None:
    root = data_root() / "meva" / _ANN
    hits = list(root.glob(f"*/*/*/{clip}.geom.yml"))
    return hits[0] if hits else None


def meva_person_boxes(clip: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """{frame: (ids, boxes)} for objects typed `person` in the clip's KPF annotation."""
    geom = _meva_ann(clip)
    if geom is None:
        return {}
    types = geom.with_name(geom.name.replace(".geom.yml", ".types.yml"))
    persons = set()
    for line in types.read_text().splitlines():
        m = _R_ID1.search(line)
        if m and "person" in line:
            persons.add(int(m.group(1)))
    per: dict[int, list[tuple[int, ...]]] = {}
    with open(geom) as f:
        for line in f:
            g, i, t = _R_G0.search(line), _R_ID1.search(line), _R_TS0.search(line)
            if not (g and i and t) or int(i.group(1)) not in persons:
                continue
            per.setdefault(int(t.group(1)), []).append((int(i.group(1)), *map(int, g.groups())))
    out = {}
    for fr, rows in per.items():
        a = np.array(rows, dtype=np.float64)
        out[fr] = (a[:, 0].astype(int), a[:, 1:5])
    return out


def meva_video(clip: str) -> Path | None:
    hits = list((data_root() / "meva" / "raw").glob(f"*/*/{clip}.r13.avi"))
    return hits[0] if hits else None


def load_meva(clip: str, start_s: float = 0.0, dur_s: float = 300.0) -> GtClip:
    frames = meva_person_boxes(clip)
    video = meva_video(clip)
    if video is None:
        raise FileNotFoundError(clip)
    fps = 30.0
    start, end = int(start_s * fps), int((start_s + dur_s) * fps)
    frames = {k: v for k, v in frames.items() if start <= k < end}
    camera = clip.split(".", 3)[-1]
    return GtClip(
        "meva",
        clip,
        camera,
        video,
        fps,
        1920,
        1080,
        frames,
        start,
        end,
        partial=True,
        notes=["only activity participants are labelled"],
    )


def meva_clips(split: str, per_camera: int = 2, min_boxes: int = 3000) -> Iterator[GtClip]:
    """The `per_camera` clips with the most labelled person boxes per camera in `split`.

    Choosing by label count (never by model output) favours clips where most people in
    view take part in an activity, i.e. where the partial labels are closest to complete.
    """
    root = data_root() / "meva" / "raw"
    for cam in MEVA_SPLITS[split]:
        clips = sorted(p.name.replace(".r13.avi", "") for p in root.glob(f"*/*/*.{cam}.r13.avi"))
        scored = []
        for c in clips:
            n = sum(len(v[0]) for v in meva_person_boxes(c).values())
            if n >= min_boxes:
                scored.append((n, c))
        for _, c in sorted(scored, reverse=True)[:per_camera]:
            yield load_meva(c)
