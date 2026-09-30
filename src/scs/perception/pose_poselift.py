"""`to_poselift()`: make our poses comparable with PoseLift-trained models (T04 -> T06).

Same keypoint order isn't compatibility. PoseLift (Ehsan et al. 2025, arXiv 2501.06591,
github.com/TeCSAR-UNCC/PoseLift, Apache-2.0) was built like this, and each choice below
matches it or states the gap:

- Keypoints: COCO17 order in both. PoseLift: HRNet top-down on YOLOv8 + ByteTrack boxes;
  ours: RTMPose top-down on T03 track boxes.
- Coordinates: PoseLift = XYC pixels of a 1920x1080 frame; ours = main-stream pixels
  (2560x1440 on the Reolink 4MP). Rescale x by 1920/W and y by 1080/H. Both are 16:9, so
  nothing is distorted; a non-16:9 source raises unless `allow_aspect_change=True`.
- Frame rate: PoseLift 15 fps; ours <= 10 fps on gated tracks, irregular under load.
  Resample onto a 15 fps grid by linear interpolation in time between the two nearest
  poses. Only grid times <= the newest input are emitted, so no future frame is read.
- Missing poses: PoseLift "linear interpolation to fill in any missing poses" (limit
  unstated). Same linear rule here, but only across gaps <= 0.5 s; longer gaps split the
  track instead of inventing motion.
- Smoothing: PoseLift "8-frame window" (method unstated; we assume a moving average).
  Input here is RAW poses (not One-Euro output) and the 8-frame mean runs on the 15 fps
  grid. `mode="offline"` = centered window, PoseLift-like, NON-CAUSAL (labeling / public
  benchmark comparison only). `mode="live"` = trailing window, causal, ~3.5 frames
  (0.23 s) extra lag.
- Confidence: PoseLift = HRNet heatmap max, unthresholded; ours = RTMPose SimCC max
  clipped to [0, 1]. Passed through unchanged. The distributions differ, so thresholds
  tuned on PoseLift don't transfer. STG-NF only uses conf to drop segments
  (`seg_conf_th`), so re-tune that threshold on our val split.
- Boxes: PoseLift XYWH pixels; ours XYXY. Converted and rescaled like keypoints.
- Model-side normalization: STG-NF `normalize_pose` centers xy on the segment mean and
  divides by the std of y (its vid_res division cancels out). `stgnf_normalize` reproduces
  it for T06.

Output mirrors one PoseLift annotation entry per (frame, person): `{"bbox": [x, y, w, h],
"keypoints": (17, 3) XYC}` keyed by 15 fps frame index, then person id.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from scs.contracts import Pose, Track
from scs.perception.pose_smooth import pose_time

POSELIFT_WH = (1920, 1080)
POSELIFT_FPS = 15.0
POSELIFT_WINDOW = 8
MAX_GAP_S = 0.5


def _resample(
    t: np.ndarray, v: np.ndarray, grid: np.ndarray, max_gap_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Linear-in-time resample of v (T, ...) at grid times; returns (valid grid mask, values)."""
    idx = np.searchsorted(t, grid, side="right")  # t[idx-1] <= g < t[idx]
    lo = np.clip(idx - 1, 0, len(t) - 1)
    hi = np.clip(idx, 0, len(t) - 1)
    exact = t[lo] == grid
    hi = np.where(exact, lo, hi)
    ok = (grid >= t[0]) & (grid <= t[-1]) & (exact | (t[hi] - t[lo] <= max_gap_s))
    span = np.where(t[hi] > t[lo], t[hi] - t[lo], 1.0)
    u = ((grid - t[lo]) / span).reshape((-1,) + (1,) * (v.ndim - 1))
    return ok, v[lo] + u * (v[hi] - v[lo])


def to_poselift(
    poses: Sequence[Pose],
    tracks: Sequence[Track] | None = None,
    mode: Literal["live", "offline"] = "live",
    person_id: int | None = None,
    t0: float | None = None,
    allow_aspect_change: bool = False,
) -> dict[int, dict[int, dict[str, list]]]:
    """Convert one track's RAW poses (time-ordered) into PoseLift-style frames.

    `tracks` (same track, any frames) supply boxes; without them the box is the keypoint
    extent. `t0` anchors the 15 fps grid (defaults to the first pose) so several tracks
    from one camera land on the same frame indices.
    """
    if not poses:
        return {}
    f = poses[0].frame
    if not allow_aspect_change and abs(f.width / f.height - POSELIFT_WH[0] / POSELIFT_WH[1]) > 1e-3:
        raise ValueError(f"source aspect {f.width}x{f.height} != 16:9; rescaling would distort poses")
    sx, sy = POSELIFT_WH[0] / f.width, POSELIFT_WH[1] / f.height
    t = np.array([pose_time(p) for p in poses])
    if np.any(np.diff(t) <= 0):
        raise ValueError("poses must be strictly increasing in time")
    kp = np.array([p.keypoints for p in poses], dtype=np.float64)
    if tracks:
        by_seq = {(tr.frame.epoch, tr.frame.seq): tr.bbox for tr in tracks}
        boxes = np.array(
            [by_seq.get((p.frame.epoch, p.frame.seq), _kp_box(k)) for p, k in zip(poses, kp, strict=True)]
        )
    else:
        boxes = np.array([_kp_box(k) for k in kp])
    t0 = t[0] if t0 is None else t0
    k0 = int(np.ceil((t[0] - t0) * POSELIFT_FPS - 1e-9))
    k1 = int(np.floor((t[-1] - t0) * POSELIFT_FPS + 1e-9))
    ks = np.arange(k0, k1 + 1)
    grid = t0 + ks / POSELIFT_FPS
    ok, kp15 = _resample(t, kp, grid, MAX_GAP_S)
    _, box15 = _resample(t, boxes, grid, MAX_GAP_S)
    ks, kp15, box15 = ks[ok], kp15[ok], box15[ok]
    kp15[..., :2] = _moving_average(kp15[..., :2], ks, POSELIFT_WINDOW, centered=(mode == "offline"))
    kp15[..., 0] *= sx
    kp15[..., 1] *= sy
    pid = poses[0].track_id if person_id is None else person_id
    out: dict[int, dict[int, dict[str, list]]] = {}
    for k, kk, b in zip(ks, kp15, box15, strict=True):
        x1, y1, x2, y2 = b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy
        out[int(k)] = {pid: {"bbox": [x1, y1, x2 - x1, y2 - y1], "keypoints": kk.tolist()}}
    return out


def _kp_box(k: np.ndarray) -> tuple[float, float, float, float]:
    return (k[:, 0].min(), k[:, 1].min(), k[:, 0].max(), k[:, 1].max())


def _moving_average(x: np.ndarray, ks: np.ndarray, window: int, centered: bool) -> np.ndarray:
    """Mean over the frames within the window that exist (runs split at grid gaps)."""
    out = np.empty_like(x)
    lo_off, hi_off = ((window - 1) // 2, window // 2) if centered else (window - 1, 0)
    for i, k in enumerate(ks):
        sel = (ks >= k - lo_off) & (ks <= k + hi_off)
        out[i] = x[sel].mean(axis=0)
    return out


def stgnf_normalize(segment: np.ndarray) -> np.ndarray:
    """STG-NF `normalize_pose` for one (T, 17, 3) segment: center xy on the segment mean,
    divide by the std of y. Confidence untouched."""
    seg = segment.astype(np.float64).copy()
    xy = seg[..., :2]
    seg[..., :2] = (xy - xy.mean(axis=(0, 1))) / max(float(seg[..., 1].std()), 1e-9)
    return seg
