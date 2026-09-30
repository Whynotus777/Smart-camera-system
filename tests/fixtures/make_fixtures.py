"""Generate tiny deterministic Track/Pose fixtures for building against the contracts.

Scenario (camera `cam1` from `configs/site.example.yaml`, main stream 2560x1440,
10 fps, 10 s, 100 frames per person):

- track 1: stands beside `shelf_a` and reaches into it with the right wrist
  (0.5-2.5 s; the wrist lands in `hv_candy`, which lies inside `shelf_a`), then walks
  down and left to `door` (3-9 s) and waits there (9-10 s). This is the "shelf → door"
  journey. Outside the reach, both wrists stay outside `shelf_a` in image space.
- track 2: walks from the aisle to `counter` (0-4 s) and stands there (4-10 s).

Boxes and keypoints are in main-stream pixels. Keypoints follow COCO17 order.
T05 (journey engine) builds and tests against these files.

Regenerate with:  python tests/fixtures/make_fixtures.py
"""

from __future__ import annotations

import random
from pathlib import Path

from scs.contracts import FrameRef, Keypoint, Pose, Track

HERE = Path(__file__).resolve().parent
TRACKS_PATH = HERE / "tracks.jsonl"
POSES_PATH = HERE / "poses.jsonl"

CAMERA_ID = "cam1"
W, H = 2560, 1440
FPS = 10
N_FRAMES = 100
T0 = 1_760_000_000.0  # fixed epoch so output is byte-stable
MONO0 = 5_000.0  # fixed host monotonic clock at the first frame
POSE_MODEL_ID = "fixture-synthetic@0"

BOX_W, BOX_H = 0.08, 0.35  # normalized person box size

# COCO17 keypoints as (dx, dy) offsets in box-normalized coords from the foot point
# (bottom-center). dy is negative = up. Arms hang at the sides by default.
_BASE_SKELETON: list[tuple[float, float]] = [
    (0.00, -0.93),  # nose
    (-0.05, -0.96), (0.05, -0.96),  # eyes
    (-0.12, -0.94), (0.12, -0.94),  # ears
    (-0.30, -0.80), (0.30, -0.80),  # shoulders
    (-0.38, -0.62), (0.38, -0.62),  # elbows
    (-0.40, -0.46), (0.40, -0.46),  # wrists
    (-0.18, -0.48), (0.18, -0.48),  # hips
    (-0.18, -0.25), (0.18, -0.25),  # knees
    (-0.18, -0.02), (0.18, -0.02),  # ankles
]
R_ELBOW, R_WRIST = 8, 10


def _lerp(a: tuple[float, float], b: tuple[float, float], u: float) -> tuple[float, float]:
    u = min(max(u, 0.0), 1.0)
    return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)


# Foot-point paths in normalized frame coords: list of (t_seconds, (x, y)) waypoints.
_PATHS: dict[int, list[tuple[float, tuple[float, float]]]] = {
    1: [(0.0, (0.52, 0.72)), (3.0, (0.52, 0.72)), (5.0, (0.52, 0.92)), (9.0, (0.10, 0.92)),
        (10.0, (0.10, 0.92))],
    2: [(0.0, (0.55, 0.60)), (4.0, (0.78, 0.85)), (10.0, (0.78, 0.85))],
}


def _foot(track_id: int, t: float) -> tuple[float, float]:
    wps = _PATHS[track_id]
    for (t_a, p_a), (t_b, p_b) in zip(wps, wps[1:], strict=False):
        if t_a <= t <= t_b:
            return _lerp(p_a, p_b, (t - t_a) / (t_b - t_a))
    return wps[-1][1]


def _reaching(track_id: int, t: float) -> bool:
    return track_id == 1 and 0.5 <= t < 2.5


def _keypoints(track_id: int, t: float, fx: float, fy: float, rng: random.Random) -> list[Keypoint]:
    offsets = list(_BASE_SKELETON)
    if _reaching(track_id, t):
        # Right arm raised up and toward the shelf (shelf_a is to the left of the foot point).
        offsets[R_ELBOW] = (-0.80, -0.95)
        offsets[R_WRIST] = (-1.40, -1.05)
    bw, bh = BOX_W * W, BOX_H * H
    kps: list[Keypoint] = []
    for dx, dy in offsets:
        x = fx * W + dx * bw + rng.gauss(0, 1.5)
        y = fy * H + dy * bh + rng.gauss(0, 1.5)
        c = round(min(1.0, max(0.0, 0.9 + rng.gauss(0, 0.03))), 3)
        kps.append((round(x, 2), round(y, 2), c))
    return kps


def generate() -> tuple[list[Track], list[Pose]]:
    rng = random.Random(0)  # noqa: S311 - deterministic test data, not crypto
    tracks: list[Track] = []
    poses: list[Pose] = []
    for i in range(N_FRAMES):
        t = i / FPS
        ref = FrameRef(camera_id=CAMERA_ID, frame_idx=i, ts=round(T0 + t, 3), width=W, height=H,
                       stream="main", ts_mono=round(MONO0 + t, 3))
        for track_id in (1, 2):
            fx, fy = _foot(track_id, t)
            x1, x2 = (fx - BOX_W / 2) * W, (fx + BOX_W / 2) * W
            y1, y2 = (fy - BOX_H) * H, fy * H
            bbox = (round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2))
            score = round(0.85 + 0.1 * rng.random(), 3)
            tracks.append(Track(frame=ref, track_id=track_id, bbox=bbox, score=score))
            poses.append(Pose(frame=ref, track_id=track_id, model_id=POSE_MODEL_ID,
                              keypoints=_keypoints(track_id, t, fx, fy, rng)))
    return tracks, poses


def render() -> tuple[str, str]:
    tracks, poses = generate()
    return (
        "".join(m.model_dump_json() + "\n" for m in tracks),
        "".join(m.model_dump_json() + "\n" for m in poses),
    )


def main() -> None:
    tracks_txt, poses_txt = render()
    TRACKS_PATH.write_text(tracks_txt)
    POSES_PATH.write_text(poses_txt)
    print(f"wrote {TRACKS_PATH} and {POSES_PATH}")


if __name__ == "__main__":
    main()
