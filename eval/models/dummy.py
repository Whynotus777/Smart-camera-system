"""Dummy behavior models: harness smoke tests, runtime checks, and chance-level baselines.

They implement `scs.behavior.base.BehaviorModel` (`model_id`, `window`, `score(poses)`)
and the optional batch fast path `score_windows(kps[N, W, 17, 3]) -> scores[N]` that
`eval.pose_bench` uses when present. Both paths must agree (tested).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from scs.contracts import BehaviorScore, Pose

LINEAGE_NONE = {"train": [], "license": "n/a (no training)"}


class _Base:
    model_id = "dummy"
    window = 24
    lineage = LINEAGE_NONE

    def score_windows(self, kps: np.ndarray) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    def score(self, poses: Sequence[Pose]) -> BehaviorScore:
        kps = np.array([p.keypoints for p in poses], dtype=float)[None]
        s = float(self.score_windows(kps)[0])
        return BehaviorScore(
            camera_id=poses[-1].frame.camera_id,
            track_id=poses[-1].track_id,
            ts_start=poses[0].frame.ts,
            ts_end=poses[-1].frame.ts,
            model_id=self.model_id,
            score=s,
        )


class ConstantBehavior(_Base):
    """Scores everything 0.5: AUC must come out exactly 0.5 (a harness sanity check)."""

    model_id = "dummy-constant@0"

    def score_windows(self, kps: np.ndarray) -> np.ndarray:
        return np.full(len(kps), 0.5)


class RandomBehavior(_Base):
    """Deterministic pseudo-random score per window content (chance baseline)."""

    model_id = "dummy-random@0"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.model_id = f"dummy-random@{seed}"

    def score_windows(self, kps: np.ndarray) -> np.ndarray:
        out = np.empty(len(kps))
        for i, w in enumerate(kps):
            h = hashlib.blake2b(np.round(w, 2).tobytes(), digest_size=8, key=str(self.seed).encode())
            out[i] = int.from_bytes(h.digest(), "little") / 2**64
        return out


class WristMotionBehavior(_Base):
    """Mean wrist speed relative to torso length, squashed to [0, 1]. A crude heuristic
    baseline (camera-agnostic normalization as T06 is asked to do), not a model."""

    model_id = "dummy-wrist-motion@0"

    def score_windows(self, kps: np.ndarray) -> np.ndarray:
        xy = kps[..., :2]
        torso = np.linalg.norm(xy[:, :, [5, 6]].mean(2) - xy[:, :, [11, 12]].mean(2), axis=-1)  # (N, W)
        scale = np.nanmedian(torso, axis=1)
        scale = np.where(np.isfinite(scale) & (scale > 1e-3), scale, 1.0)
        v = np.linalg.norm(np.diff(xy[:, :, [9, 10]], axis=1), axis=-1).mean(axis=(1, 2)) / scale
        return 1.0 - np.exp(-np.nan_to_num(v) * 10.0)
