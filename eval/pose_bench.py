"""Pose-sequence benchmark machinery + the framework-default `public_pose` suite.

`eval/suites/public_pose.py` belongs to T06; until it lands (and as the reference T06
builds on), this module registers a default `public_pose` with `builtin=True`. T06's
file replaces it automatically. Everything T06 needs is here as functions:
`window_scores()` (per-track causal windows -> scores) and `clip_frame_scores()`.

Protocol (so numbers are comparable across models):
- Windows are `model.window` consecutive rows of one track; each window's score is
  assigned to its LAST frame (causal: no future frames). Frames of a track before its
  first full window get no score from that track.
- Frame score = max over tracks at that frame; frames with no scored person get the
  clip's minimum score (STG-NF convention).
- `--opt sigma=S` applies a Gaussian over frames. It is NON-CAUSAL, off by default, and
  exists only to reproduce offline published numbers; the report records it.
- Model input: raw main-stream pixel keypoints `(x, y, conf)`; normalization is the
  model's job (T06: camera-agnostic normalization).
- Fast path: if the model has `score_windows(kps[N, W, 17, 3]) -> [N]`, it is used;
  otherwise each window goes through `BehaviorModel.score(Sequence[Pose])`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from eval.datasets import Clip, load_dataset
from eval.metrics.frame import frame_auc, frame_scores_from_tracks
from eval.metrics.stats import MetricValue
from eval.suites.base import RunContext, Suite, SuiteResult, register_suite
from scs.contracts import FrameRef, Pose

PUBLISHED = {
    "poselift": {"STG-NF": 67.5},
    "retails_real": {"STG-NF": 63.2},
    "retails_staged": {"STG-NF": 87.2},
}


def _windows(kps: np.ndarray, w: int) -> np.ndarray:
    """(T, 17, 3) -> (T - w + 1, w, 17, 3) view."""
    v = np.lib.stride_tricks.sliding_window_view(kps, w, axis=0)  # (N, 17, 3, w)
    return np.moveaxis(v, -1, 1)


def _poses(
    camera_id: str,
    track_id: int,
    fi: np.ndarray,
    ts: np.ndarray,
    kps: np.ndarray,
    width: int,
    height: int,
    model_id: str,
) -> list[Pose]:
    k = kps.copy()
    k[..., 2] = np.clip(np.nan_to_num(k[..., 2]), 0.0, 1.0)
    k[..., :2] = np.nan_to_num(k[..., :2])
    return [
        Pose(
            frame=FrameRef(
                camera_id=camera_id,
                epoch=0,
                seq=int(f),
                frame_idx=int(f),
                ts=float(t),
                width=width,
                height=height,
            ),
            track_id=int(track_id),
            keypoints=[tuple(map(float, r)) for r in k[i]],
            model_id=model_id,
        )
        for i, (f, t) in enumerate(zip(fi, ts, strict=True))
    ]


@dataclass
class ClipScores:
    clip_id: str
    camera_id: str
    frame_idx: np.ndarray  # window-end frames
    scores: np.ndarray
    n_windows: int


def window_scores(clip: Clip, model: Any, stride: int = 1, batch: bool = True) -> ClipScores:
    w = int(model.window)
    fis, scs = [], []
    lab = clip.labels
    for seq in clip.pose_sequences(min_len=w):
        if batch and hasattr(model, "score_windows"):
            wins = _windows(seq.kps, w)[::stride]
            s = np.asarray(model.score_windows(np.ascontiguousarray(wins)), dtype=float)
        else:
            poses = _poses(
                seq.camera_id,
                seq.track_id,
                seq.frame_idx,
                seq.ts,
                seq.kps,
                lab.width or 1,
                lab.height or 1,
                "gt-or-upstream",
            )
            s = np.array(
                [model.score(poses[e - w + 1 : e + 1]).score for e in range(w - 1, len(poses), stride)]
            )
        ends = seq.frame_idx[w - 1 :: stride][: len(s)]
        if len(s) != len(ends):
            raise ValueError(f"{clip.clip_id}: model returned {len(s)} scores for {len(ends)} windows")
        fis.append(ends)
        scs.append(s)
    fi = np.concatenate(fis) if fis else np.zeros(0, dtype=int)
    sc = np.concatenate(scs) if scs else np.zeros(0)
    if sc.size and (np.isnan(sc).any() or sc.min() < 0 or sc.max() > 1):
        raise ValueError(f"{clip.clip_id}: scores must be finite and in [0, 1]")
    return ClipScores(clip.clip_id, lab.camera_id or "", fi, sc, int(sc.size))


def clip_frame_scores(cs: ClipScores, n_frames: int, sigma: float = 0.0) -> np.ndarray:
    return frame_scores_from_tracks(n_frames, cs.frame_idx, cs.scores, fill=None, sigma=sigma)


def pose_clip_outputs(
    dataset: str, split: str, model: Any, ctx: RunContext, sigma: float, stride: int, batch: bool = True
) -> tuple[list[tuple[str, str, np.ndarray, np.ndarray]], dict[str, Any]]:
    """[(clip_id, camera_id, frame_labels, frame_scores)] for every clip of `split` with frame labels."""
    ds = load_dataset(dataset, ctx.data_root, ctx.split_dir)
    ids = ds.clip_ids(split)
    if ctx.limit:
        ids = ids[: ctx.limit]
    out, n_win, skipped = [], 0, []
    for cid in ids:
        clip = ds.clip(cid)
        y = clip.frame_labels()
        if y is None:
            skipped.append(cid)
            continue
        cs = window_scores(clip, model, stride, batch)
        n_win += cs.n_windows
        out.append((cid, cs.camera_id, y.astype(int), clip_frame_scores(cs, len(y), sigma)))
    return out, {"windows": n_win, "clips_without_frame_labels": skipped, "dataset": ds.version()}


@register_suite(builtin=True)
class PublicPose(Suite):
    name = "public_pose"
    description = (
        "Frame AUC-ROC/PR on PoseLift test (pose-sequence model comparability; not system validation)"
    )
    datasets = ("poselift",)
    labels = ("real store, pose only: benchmark comparability, not system-level validation",)
    gated = ("auc_roc",)

    def run(self, ctx: RunContext) -> SuiteResult:
        if "behavior" not in ctx.models:
            return self.unavailable("needs --models behavior=<module:attr>")
        m = ctx.models["behavior"]
        sigma = ctx.opt("sigma", 0.0, float)
        stride = ctx.opt("stride", 1, int)
        dataset = ctx.opt("dataset", "poselift")
        try:
            t0 = time.perf_counter()
            clips, info = pose_clip_outputs(dataset, ctx.split, m.obj, ctx, sigma, stride)
            dt = time.perf_counter() - t0
        except (FileNotFoundError, KeyError) as e:
            return self.unavailable(f"{dataset} not available: {e}")
        if not clips:
            return self.unavailable(f"no {ctx.split} clips with frame labels in {dataset}")
        auc = frame_auc([(y, s) for _, _, y, s in clips], b=ctx.bootstrap)
        per_cam: dict[str, dict[str, MetricValue]] = {}
        for cam in sorted({c for _, c, _, _ in clips}):
            per_cam[cam] = frame_auc([(y, s) for _, c, y, s in clips if c == cam], b=ctx.bootstrap)
        notes = [f"Published STG-NF on PoseLift: {PUBLISHED['poselift']['STG-NF']} AUC-ROC (x100)."]
        if sigma > 0:
            notes.append(f"sigma={sigma}: NON-CAUSAL Gaussian smoothing (offline comparability only)")
        return self.result(
            "ok",
            {"auc_roc": auc["auc_roc"], "auc_pr": auc["auc_pr"]},
            metrics={
                "per_camera": per_cam,
                "windows_scored": info["windows"],
                "scoring_seconds": round(dt, 3),
                "clips_without_frame_labels": info["clips_without_frame_labels"],
            },
            datasets=[info["dataset"]],
            notes=notes,
            params={
                "sigma": sigma,
                "stride": stride,
                "window": int(m.obj.window),
                "dataset": dataset,
                "split": ctx.split,
                "causal": sigma == 0,
            },
        )
