"""Frame-level AUC-ROC and AUC-PR (average precision) for `public_pose`-style benchmarks.

These are the numbers published for STG-NF on PoseLift/RetailS, so the definitions
match scikit-learn exactly (tests cross-check when sklearn is installed):
- AUC-ROC: probability a random positive frame outscores a random negative one,
  ties counting 1/2 (Mann-Whitney U with average ranks).
- AUC-PR: average precision, sum over distinct thresholds of (R_k - R_{k-1}) * P_k
  (step function, no interpolation), i.e. `sklearn.metrics.average_precision_score`.

Per-clip score aggregation (max over tracks per frame, fill for frames with no
person, optional smoothing) lives in `frame_scores_from_tracks` so every model is
aggregated identically. Gaussian smoothing looks at future frames; it exists only for
comparability with published offline numbers and is recorded as non-causal.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from eval.metrics.stats import DEFAULT_B, DEFAULT_SEED, LOW_N, MetricValue, bootstrap_stat


def _rankdata_avg(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=float)
    # boundaries of tie blocks in sorted order
    starts = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])
    ends = np.r_[starts[1:], len(xs)]
    for s, e in zip(starts, ends, strict=True):
        ranks[order[s:e]] = (s + e + 1) / 2.0  # average of 1-based ranks s+1..e
    return ranks


def auc_roc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    r = _rankdata_avg(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return math.nan
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    last = np.r_[ss[1:] != ss[:-1], True]  # last index of each distinct score
    tp = np.cumsum(ys)[last]
    fp = np.cumsum(~ys)[last]
    prec = tp / (tp + fp)
    rec = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def frame_auc(
    clips: Sequence[tuple[np.ndarray, np.ndarray]], b: int = DEFAULT_B, seed: int = DEFAULT_SEED
) -> dict[str, MetricValue]:
    """AUC-ROC and AP over all frames of all clips, CI by resampling clips.

    `clips` = [(labels[T], scores[T]), ...], one pair per clip (the bootstrap unit).
    """
    if not clips:
        return {"auc_roc": MetricValue.unavailable("no clips"), "auc_pr": MetricValue.unavailable("no clips")}
    for y, s in clips:
        if len(y) != len(s):
            raise ValueError(f"labels/scores length mismatch: {len(y)} vs {len(s)}")
        if np.isnan(np.asarray(s, dtype=float)).any():
            raise ValueError("NaN frame scores: fill frames without a person before scoring")
    y_all = np.concatenate([c[0] for c in clips]).astype(bool)
    n = {
        "frames": int(y_all.size),
        "positive_frames": int(y_all.sum()),
        "clips": len(clips),
        "positive_clips": sum(1 for c in clips if np.any(c[0])),
    }
    out: dict[str, MetricValue] = {}
    for name, fn in (("auc_roc", auc_roc), ("auc_pr", average_precision)):

        def stat(cs: list[tuple[np.ndarray, np.ndarray]], fn=fn) -> float:
            return fn(np.concatenate([c[0] for c in cs]), np.concatenate([c[1] for c in cs]))

        v, ci = bootstrap_stat(list(clips), stat, b, seed)
        if math.isnan(v):
            out[name] = MetricValue.unavailable("needs both positive and negative frames", **n)
        else:
            out[name] = MetricValue(v, ci, n, low_n=n["positive_clips"] < LOW_N)
    return out


def gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """1-D Gaussian filter, 'reflect' borders, truncate 4 sigma (= scipy.ndimage defaults).
    NON-CAUSAL: uses future frames. Benchmark comparability only."""
    if sigma <= 0:
        return np.asarray(x, dtype=float)
    radius = int(4.0 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    xp = np.pad(np.asarray(x, dtype=float), radius, mode="symmetric")
    return np.convolve(xp, k, mode="valid")


def frame_scores_from_tracks(
    n_frames: int, frame_idx: np.ndarray, scores: np.ndarray, fill: float | None = None, sigma: float = 0.0
) -> np.ndarray:
    """Per-frame clip score = max over tracks' scores at that frame.

    Frames with no scored person get `fill` (default: the clip's minimum score, or 0 if
    nothing was scored), as in the STG-NF evaluation code.
    """
    frame_idx = np.asarray(frame_idx, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if frame_idx.size and (frame_idx.min() < 0 or frame_idx.max() >= n_frames):
        raise ValueError("frame_idx out of range")
    out = np.full(n_frames, -np.inf)
    np.maximum.at(out, frame_idx, scores)
    missing = np.isinf(out)
    if fill is None:
        fill = float(scores.min()) if scores.size else 0.0
    out[missing] = fill
    return gaussian_smooth(out, sigma) if sigma > 0 else out
