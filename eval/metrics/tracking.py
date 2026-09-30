"""Tracking metrics: IDF1, ID switches (CLEAR-MOT), MOTA, detection AP@IoU.

Definitions follow the references that published numbers use, so ours are comparable:
- IDF1 (Ristani et al. 2016, as in `motmetrics`/TrackEval): a single global one-to-one
  mapping of GT ids to predicted ids maximizing IDTP, where IDTP(g, p) counts frames in
  which both exist with IoU >= `iou`. IDF1 = 2 IDTP / (n_gt_dets + n_pred_dets).
- ID switches (CLEAR-MOT, `motmetrics` conventions): per frame, a GT keeps last frame's
  last matched predicted id if that pair is present with IoU >= `iou`; the rest are assigned
  by Hungarian on 1 - IoU. A switch is a GT matched to a predicted id different from
  the one it was last matched to (gaps in between don't reset it).
- Detection AP: greedy by score per frame, all-point interpolated precision envelope
  (VOC2010+/COCO style, single IoU).

All inputs are `TrackFrames`: rows of (frame, id, x1, y1, x2, y2[, score]) for one
sequence (one camera clip). Sequence-level sufficient statistics add up, so CIs come
from resampling sequences.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from eval.metrics.stats import DEFAULT_B, DEFAULT_SEED, MetricValue, bootstrap_ratio

try:  # scipy is optional; the fallback is exact and cross-checked in tests
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
except ImportError:  # pragma: no cover - exercised in CI without scipy
    _scipy_lsa = None


def _lsa_numpy(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Min-cost rectangular assignment (shortest augmenting path, O(n^2 m))."""
    c = np.asarray(cost, dtype=float)
    transposed = c.shape[0] > c.shape[1]
    if transposed:
        c = c.T
    n, m = c.shape
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    inf = np.inf
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)  # p[j] = row (1-based) assigned to column j
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            cur = c[i0 - 1, :] - u[i0] - v[1:]
            upd = free & (cur < minv[1:])
            minv[1:][upd] = cur[upd]
            way[1:][upd] = j0
            cand = np.where(free, minv[1:], inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    rows, cols = [], []
    for j in range(1, m + 1):
        if p[j]:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    r, cc = np.array(rows, dtype=int), np.array(cols, dtype=int)
    if transposed:
        r, cc = cc, r
    o = np.argsort(r)
    return r[o], cc[o]


def linear_sum_assignment(cost: np.ndarray, use_scipy: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if use_scipy and _scipy_lsa is not None:
        r, c = _scipy_lsa(np.asarray(cost, dtype=float))
        return np.asarray(r), np.asarray(c)
    return _lsa_numpy(cost)


@dataclass(frozen=True)
class TrackFrames:
    """One sequence of boxes: arrays of equal length."""

    frame: np.ndarray  # int
    ids: np.ndarray  # int
    boxes: np.ndarray  # (N, 4) x1 y1 x2 y2
    scores: np.ndarray | None = None

    @classmethod
    def from_rows(cls, rows: list[tuple]) -> TrackFrames:
        a = np.array(rows, dtype=float).reshape(-1, len(rows[0]) if rows else 6)
        sc = a[:, 6] if a.shape[1] > 6 else None
        return cls(a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2:6], sc)

    def __post_init__(self) -> None:
        n = len(self.frame)
        if len(self.ids) != n or self.boxes.shape != (n, 4):
            raise ValueError("TrackFrames arrays have inconsistent lengths")
        key = self.frame.astype(np.int64) * (1 << 32) + self.ids.astype(np.int64)
        if len(np.unique(key)) != n:
            raise ValueError("duplicate (frame, id) rows in a sequence")


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def _by_frame(t: TrackFrames) -> dict[int, np.ndarray]:
    d: dict[int, list[int]] = defaultdict(list)
    for i, f in enumerate(t.frame):
        d[int(f)].append(i)
    return {f: np.array(ix) for f, ix in d.items()}


@dataclass(frozen=True)
class SeqStats:
    """Additive per-sequence statistics."""

    n_gt: int
    n_pred: int
    idtp: int
    tp: int  # CLEAR matches
    fp: int
    fn: int
    idsw: int
    gt_frames_seconds: float  # person-seconds of GT (for switches per person-minute)


def sequence_stats(gt: TrackFrames, pred: TrackFrames, iou: float = 0.5, fps: float = 30.0,
                   use_scipy: bool = True) -> SeqStats:
    gtf, prf = _by_frame(gt), _by_frame(pred)
    gt_ids = {int(x): k for k, x in enumerate(np.unique(gt.ids))}
    pr_ids = {int(x): k for k, x in enumerate(np.unique(pred.ids))}
    co = np.zeros((len(gt_ids), len(pr_ids)))  # frames of co-occurrence with IoU >= thr
    last_match: dict[int, int] = {}  # gt id -> pred id it was last matched to (any earlier frame)
    tp = fp = fn = idsw = 0
    for f in sorted(set(gtf) | set(prf)):
        gi = gtf.get(f, np.array([], dtype=int))
        pi = prf.get(f, np.array([], dtype=int))
        g_id = gt.ids[gi].astype(int)
        p_id = pred.ids[pi].astype(int)
        m = iou_matrix(gt.boxes[gi], pred.boxes[pi]) if len(gi) and len(pi) else np.zeros((len(gi), len(pi)))
        ok = m >= iou - 1e-12
        # IDF1 co-occurrence
        for a, b in zip(*np.nonzero(ok), strict=True):
            co[gt_ids[int(g_id[a])], pr_ids[int(p_id[b])]] += 1
        # CLEAR: re-establish each GT's last correspondence if still valid (motmetrics step 1)
        pairs: dict[int, int] = {}
        used_g, used_p = set(), set()
        gpos = {int(x): k for k, x in enumerate(g_id)}
        ppos = {int(x): k for k, x in enumerate(p_id)}
        for g, p in last_match.items():
            if g in gpos and p in ppos and ppos[p] not in used_p and ok[gpos[g], ppos[p]]:
                pairs[g] = p
                used_g.add(gpos[g])
                used_p.add(ppos[p])
        rg = [k for k in range(len(gi)) if k not in used_g]
        rp = [k for k in range(len(pi)) if k not in used_p]
        if rg and rp:
            sub = ok[np.ix_(rg, rp)]
            cost = np.where(sub, 1.0 - m[np.ix_(rg, rp)], 1e6)
            r, c = linear_sum_assignment(cost, use_scipy)
            for a, b in zip(r, c, strict=True):
                if sub[a, b]:
                    g, p = int(g_id[rg[a]]), int(p_id[rp[b]])
                    pairs[g] = p
        for g, p in pairs.items():
            if g in last_match and last_match[g] != p:
                idsw += 1
            last_match[g] = p
        tp += len(pairs)
        fn += len(gi) - len(pairs)
        fp += len(pi) - len(pairs)
    idtp = 0
    if co.size:
        r, c = linear_sum_assignment(-co, use_scipy)
        idtp = int(co[r, c].sum())
    return SeqStats(len(gt.frame), len(pred.frame), idtp, tp, fp, fn, idsw, len(gt.frame) / fps)


def tracking_metrics(seqs: list[SeqStats], b: int = DEFAULT_B,
                     seed: int = DEFAULT_SEED) -> dict[str, MetricValue]:
    if not seqs:
        return {k: MetricValue.unavailable("no sequences") for k in ("idf1", "idsw_per_person_min", "mota")}
    a = {k: np.array([getattr(s, k) for s in seqs], dtype=float) for k in SeqStats.__dataclass_fields__}
    n = {"sequences": len(seqs), "gt_boxes": int(a["n_gt"].sum()), "pred_boxes": int(a["n_pred"].sum())}
    out: dict[str, MetricValue] = {}
    v, ci = bootstrap_ratio(2 * a["idtp"], a["n_gt"] + a["n_pred"], b, seed)
    out["idf1"] = MetricValue(v, ci, n)
    v, ci = bootstrap_ratio(a["idsw"], a["gt_frames_seconds"] / 60.0, b, seed)
    out["idsw_per_person_min"] = MetricValue(v, ci, n, unit="per person-minute")
    out["id_switches"] = MetricValue(float(a["idsw"].sum()), None, n)
    v, ci = bootstrap_ratio(a["n_gt"] - a["fn"] - a["fp"] - a["idsw"], a["n_gt"], b, seed)
    out["mota"] = MetricValue(v, ci, n)
    return out


def detection_ap(gt: list[TrackFrames], pred: list[TrackFrames], iou: float = 0.5) -> MetricValue:
    """AP at one IoU over all sequences (each GT box matched at most once per frame)."""
    recs: list[tuple[float, int]] = []  # (score, is_tp)
    n_gt = 0
    for g, p in zip(gt, pred, strict=True):
        if p.scores is None:
            raise ValueError("detection AP needs prediction scores")
        gtf, prf = _by_frame(g), _by_frame(p)
        n_gt += len(g.frame)
        for f, pi in prf.items():
            pi = pi[np.argsort(-p.scores[pi], kind="mergesort")]
            gi = gtf.get(f, np.array([], dtype=int))
            m = iou_matrix(p.boxes[pi], g.boxes[gi]) if len(gi) else np.zeros((len(pi), 0))
            taken = np.zeros(len(gi), dtype=bool)
            for k in range(len(pi)):
                best, bj = iou - 1e-12, -1
                for j in range(len(gi)):
                    if not taken[j] and m[k, j] >= best:
                        best, bj = m[k, j], j
                if bj >= 0:
                    taken[bj] = True
                recs.append((float(p.scores[pi[k]]), int(bj >= 0)))
    if n_gt == 0:
        return MetricValue.unavailable("no GT boxes")
    recs.sort(key=lambda r: -r[0])
    tps = np.cumsum([r[1] for r in recs]) if recs else np.array([0])
    fps_ = np.cumsum([1 - r[1] for r in recs]) if recs else np.array([0])
    rec = tps / n_gt
    prec = tps / np.maximum(tps + fps_, 1)
    mrec = np.r_[0.0, rec, 1.0]
    mpre = np.r_[0.0, prec, 0.0]
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.flatnonzero(mrec[1:] != mrec[:-1])
    ap = float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    return MetricValue(ap, None, {"gt_boxes": n_gt, "pred_boxes": len(recs), "sequences": len(gt)},
                       reason="CI unavailable for AP (not additive over sequences)")
