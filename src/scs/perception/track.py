"""Tracking building blocks shared by ByteTrack and the appearance-aware tracker.

Ported from the ByteTrack reference implementation (github.com/ifzhang/ByteTrack,
yolox/tracker/{kalman_filter,matching,basetrack,byte_tracker}.py, MIT License,
Copyright (c) 2021 Yifu Zhang) and BoT-SORT (github.com/NirAharon/BoT-SORT, MIT License,
Copyright (c) 2022 Nir Aharon). Changes: numpy-only (no lap/cython_bbox, so CPU CI can run
it), per-instance track-id counters (one tracker per camera, ids are camera-local per
contracts.Track), boxes in main-stream pixels, output as `contracts.Track`.

Why a port instead of a pip package: boxmot is AGPL-3.0 and ultralytics' trackers ship
inside an AGPL package; these reference implementations are MIT and small enough to own.
Everything here is causal (AGENTS.md rule 11): state at frame t uses frames <= t only.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np

from scs.contracts import FrameRef, Track
from scs.perception.detect import iou_matrix

# --------------------------------------------------------------------------- assignment


def _hungarian(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Min-cost assignment for a rectangular matrix (O(n^3) shortest augmenting path).

    Fallback for when scipy isn't installed (CPU CI installs only numpy). Returns row and
    column indices like scipy.optimize.linear_sum_assignment.
    """
    transposed = cost.shape[0] > cost.shape[1]
    c = cost.T if transposed else cost
    n, m = c.shape
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)  # p[j] = row (1-based) assigned to column j
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = c[i0 - 1] - u[i0] - v[1:]
            free = ~used[1:]
            upd = free & (cur < minv[1:])
            minv[1:][upd] = cur[upd]
            way[1:][upd] = j0
            cand = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    cols = np.nonzero(p[1:])[0]
    rows = p[1:][cols] - 1
    order = np.argsort(rows)
    rows, cols = rows[order], cols[order]
    return (cols, rows) if transposed else (rows, cols)


def linear_assignment(cost: np.ndarray, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Optimal matching with costs > `thresh` forbidden. Returns (matches, unmatched_a, unmatched_b)."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    big = thresh + 1e5
    c = np.where(cost > thresh, big, cost)
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(c)
    except ImportError:
        rows, cols = _hungarian(c)
    matches = [(int(r), int(k)) for r, k in zip(rows, cols, strict=True) if c[r, k] <= thresh]
    ma, mb = {r for r, _ in matches}, {k for _, k in matches}
    return matches, [i for i in range(n) if i not in ma], [j for j in range(m) if j not in mb]


# --------------------------------------------------------------------------- Kalman filters


class KalmanFilterXYAH:
    """Constant-velocity KF on (cx, cy, aspect, h); ByteTrack's filter."""

    ndim = 4
    std_pos = 1.0 / 20
    std_vel = 1.0 / 160

    def __init__(self) -> None:
        self._F = np.eye(8)
        self._F[:4, 4:] = np.eye(4)
        self._H = np.eye(4, 8)

    def _scale(self, mean: np.ndarray) -> np.ndarray:
        """Per-dimension scale for noise; (N,4). XYAH: h, h, 1(aspect), h."""
        h = mean[..., 3]
        return np.stack([h, h, np.full_like(h, np.nan), h], -1)

    def _std(self, mean: np.ndarray, k: float, aspect: float) -> np.ndarray:
        s = self._scale(mean) * k
        s[..., 2] = aspect
        return s

    def initiate(self, meas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[meas, np.zeros(4)]
        std = np.r_[self._std(mean, 2 * self.std_pos, 1e-2), self._std(mean, 10 * self.std_vel, 1e-5)]
        return mean, np.diag(std**2)

    def multi_predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        std = np.concatenate([self._std(mean, self.std_pos, 1e-2), self._std(mean, self.std_vel, 1e-5)], 1)
        q = np.zeros((len(mean), 8, 8))
        q[:, np.arange(8), np.arange(8)] = std**2
        mean = mean @ self._F.T
        cov = self._F @ cov @ self._F.T + q
        return mean, cov

    def update(self, mean: np.ndarray, cov: np.ndarray, meas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        std = self._std(mean, self.std_pos, 1e-1)
        proj_mean = self._H @ mean
        proj_cov = self._H @ cov @ self._H.T + np.diag(std**2)
        gain = np.linalg.solve(proj_cov, (cov @ self._H.T).T).T
        mean = mean + (meas - proj_mean) @ gain.T
        cov = cov - gain @ proj_cov @ gain.T
        return mean, cov

    @staticmethod
    def to_meas(tlwh: np.ndarray) -> np.ndarray:
        x, y, w, h = tlwh
        return np.array([x + w / 2, y + h / 2, w / max(h, 1e-6), h])

    @staticmethod
    def to_tlwh(mean: np.ndarray) -> np.ndarray:
        cx, cy, a, h = mean[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, w, h])


class KalmanFilterXYWH(KalmanFilterXYAH):
    """Constant-velocity KF on (cx, cy, w, h); BoT-SORT's filter (width noise scales with w)."""

    def _std(self, mean: np.ndarray, k: float, aspect: float) -> np.ndarray:
        w, h = mean[..., 2], mean[..., 3]
        return np.stack([w, h, w, h], -1) * k

    @staticmethod
    def to_meas(tlwh: np.ndarray) -> np.ndarray:
        x, y, w, h = tlwh
        return np.array([x + w / 2, y + h / 2, w, h])

    @staticmethod
    def to_tlwh(mean: np.ndarray) -> np.ndarray:
        cx, cy, w, h = mean[:4]
        return np.array([cx - w / 2, cy - h / 2, w, h])


# --------------------------------------------------------------------------- track state


class TrackState(IntEnum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class STrack:
    """One tracklet. `feat` is the EMA-smoothed, L2-normalized appearance embedding (or None)."""

    def __init__(
        self,
        tlwh: np.ndarray,
        score: float,
        frame: FrameRef | None = None,
        feat: np.ndarray | None = None,
        feat_alpha: float = 0.9,
    ) -> None:
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.score = float(score)
        self.frame = frame
        self.kf: KalmanFilterXYAH | None = None
        self.mean: np.ndarray | None = None
        self.cov: np.ndarray | None = None
        self.is_activated = False
        self.state = TrackState.NEW
        self.track_id = -1
        self.tracklet_len = 0
        self.start_frame = 0
        self.frame_id = 0
        self.feat_alpha = feat_alpha
        self.curr_feat = None if feat is None else feat / (np.linalg.norm(feat) + 1e-12)
        self.feat = self.curr_feat

    # ---- geometry
    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None or self.kf is None:
            return self._tlwh.copy()
        return self.kf.to_tlwh(self.mean)

    @property
    def tlbr(self) -> np.ndarray:
        t = self.tlwh
        return np.r_[t[:2], t[:2] + t[2:]]

    @property
    def end_frame(self) -> int:
        return self.frame_id

    # ---- appearance
    def _update_feat(self, feat: np.ndarray | None) -> None:
        if feat is None:
            return
        feat = feat / (np.linalg.norm(feat) + 1e-12)
        self.curr_feat = feat
        self.feat = feat if self.feat is None else self.feat_alpha * self.feat + (1 - self.feat_alpha) * feat
        self.feat = self.feat / (np.linalg.norm(self.feat) + 1e-12)

    # ---- lifecycle (same semantics as ByteTrack's STrack)
    @staticmethod
    def multi_predict(tracks: list[STrack]) -> None:
        if not tracks:
            return
        kf = tracks[0].kf
        assert kf is not None
        mean = np.stack([t.mean for t in tracks if t.mean is not None]).copy()
        cov = np.stack([t.cov for t in tracks if t.cov is not None])
        for i, t in enumerate(tracks):
            if t.state != TrackState.TRACKED:
                mean[i, 7] = 0  # freeze height velocity while lost (reference behavior)
        mean, cov = kf.multi_predict(mean, cov)
        for i, t in enumerate(tracks):
            t.mean, t.cov = mean[i], cov[i]

    def activate(self, kf: KalmanFilterXYAH, frame_id: int, track_id: int) -> None:
        self.kf, self.track_id = kf, track_id
        self.mean, self.cov = kf.initiate(kf.to_meas(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        self.is_activated = frame_id == 1
        self.frame_id = self.start_frame = frame_id

    def re_activate(self, new: STrack, frame_id: int, new_id: int | None = None) -> None:
        self._step(new, frame_id)
        self.tracklet_len = 0
        if new_id is not None:
            self.track_id = new_id

    def update(self, new: STrack, frame_id: int) -> None:
        self._step(new, frame_id)
        self.tracklet_len += 1

    def _step(self, new: STrack, frame_id: int) -> None:
        assert self.kf is not None and self.mean is not None and self.cov is not None
        self.frame_id = frame_id
        self.mean, self.cov = self.kf.update(self.mean, self.cov, self.kf.to_meas(new._tlwh))
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.score = new.score
        self.frame = new.frame
        self._update_feat(new.curr_feat)

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED

    def to_track(self, state: str = "confirmed") -> Track | None:
        assert self.frame is not None
        x1, y1, x2, y2 = self.tlbr
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(self.frame.width), x2), min(float(self.frame.height), y2)
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None
        return Track(
            frame=self.frame,
            track_id=self.track_id,
            bbox=(float(x1), float(y1), float(x2), float(y2)),
            score=min(max(self.score, 0.0), 1.0),
            state=state,
        )  # type: ignore[arg-type]


# --------------------------------------------------------------------------- distances & list ops


def iou_distance(a: list[STrack], b: list[STrack]) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)))
    return 1.0 - iou_matrix(np.stack([t.tlbr for t in a]), np.stack([t.tlbr for t in b]))


def fuse_score(cost: np.ndarray, dets: list[STrack]) -> np.ndarray:
    if cost.size == 0:
        return cost
    scores = np.array([d.score for d in dets])[None, :]
    return 1.0 - (1.0 - cost) * scores


def embedding_distance(tracks: list[STrack], dets: list[STrack]) -> np.ndarray:
    """Cosine distance in [0, 2] between smoothed track features and detection features; 1 if missing."""
    cost = np.ones((len(tracks), len(dets)))
    if cost.size == 0:
        return cost
    ti = [i for i, t in enumerate(tracks) if t.feat is not None]
    di = [j for j, d in enumerate(dets) if d.curr_feat is not None]
    if ti and di:
        tf = np.stack([tracks[i].feat for i in ti])  # type: ignore[misc]
        df = np.stack([dets[j].curr_feat for j in di])  # type: ignore[misc]
        cost[np.ix_(ti, di)] = np.clip(1.0 - tf @ df.T, 0.0, 2.0)
    return cost


def joint(a: list[STrack], b: list[STrack]) -> list[STrack]:
    seen = {t.track_id for t in a}
    return a + [t for t in b if t.track_id not in seen]


def sub(a: list[STrack], b: list[STrack]) -> list[STrack]:
    drop = {t.track_id for t in b}
    return [t for t in a if t.track_id not in drop]


def remove_duplicates(a: list[STrack], b: list[STrack]) -> tuple[list[STrack], list[STrack]]:
    d = iou_distance(a, b)
    dup_a, dup_b = set(), set()
    for p, q in zip(*np.where(d < 0.15), strict=True):
        if a[p].frame_id - a[p].start_frame > b[q].frame_id - b[q].start_frame:
            dup_b.add(q)
        else:
            dup_a.add(p)
    return [t for i, t in enumerate(a) if i not in dup_a], [t for i, t in enumerate(b) if i not in dup_b]


def dets_to_stracks(dets: Any, feats: np.ndarray | None = None, feat_alpha: float = 0.9) -> list[STrack]:
    """`contracts.Detection`s (+ optional (N,D) features) → STracks."""
    out = []
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = d.bbox
        out.append(
            STrack(
                np.array([x1, y1, x2 - x1, y2 - y1]),
                d.score,
                d.frame,
                None if feats is None else feats[i],
                feat_alpha,
            )
        )
    return out
