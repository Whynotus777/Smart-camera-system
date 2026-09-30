"""Appearance-aware tracker for shelf occlusions: BoT-SORT-style association + lost-track re-ID.

Why: in a c-store the dominant ID-switch cause is a shopper disappearing behind a shelf
end or another shopper for 1–3 s and reappearing a few body-widths away. Motion alone
(ByteTrack) either loses the track or grabs the wrong nearby person. This tracker adds:

1. BoT-SORT association (MIT, github.com/NirAharon/BoT-SORT): XYWH Kalman filter, and
   cost = min(IoU distance, appearance distance) where appearance only counts for
   candidates that are spatially plausible (IoU gate) and similar enough.
2. A **re-ID stage for lost tracks** before new tracks are spawned: an unmatched
   high-score detection is matched to a lost track if their appearance is close and
   the detection lies within a motion gate that widens with time lost. This is what
   turns "new ID after the occlusion" into "same ID".
3. Camera-motion compensation is off: store cameras are fixed.

Appearance comes from an `AppearanceEmbedder`. The default, `HsvEmbedder`, is a
torso/legs colour histogram with no learned weights, so it carries **no dataset
licence lineage** (learned ReID weights are typically trained on Market-1501/MSMT17/
DukeMTMC, none of which is cleared for us; see the T03 report). It's weak for
look-alike clothing, which is why association is gated by motion first.
Embeddings live only in tracker memory (AGENTS.md rule 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from scs.contracts import Detection, Track
from scs.perception.detect import iou_matrix
from scs.perception.track import (
    KalmanFilterXYWH,
    STrack,
    TrackState,
    embedding_distance,
    fuse_score,
    iou_distance,
    linear_assignment,
)
from scs.perception.track_bytetrack import ByteTrackConfig, ByteTracker


class AppearanceEmbedder(Protocol):
    dim: int

    def embed(self, frame: Any, boxes: np.ndarray, frame_wh: tuple[int, int]) -> np.ndarray:
        """(N,4) xyxy main-stream boxes → (N, dim) float features (any scale)."""
        ...


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """uint8/float RGB (...,3) → h in [0,1), s, v in [0,1]."""
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    v = x.max(-1)
    c = v - x.min(-1)
    s = np.where(v > 0, c / np.maximum(v, 1e-6), 0)
    cc = np.maximum(c, 1e-6)
    h = np.where(v == r, ((g - b) / cc) % 6, np.where(v == g, (b - r) / cc + 2, (r - g) / cc + 4)) / 6.0
    return np.where(c > 0, h, 0) % 1.0, s, v


class HsvEmbedder:
    """Colour histogram of the torso and legs stripes of a person box. No weights, no training data."""

    def __init__(
        self,
        h_bins: int = 16,
        s_bins: int = 4,
        v_bins: int = 8,
        grid: tuple[int, int] = (32, 64),
        stripes: tuple[tuple[float, float], ...] = ((0.15, 0.55), (0.55, 1.0)),
        channel_order: str = "rgb",
    ) -> None:
        self.h_bins, self.s_bins, self.v_bins = h_bins, s_bins, v_bins
        self.grid, self.stripes, self.channel_order = grid, stripes, channel_order
        self.dim = len(stripes) * (h_bins * s_bins + v_bins)

    def _sample(self, frame: Any, box: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
        """Nearest-neighbour sample a grid of pixels inside `box` (main-stream coords) → (gh, gw, 3)."""
        fh, fw = (
            (frame.shape[0], frame.shape[1]) if frame.shape[-1] == 3 else (frame.shape[1], frame.shape[2])
        )
        sx, sy = fw / wh[0], fh / wh[1]  # the image may be smaller than main-stream (sub-stream decode)
        x1, y1, x2, y2 = box
        gw, gh = self.grid
        xs = np.clip(((x1 + (np.arange(gw) + 0.5) * (x2 - x1) / gw) * sx).astype(int), 0, fw - 1)
        ys = np.clip(((y1 + (np.arange(gh) + 0.5) * (y2 - y1) / gh) * sy).astype(int), 0, fh - 1)
        if frame.shape[-1] == 3:
            patch = frame[ys][:, xs]
        else:  # CHW tensor
            patch = frame[:, ys][:, :, xs]
            patch = patch.permute(1, 2, 0) if hasattr(patch, "permute") else patch.transpose(1, 2, 0)
        if hasattr(patch, "cpu"):
            patch = patch.cpu().numpy()
        patch = np.asarray(patch)
        return patch[..., ::-1] if self.channel_order == "bgr" else patch

    def embed(self, frame: Any, boxes: np.ndarray, frame_wh: tuple[int, int]) -> np.ndarray:
        out = np.zeros((len(boxes), self.dim), np.float32)
        if frame is None:
            return out
        for i, box in enumerate(np.asarray(boxes, dtype=np.float64).reshape(-1, 4)):
            h, s, v = _rgb_to_hsv(self._sample(frame, box, frame_wh))
            feats = []
            for a, b in self.stripes:
                r0, r1 = int(a * len(h)), int(b * len(h))
                hh, ss, vv = h[r0:r1].ravel(), s[r0:r1].ravel(), v[r0:r1].ravel()
                chroma = (ss > 0.15) & (vv > 0.15)
                hs = np.histogram2d(
                    hh[chroma], ss[chroma], bins=(self.h_bins, self.s_bins), range=((0, 1), (0, 1))
                )[0].ravel()
                vh = np.histogram(vv[~chroma], bins=self.v_bins, range=(0, 1))[0]
                f = np.r_[hs, vh].astype(np.float32)
                feats.append(f / max(f.sum(), 1.0))
            out[i] = np.sqrt(np.concatenate(feats))  # Hellinger: cosine on sqrt-histograms
        return out


@dataclass(frozen=True)
class AppearanceTrackConfig(ByteTrackConfig):
    track_thresh: float = 0.6  # BoT-SORT track_high_thresh
    proximity_thresh: float = 0.5  # IoU-distance gate for using appearance
    appearance_thresh: float = 0.25  # cosine-distance gate
    feat_alpha: float = 0.9  # EMA of track features
    track_buffer: int = 60
    # lost-track re-ID (stage 3b)
    reid_lost: bool = True
    reid_thresh: float = 0.15  # cosine distance (Hellinger features are dense, keep this tight)
    reid_gate_heights: float = 1.0  # base motion gate, in box heights
    reid_gate_growth: float = 1.0  # + box heights per second lost
    # long-term memory for re-entries: expired tracks' appearance kept in memory only (rule 6)
    reid_memory_s: float = 0.0  # 0 = off; <= 1800 (30 min, AGENTS.md rule 6)
    reid_long_after_s: float = 3.0  # gaps longer than this use reid_long_thresh
    reid_long_thresh: float = 0.08
    # suppress near-duplicate boxes before association (D-FINE is NMS-free and emits
    # low-score duplicates that can spawn spurious tracks); None = off
    dedup_iou: float | None = None


def _dedup(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    """Greedy NMS; returns kept indices in original order."""
    order = list(np.argsort(-scores))
    m = iou_matrix(boxes, boxes)
    keep: list[int] = []
    for i in order:
        if all(m[i, j] < iou for j in keep):
            keep.append(int(i))
    return sorted(keep)


class AppearanceTracker(ByteTracker):
    """Implements `scs.perception.base.Tracker`; needs `frame` (HxWx3 RGB) for appearance."""

    def __init__(
        self, cfg: AppearanceTrackConfig | None = None, embedder: AppearanceEmbedder | None = None
    ) -> None:
        super().__init__(cfg or AppearanceTrackConfig())
        self.acfg: AppearanceTrackConfig = self.cfg  # type: ignore[assignment]
        self.kf = KalmanFilterXYWH()
        self.embedder = embedder or HsvEmbedder()
        if not 0 <= self.acfg.reid_memory_s <= 1800:
            raise ValueError("reid_memory_s must be within [0, 1800] s (AGENTS.md rule 6)")
        self.gallery: dict[int, STrack] = {}  # expired tracks kept for re-entry re-ID

    def _on_removed(self, tracks: list[STrack]) -> None:
        cfg = self.acfg
        if cfg.reid_memory_s > 0:
            for t in tracks:
                if t.feat is not None and t.is_activated:
                    self.gallery[t.track_id] = t
        horizon = cfg.reid_memory_s * cfg.frame_rate
        self.gallery = {k: t for k, t in self.gallery.items() if self.frame_id - t.end_frame <= horizon}

    def update(self, dets: list[Detection], frame: Any = None) -> list[Track]:
        if self.acfg.dedup_iou is not None and len(dets) > 1:
            keep = _dedup(
                np.array([d.bbox for d in dets]), np.array([d.score for d in dets]), self.acfg.dedup_iou
            )
            if (
                frame is not None and isinstance(frame, np.ndarray) and frame.ndim == 2
            ):  # precomputed features
                frame = frame[keep]
            dets = [dets[i] for i in keep]
        return super().update(dets, frame)

    def _features(self, dets: list[Detection], frame: Any) -> np.ndarray | None:
        if frame is None or not dets:
            return None
        f = dets[0].frame
        return self.embedder.embed(frame, np.array([d.bbox for d in dets]), (f.width, f.height))

    def _first_cost(self, pool: list[STrack], dets: list[STrack]) -> np.ndarray:
        iou_d = iou_distance(pool, dets)
        far = iou_d > self.acfg.proximity_thresh
        if not self.cfg.mot20:
            iou_d = fuse_score(iou_d, dets)
        emb = embedding_distance(pool, dets) / 2.0
        emb[emb > self.acfg.appearance_thresh] = 1.0
        emb[far] = 1.0
        return np.minimum(iou_d, emb)

    def _reidentify(self, dets: list[STrack], refind: list[STrack]) -> list[STrack]:
        cfg = self.acfg
        lost = [t for t in self.lost if t.state == TrackState.LOST and t.feat is not None]
        seen = {t.track_id for t in lost}
        pool = lost + [t for k, t in self.gallery.items() if k not in seen]
        cand = [d for d in dets if d.score >= self.det_thresh and d.curr_feat is not None]
        if not cfg.reid_lost or not pool or not cand:
            return dets
        cost = embedding_distance(pool, cand) / 2.0
        for i, t in enumerate(pool):
            tx, ty, tw, th = t.tlwh
            gap_s = (self.frame_id - t.end_frame) / self.cfg.frame_rate
            gate = th * (cfg.reid_gate_heights + cfg.reid_gate_growth * gap_s)
            thresh = cfg.reid_long_thresh if gap_s > cfg.reid_long_after_s else cfg.reid_thresh
            for j, d in enumerate(cand):
                dx, dy, dw, dh = d._tlwh
                dist = np.hypot((dx + dw / 2) - (tx + tw / 2), (dy + dh) - (ty + th))  # foot points
                if dist > gate or not (0.5 < dh / max(th, 1e-6) < 2.0) or cost[i, j] > thresh:
                    cost[i, j] = 1.0
        matches, _, _ = linear_assignment(cost, max(cfg.reid_thresh, cfg.reid_long_thresh))
        used = set()
        for it, idet in matches:
            t, d = pool[it], cand[idet]
            if t.track_id in self.gallery:  # revive an expired track: fresh motion state, same id
                del self.gallery[t.track_id]
                self.removed = [r for r in self.removed if r.track_id != t.track_id]
                t.mean, t.cov = self.kf.initiate(self.kf.to_meas(d._tlwh))
            t.re_activate(d, self.frame_id)
            refind.append(t)
            used.add(id(d))
        return [d for d in dets if id(d) not in used]
