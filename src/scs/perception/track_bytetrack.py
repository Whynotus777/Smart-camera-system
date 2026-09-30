"""ByteTrack (MIT reference implementation, see track.py for attribution), one instance per camera.

Why ByteTrack: it's the simplest strong tracker (motion + IoU only) and associates the
*low-score* detections in a second pass, which is what keeps a person tracked while a
shelf or another shopper partly hides them. `ByteTrackConfig()` defaults equal the
reference repo's defaults; that's the **baseline** for the T03 ID-switch acceptance
criterion, so don't change them. Tune via a different config instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scs.contracts import Detection, Track
from scs.perception.track import (
    KalmanFilterXYAH,
    STrack,
    TrackState,
    dets_to_stracks,
    fuse_score,
    iou_distance,
    joint,
    linear_assignment,
    remove_duplicates,
    sub,
)


@dataclass(frozen=True)
class ByteTrackConfig:
    track_thresh: float = 0.5  # high/low split; new tracks need track_thresh + 0.1
    low_thresh: float = 0.1
    match_thresh: float = 0.8
    second_match_thresh: float = 0.5
    unconfirmed_match_thresh: float = 0.7
    track_buffer: int = 30  # frames at 30 fps; scaled by frame_rate / 30
    frame_rate: float = 30.0
    mot20: bool = False  # True disables score fusion


class ByteTracker:
    """Implements `scs.perception.base.Tracker`. `frame` is ignored (motion-only)."""

    def __init__(self, cfg: ByteTrackConfig | None = None) -> None:
        self.cfg = cfg or ByteTrackConfig()
        self.det_thresh = self.cfg.track_thresh + 0.1
        self.max_time_lost = int(self.cfg.frame_rate / 30.0 * self.cfg.track_buffer)
        self.kf = KalmanFilterXYAH()
        self.frame_id = 0
        self._next_id = 0
        self.tracked: list[STrack] = []
        self.lost: list[STrack] = []
        self.removed: list[STrack] = []

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _features(self, dets: list[Detection], frame: Any) -> np.ndarray | None:
        return None

    def update(self, dets: list[Detection], frame: Any = None) -> list[Track]:
        self.frame_id += 1
        cfg = self.cfg
        scores = np.array([d.score for d in dets])
        feats = self._features(dets, frame)
        allst = dets_to_stracks(dets, feats)
        high = [s for s, sc in zip(allst, scores, strict=True) if sc > cfg.track_thresh]
        low = [s for s, sc in zip(allst, scores, strict=True) if cfg.low_thresh < sc <= cfg.track_thresh]

        activated: list[STrack] = []
        refind: list[STrack] = []
        lost_now: list[STrack] = []
        removed_now: list[STrack] = []
        unconfirmed = [t for t in self.tracked if not t.is_activated]
        tracked = [t for t in self.tracked if t.is_activated]

        # 1) high-score detections vs tracked + lost
        pool = joint(tracked, self.lost)
        STrack.multi_predict(pool)
        matches, u_track, u_det = linear_assignment(self._first_cost(pool, high), cfg.match_thresh)
        for it, idet in matches:
            self._apply(pool[it], high[idet], activated, refind)

        # 2) low-score detections vs still-unmatched *tracked* tracks (IoU only)
        r_tracked = [pool[i] for i in u_track if pool[i].state == TrackState.TRACKED]
        matches, u_track2, _ = linear_assignment(iou_distance(r_tracked, low), cfg.second_match_thresh)
        for it, idet in matches:
            self._apply(r_tracked[it], low[idet], activated, refind)
        for it in u_track2:
            t = r_tracked[it]
            if t.state != TrackState.LOST:
                t.mark_lost()
                lost_now.append(t)

        # 3) unconfirmed (one-frame-old) tracks vs leftover high detections
        left = [high[i] for i in u_det]
        matches, u_unconf, u_det = linear_assignment(
            self._unconfirmed_cost(unconfirmed, left), cfg.unconfirmed_match_thresh
        )
        for it, idet in matches:
            unconfirmed[it].update(left[idet], self.frame_id)
            activated.append(unconfirmed[it])
        for it in u_unconf:
            unconfirmed[it].mark_removed()
            removed_now.append(unconfirmed[it])
        left = [left[i] for i in u_det]

        # 3b) hook for appearance re-identification of lost tracks (no-op here)
        left = self._reidentify(left, refind)

        # 4) new tracks
        for d in left:
            if d.score >= self.det_thresh:
                d.activate(self.kf, self.frame_id, self._new_id())
                activated.append(d)

        # 5) expire lost tracks
        for t in self.lost:
            if self.frame_id - t.end_frame > self.max_time_lost:
                t.mark_removed()
                removed_now.append(t)

        self.tracked = [t for t in self.tracked if t.state == TrackState.TRACKED]
        self.tracked = joint(joint(self.tracked, activated), refind)
        # Same order as the reference: tracks expired *this* frame leave `lost` next frame.
        self.lost = sub(sub(self.lost, self.tracked) + lost_now, self.removed)
        self.removed = (self.removed + removed_now)[-1000:]  # bounded; ids only grow, so old ones never recur
        self._on_removed(removed_now)
        self.tracked, self.lost = remove_duplicates(self.tracked, self.lost)
        out = [t.to_track() for t in self.tracked if t.is_activated]
        return [t for t in out if t is not None]

    # ---- overridable pieces (appearance-aware subclass)
    def _first_cost(self, pool: list[STrack], dets: list[STrack]) -> np.ndarray:
        d = iou_distance(pool, dets)
        return d if self.cfg.mot20 else fuse_score(d, dets)

    def _unconfirmed_cost(self, tracks: list[STrack], dets: list[STrack]) -> np.ndarray:
        return self._first_cost(tracks, dets)

    def _reidentify(self, dets: list[STrack], refind: list[STrack]) -> list[STrack]:
        return dets

    def _on_removed(self, tracks: list[STrack]) -> None:
        pass

    def _apply(self, t: STrack, d: STrack, activated: list[STrack], refind: list[STrack]) -> None:
        if t.state == TrackState.TRACKED:
            t.update(d, self.frame_id)
            activated.append(t)
        else:
            t.re_activate(d, self.frame_id)
            refind.append(t)
