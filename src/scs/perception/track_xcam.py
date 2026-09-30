"""Optional cross-camera association: assigns `Track.global_id` from appearance, in memory only.

**Off by default** (`enabled=False` makes `assign` a pass-through). Why it exists: the
journey state machine (T05) is strongest when "picked up at the shelf cam" and "left
through the door cam" are the same person. Why it's off: with colour-histogram
appearance, similar clothing (staff uniforms, dark hoodies) merges different people,
and a wrong merge is worse for T05 than no merge. Turn it on for sites where the
cameras hand people off through a doorway/aisle with little overlap and T05 is using
`global_id`, after checking the merge rate on that site's footage.

Privacy (AGENTS.md rule 6): embeddings are held in a dict in this process, never
persisted or published, and each identity expires `ttl_s` (≤ 30 min) after it was last
seen. There is no face data; the default embedder sees clothing colour only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from scs.contracts import Track
from scs.perception.track import linear_assignment

MAX_TTL_S = 30 * 60.0


@dataclass
class _Identity:
    global_id: int
    feat: np.ndarray
    last_seen: float
    cameras: dict[str, int] = field(default_factory=dict)  # camera_id -> local track_id last seen


class CrossCameraAssociator:
    def __init__(
        self,
        enabled: bool = False,
        ttl_s: float = MAX_TTL_S,
        match_thresh: float = 0.15,
        min_track_frames: int = 5,
        feat_alpha: float = 0.9,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if ttl_s > MAX_TTL_S:
            raise ValueError("ttl_s must be <= 30 min (AGENTS.md rule 6)")
        self.enabled, self.ttl_s, self.match_thresh = enabled, ttl_s, match_thresh
        self.min_track_frames, self.feat_alpha = min_track_frames, feat_alpha
        self.clock = clock
        self._ids: dict[int, _Identity] = {}
        self._local: dict[tuple[str, int], int] = {}  # (camera, local id) -> global id
        self._pending: dict[tuple[str, int], tuple[int, np.ndarray, float]] = {}  # frames, feat sum, last ts
        self._next = 0

    def __len__(self) -> int:
        return len(self._ids)

    def expire(self, now: float) -> None:
        dead = [g for g, ident in self._ids.items() if now - ident.last_seen > self.ttl_s]
        for g in dead:
            del self._ids[g]
        self._local = {k: g for k, g in self._local.items() if g in self._ids}
        self._pending = {k: p for k, p in self._pending.items() if now - p[2] <= self.ttl_s}

    def assign(self, tracks: list[Track], feats: np.ndarray | None) -> list[Track]:
        """Tracks from ONE camera at one frame + their (N,D) appearance features → tracks with `global_id`."""
        if not self.enabled or not tracks or feats is None:
            return tracks
        now = self.clock() if self.clock else tracks[0].frame.ts
        self.expire(now)
        cam = tracks[0].frame.camera_id
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
        out: list[Track | None] = [None] * len(tracks)
        new_keys = []
        for i, t in enumerate(tracks):
            key = (cam, t.track_id)
            g = self._local.get(key)
            if g is not None:
                self._observe(g, cam, t.track_id, feats[i], now)
                out[i] = t.model_copy(update={"global_id": g})
                continue
            n, acc, _ = self._pending.get(key, (0, np.zeros_like(feats[i]), now))
            self._pending[key] = (n + 1, acc + feats[i], now)
            if n + 1 >= self.min_track_frames:  # wait for a stable appearance before matching
                new_keys.append(i)
        if new_keys:
            self._match_new(cam, tracks, new_keys, now)
            for i in new_keys:
                out[i] = tracks[i].model_copy(update={"global_id": self._local[(cam, tracks[i].track_id)]})
        return [o if o is not None else t for o, t in zip(out, tracks, strict=True)]

    def _observe(self, g: int, cam: str, local: int, feat: np.ndarray, now: float) -> None:
        ident = self._ids[g]
        ident.feat = self.feat_alpha * ident.feat + (1 - self.feat_alpha) * feat
        ident.feat /= np.linalg.norm(ident.feat) + 1e-12
        ident.last_seen = now
        ident.cameras[cam] = local

    def _match_new(self, cam: str, tracks: list[Track], idx: list[int], now: float) -> None:
        # Identities currently present on this camera can't be a second person here.
        busy = {self._local[(cam, t.track_id)] for t in tracks if (cam, t.track_id) in self._local}
        gallery = [g for g in self._ids if g not in busy]
        q = []
        for i in idx:
            _, acc, _ = self._pending.pop((cam, tracks[i].track_id))
            q.append(acc / (np.linalg.norm(acc) + 1e-12))
        qf = np.stack(q)
        matches: list[tuple[int, int]] = []
        unmatched = list(range(len(idx)))
        if gallery:
            gf = np.stack([self._ids[g].feat for g in gallery])
            matches, unmatched, _ = linear_assignment((1.0 - qf @ gf.T) / 2.0, self.match_thresh)
        for qi, gi in matches:
            g = gallery[gi]
            self._local[(cam, tracks[idx[qi]].track_id)] = g
            self._observe(g, cam, tracks[idx[qi]].track_id, qf[qi], now)
        for qi in unmatched:
            self._next += 1
            self._ids[self._next] = _Identity(self._next, qf[qi], now, {cam: tracks[idx[qi]].track_id})
            self._local[(cam, tracks[idx[qi]].track_id)] = self._next
