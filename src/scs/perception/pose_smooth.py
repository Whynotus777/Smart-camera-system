"""Temporal smoothing and gap filling for per-track pose (T04, AGENTS.md rule 11).

Live path = `LivePoseSmoother`: strictly causal. `update(pose)` sees one pose at a
time and its output depends only on that pose and earlier ones (a unit test proves
prefix invariance). It keeps raw and smoothed poses side by side.

- One-Euro filter (Casiez et al., CHI 2012) per keypoint coordinate. Its speed term is
  measured in box-heights per second so `beta` doesn't depend on how far the person is.
- Gap filling is limited by ELAPSED TIME (`max_gap_s`, default 0.5 s), never by frame
  count, so it behaves the same at 5, 10 or 15 fps and across dropped frames. A
  keypoint below `min_conf` inside the window is imputed by holding the last smoothed
  position, and its confidence decays linearly to 0 over the window. Every keypoint
  carries a mask entry: OBSERVED (from the model on this frame) or IMPUTED.
- The filter resets on a new `epoch` (source reconnect), on `seq` going backwards, or
  after a gap longer than `max_gap_s`.

`offline_interpolate_smooth` is NON-CAUSAL (it reads future frames: linear interpolation
across gaps and a centered window). Labeling/offline analysis only; never on the live
path, and eval must run the live version (rule 11).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from scs.contracts import Keypoint, Pose


class KpMask(IntEnum):
    OBSERVED = 0
    IMPUTED = 1


@dataclass(frozen=True)
class SmoothedPose:
    """Live smoother output: both versions plus the per-keypoint observed/imputed mask."""

    raw: Pose
    smoothed: Pose
    mask: tuple[KpMask, ...]


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


@dataclass
class OneEuro:
    """Vectorized One-Euro filter over an array of signals sharing one clock."""

    min_cutoff: float = 1.0  # Hz; lower = smoother when still
    beta: float = 0.7  # speed coefficient (per box-height/s); higher = less lag when moving
    d_cutoff: float = 1.0  # Hz, for the derivative
    x: np.ndarray | None = None
    dx: np.ndarray | None = None

    def reset(self) -> None:
        self.x = self.dx = None

    def __call__(self, value: np.ndarray, dt: float, speed_scale: float = 1.0) -> np.ndarray:
        if self.x is None or dt <= 0:
            self.x, self.dx = value.copy(), np.zeros_like(value)
            return self.x.copy()
        assert self.dx is not None
        dx = (value - self.x) / dt
        self.dx = self.dx + _alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx) / max(speed_scale, 1e-6)
        a = 1.0 / (1.0 + (1.0 / (2 * np.pi * cutoff)) / dt)
        self.x = self.x + a * (value - self.x)
        return self.x.copy()


def pose_time(p: Pose) -> float:
    """Clock used for elapsed time: host monotonic when present, else wall clock at decode."""
    return p.frame.ts_mono if p.frame.ts_mono is not None else p.frame.ts


def _box_height(kps: np.ndarray, conf_ok: np.ndarray) -> float:
    ys = kps[conf_ok, 1] if conf_ok.any() else kps[:, 1]
    return float(max(ys.max() - ys.min(), 1.0))


@dataclass
class _TrackState:
    filt: OneEuro
    epoch: int
    seq: int
    t_last: float
    last_xy: np.ndarray  # smoothed (17, 2)
    last_conf: np.ndarray  # conf of last observation (17,)
    t_obs: np.ndarray  # time each keypoint was last observed (17,)
    scale: float


@dataclass
class LivePoseSmoother:
    """Causal per-track smoother for one camera. Call `update` in frame order."""

    min_cutoff: float = 1.0
    beta: float = 0.7
    d_cutoff: float = 1.0
    min_conf: float = 0.3
    max_gap_s: float = 0.5
    _tracks: dict[int, _TrackState] = field(default_factory=dict)

    @property
    def tag(self) -> str:
        return f"oneeuro(mc={self.min_cutoff},b={self.beta},gap={self.max_gap_s}s)"

    def track_ids(self) -> list[int]:
        return list(self._tracks)

    def drop(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)

    def update(self, pose: Pose) -> SmoothedPose:
        kp = np.asarray(pose.keypoints, dtype=np.float64)
        xy, conf = kp[:, :2], kp[:, 2]
        t = pose_time(pose)
        obs = conf >= self.min_conf
        st = self._tracks.get(pose.track_id)
        if st is not None and (
            pose.frame.epoch != st.epoch or pose.frame.seq <= st.seq or t - st.t_last > self.max_gap_s
        ):
            st = None  # reconnect, reorder, or gap too long: start over
        if st is None:
            st = _TrackState(
                OneEuro(self.min_cutoff, self.beta, self.d_cutoff),
                pose.frame.epoch,
                pose.frame.seq,
                t,
                xy.copy(),
                conf.copy(),
                np.full(17, t),
                _box_height(xy, obs),
            )
            st.filt(xy.copy(), 0.0)
            self._tracks[pose.track_id] = st
            mask = np.zeros(17, dtype=int)
            return self._emit(pose, xy, conf, mask)

        dt = t - st.t_last
        age = t - st.t_obs
        impute = ~obs & (age <= self.max_gap_s)
        target = np.where(impute[:, None], st.last_xy, xy)
        st.scale = 0.9 * st.scale + 0.1 * _box_height(xy, obs)
        sm = st.filt(target, dt, speed_scale=st.scale)
        # Imputed points hold the last smoothed position (no drift); observed ones are filtered.
        sm = np.where(impute[:, None], st.last_xy, sm)
        out_conf = np.where(impute, st.last_conf * np.clip(1 - age / self.max_gap_s, 0, 1), conf)

        st.last_xy = sm.copy()
        st.last_conf = np.where(obs, conf, st.last_conf)
        st.t_obs = np.where(obs, t, st.t_obs)
        st.t_last, st.seq = t, pose.frame.seq
        return self._emit(pose, sm, out_conf, impute.astype(int))

    def _emit(self, raw: Pose, xy: np.ndarray, conf: np.ndarray, mask: np.ndarray) -> SmoothedPose:
        kps: list[Keypoint] = [(float(x), float(y), float(c)) for (x, y), c in zip(xy, conf, strict=True)]
        sm = raw.model_copy(update={"keypoints": kps, "model_id": f"{raw.model_id}+{self.tag}"})
        return SmoothedPose(raw=raw, smoothed=sm, mask=tuple(KpMask(int(m)) for m in mask))


def offline_interpolate_smooth(
    poses: Sequence[Pose], window: int = 8, max_gap_s: float = 0.5, min_conf: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """NON-CAUSAL (reads future frames). Offline labeling / dataset matching only.

    One track's poses in time order -> (T, 17, 3) keypoints and (T, 17) mask. Keypoints
    below `min_conf` are linearly interpolated between the surrounding observed values
    when both lie within `max_gap_s`, then a centered `window`-frame moving average is
    applied to x, y. Mirrors PoseLift's "linear interpolation + 8-frame window".
    """
    if not poses:
        return np.zeros((0, 17, 3)), np.zeros((0, 17), dtype=int)
    t = np.array([pose_time(p) for p in poses])
    kp = np.array([p.keypoints for p in poses], dtype=np.float64)
    mask = np.zeros(kp.shape[:2], dtype=int)
    for j in range(17):
        obs = np.flatnonzero(kp[:, j, 2] >= min_conf)
        for a, b in zip(obs[:-1], obs[1:], strict=True):
            if b - a > 1 and t[b] - t[a] <= max_gap_s:
                u = (t[a + 1 : b] - t[a]) / (t[b] - t[a])
                kp[a + 1 : b, j, :2] = kp[a, j, :2] + u[:, None] * (kp[b, j, :2] - kp[a, j, :2])
                kp[a + 1 : b, j, 2] = np.minimum(kp[a, j, 2], kp[b, j, 2])
                mask[a + 1 : b, j] = KpMask.IMPUTED
    out = kp.copy()
    half_lo, half_hi = (window - 1) // 2, window // 2
    for i in range(len(kp)):
        lo, hi = max(0, i - half_lo), min(len(kp), i + half_hi + 1)
        out[i, :, :2] = kp[lo:hi, :, :2].mean(axis=0)
    return out, mask
