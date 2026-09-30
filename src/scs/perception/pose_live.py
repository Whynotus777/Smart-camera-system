"""The T04 live path for one camera: zone gate -> top-down pose -> causal smoothing -> crop export.

`LivePosePipeline.step` is called once per decoded frame, in order, and only ever
sees that frame plus state built from earlier ones (AGENTS.md rule 11). It returns
raw and smoothed poses with the observed/imputed mask; the caller publishes the
smoothed `Pose` to `Streams.POSES` (raw stays available for eval and debugging).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from scs.contracts import FrameRef, Track, Zone
from scs.perception.pose import TopDownPoseEstimator
from scs.perception.pose_export import CropExporter
from scs.perception.pose_gate import gate_tracks
from scs.perception.pose_smooth import LivePoseSmoother, SmoothedPose

if TYPE_CHECKING:
    import torch


@dataclass
class LivePosePipeline:
    estimator: TopDownPoseEstimator
    zones: list[Zone]
    smoother: LivePoseSmoother = field(default_factory=LivePoseSmoother)
    exporter: CropExporter = field(default_factory=CropExporter)  # disabled by default

    def step(
        self, frame: FrameRef, image: torch.Tensor | np.ndarray, tracks: list[Track]
    ) -> list[SmoothedPose]:
        gated = gate_tracks(tracks, self.zones)
        raw = self.estimator.estimate(frame, image, gated)
        out = [self.smoother.update(p) for p in raw]
        if self.exporter.enabled:
            host = image if isinstance(image, np.ndarray) else image.cpu().numpy()
            self.exporter(host, gated, raw)
        live_ids = {t.track_id for t in tracks if t.state != "lost"}
        for tid in [tid for tid in self.smoother.track_ids() if tid not in live_ids]:
            self.smoother.drop(tid)
        return out
