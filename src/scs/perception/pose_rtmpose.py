"""RTMPose (OpenMMLab, Apache-2.0 code) as a `PoseBackend`: the primary T04 model.

Runtime needs only TensorRT + torch; mmpose/mmcv (compiled ops, sm_120 risk) are not
imported. The ONNX comes from OpenMMLab's official SDK export, whose `pipeline.json`
fixes the preprocessing we reproduce here:
TopDownGetBboxCenterScale(padding=1.25) -> TopDownAffine(192x256) -> BGR->RGB ->
Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]).

Decoding is mmpose `get_simcc_maximal`: x = argmax(simcc_x) / split_ratio (2.0), same for
y; confidence = min(max simcc_x, max simcc_y). SimCC maxima aren't probabilities and can
exceed 1, so they're clipped to [0, 1] to satisfy `Pose` (contracts.Keypoint).

Weight lineage (docs/DATA.md): the body7 checkpoints are trained on 7 public datasets,
some with research-only terms, so they are R&D until that is cleared; the code license
alone doesn't make the weights shippable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scs.perception.pose_trt import TRTEngine

MEAN_RGB = (123.675, 116.28, 103.53)
STD_RGB = (58.395, 57.12, 57.375)
SIMCC_SPLIT = 2.0


def simcc_decode(simcc_x: torch.Tensor, simcc_y: torch.Tensor) -> torch.Tensor:
    """(N, K, Wx), (N, K, Wy) -> (N, K, 3) (x, y in input px, conf = min of the two maxima)."""
    vx, ix = simcc_x.float().max(dim=-1)
    vy, iy = simcc_y.float().max(dim=-1)
    conf = torch.minimum(vx, vy)
    return torch.stack((ix.float() / SIMCC_SPLIT, iy.float() / SIMCC_SPLIT, conf), dim=-1)


class RTMPoseTRT:
    """`PoseBackend` running an RTMPose SimCC TensorRT engine."""

    def __init__(self, engine_path: str | Path, model_id: str, color_order: str = "bgr") -> None:
        self.engine = TRTEngine(engine_path)
        c, h, w = self.engine.input_chw
        self.input_wh = (int(w), int(h))
        self.max_batch = self.engine.max_batch
        self.model_id = model_id
        self._flip = color_order == "bgr"
        self._mean = torch.tensor(MEAN_RGB).view(1, 3, 1, 1)
        self._std = torch.tensor(STD_RGB).view(1, 3, 1, 1)

    def infer(self, crops: torch.Tensor | np.ndarray) -> np.ndarray:
        return self.infer_gpu(torch.as_tensor(crops, device="cuda")).cpu().numpy()

    def infer_gpu(self, crops: torch.Tensor) -> torch.Tensor:
        if self._mean.device != crops.device:
            self._mean, self._std = self._mean.to(crops.device), self._std.to(crops.device)
        x = crops.flip(1) if self._flip else crops
        out = self.engine((x - self._mean) / self._std)
        (sx,) = (v for v in out.values() if v.shape[-1] == self.input_wh[0] * SIMCC_SPLIT)
        (sy,) = (v for v in out.values() if v.shape[-1] == self.input_wh[1] * SIMCC_SPLIT)
        return simcc_decode(sx, sy)
