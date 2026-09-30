"""YOLO11-pose as a top-down `PoseBackend`: comparison model only. [license-risk]

Ultralytics code and weights are AGPL-3.0 (docs/DATA.md "Pretrained models"): R&D and
eval only, behind the same `PoseBackend` interface so it can be dropped without touching
callers. Nothing here imports `ultralytics`; the engine is built offline from an ONNX
exported with `yolo export format=onnx imgsz=256,192 dynamic=True`.

YOLO11-pose is a one-stage multi-person model. To use it top-down on the same crops as
RTMPose (so the comparison measures the model, not the cropping), each crop's output
candidates are scored by `conf x IoU(candidate box, expected track box)`, where the
expected box is the track box in crop pixels (the central 1/padding of the crop). That
picks the tracked person even when a neighbour is in the crop.

Output layout of the exported head: (N, 5 + 17*3, A) = cx, cy, w, h, person conf, then
x, y, visibility (already sigmoid) per keypoint, all in input pixels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scs.perception.pose import BBOX_PADDING
from scs.perception.pose_trt import TRTEngine


class YOLOPoseTRT:
    """`PoseBackend` for a YOLO11-pose TensorRT engine (RGB, 0..1 input)."""

    def __init__(
        self, engine_path: str | Path, model_id: str, color_order: str = "bgr", padding: float = BBOX_PADDING
    ) -> None:
        self.engine = TRTEngine(engine_path)
        _, h, w = self.engine.input_chw
        self.input_wh = (int(w), int(h))
        self.max_batch = self.engine.max_batch
        self.model_id = model_id
        self._flip = color_order == "bgr"
        # expected person box inside the crop: the track box, shrunk back by the padding
        ew, eh = w / padding, h / padding
        self._expected = torch.tensor([(w - ew) / 2, (h - eh) / 2, (w + ew) / 2, (h + eh) / 2])

    def infer(self, crops: torch.Tensor | np.ndarray) -> np.ndarray:
        return self.infer_gpu(torch.as_tensor(crops, device="cuda")).cpu().numpy()

    def infer_gpu(self, crops: torch.Tensor) -> torch.Tensor:
        x = crops.flip(1) if self._flip else crops
        (raw,) = self.engine(x / 255.0).values()
        raw = raw.float().transpose(1, 2)  # (N, A, 56)
        cx, cy, bw, bh, conf = raw[..., 0], raw[..., 1], raw[..., 2], raw[..., 3], raw[..., 4]
        boxes = torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), dim=-1)
        e = self._expected.to(raw.device)
        iw = (torch.minimum(boxes[..., 2], e[2]) - torch.maximum(boxes[..., 0], e[0])).clamp(min=0)
        ih = (torch.minimum(boxes[..., 3], e[3]) - torch.maximum(boxes[..., 1], e[1])).clamp(min=0)
        inter = iw * ih
        union = bw * bh + (e[2] - e[0]) * (e[3] - e[1]) - inter
        best = (conf * inter / union.clamp(min=1e-6)).argmax(dim=1)  # (N,)
        kp = raw[torch.arange(len(raw), device=raw.device), best, 5:].view(-1, 17, 3)
        return kp
