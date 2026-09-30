"""Ultralytics YOLO11 person detector, TensorRT FP16. **R&D only: AGPL-3.0** [license-risk].

Why it's here: YOLO11 is the velocity baseline and what most retail-CV examples use, so
it's the reference the Apache-2.0 option (detect_dfine.py) has to beat or match before
we ship. The `ultralytics` package is needed only to *export* ONNX; inference runs the
TRT engine through `TrtRunner`, and NMS is torchvision's, so swapping it out removes no
runtime code path.

Raw head output is (B, 4+nc, anchors): cx, cy, w, h in input pixels then per-class
scores. We keep class 0 ("person" in COCO and in our 1-class fine-tunes).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scs.perception.detect import BatchedDetector, DetectorConfig
from scs.perception.detect_trt import BatchProfile, TrtRunner, build_engine

if TYPE_CHECKING:
    from torch import Tensor

MODELS = Path("models")


def export_yolo11(
    weights: str = "yolo11s.pt", out_dir: Path = MODELS / "yolo11s", profile: BatchProfile | None = None
) -> Path:
    """.pt → ONNX (dynamic batch, opset 17) → TRT FP16 engine. Returns the engine path."""
    from ultralytics import YOLO

    profile = profile or BatchProfile()
    out_dir.mkdir(parents=True, exist_ok=True)
    pt = Path(weights)
    if not pt.exists() and not pt.is_absolute() and (out_dir / pt.name).exists():
        pt = out_dir / pt.name
    model = YOLO(str(pt))  # downloads official weights on first use (AGPL-3.0)
    if not pt.is_absolute() and pt.parent == Path(".") and pt.exists():  # keep weights under models/
        pt = pt.rename(out_dir / pt.name)
        model = YOLO(str(pt))
    onnx = Path(
        model.export(
            format="onnx",
            dynamic=True,
            imgsz=profile.height,
            opset=17,
            simplify=True,
            half=False,
            batch=profile.max_batch,
        )
    )
    onnx = onnx.rename(out_dir / onnx.name) if onnx.parent != out_dir else onnx
    engine = out_dir / f"{onnx.stem}_fp16_b{profile.max_batch}.engine"
    build_engine(
        onnx,
        engine,
        profile,
        extra={"model": "ultralytics-yolo11", "weights": str(pt), "license": "AGPL-3.0", "use": "R&D"},
    )
    return engine


class Yolo11Detector(BatchedDetector):
    def __init__(
        self,
        engine: str | Path = MODELS / "yolo11s" / "yolo11s_fp16_b16.engine",
        cfg: DetectorConfig | None = None,
        nms_iou: float = 0.6,
    ) -> None:
        super().__init__(cfg)
        self.runner = TrtRunner(Path(engine), self.cfg.device)
        self.nms_iou = nms_iou
        self.model_id = f"yolo11:{Path(engine).stem}"

    def _infer(self, x: Tensor) -> list[np.ndarray]:
        import torch
        from torchvision.ops import nms

        out = next(iter(self.runner(x).values())).float()  # (B, 4+nc, A)
        res = []
        for p in out:
            scores = p[4]
            keep = scores >= self.cfg.score_thresh
            if not keep.any():
                res.append(np.zeros((0, 5), np.float32))
                continue
            cx, cy, w, h = p[0, keep], p[1, keep], p[2, keep], p[3, keep]
            boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
            s = scores[keep]
            k = nms(boxes, s, self.nms_iou)[: self.cfg.max_dets]
            res.append(torch.cat([boxes[k], s[k, None]], 1).cpu().numpy())
        return res
