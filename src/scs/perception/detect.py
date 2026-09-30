"""Person detection shared code: input transform, box mapping, batched detector base.

Why this module exists
- ADR 0002: detectors run on a ~640 px input but every box that leaves T03 is in
  main-stream full-resolution pixels. `InputTransform` records exactly how a frame was
  resized/padded so the mapping back is exact and unit-tested, and its `tag` goes into
  `FrameRef.transform` so downstream stages can see what the detector looked at.
- ARCHITECTURE D6: YOLO11 (AGPL) and an Apache-2.0 detector sit behind the same
  `Detector` Protocol. `BatchedDetector` holds everything they share (preprocessing on
  the GPU, score/size filtering, conversion to `Detection`), so the two implementations
  only differ in the engine call and the raw-output decoding.

Image convention (see "Blockers" in the T03 PR: T02 owns this): frames are uint8,
HxWx3, **RGB**, as numpy arrays or torch tensors on any device. `channel_order="bgr"`
covers OpenCV-decoded input. If the image is smaller than `FrameRef.width/height`
(e.g. the sub-stream was decoded), boxes are scaled up to the main-stream size.

torch is imported lazily so the pure-geometry parts import on CPU-only CI.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from scs.contracts import Detection, FrameRef

if TYPE_CHECKING:
    import torch
    from torch import Tensor

ResizeMode = Literal["letterbox", "stretch"]


@dataclass(frozen=True)
class InputTransform:
    """Maps detector-input pixels ↔ source-image pixels ↔ main-stream pixels.

    detector_xy = source_xy * scale + pad;  main_xy = source_xy * main_scale.
    """

    src_w: int
    src_h: int
    dst_w: int
    dst_h: int
    scale_x: float
    scale_y: float
    pad_x: float
    pad_y: float
    main_w: int
    main_h: int
    mode: ResizeMode

    @classmethod
    def fit(
        cls,
        src_w: int,
        src_h: int,
        dst: int | tuple[int, int],
        mode: ResizeMode = "letterbox",
        main_wh: tuple[int, int] | None = None,
    ) -> InputTransform:
        dst_w, dst_h = (dst, dst) if isinstance(dst, int) else dst
        if mode == "letterbox":
            s = min(dst_w / src_w, dst_h / src_h)
            new_w, new_h = round(src_w * s), round(src_h * s)
            sx, sy = new_w / src_w, new_h / src_h
            px, py = (dst_w - new_w) // 2, (dst_h - new_h) // 2
        else:
            sx, sy, px, py = dst_w / src_w, dst_h / src_h, 0, 0
        mw, mh = main_wh or (src_w, src_h)
        return cls(src_w, src_h, dst_w, dst_h, sx, sy, float(px), float(py), mw, mh, mode)

    @property
    def resized_wh(self) -> tuple[int, int]:
        return round(self.src_w * self.scale_x), round(self.src_h * self.scale_y)

    @property
    def tag(self) -> str:
        """Value for `FrameRef.transform`, e.g. ``resize640x640:letterbox``."""
        return f"resize{self.dst_w}x{self.dst_h}:{self.mode}"

    def to_main(self, boxes: np.ndarray) -> np.ndarray:
        """(N,4) xyxy detector-input pixels → main-stream pixels, clipped to the frame."""
        b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
        b[:, [0, 2]] = (b[:, [0, 2]] - self.pad_x) / self.scale_x * (self.main_w / self.src_w)
        b[:, [1, 3]] = (b[:, [1, 3]] - self.pad_y) / self.scale_y * (self.main_h / self.src_h)
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, self.main_w)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, self.main_h)
        return b

    def to_input(self, boxes: np.ndarray) -> np.ndarray:
        """(N,4) xyxy main-stream pixels → detector-input pixels (inverse of `to_main`, unclipped)."""
        b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
        b[:, [0, 2]] = b[:, [0, 2]] * (self.src_w / self.main_w) * self.scale_x + self.pad_x
        b[:, [1, 3]] = b[:, [1, 3]] * (self.src_h / self.main_h) * self.scale_y + self.pad_y
        return b


def _as_chw_float(image: Any, device: torch.device, channel_order: str) -> Tensor:
    """HxWx3 uint8 (numpy or tensor, RGB/BGR) → 3xHxW float32 RGB in [0, 255] on `device`."""
    import torch

    t = image if isinstance(image, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(image))
    t = t.to(device, non_blocking=True)
    if t.ndim != 3:
        raise ValueError(f"expected an HxWx3 or 3xHxW image, got shape {tuple(t.shape)}")
    if t.shape[-1] == 3 and t.shape[0] != 3:
        t = t.permute(2, 0, 1)
    if channel_order == "bgr":
        t = t.flip(0)
    return t.float()


def preprocess_batch(
    batch: Sequence[tuple[FrameRef, Any]],
    size: int | tuple[int, int],
    mode: ResizeMode,
    device: torch.device,
    channel_order: str = "rgb",
    pad_value: float = 114.0,
) -> tuple[Tensor, list[InputTransform]]:
    """Resize/pad every frame on `device` → (B,3,H,W) float in [0,1] + one transform per frame."""
    import torch
    import torch.nn.functional as F  # noqa: N812

    dst_w, dst_h = (size, size) if isinstance(size, int) else size
    out = torch.full((len(batch), 3, dst_h, dst_w), pad_value, device=device, dtype=torch.float32)
    transforms = []
    for i, (ref, img) in enumerate(batch):
        chw = _as_chw_float(img, device, channel_order)
        h, w = int(chw.shape[1]), int(chw.shape[2])
        tf = InputTransform.fit(w, h, (dst_w, dst_h), mode, main_wh=(ref.width, ref.height))
        rw, rh = tf.resized_wh
        resized = F.interpolate(
            chw[None], size=(rh, rw), mode="bilinear", align_corners=False, antialias=True
        )[0]
        px, py = int(tf.pad_x), int(tf.pad_y)
        out[i, :, py : py + rh, px : px + rw] = resized
        transforms.append(tf)
    return out.div_(255.0), transforms


@dataclass
class DetectorConfig:
    input_size: int = 640
    resize_mode: ResizeMode = "letterbox"
    score_thresh: float = 0.1  # low on purpose: ByteTrack uses the low-score band
    max_dets: int = 100
    min_box_px: float = 4.0  # main-stream pixels, either side
    channel_order: Literal["rgb", "bgr"] = "rgb"
    device: str = "cuda"


class BatchedDetector:
    """Shared `Detector` plumbing. Subclasses implement `_infer` on a preprocessed batch.

    `_infer` returns, per image, an (N,5) float array ``x1,y1,x2,y2,score`` of **person**
    boxes in detector-input pixels. Everything else (mapping to main-stream pixels,
    filtering, `Detection` construction) happens here so both detectors behave the same.
    """

    model_id: str = "base"

    def __init__(self, cfg: DetectorConfig | None = None) -> None:
        self.cfg = cfg or DetectorConfig()

    def _infer(self, x: Tensor) -> list[np.ndarray]:
        raise NotImplementedError

    def detect(self, batch: list[tuple[FrameRef, Any]]) -> list[list[Detection]]:
        if not batch:
            return []
        import torch

        x, tfs = preprocess_batch(
            batch,
            self.cfg.input_size,
            self.cfg.resize_mode,
            torch.device(self.cfg.device),
            self.cfg.channel_order,
        )
        raw = self._infer(x)
        return [to_detections(ref, r, tf, self.cfg) for (ref, _), r, tf in zip(batch, raw, tfs, strict=True)]


def to_detections(ref: FrameRef, raw: np.ndarray, tf: InputTransform, cfg: DetectorConfig) -> list[Detection]:
    """(N,5) detector-input boxes+scores → `Detection`s in main-stream pixels, best first."""
    if raw.size == 0:
        return []
    raw = raw[raw[:, 4] >= cfg.score_thresh]
    raw = raw[np.argsort(-raw[:, 4])][: cfg.max_dets]
    boxes = tf.to_main(raw[:, :4])
    ok = ((boxes[:, 2] - boxes[:, 0]) >= cfg.min_box_px) & ((boxes[:, 3] - boxes[:, 1]) >= cfg.min_box_px)
    frame = ref if ref.transform == tf.tag else ref.model_copy(update={"transform": tf.tag})
    return [
        Detection(
            frame=frame,
            bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
            score=float(min(max(s, 0.0), 1.0)),
        )
        for b, s in zip(boxes[ok], raw[ok, 4], strict=True)
    ]


def make_detector(name: str, **kwargs: Any) -> BatchedDetector:
    """Factory used by the benchmark/eval CLIs: ``yolo11`` (R&D, AGPL) or ``dfine`` (Apache-2.0)."""
    if name.startswith("yolo"):
        from scs.perception.detect_yolo import Yolo11Detector

        return Yolo11Detector(**kwargs)
    if name.startswith("dfine"):
        from scs.perception.detect_dfine import DFineDetector

        return DFineDetector(**kwargs)
    raise ValueError(f"unknown detector {name!r}")


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of (N,4) and (M,4) xyxy boxes."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


__all__ = [
    "BatchedDetector",
    "DetectorConfig",
    "InputTransform",
    "iou_matrix",
    "make_detector",
    "preprocess_batch",
    "to_detections",
]
