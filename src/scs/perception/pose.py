"""Top-down COCO17 pose on high-resolution main-stream crops (T04).

Why top-down on main-stream crops (ARCHITECTURE D1, ADR 0002): at 5 m a hand is
~6 px on the 640x360 sub-stream but ~25 px on the 2560x1440 main stream. Track
boxes are already main-stream pixels, so we crop the decoded main-stream frame
with the box as-is (no rescaling) and map keypoints back into the same pixels.

This module holds everything model-agnostic:
- crop geometry (mmpose-compatible center/scale + aspect fix + 1.25 padding),
- a batched GPU crop (one `grid_sample` for the whole batch, zero padding outside
  the frame, like `cv2.warpAffine(borderValue=0)`),
- `TopDownPoseEstimator`, which implements `scs.perception.base.PoseEstimator` on
  top of any `PoseBackend` (RTMPose / YOLO11-pose TensorRT engines, or a fake in tests).

Frame identity (ARCHITECTURE D10): every track handed to `estimate` must carry the
same `(camera_id, epoch, seq)` as the frame being cropped; a mismatch means the box
and the pixels come from different images, so we raise instead of guessing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np

from scs.contracts import BBox, FrameRef, Pose, Track

if TYPE_CHECKING:
    import torch

COCO_WRISTS = (9, 10)
BBOX_PADDING = 1.25  # mmpose GetBBoxCenterScale default used by RTMPose top-down configs


class FrameIdentityError(ValueError):
    """A track's frame identity doesn't match the image being cropped."""


@dataclass(frozen=True)
class CropGeometry:
    """Affine crop of one box: input pixel (u, v) <- frame pixel (cx + (u - W/2) * sx, ...)."""

    center: tuple[float, float]
    scale: tuple[float, float]  # (w, h) of the crop window in frame pixels, aspect = input aspect
    input_wh: tuple[int, int]

    def to_frame(self, pts: np.ndarray) -> np.ndarray:
        """Input-pixel points (..., 2) -> main-stream frame pixels."""
        w, h = self.input_wh
        out = np.empty_like(pts, dtype=np.float64)
        out[..., 0] = (pts[..., 0] - w / 2) * (self.scale[0] / w) + self.center[0]
        out[..., 1] = (pts[..., 1] - h / 2) * (self.scale[1] / h) + self.center[1]
        return out

    def to_input(self, pts: np.ndarray) -> np.ndarray:
        """Main-stream frame pixels (..., 2) -> input pixels (inverse of `to_frame`)."""
        w, h = self.input_wh
        out = np.empty_like(pts, dtype=np.float64)
        out[..., 0] = (pts[..., 0] - self.center[0]) * (w / self.scale[0]) + w / 2
        out[..., 1] = (pts[..., 1] - self.center[1]) * (h / self.scale[1]) + h / 2
        return out


def crop_geometry(bbox: BBox, input_wh: tuple[int, int], padding: float = BBOX_PADDING) -> CropGeometry:
    """Box -> crop window: pad by `padding`, then grow the short side to the input aspect (no squashing)."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * padding, (y2 - y1) * padding
    aspect = input_wh[0] / input_wh[1]
    if bw > bh * aspect:
        bh = bw / aspect
    else:
        bw = bh * aspect
    return CropGeometry((cx, cy), (bw, bh), input_wh)


def crop_batch(image: torch.Tensor, geoms: Sequence[CropGeometry]) -> torch.Tensor:
    """Crop + resize all boxes from one HWC uint8 frame in a single `grid_sample` call.

    Returns float32 (N, 3, H, W) in the image's channel order and 0..255 range.
    Pixels outside the frame are 0. Bilinear, pixel centers at integer coords (matches
    `cv2.warpAffine` with the same matrix to within interpolation rounding).
    """
    import torch
    import torch.nn.functional as F  # noqa: N812

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HWC image with 3 channels, got {tuple(image.shape)}")
    ih, iw = int(image.shape[0]), int(image.shape[1])
    w, h = geoms[0].input_wh
    dev = image.device
    # Only the region the crops touch is converted to float (+2 px so bilinear taps at the
    # window edge still see real neighbours); everything outside it is outside every window.
    x0 = max(int(min(gm.center[0] - gm.scale[0] / 2 for gm in geoms)) - 2, 0)
    y0 = max(int(min(gm.center[1] - gm.scale[1] / 2 for gm in geoms)) - 2, 0)
    x1 = min(int(max(gm.center[0] + gm.scale[0] / 2 for gm in geoms)) + 3, iw)
    y1 = min(int(max(gm.center[1] + gm.scale[1] / 2 for gm in geoms)) + 3, ih)
    if x1 <= x0 or y1 <= y0:
        return torch.zeros((len(geoms), 3, h, w), dtype=torch.float32, device=dev)
    roi = image[y0:y1, x0:x1]
    rh, rw = y1 - y0, x1 - x0
    g = torch.tensor([[*gm.center, *gm.scale] for gm in geoms], dtype=torch.float32, device=dev)
    u = torch.arange(w, dtype=torch.float32, device=dev)
    v = torch.arange(h, dtype=torch.float32, device=dev)
    xs = (u[None, :] - w / 2) * (g[:, 2:3] / w) + g[:, 0:1] - x0  # (N, W) roi px
    ys = (v[None, :] - h / 2) * (g[:, 3:4] / h) + g[:, 1:2] - y0  # (N, H)
    gx = (2 * xs + 1) / rw - 1  # align_corners=False normalization
    gy = (2 * ys + 1) / rh - 1
    # All crops as one tall (N*H, W) grid over a single source: no per-crop copy of the frame.
    grid = torch.stack((gx[:, None, :].expand(-1, h, -1), gy[:, :, None].expand(-1, -1, w)), dim=-1)
    src = roi.permute(2, 0, 1)[None].float()  # (1, 3, rh, rw)
    out = F.grid_sample(
        src, grid.reshape(1, -1, w, 2), mode="bilinear", padding_mode="zeros", align_corners=False
    )  # (1, 3, N*H, W)
    return out.view(3, len(geoms), h, w).transpose(0, 1).contiguous()


def crop_batch_np(image: np.ndarray, geoms: Sequence[CropGeometry]) -> np.ndarray:
    """CPU/numpy twin of `crop_batch` (same sampling grid, bilinear, zeros outside)."""
    ih, iw = image.shape[:2]
    w, h = geoms[0].input_wh
    out = np.zeros((len(geoms), 3, h, w), dtype=np.float32)
    img = image.astype(np.float32)
    for n, gm in enumerate(geoms):
        xs = (np.arange(w) - w / 2) * (gm.scale[0] / w) + gm.center[0]
        ys = (np.arange(h) - h / 2) * (gm.scale[1] / h) + gm.center[1]
        x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
        fx, fy = (xs - x0)[None, :, None], (ys - y0)[:, None, None]

        def tap(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
            ok = ((yy >= 0) & (yy < ih))[:, None] & ((xx >= 0) & (xx < iw))[None, :]
            v = img[np.clip(yy, 0, ih - 1)[:, None], np.clip(xx, 0, iw - 1)[None, :]]
            return v * ok[..., None]

        v = (
            tap(y0, x0) * (1 - fx) * (1 - fy)
            + tap(y0, x0 + 1) * fx * (1 - fy)
            + tap(y0 + 1, x0) * (1 - fx) * fy
            + tap(y0 + 1, x0 + 1) * fx * fy
        )
        out[n] = v.transpose(2, 0, 1)
    return out


class PoseBackend(Protocol):
    """One model's forward pass on already-cropped inputs.

    `infer` gets float32 (N, 3, H, W) crops (a torch tensor on the frame's device, or a
    numpy array when the frame is numpy) in 0..255, channel order `color_order` of the
    SOURCE frame, and returns (N, 17, 3) keypoints (x, y in input pixels, conf in [0, 1]).
    N may exceed `max_batch`; the estimator chunks before calling.
    """

    model_id: str
    input_wh: tuple[int, int]
    max_batch: int

    def infer(self, crops: torch.Tensor | np.ndarray) -> np.ndarray: ...


class TopDownPoseEstimator:
    """`PoseEstimator` (base.py) over any `PoseBackend`; emits keypoints in main-stream pixels.

    `min_box_px`: tracks whose box is smaller than this (short side) are skipped; a
    top-down model can't recover wrists from a handful of pixels, and T05 treats a
    missing pose as "unknown", which is the honest answer.
    """

    def __init__(
        self,
        backend: PoseBackend,
        padding: float = BBOX_PADDING,
        min_box_px: float = 8.0,
        stream_check: Literal["strict", "off"] = "strict",
    ) -> None:
        self.backend = backend
        self.padding = padding
        self.min_box_px = min_box_px
        self.stream_check = stream_check

    @property
    def model_id(self) -> str:
        return self.backend.model_id

    def check_inputs(
        self, frame: FrameRef, image: torch.Tensor | np.ndarray, tracks: Sequence[Track]
    ) -> None:
        """Raise if boxes and pixels can't be proven to come from the same main-stream image."""
        for t in tracks:
            if t.frame.identity != frame.identity:
                raise FrameIdentityError(
                    f"track {t.track_id} is from {t.frame.identity}, image is {frame.identity}"
                )
        if self.stream_check == "strict":
            if frame.stream != "main" or frame.transform is not None:
                raise FrameIdentityError(
                    f"pose needs the untransformed main-stream frame, got stream={frame.stream!r} "
                    f"transform={frame.transform!r}"
                )
            if tuple(image.shape[:2]) != (frame.height, frame.width):
                raise FrameIdentityError(
                    f"image {tuple(image.shape[:2])} != main-stream size {(frame.height, frame.width)}"
                )

    def estimate(self, frame: FrameRef, image: torch.Tensor | np.ndarray, tracks: list[Track]) -> list[Pose]:
        """Crop from `image` (HWC uint8 main-stream frame; torch on GPU for speed, numpy works)."""
        return self.estimate_many([(frame, image, tracks)])[0]

    def estimate_many(
        self, items: Sequence[tuple[FrameRef, torch.Tensor | np.ndarray, list[Track]]]
    ) -> list[list[Pose]]:
        """Micro-batch across frames/cameras (ARCHITECTURE D9): crops from every item share
        one backend batch (chunked at `max_batch`). Returns one list of poses per item."""
        crops, owners = [], []
        for n, (frame, image, tracks) in enumerate(items):
            self.check_inputs(frame, image, tracks)
            keep = [t for t in tracks if min(t.bbox[2] - t.bbox[0], t.bbox[3] - t.bbox[1]) >= self.min_box_px]
            if not keep:
                continue
            geoms = [crop_geometry(t.bbox, self.backend.input_wh, self.padding) for t in keep]
            crops.append(
                crop_batch_np(image, geoms) if isinstance(image, np.ndarray) else crop_batch(image, geoms)
            )
            owners += [(n, t, g) for t, g in zip(keep, geoms, strict=True)]
        out: list[list[Pose]] = [[] for _ in items]
        if not owners:
            return out
        batch: Any
        if isinstance(crops[0], np.ndarray):
            batch = np.concatenate(crops)
        else:
            import torch

            batch = torch.cat(crops)  # type: ignore[arg-type]
        mb = self.backend.max_batch
        kps = np.concatenate([self.backend.infer(batch[i : i + mb]) for i in range(0, len(batch), mb)])
        for (n, t, g), k in zip(owners, kps, strict=True):
            out[n].append(self._to_pose(items[n][0], t, g, k))
        return out

    def _to_pose(self, frame: FrameRef, track: Track, geom: CropGeometry, kp: np.ndarray) -> Pose:
        xy = geom.to_frame(kp[:, :2])
        conf = np.clip(kp[:, 2], 0.0, 1.0)
        return Pose(
            frame=frame,
            track_id=track.track_id,
            model_id=self.model_id,
            keypoints=[(float(x), float(y), float(c)) for (x, y), c in zip(xy, conf, strict=True)],
        )
