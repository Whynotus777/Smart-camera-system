"""Raw decoder output → the image a consumer asked for (RGB numpy or RGB CUDA tensor).

Decoders hand over NV12 (NVDEC's native format, 1.5 bytes/px) or I420 (software
decoders). Conversion to RGB is deferred to the consumer's thread (`StreamSource.frames`)
so a frame evicted from the analytics queue costs nothing beyond the decode.

- `"numpy"`: HxWx3 uint8 RGB via OpenCV (BT.601 limited range, OpenCV's NV12 convention).
- `"torch"`: HxWx3 uint8 RGB CUDA tensor; NV12 is uploaded (1.5 B/px over PCIe rather
  than 3) and converted on the GPU with the same BT.601 coefficients.
- `"raw"`: the `RawImage` itself, for callers that batch conversion.

Known gap: NVDEC output currently round-trips through host memory (decoder → NV12 host
buffer → GPU). Zero-copy CUDA buffers from GStreamer need the GstCuda bindings mapped to
DLPack; that's a follow-up if T12's budget says the PCIe copy matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import torch

Output = Literal["numpy", "torch", "raw"]
PixFmt = Literal["nv12", "i420", "rgb"]


@dataclass(frozen=True, slots=True)
class RawImage:
    """Tightly packed planes: NV12/I420 as a (H*3/2, W) uint8 array; RGB as (H, W, 3)."""

    data: np.ndarray
    fmt: PixFmt
    width: int
    height: int


def pack_planes(buf: memoryview | bytes | np.ndarray, fmt: PixFmt, width: int, height: int,
                strides: tuple[int, ...], offsets: tuple[int, ...]) -> RawImage:
    """Copy a decoder buffer with (possibly padded) strides into a tightly packed RawImage.

    `buf` may be a view into mapped decoder memory: every output plane is a fresh copy,
    so nothing references `buf` after this returns.
    """
    src = buf if isinstance(buf, np.ndarray) else np.frombuffer(buf, dtype=np.uint8)
    if fmt == "rgb":
        rows = src[offsets[0]: offsets[0] + strides[0] * height].reshape(height, strides[0])
        return RawImage(rows[:, : width * 3].reshape(height, width, 3).copy(), fmt, width, height)
    out = np.empty((height * 3 // 2, width), dtype=np.uint8)
    y = src[offsets[0]: offsets[0] + strides[0] * height].reshape(height, strides[0])
    out[:height] = y[:, :width]
    ch = height // 2
    if fmt == "nv12":
        uv = src[offsets[1]: offsets[1] + strides[1] * ch].reshape(ch, strides[1])
        out[height:] = uv[:, :width]
    else:  # i420: U plane then V plane, each (H/2, W/2), stored as W/2-wide rows
        cw = width // 2
        u = src[offsets[1]: offsets[1] + strides[1] * ch].reshape(ch, strides[1])[:, :cw]
        v = src[offsets[2]: offsets[2] + strides[2] * ch].reshape(ch, strides[2])[:, :cw]
        out[height:].reshape(-1)[: ch * cw] = u.reshape(-1)
        out[height:].reshape(-1)[ch * cw: 2 * ch * cw] = v.reshape(-1)
    return RawImage(out, fmt, width, height)


def convert(raw: RawImage, output: Output) -> Any:
    if output == "raw":
        return raw
    if output == "numpy":
        return to_rgb_numpy(raw)
    if output == "torch":
        return to_rgb_torch(raw)
    raise ValueError(f"unknown output {output!r}")


def to_rgb_numpy(raw: RawImage) -> np.ndarray:
    if raw.fmt == "rgb":
        return raw.data
    import cv2

    code = cv2.COLOR_YUV2RGB_NV12 if raw.fmt == "nv12" else cv2.COLOR_YUV2RGB_I420
    return cv2.cvtColor(raw.data, code)


_TORCH_COEF: dict[Any, Any] = {}


def to_rgb_torch(raw: RawImage, device: str = "cuda") -> torch.Tensor:
    import torch

    if raw.fmt == "rgb":
        return torch.from_numpy(raw.data).to(device, non_blocking=True)
    h, w = raw.height, raw.width
    t = torch.from_numpy(raw.data).pin_memory().to(device, non_blocking=True)
    y = t[:h].float()
    if raw.fmt == "nv12":
        uv = t[h:].view(h // 2, w // 2, 2).float()
        u, v = uv[..., 0], uv[..., 1]
    else:
        flat = t[h:].reshape(-1)
        n = (h // 2) * (w // 2)
        u = flat[:n].view(h // 2, w // 2).float()
        v = flat[n: 2 * n].view(h // 2, w // 2).float()
    u = u.repeat_interleave(2, 0).repeat_interleave(2, 1) - 128.0
    v = v.repeat_interleave(2, 0).repeat_interleave(2, 1) - 128.0
    c = 1.164 * (y - 16.0)
    r = c + 1.596 * v
    g = c - 0.391 * u - 0.813 * v
    b = c + 2.018 * u
    return torch.stack((r, g, b), dim=-1).clamp_(0, 255).to(torch.uint8)
