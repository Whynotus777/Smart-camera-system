"""NV12/I420 packing and RGB conversion: numpy (OpenCV) and CUDA (torch) paths agree."""

from __future__ import annotations

import numpy as np
import pytest

from scs.ingest.convert import RawImage, pack_planes, to_rgb_numpy, to_rgb_torch


def _nv12(w: int = 64, h: int = 32, seed: int = 0) -> RawImage:
    rng = np.random.default_rng(seed)
    return RawImage(rng.integers(16, 236, size=(h * 3 // 2, w), dtype=np.uint8), "nv12", w, h)


def test_pack_planes_strips_stride_padding():
    w, h, stride = 6, 4, 8
    y = np.arange(h * stride, dtype=np.uint8).reshape(h, stride)
    uv = (100 + np.arange(h // 2 * stride, dtype=np.uint8)).reshape(h // 2, stride)
    buf = np.concatenate([y.ravel(), uv.ravel()]).tobytes()
    img = pack_planes(buf, "nv12", w, h, (stride, stride), (0, h * stride))
    assert img.data.shape == (6, 6)
    assert (img.data[:h] == y[:, :w]).all() and (img.data[h:] == uv[:, :w]).all()


def test_pack_planes_i420_layout():
    w, h = 4, 4
    y = np.full((h, w), 50, np.uint8)
    u = np.full((2, 2), 60, np.uint8)
    v = np.full((2, 2), 70, np.uint8)
    buf = np.concatenate([y.ravel(), u.ravel(), v.ravel()]).tobytes()
    img = pack_planes(buf, "i420", w, h, (4, 2, 2), (0, 16, 20))
    flat = img.data[h:].ravel()
    assert (flat[:4] == 60).all() and (flat[4:8] == 70).all()


def test_numpy_rgb_shape_and_dtype():
    pytest.importorskip("cv2")
    rgb = to_rgb_numpy(_nv12())
    assert rgb.shape == (32, 64, 3) and rgb.dtype == np.uint8


@pytest.mark.gpu
@pytest.mark.parametrize("fmt", ["nv12", "i420"])
def test_torch_matches_opencv(fmt):
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    raw = _nv12(640, 360, seed=1)
    raw = RawImage(raw.data, fmt, raw.width, raw.height)
    ref = to_rgb_numpy(raw).astype(np.int16)
    got = to_rgb_torch(raw)
    assert got.is_cuda and got.dtype == torch.uint8 and tuple(got.shape) == (360, 640, 3)
    diff = np.abs(got.cpu().numpy().astype(np.int16) - ref)
    assert diff.max() <= 3 and diff.mean() < 1.0  # same BT.601 limited-range matrix, rounding aside
