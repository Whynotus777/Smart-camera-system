"""Perceptual hashes for images and video clips (numpy + OpenCV only).

pHash (32x32 grayscale -> DCT -> top-left 8x8 -> median threshold) is robust to re-encoding,
resizing and mild color shifts, which is exactly what separates an original clip from its
re-encoded, emulated or cropped-then-resized copies. A clip's signature is the pHash of
N frames sampled at fixed *relative* positions, so two encodes of the same clip line up
even if their fps or length differ slightly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLES = 8  # frames per clip signature


def phash(img: np.ndarray) -> int:
    """64-bit pHash of a BGR or grayscale image."""
    import cv2

    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(g)[:8, :8].flatten()
    bits = low > np.median(low[1:])  # exclude DC from the threshold
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def video_signature(path: Path, samples: int = SAMPLES) -> list[int]:
    """pHashes of `samples` frames at evenly spaced relative positions (skips unreadable frames)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sig = []
    for k in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * n / samples))
        ok, img = cap.read()
        if ok:
            sig.append(phash(img))
    cap.release()
    return sig


def signature_distance(a: list[int], b: list[int]) -> float:
    """Median Hamming distance over aligned samples (64 = unrelated worst case)."""
    m = min(len(a), len(b))
    return float(np.median([hamming(x, y) for x, y in zip(a[:m], b[:m], strict=True)])) if m else 64.0


NEAR_DUP = 10  # bits; re-encodes land ~0-6, different scenes ~25-35
