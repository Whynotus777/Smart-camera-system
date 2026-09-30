"""Crop export hook for model development (ARCHITECTURE D11). OFF by default.

T06 compares pose-only against visual and fused behavior models, which need person
and hand-region pixels over time. Re-ingesting video later is expensive, so on gated
tracks this hook can save:
- the person crop (track box, padded), and
- one square hand-region crop per wrist, side = `hand_frac` x torso diameter
  (left shoulder <-> right hip; see pose_eval) with a floor of `min_hand_px`.

Privacy (AGENTS.md rule 6, D11): persisting pixels is the exception. The exporter is
disabled unless constructed with `enabled=True` AND a `source_kind` of "lab" or "sim";
"store" additionally needs `store_consent=True`. Writes happen on a background thread
through a bounded queue that DROPS when full, so export never slows the live path.

Layout: `<root>/<camera_id>/<epoch>/<seq>_<track_id>_{person,lwrist,rwrist}.jpg` plus
`<root>/<camera_id>/index.jsonl` with frame identity, boxes, keypoints and model id.
Rows carry `group_id` (the source clip id) so T09 keeps derivatives in one split.
"""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from scs.contracts import BBox, Pose, Track

SourceKind = Literal["lab", "sim", "store"]
DEFAULT_ROOT = Path("data/crops")


def torso_diameter(kps: np.ndarray) -> float:
    """Distance left shoulder (5) <-> right hip (12) in pixels."""
    return float(np.hypot(*(kps[5, :2] - kps[12, :2])))


def hand_boxes(
    pose: Pose, hand_frac: float = 0.5, min_hand_px: float = 32.0, min_conf: float = 0.3
) -> dict[str, BBox]:
    """Square boxes around each confident wrist, in main-stream pixels."""
    kp = np.asarray(pose.keypoints, dtype=np.float64)
    side = max(hand_frac * torso_diameter(kp), min_hand_px)
    out: dict[str, BBox] = {}
    for name, j in (("lwrist", 9), ("rwrist", 10)):
        if kp[j, 2] >= min_conf:
            x, y = kp[j, :2]
            out[name] = (x - side / 2, y - side / 2, x + side / 2, y + side / 2)
    return out


def _clip_box(b: BBox, w: int, h: int) -> tuple[int, int, int, int] | None:
    x1, y1 = max(int(b[0]), 0), max(int(b[1]), 0)
    x2, y2 = min(int(np.ceil(b[2])), w), min(int(np.ceil(b[3])), h)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


@dataclass
class CropExporter:
    root: Path = DEFAULT_ROOT
    enabled: bool = False
    source_kind: SourceKind | None = None
    store_consent: bool = False
    group_id: str | None = None
    person_pad: float = 1.1
    hand_frac: float = 0.5
    max_queue: int = 256
    dropped: int = 0
    _q: queue.Queue[tuple[str, Path, np.ndarray | dict] | None] = field(init=False)
    _thread: threading.Thread | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._q = queue.Queue(self.max_queue)
        if self.enabled:
            if self.source_kind not in ("lab", "sim", "store"):
                raise ValueError("crop export needs source_kind 'lab' | 'sim' | 'store'")
            if self.source_kind == "store" and not self.store_consent:
                raise PermissionError(
                    "store crops need store_consent=True (ARCHITECTURE D11, privacy review)"
                )
            self._thread = threading.Thread(target=self._run, name="crop-export", daemon=True)
            self._thread.start()

    def __call__(self, image: np.ndarray, tracks: list[Track], poses: list[Pose]) -> int:
        """Queue crops for `tracks` (already gated) and their poses. Returns crops queued.

        `image` is the main-stream HWC uint8 frame (host memory). No-op when disabled.
        """
        if not self.enabled:
            return 0
        by_id = {p.track_id: p for p in poses}
        n = 0
        h, w = image.shape[:2]
        for t in tracks:
            f = t.frame
            base = self.root / f.camera_id / str(f.epoch) / f"{f.seq}_{t.track_id}"
            cx, cy = (t.bbox[0] + t.bbox[2]) / 2, (t.bbox[1] + t.bbox[3]) / 2
            hw, hh = (
                (t.bbox[2] - t.bbox[0]) * self.person_pad / 2,
                (t.bbox[3] - t.bbox[1]) * self.person_pad / 2,
            )
            boxes: dict[str, BBox] = {"person": (cx - hw, cy - hh, cx + hw, cy + hh)}
            pose = by_id.get(t.track_id)
            if pose is not None:
                boxes |= hand_boxes(pose, self.hand_frac)
            written = {}
            for name, b in boxes.items():
                cb = _clip_box(b, w, h)
                if cb is None:
                    continue
                n += self._put(
                    (
                        "img",
                        base.with_name(f"{base.name}_{name}.jpg"),
                        image[cb[1] : cb[3], cb[0] : cb[2]].copy(),
                    )
                )
                written[name] = cb
            row = {
                "camera_id": f.camera_id,
                "epoch": f.epoch,
                "seq": f.seq,
                "frame_idx": f.frame_idx,
                "ts": f.ts,
                "track_id": t.track_id,
                "group_id": self.group_id,
                "source_kind": self.source_kind,
                "bbox": list(t.bbox),
                "crops": written,
                "keypoints": pose.keypoints if pose else None,
                "pose_model": pose.model_id if pose else None,
            }
            self._put(("row", self.root / f.camera_id / "index.jsonl", row))
        return n

    def _put(self, item: tuple[str, Path, np.ndarray | dict]) -> int:
        try:
            self._q.put_nowait(item)
            return 1
        except queue.Full:
            self.dropped += 1
            return 0

    def close(self, timeout: float = 10.0) -> None:
        if self._thread is not None:
            self._q.put(None)
            self._thread.join(timeout)
            self._thread = None

    def _run(self) -> None:
        import cv2  # type: ignore[import-not-found,unused-ignore]

        while (item := self._q.get()) is not None:
            kind, path, payload = item
            path.parent.mkdir(parents=True, exist_ok=True)
            if kind == "img":
                cv2.imwrite(str(path), payload, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                with path.open("a") as fh:
                    fh.write(json.dumps(payload) + "\n")
