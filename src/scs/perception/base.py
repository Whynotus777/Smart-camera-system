"""Interfaces for detection, tracking and pose estimation.

Several of the strongest detector/pose options are AGPL or non-commercial
(ARCHITECTURE D6), so every model sits behind one of these Protocols and can be
swapped for a permissively licensed one without touching callers.
Signatures are fixed by docs/ARCHITECTURE.md §4; change them only via ADR.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from scs.contracts import Detection, FrameRef, Pose, Track

if TYPE_CHECKING:
    from torch import Tensor


@runtime_checkable
class Detector(Protocol):
    """Batched person detector across cameras (T03)."""

    def detect(self, batch: list[tuple[FrameRef, Tensor]]) -> list[list[Detection]]:
        """Return one list of detections per input frame, in input order."""
        ...


@runtime_checkable
class Tracker(Protocol):
    """Multi-object tracker; one instance per camera (T03)."""

    def update(self, dets: list[Detection], frame: Tensor | None) -> list[Track]:
        """Advance one frame. `frame` is optional, for appearance-aware trackers."""
        ...


@runtime_checkable
class PoseEstimator(Protocol):
    """Top-down COCO17 pose on crops of the given tracks (T04)."""

    def estimate(self, frame: FrameRef, image: Tensor, tracks: list[Track]) -> list[Pose]:
        """Return at most one `Pose` per input track."""
        ...
