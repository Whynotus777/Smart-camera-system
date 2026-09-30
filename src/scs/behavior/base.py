"""Interface for behavior models that score windows of one track's poses.

Behavior runs on pose sequences first (ARCHITECTURE D3): public retail-theft data is
pose-only, and pose is privacy-preserving and camera-agnostic. Keeping models behind
this Protocol lets unsupervised, supervised and future video models be compared in
eval and swapped at runtime. Signature fixed by docs/ARCHITECTURE.md §4.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from scs.contracts import BehaviorScore, Pose


@runtime_checkable
class BehaviorModel(Protocol):
    """Scores a window of consecutive poses from a single track (T06)."""

    model_id: str
    window: int  # frames

    def score(self, poses: Sequence[Pose]) -> BehaviorScore:
        """Return a calibrated score in [0, 1]; higher = more suspicious."""
        ...
