"""Interface for an optional second-opinion verifier on alert clips.

A VLM looks at the evidence clip before a human does, to shorten the review queue
(ARCHITECTURE §2). Its output lands in `Alert.verifier`; a verifier failure must
never block an alert. Signature fixed by docs/ARCHITECTURE.md §4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from scs.contracts import Alert


@runtime_checkable
class Verifier(Protocol):
    """Second-stage verifier over an alert's clips (T11)."""

    def verify(self, alert: Alert, clip_paths: list[Path]) -> dict:
        """Return a JSON-serializable verdict to store in `Alert.verifier`."""
        ...
