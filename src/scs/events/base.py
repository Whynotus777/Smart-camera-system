"""Interface for the journey engine that turns perception outputs into events and alerts.

Theft is a sequence across zones (shelf → pickup → conceal → exit without checkout),
not a single-frame gesture (ARCHITECTURE D2). The engine is the one place that holds
per-person journey state; everything upstream is stateless per frame/window.
Signature fixed by docs/ARCHITECTURE.md §4.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from scs.contracts import Alert, BehaviorScore, Event, Pose, Track


@runtime_checkable
class JourneyEngine(Protocol):
    """Zone + journey state machine per (camera, track) or global id (T05)."""

    def on_track(self, t: Track) -> list[Event]:
        """Consume one track observation; return events it triggers."""
        ...

    def on_pose(self, p: Pose) -> list[Event]:
        """Consume one pose; return events it triggers (e.g. shelf interaction)."""
        ...

    def on_behavior(self, b: BehaviorScore) -> list[Event]:
        """Consume one behavior score; return events it triggers (e.g. conceal candidate)."""
        ...

    def poll_alerts(self, now: float) -> list[Alert]:
        """Return alerts whose journeys have resolved as of `now` (epoch seconds)."""
        ...
