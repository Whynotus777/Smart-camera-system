"""CAMERA_HEALTH events: connect, disconnect, stall, recovery, published to `Streams.HEALTH`.

Health goes on the bus (not just a log) so T12's supervisor, T13's durability checks and
the store dashboard all see the same record, and it survives consumer restarts
(ARCHITECTURE D7). `data.state` is one of `HEALTH_STATES`; every event carries the
camera's epoch and a stats snapshot so a reader can tell what was lost.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from scs.bus import Bus
from scs.contracts import Event, EventType, Streams
from scs.ingest.urls import redact_text

log = logging.getLogger("scs.ingest")

SOURCE = "scs.ingest@0.1"
HEALTH_STATES = (
    "connected",  # first frames of an epoch arrived
    "disconnected",  # pipeline ended or errored; a reconnect will follow
    "stalled",  # session up but no fresh frame for stall_timeout_s
    "reconnecting",  # about to retry after a backoff delay
    "eos",  # a finite source (file without loop) ended normally
    "stopped",  # close() was called
)


class HealthReporter:
    """Builds CAMERA_HEALTH events for one camera and publishes them (bus optional)."""

    def __init__(self, camera_id: str, bus: Bus | None = None, keep: int = 1000) -> None:
        self.camera_id = camera_id
        self.bus = bus
        self.events: list[Event] = []  # recent events, for tests and status pages
        self._keep = keep

    def emit(self, state: str, epoch: int, reason: str = "", **data: Any) -> Event:
        if state not in HEALTH_STATES:
            raise ValueError(f"unknown health state {state!r}")
        ev = Event(
            type=EventType.CAMERA_HEALTH,
            camera_id=self.camera_id,
            ts=time.time(),
            source=SOURCE,
            data={"state": state, "epoch": epoch, "reason": redact_text(reason), **data},
        )
        self.events.append(ev)
        del self.events[: -self._keep]
        level = logging.INFO if state in ("connected", "eos", "stopped") else logging.WARNING
        log.log(level, "camera %s %s (epoch %d) %s", self.camera_id, state, epoch, ev.data["reason"])
        if self.bus is not None:
            try:
                self.bus.publish(Streams.HEALTH, ev)
            except Exception as e:  # noqa: BLE001 - a dead bus must not kill ingest
                log.error("camera %s: health publish failed: %s", self.camera_id, e)
        return ev
