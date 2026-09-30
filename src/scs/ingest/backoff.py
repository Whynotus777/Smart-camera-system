"""Reconnect backoff for camera sources.

Cameras reboot, lose Wi-Fi, or get power-cycled by staff. Retrying in a tight loop
floods logs and the camera's RTSP server; never retrying loses the camera. Exponential
backoff with a 30 s cap (T02 brief) and jitter, so ten cameras behind one switch that
drop together don't reconnect in lockstep.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Backoff:
    initial_s: float = 0.5
    factor: float = 2.0
    cap_s: float = 30.0
    jitter: float = 0.1  # ± fraction of the delay
    _attempt: int = field(default=0, init=False)
    _rng: random.Random = field(default_factory=random.Random, repr=False)

    def next_delay(self) -> float:
        """Delay before the next attempt; grows until `cap_s`. Never exceeds the cap."""
        base = min(self.cap_s, self.initial_s * self.factor**self._attempt)
        self._attempt += 1
        j = base * self.jitter * (2 * self._rng.random() - 1)
        return max(0.0, min(self.cap_s, base + j))

    def reset(self) -> None:
        """Call after a connection has delivered frames again."""
        self._attempt = 0

    @property
    def attempt(self) -> int:
        return self._attempt
