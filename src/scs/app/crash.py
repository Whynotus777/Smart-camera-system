"""Crash-point injection for the durability test (no-op unless SCS_CRASHPOINTS is set).

Random external kill -9s mostly land where processes spend their time (decoding,
sleeping), and rarely in the few milliseconds between two durable steps, which is where
durability bugs live. Each such boundary calls `crashpoint("<role>.<step>")`; the chaos
test sets e.g. `SCS_CRASHPOINTS="ingest.*.event=0.5,web.*=0.1"` (first matching pattern
wins) and the process SIGKILLs *itself* there with that probability, exactly like an OOM
kill of that process would.
"""

from __future__ import annotations

import fnmatch
import os
import random
import signal

_SPEC = os.environ.get("SCS_CRASHPOINTS", "")
_RULES: list[tuple[str, float]] = []
for part in filter(None, (p.strip() for p in _SPEC.split(","))):
    pattern, _, prob = part.partition("=")
    _RULES.append((pattern, float(prob or 1.0)))
_RNG = random.Random()  # noqa: S311 (fault injection, not security)


def crashpoint(name: str) -> None:
    for pattern, prob in _RULES:
        if fnmatch.fnmatch(name, pattern):
            if _RNG.random() < prob:
                os.kill(os.getpid(), signal.SIGKILL)
            return
