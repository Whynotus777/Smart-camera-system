"""Zone gating for pose (ARCHITECTURE D4).

Pose is the most expensive per-person stage and most people in a c-store aren't at a
shelf at any given moment, so only tracks whose box touches a SHELF, HIGH_VALUE,
CHECKOUT or ENTRY_EXIT zone (each expanded by 10%) get pose. All polygon math is
`scs.geometry` so T04 and T05 agree on edge cases; nothing here re-implements it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from scs.contracts import Track, Zone, ZoneType
from scs.geometry import bbox_touches_zone

GATED_ZONE_TYPES: frozenset[ZoneType] = frozenset(
    {ZoneType.SHELF, ZoneType.HIGH_VALUE, ZoneType.CHECKOUT, ZoneType.ENTRY_EXIT}
)
GATE_EXPAND = 0.10


def track_is_gated(
    track: Track,
    zones: Iterable[Zone],
    types: frozenset[ZoneType] = GATED_ZONE_TYPES,
    expand: float = GATE_EXPAND,
) -> bool:
    """True if the track box touches any zone of a gated type (zone expanded by `expand`)."""
    return any(
        z.type in types and bbox_touches_zone(track.bbox, track.frame, z, expand=expand) for z in zones
    )


def gate_tracks(
    tracks: Sequence[Track],
    zones: Sequence[Zone],
    types: frozenset[ZoneType] = GATED_ZONE_TYPES,
    expand: float = GATE_EXPAND,
    include_lost: bool = False,
) -> list[Track]:
    """Tracks that should get pose this frame, in input order.

    `lost` tracks are dropped by default: their box is a prediction, not a detection,
    so cropping it spends GPU on a region that may not contain the person.
    """
    return [
        t for t in tracks if (include_lost or t.state != "lost") and track_is_gated(t, zones, types, expand)
    ]
