"""Polygon and zone helpers shared by pose gating (T04) and the journey engine (T05).

Both stages ask the same questions: is this foot point in the checkout zone, is this
wrist in a shelf polygon, does this box touch a gated zone? If they each implement
point-in-polygon, they'll disagree on edge cases and the eval numbers stop meaning
anything. So T04 and T05 must use these functions rather than their own.

Coordinate rules (contracts.py "Conventions", ADR 0002):
- Zone polygons are NORMALIZED [0, 1] (x right, y down).
- Boxes and keypoints are main-stream pixels; convert with `normalize_point` /
  `normalize_bbox` using the `FrameRef` width/height before testing against zones.
- A point exactly on a polygon edge or vertex counts as inside.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from scs.contracts import BBox, FrameRef, Zone, ZoneType

Point = tuple[float, float]
Polygon = Sequence[Point]

_EPS = 1e-12


# --------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------


def normalize_point(pt: Point, frame: FrameRef) -> Point:
    """Main-stream pixel point → normalized [0, 1] coords (not clamped)."""
    return (pt[0] / frame.width, pt[1] / frame.height)


def normalize_bbox(bbox: BBox, frame: FrameRef) -> BBox:
    """Main-stream pixel box → normalized box (not clamped)."""
    x1, y1, x2, y2 = bbox
    return (x1 / frame.width, y1 / frame.height, x2 / frame.width, y2 / frame.height)


def to_pixels(polygon: Polygon, width: int, height: int) -> list[Point]:
    """Normalized polygon → pixel polygon for a frame of the given size."""
    return [(x * width, y * height) for x, y in polygon]


def bbox_polygon(bbox: BBox) -> list[Point]:
    """Box → its 4 corners, clockwise from top-left (in the box's own coord system)."""
    x1, y1, x2, y2 = bbox
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def foot_point(bbox: BBox) -> Point:
    """Bottom-center of a box: where a standing person touches the floor."""
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2, y2)


# --------------------------------------------------------------------------
# Polygon primitives
# --------------------------------------------------------------------------


def polygon_area(polygon: Polygon) -> float:
    """Unsigned area (shoelace)."""
    n = len(polygon)
    s = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def polygon_centroid(polygon: Polygon) -> Point:
    """Area centroid; falls back to the vertex mean for degenerate polygons."""
    n = len(polygon)
    a = cx = cy = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        a += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(a) < _EPS:
        return (sum(p[0] for p in polygon) / n, sum(p[1] for p in polygon) / n)
    a *= 0.5
    return (cx / (6 * a), cy / (6 * a))


def _on_segment(p: Point, a: Point, b: Point) -> bool:
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    if abs(cross) > 1e-9:
        return False
    return (min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9
            and min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9)


def point_in_polygon(pt: Point, polygon: Polygon) -> bool:
    """Even-odd ray casting; points on an edge or vertex count as inside."""
    n = len(polygon)
    if n < 3:
        raise ValueError("polygon needs at least 3 points")
    x, y = pt
    inside = False
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        if _on_segment(pt, a, b):
            return True
        if (a[1] > y) != (b[1] > y):
            x_cross = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x < x_cross:
                inside = not inside
    return inside


def _segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    def orient(a: Point, b: Point, c: Point) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and 0 not in (d1, d2, d3, d4):
        return True
    return (_on_segment(p1, q1, q2) or _on_segment(p2, q1, q2)
            or _on_segment(q1, p1, p2) or _on_segment(q2, p1, p2))


def polygons_intersect(a: Polygon, b: Polygon) -> bool:
    """True if the polygons overlap or touch (including one containing the other)."""
    if any(point_in_polygon(p, b) for p in a) or any(point_in_polygon(p, a) for p in b):
        return True
    na, nb = len(a), len(b)
    return any(
        _segments_intersect(a[i], a[(i + 1) % na], b[j], b[(j + 1) % nb])
        for i in range(na) for j in range(nb)
    )


def expand_polygon(polygon: Polygon, fraction: float, clamp: bool = True) -> list[Point]:
    """Scale a polygon about its centroid by (1 + fraction), e.g. 0.10 for "expanded by 10%".

    With `clamp`, the result is clipped to the [0, 1] normalized frame.
    """
    if fraction <= -1:
        raise ValueError("fraction must be > -1")
    cx, cy = polygon_centroid(polygon)
    k = 1 + fraction
    out = [(cx + (x - cx) * k, cy + (y - cy) * k) for x, y in polygon]
    if clamp:
        out = [(min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)) for x, y in out]
    return out


# --------------------------------------------------------------------------
# Zone queries (normalized coords)
# --------------------------------------------------------------------------


def zones_containing(pt: Point, zones: Iterable[Zone], types: Iterable[ZoneType] | None = None) -> list[Zone]:
    """Zones (optionally filtered by type) that contain a normalized point, in input order."""
    wanted = set(types) if types is not None else None
    return [z for z in zones if (wanted is None or z.type in wanted) and point_in_polygon(pt, z.polygon)]


def bbox_touches_zone(bbox: BBox, frame: FrameRef, zone: Zone, expand: float = 0.0) -> bool:
    """Does a main-stream pixel box overlap `zone` (optionally expanded by `expand`, e.g. 0.10)?"""
    poly = expand_polygon(zone.polygon, expand) if expand else list(zone.polygon)
    return polygons_intersect(bbox_polygon(normalize_bbox(bbox, frame)), poly)
