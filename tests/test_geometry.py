from pathlib import Path

import pytest

from scs import geometry as g
from scs.contracts import FrameRef, SiteConfig, Zone, ZoneType

ROOT = Path(__file__).resolve().parents[1]
FRAME = FrameRef(camera_id="cam1", frame_idx=0, ts=0.0, width=2000, height=1000)
SQUARE = [(0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)]
# Concave "C" shape: the notch (0.5, 0.5) is outside.
C_SHAPE = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.3), (0.3, 0.3), (0.3, 0.7), (0.9, 0.7), (0.9, 0.9), (0.1, 0.9)]


@pytest.mark.parametrize(("pt", "inside"), [
    ((0.4, 0.4), True), ((0.1, 0.4), False), ((0.7, 0.7), False),
    ((0.2, 0.4), True),   # on left edge
    ((0.6, 0.6), True),   # on vertex
    ((0.4, 0.2), True),   # on top edge
    ((0.6000001, 0.4), False),
])
def test_point_in_square(pt, inside):
    assert g.point_in_polygon(pt, SQUARE) is inside


def test_point_in_concave():
    assert g.point_in_polygon((0.2, 0.5), C_SHAPE)
    assert not g.point_in_polygon((0.5, 0.5), C_SHAPE)
    assert g.point_in_polygon((0.8, 0.8), C_SHAPE)


def test_degenerate_polygon_rejected():
    with pytest.raises(ValueError):
        g.point_in_polygon((0, 0), [(0, 0), (1, 1)])


def test_area_and_centroid():
    assert g.polygon_area(SQUARE) == pytest.approx(0.16)
    assert g.polygon_centroid(SQUARE) == pytest.approx((0.4, 0.4))
    assert g.polygon_centroid(list(reversed(SQUARE))) == pytest.approx((0.4, 0.4))
    tri = [(0, 0), (3, 0), (0, 3)]
    assert g.polygon_centroid(tri) == pytest.approx((1.0, 1.0))


def test_expand_polygon():
    big = g.expand_polygon(SQUARE, 0.10)
    assert g.polygon_area(big) == pytest.approx(0.16 * 1.1**2)
    assert g.polygon_centroid(big) == pytest.approx((0.4, 0.4))
    edge = g.expand_polygon([(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)], 0.5)
    assert all(0 <= x <= 1 and 0 <= y <= 1 for x, y in edge)
    assert min(x for x, _ in g.expand_polygon(SQUARE, 3.0, clamp=False)) < 0
    with pytest.raises(ValueError):
        g.expand_polygon(SQUARE, -1)


def test_polygons_intersect():
    assert g.polygons_intersect(SQUARE, [(0.5, 0.5), (0.9, 0.5), (0.9, 0.9)])       # overlap
    assert g.polygons_intersect(SQUARE, [(0.3, 0.3), (0.35, 0.3), (0.35, 0.35)])    # contained
    assert g.polygons_intersect([(0.3, 0.3), (0.35, 0.3), (0.35, 0.35)], SQUARE)    # contains
    assert g.polygons_intersect(SQUARE, [(0.6, 0.3), (0.8, 0.3), (0.8, 0.5)])       # touches edge
    assert g.polygons_intersect(SQUARE, [(0.0, 0.4), (1.0, 0.4), (1.0, 0.45), (0.0, 0.45)])  # crossing bar
    assert not g.polygons_intersect(SQUARE, [(0.7, 0.7), (0.9, 0.7), (0.9, 0.9)])


def test_conversions_and_foot_point():
    assert g.normalize_point((1000, 250), FRAME) == (0.5, 0.25)
    assert g.normalize_bbox((200, 100, 400, 500), FRAME) == (0.1, 0.1, 0.2, 0.5)
    assert g.to_pixels(SQUARE, 2000, 1000)[2] == pytest.approx((1200, 600))
    assert g.foot_point((10, 20, 30, 80)) == (20, 80)
    assert g.bbox_polygon((1, 2, 3, 4)) == [(1, 2), (3, 2), (3, 4), (1, 4)]


def test_zone_queries():
    shelf = Zone(id="s", type=ZoneType.SHELF, polygon=SQUARE)
    door = Zone(id="d", type=ZoneType.ENTRY_EXIT, polygon=[(0.0, 0.8), (0.2, 0.8), (0.2, 1.0), (0.0, 1.0)])
    assert g.zones_containing((0.4, 0.4), [shelf, door]) == [shelf]
    assert g.zones_containing((0.4, 0.4), [shelf, door], types=[ZoneType.ENTRY_EXIT]) == []
    assert g.zones_containing((0.1, 0.9), [shelf, door]) == [door]
    # Box (pixels) just right of the shelf: misses it, but hits the 10%-expanded shelf.
    box = (1225, 400, 1400, 700)  # x1 = 0.6125 normalized
    assert not g.bbox_touches_zone(box, FRAME, shelf)
    assert g.bbox_touches_zone(box, FRAME, shelf, expand=0.10)


def test_example_site_zones_are_consistent():
    site = SiteConfig.from_yaml(ROOT / "configs" / "site.example.yaml")
    zones = {z.id: z for z in site.cameras[0].zones}
    # hv_candy sits inside shelf_a in the example; a point in it hits both, in file order.
    hits = g.zones_containing((0.4, 0.3), site.cameras[0].zones)
    assert [z.id for z in hits] == ["shelf_a", "hv_candy"]
    assert g.polygons_intersect(zones["hv_candy"].polygon, zones["shelf_a"].polygon)
    assert not g.polygons_intersect(zones["door"].polygon, zones["counter"].polygon)
