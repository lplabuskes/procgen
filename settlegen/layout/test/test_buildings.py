"""Buildings on the lots: lot frames, the typology, the church."""
import dataclasses

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon, box

from settlegen.layout.buildings import TYPOLOGY, LotFrames, church, lot_frames, lot_type, place_buildings, square_frames
from settlegen.layout.builder import build_layout
from settlegen.layout.describe import describe_at
from settlegen.layout.params import SettlementLayoutParams
from settlegen.layout.test.test_layout import HALF, _profile, _site, _slope
from settlegen.layout.types import Block, SettlementLayout
from settlegen.profile.builder import PolityRelation


def test_a_strip_lot_fronts_its_street():
    lot = box(10, 3, 18, 25)                                  # 8 m on a street along y = 0 (6 m wide), 22 deep
    f = lot_frames(np.asarray([lot], dtype=object), [([LineString([(0, 0), (100, 0)])], 3.0 + 1.5, 1)])
    assert abs(f.u[0, 0]) == pytest.approx(1.0) and f.n[0] == pytest.approx([0.0, 1.0])
    assert f.width[0] == pytest.approx(8.0) and f.depth[0] == pytest.approx(22.0)


def test_a_corner_lot_fronts_its_narrow_end_not_its_flank():
    lot = box(0, 3, 8, 25)                                    # a cross street runs down its long side (x = -3)
    streets = [LineString([(-50, 0), (100, 0)]), LineString([(-3, -50), (-3, 100)])]
    f = lot_frames(np.asarray([lot], dtype=object), [(streets[:1], 3.0 + 1.5, 1), (streets[1:], 3.0 + 1.5, 3)])
    assert f.width[0] == pytest.approx(8.0) and f.n[0] == pytest.approx([0.0, 1.0])


def test_a_plot_between_two_streets_faces_the_grander():
    lot = box(0, 5, 7, 60)                                    # a burgage plot: market street below, back lane above
    fronts = [([LineString([(-50, 0), (50, 0)])], 12.0 + 1.5, 3), ([LineString([(-50, 62), (50, 62)])], 1.5 + 1.5, 0)]
    f = lot_frames(np.asarray([lot], dtype=object), fronts)
    assert f.n[0] == pytest.approx([0.0, 1.0])               # faces down, onto the market street


def test_with_no_street_a_lot_fronts_its_narrow_side():
    f = lot_frames(np.asarray([box(0, 0, 8, 30)], dtype=object), [])
    assert f.width[0] == pytest.approx(8.0) and f.depth[0] == pytest.approx(30.0)


def _skewed_lot(skew_deg: float) -> Polygon:
    """A 12 m front on a street along y = 0 (5 m wide), its sides leaning ``skew_deg`` off square, 25 m deep."""
    t = 25.0 * np.tan(np.radians(skew_deg))
    return Polygon([(0, 3), (12, 3), (12 + t, 28), (t, 28)])


def test_a_square_lot_keeps_its_street_frame():
    lots = np.asarray([box(0, 3, 12, 28)], dtype=object)
    f = lot_frames(lots, [([LineString([(-50, 0), (50, 0)])], 2.5 + 1.5, 1)])
    sq, skewed = square_frames(lots, f)
    assert not skewed[0] and sq.n[0] == pytest.approx(f.n[0]) and sq.width[0] == pytest.approx(f.width[0])


def test_a_skewed_lot_frames_square_to_its_sides():
    lots = np.asarray([_skewed_lot(30.0)], dtype=object)
    f = lot_frames(lots, [([LineString([(-50, 0), (50, 0)])], 2.5 + 1.5, 1)])
    sq, skewed = square_frames(lots, f)
    side = np.array([np.sin(np.radians(30.0)), np.cos(np.radians(30.0))])
    assert skewed[0] and sq.n[0] == pytest.approx(side)              # into the lot, along its sides
    assert sq.width[0] == pytest.approx(12.0 * np.cos(np.radians(30.0)))


@pytest.mark.parametrize("margin, square", [(0.1, True), (10.0, False)])
def test_a_house_on_a_skewed_lot_stands_square_to_the_lot(margin: float, square: bool):
    lot = _skewed_lot(30.0)
    layout = SettlementLayout.empty()
    layout.streets = [LineString([(-50, 0), (50, 0)])]
    layout.blocks = [Block.of_lots(lot, "suburb", False, [lot])]
    params = SettlementLayoutParams(lot_align_margin=margin)
    b = place_buildings(layout, 100.0, 2, np.random.default_rng(1), params)
    house = b.polygons[0]
    # Square to the lot: a whole rectangle, its sides along the lot's; square to the street
    # (the margin out of reach): a sliver the lot's sides cut.
    rect = house.minimum_rotated_rectangle
    edges = np.diff(np.asarray(rect.exterior.coords), axis=0)
    lean = np.degrees(np.arctan2(np.abs(edges[:, 0]), np.abs(edges[:, 1])))       # off the street's normal
    if square:
        assert house.area > 0.97 * rect.area and np.min(np.abs(lean - 30.0)) < 1.0
    else:
        assert house.area < 0.85 * rect.area


def test_lot_types_by_use_then_district():
    blk = Block(polygon=box(0, 0, 10, 10), district="residential", core=True, seeds=np.array([[1.0, 1.0]] * 4),
                uses=np.asarray(["", "smith", "farm", "manor"], dtype=object))
    assert lot_type(blk, 0, 2) == "townhouse" and lot_type(blk, 1, 2) == "workshop"
    assert [lot_type(blk, i, 5) for i in range(4)] == ["cottage", "craft", "farm", "manor"]
    assert lot_type(dataclasses.replace(blk, district="graveyard"), 0, 2) is None
    assert lot_type(dataclasses.replace(blk, district="tanners"), 0, 2) == "workshop"
    assert set(TYPOLOGY) >= {"townhouse", "suburb", "workshop", "elite", "garden", "farm", "cottage", "craft", "manor"}


def _check(lay) -> None:  # type: ignore[no-untyped-def]
    """Every building stands inside its lot, and none overlap."""
    b = lay.buildings
    assert len(b) > 0
    for poly, (bi, li) in zip(b.polygons, b.lots):
        if bi >= 0:
            assert poly.difference(lay.blocks[bi].lots[li]).area < 1e-6
    lot_built = [p for p, (bi, _) in zip(b.polygons, b.lots) if bi >= 0]
    # On a fixed grid: GEOS's floating overlay can swallow a building sharing a near-collinear edge.
    assert shapely.union_all(lot_built, grid_size=1e-9).area == pytest.approx(sum(p.area for p in lot_built), rel=1e-6)


def test_a_city_is_built_up_its_core_terraced():
    lay = build_layout(_site(), _profile(), _slope(), (1, 2, 3))
    _check(lay)
    kinds = list(lay.buildings.kinds)
    assert {"row", "house", "workshop", "church"} <= set(kinds)
    # Nearly every house lot has a house on it.
    built = {(int(bi), int(li)) for bi, li in lay.buildings.lots}
    house_lots = [(bi, li) for bi, b in enumerate(lay.blocks) for li in range(len(b.lots))
                  if lot_type(b, li, 2) == "townhouse"]
    assert sum(r in built for r in house_lots) >= 0.9 * len(house_lots)
    # The church stands on the precinct; hover names it.
    nave = next(p for p, k in zip(lay.buildings.polygons, kinds) if k == "church")
    assert nave.difference(lay.precinct).area < 1e-6
    assert describe_at(lay, _site(), np.asarray(nave.representative_point().coords[0]), 2) == "Temple"


def test_a_village_is_farmhouses_and_barns():
    profile = _profile(tier=5, tier_name="Village", kind="farming", walls=False, wall_strength=0.0,
                       population=600.0, polity=PolityRelation(), industries=[])
    site = _site()
    site = dataclasses.replace(site, footprint=MultiPolygon([box(-200, -250, 200, 50)]),
                               buildable=MultiPolygon(list(shapely.get_parts(
                                   box(-HALF, -HALF, HALF, HALF).difference(site.water.channels)))))
    lay = build_layout(site, profile, _slope(), (1, 2, 3))
    _check(lay)
    kinds = list(lay.buildings.kinds)
    farms = sum(b.use(i) == "farm" for b in lay.blocks for i in range(len(b.lots)))
    assert kinds.count("farmhouse") >= 0.9 * farms and kinds.count("barn") >= 0.5 * farms
    assert "church" in kinds
    # Houses front their streets: each farmhouse within a few metres of one.
    streets = shapely.union_all(lay.arterials)
    houses = [p for p, k in zip(lay.buildings.polygons, kinds) if k == "farmhouse"]
    assert np.median([h.distance(streets) for h in houses]) < 0.5 * lay.arterial_m + 6.0


def test_the_church_fits_its_precinct_or_shrinks():
    small = box(0, 0, 30, 20)
    parts = church(small, 5000.0)                              # too big at full size: shrunk to fit
    assert parts and all(small.buffer(-1.0).contains(g) for _, g in parts)
    assert {k for k, _ in parts} == {"church", "chancel", "tower"}
    assert church(box(0, 0, 3, 3), 500.0) == []


def test_a_strip_along_its_street_fronts_its_long_side_not_its_tapered_end():
    # 30 m along a street at y = 0 (reach 4.5), 5 m deep, one end tapered to 2 m.
    lot = shapely.Polygon([(0, 3), (30, 3), (30, 8), (3, 8), (0, 5)])
    f = lot_frames(np.asarray([lot], dtype=object), [([LineString([(-50, 0), (100, 0)])], 3.0 + 1.5, 2)])
    assert abs(f.u[0, 0]) == pytest.approx(1.0) and f.depth[0] == pytest.approx(5.0)


def test_a_narrow_lot_builds_its_back_across_not_a_thin_range():
    p = SettlementLayoutParams()
    k = 200
    f = LotFrames(np.zeros((k, 2)), np.tile([1.0, 0.0], (k, 1)), np.tile([0.0, 1.0], (k, 1)),
                  np.full(k, 5.0), np.full(k, 40.0))
    placed = TYPOLOGY["townhouse"](f, np.ones(k, dtype=bool), np.random.default_rng(0), p)
    wings = [r for kind, rects, _ in placed if kind == "wing" for r in rects]
    assert wings
    for w in wings:
        x0, _, x1, _ = w.bounds
        assert x1 - x0 == pytest.approx(5.0)                    # the lot's whole width
    houses = [r for kind, rects, _ in placed if kind in ("row", "house") for r in rects]
    assert all(h.bounds[2] - h.bounds[0] == pytest.approx(5.0) for h in houses)     # terraced, no gaps


def test_a_city_scatters_temple_precincts_by_its_size():
    small = build_layout(_site(), _profile(population=1500.0), _slope(), (1, 2, 3))
    big = build_layout(_site(), _profile(), _slope(), (1, 2, 3))
    n_small, n_big = (len(shapely.get_parts(lay.precinct)) for lay in (small, big))
    assert n_small == 1 and n_big > 3
    # Each extra piece stands apart, has a temple on it, and hover names it.
    kinds = np.asarray(big.buildings.kinds)
    sacred = [p for p, k in zip(big.buildings.polygons, kinds) if k in ("temple", "rotunda")]
    assert len(sacred) >= n_big - 2
    assert all(p.difference(big.precinct).area < 1e-6 for p in sacred)
    assert all(t.distance(b.polygon) > 0.5 for t in sacred for b in big.blocks)
    xy = np.asarray(sacred[0].representative_point().coords[0])
    assert describe_at(big, _site(), xy, 2) == "Temple"
