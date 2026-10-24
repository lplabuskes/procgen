"""Buildings on the lots: lot frames, the typology, the church."""
import dataclasses

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiPolygon, box

from settlegen.layout.buildings import TYPOLOGY, church, lot_frames, lot_type
from settlegen.layout.builder import build_layout
from settlegen.layout.describe import describe_at
from settlegen.layout.test.test_layout import HALF, _profile, _site, _slope
from settlegen.layout.types import Block
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
    assert shapely.union_all(lot_built).area == pytest.approx(sum(p.area for p in lot_built), rel=1e-6)


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
