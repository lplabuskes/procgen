"""The layout on a hand-built site: a square town split by a river, two roads in."""
import dataclasses

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiLineString, MultiPolygon, box

from settlegen.anchors import Anchor, AnchorCandidates
from settlegen.frame import Frame
from settlegen.layout import (
    OPEN_DISTRICTS, SettlementLayoutParams, build_layout, chaikin, core_polygon, describe_at, lots_of, route_arterials,
    split_blocks,
)
from settlegen.profile import Industry, PolityRelation, SettlementProfile
from settlegen.roads import Gate, RoadLine, SiteRoads
from settlegen.site import SettlementSite
from settlegen.village_layout import back_lanes
from settlegen.water import RiverLine, SiteWater

HALF = 1000.0


def _site() -> SettlementSite:
    river = np.array([[-HALF, 100.0], [HALF, 100.0]])
    line = RiverLine(xy=river, flows=np.array([50.0]), width_m=np.array([40.0]), in_water=np.array([False]),
                     minor=False)
    channels = MultiPolygon([LineString(river).buffer(20.0, cap_style="flat")])
    water = SiteWater(sea=MultiPolygon(), lakes=MultiPolygon(), shorelines=MultiLineString(), lines=[line],
                      channels=channels, rivers=None)  # type: ignore[arg-type]
    footprint = MultiPolygon(list(shapely.get_parts(box(-400, -400, 400, 400).difference(channels))))
    roads = [RoadLine(xy=np.array([[0.0, -HALF], [0.0, 0.0]]), flows=np.array([10.0])),
             RoadLine(xy=np.array([[HALF, -50.0], [0.0, 0.0]]), flows=np.array([5.0]))]
    gates = [Gate(xy=np.array([0.0, -400.0]), direction=np.array([0.0, -1.0]), flow=10.0, road=0),
             Gate(xy=np.array([400.0, -40.0]), direction=np.array([1.0, 0.0]), flow=5.0, road=1)]
    site_roads = SiteRoads(lines=roads, water=[], sea=[], landings=[], crossings=[], gates=gates,
                           roads=None)  # type: ignore[arg-type]
    anchors = AnchorCandidates({
        "market": [Anchor("market", np.array([0.0, -60.0]), 1.0)],
        "citadel": [Anchor("citadel", np.array([-250.0, -250.0]), 1.0)],
        "waterfront": [Anchor("waterfront", np.array([150.0, 130.0]), 1.0)],
        "sacred": [Anchor("sacred", np.array([-100.0, -150.0]), 1.0)],
    })
    return SettlementSite(
        frame=Frame(geometry=None, side_m=2.0 * HALF), water=water, footprint=footprint,  # type: ignore[arg-type]
        density_per_ha=100.0, sparse=MultiPolygon(), buildable=footprint, roads=site_roads, anchors=anchors,
        hinterland=None,  # type: ignore[arg-type]
    )


def _profile(**overrides) -> SettlementProfile:
    base = SettlementProfile(
        population=40_000.0, tier=2, tier_name="City", kind="river", site_kinds=("river",), walls=True,
        wall_strength=0.8, polity=PolityRelation(polity_id=0, control="direct", capital=True),
        wealth=0.6, trade=0.0,
        industries=[Industry("shipping", 0.5), Industry("tanning", 0.6), Industry("milling", 0.4)],
        wind_xy=np.array([5.0, 0.0]), wind_steadiness=0.8,
        materials={"stone": 0.5, "timber": 0.2, "brick": 0.3, "adobe": 0.0}, danger=0.0, biome="Temperate",
        founding_age=600, planned=False,
    )
    return dataclasses.replace(base, **overrides)


def _slope() -> np.ndarray:
    return np.zeros((100, 100))


@pytest.fixture(scope="module")
def layout():
    return build_layout(_site(), _profile(), _slope(), (1, 2, 3))


def test_blocks_and_lots_stay_on_the_footprint_off_the_water(layout):
    site = _site()
    assert len(layout.blocks) > 50 and layout.n_lots > 500
    blocks = shapely.union_all([b.polygon for b in layout.blocks])
    assert blocks.difference(site.footprint.buffer(1e-6)).area < 1.0
    assert blocks.intersection(site.water.channels).area < 1.0
    lots = shapely.union_all([g for b in layout.blocks for g in b.lots])
    assert lots.difference(blocks.buffer(1e-6)).area < 1.0
    # Streets and open ground leave a share of the footprint unbuilt, not most of it.
    assert 0.5 < blocks.area / site.footprint.area < 0.95


def test_open_districts_have_no_lots(layout):
    assert any(b.district == "graveyard" for b in layout.blocks)
    assert all(len(b.lots) == 0 for b in layout.blocks if b.district in OPEN_DISTRICTS)


def test_anchors_by_profile(layout):
    assert set(layout.anchors) == {"market", "citadel", "waterfront", "sacred"}
    plain = build_layout(_site(), _profile(polity=PolityRelation(), walls=False, wall_strength=0.0,
                                           industries=[]), _slope(), (1, 2, 3))
    assert set(plain.anchors) == {"market", "sacred"}
    assert plain.walls is None and plain.citadel.is_empty


def test_walls_ring_the_core_open_to_the_river_with_gates(layout):
    walls = layout.walls
    assert walls is not None and walls.kind == "stone"
    assert walls.line.distance(_site().water.channels) >= SettlementLayoutParams().wall_water_m - 1.0
    assert len(walls.towers) > 5
    assert len(walls.gates) >= 1


def test_arterials_reach_every_gate_and_bridge_the_river(layout):
    net = shapely.union_all(layout.arterials)
    for g in _site().roads.gates:
        assert net.distance(shapely.Point(g.xy)) < 1.0
    assert net.distance(shapely.Point(layout.anchors["market"])) < 1.0
    # The waterfront lies across the river from the market.
    assert len(layout.bridges) >= 1


def test_noxious_trades_go_downwind(layout):
    tanners = [b.polygon.centroid.x for b in layout.blocks if b.district == "tanners"]
    elite = [b.polygon.centroid.x for b in layout.blocks if b.district == "elite"]
    assert tanners and elite
    assert np.mean(tanners) > np.mean(elite)      # the wind blows toward +x


def test_layout_is_deterministic_per_seed():
    a = build_layout(_site(), _profile(), _slope(), (1, 2, 3))
    b = build_layout(_site(), _profile(), _slope(), (1, 2, 3))
    c = build_layout(_site(), _profile(), _slope(), (1, 2, 4))
    assert [x.polygon.wkb for x in a.blocks] == [x.polygon.wkb for x in b.blocks]
    assert [x.polygon.wkb for x in a.blocks] != [x.polygon.wkb for x in c.blocks]


def test_core_holds_its_share():
    footprint = MultiPolygon([box(-500, -300, 500, 300)])
    core = core_polygon(footprint, np.array([100.0, 0.0]), 0.4, np.random.default_rng(0), 0.1)
    assert core.area == pytest.approx(0.4 * footprint.area, rel=0.01)
    assert core.contains(shapely.Point(100.0, 0.0))


def test_planned_blocks_follow_the_grid():
    params = SettlementLayoutParams()
    ward = box(0, 0, 400, 300).buffer(0)
    blocks, streets = split_blocks(ward, 3000.0, np.random.default_rng(0), params, True, 0.0)
    assert len(blocks) > 20
    for s in streets:
        d = np.diff(np.asarray(s.coords)[[0, -1]], axis=0)[0]
        assert min(abs(d[0]), abs(d[1])) < 1e-6 * max(abs(d[0]), abs(d[1]), 1.0)
    assert sum(b.area for b in blocks) < ward.area


def test_lots_front_the_long_sides():
    block = box(0, 0, 80, 40)
    lots = lots_of(block, 8.0, 20.0, np.random.default_rng(0), True)
    assert len(lots) == 20
    assert sum(g.area for g in lots) == pytest.approx(block.area)


def test_routing_avoids_steep_ground():
    slope = np.zeros((100, 100))
    slope[:, 45:55] = 1.0
    slope[80:, 45:55] = 0.0                 # a pass in the south
    params = SettlementLayoutParams()
    lines = route_arterials(HALF, slope, MultiPolygon(), MultiPolygon(), [], np.array([-500.0, 500.0]),
                            [np.array([500.0, 500.0])], params)
    net = shapely.union_all(lines)
    assert net.bounds[1] < -500.0           # went round by the pass


def test_chaikin_keeps_ends():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    out = chaikin(xy, 2)
    assert np.allclose(out[0], xy[0]) and np.allclose(out[-1], xy[-1])
    assert len(out) > len(xy)


def _sea_site() -> SettlementSite:
    """The same town on a shore: the sea north of y = 100, no river."""
    site = _site()
    sea = MultiPolygon([box(-HALF, 100.0, HALF, HALF)])
    water = SiteWater(sea=sea, lakes=MultiPolygon(), shorelines=MultiLineString(), lines=[],
                      channels=MultiPolygon(), rivers=None)  # type: ignore[arg-type]
    footprint = MultiPolygon([box(-400, -400, 400, 100)])
    anchors = AnchorCandidates({**site.anchors.by_kind,
                                "waterfront": [Anchor("waterfront", np.array([100.0, 100.0]), 1.0)]})
    return dataclasses.replace(site, water=water, footprint=footprint, buildable=footprint, anchors=anchors)


def test_a_port_gets_quays_and_piers_over_the_water():
    profile = _profile(walls=False, wall_strength=0.0, industries=[Industry("shipping", 0.9)])
    lay = build_layout(_sea_site(), profile, _slope(), (1, 2, 3))
    assert len(lay.piers) >= 2
    sea = _sea_site().water.sea
    for pier in lay.piers:
        assert pier.intersection(sea).area >= 0.75 * pier.area
    assert not lay.quays.is_empty and lay.quays.difference(_sea_site().footprint.buffer(1e-6)).area < 1.0
    blocks = shapely.union_all([b.polygon for b in lay.blocks])
    assert blocks.intersection(lay.quays).area < 1.0


def test_an_unwalled_city_has_no_suburb_ring():
    lay = build_layout(_site(), _profile(walls=False, wall_strength=0.0), _slope(), (1, 2, 3))
    assert lay.walls is None
    assert not any(b.district == "suburb" for b in lay.blocks)


def test_a_town_has_a_market_street_and_few_quarters():
    profile = _profile(tier=4, tier_name="Town", kind="market", walls=False, wall_strength=0.0, population=4000.0,
                       polity=PolityRelation(polity_id=0, control="direct"))
    lay = build_layout(_site(), profile, _slope(), (1, 2, 3))
    assert describe_at(lay, _site(), np.asarray(lay.plaza.representative_point().coords[0]), 4) == "Market street"
    districts = {b.district for b in lay.blocks}
    assert not districts & {"square", "elite", "suburb", "butchers"}
    assert sum(b.district == "graveyard" for b in lay.blocks) <= 1
    assert len(districts - {"residential", "market", "quay", "graveyard", "gardens"}) <= 2


def test_a_village_is_farmsteads_along_its_streets():
    profile = _profile(tier=5, tier_name="Village", kind="farming", walls=False, wall_strength=0.0,
                       population=600.0, polity=PolityRelation(), industries=[])
    site = _site()
    # As on a real site: buildable ground is the frame less the water; the footprint a village's size.
    site = dataclasses.replace(site, footprint=MultiPolygon([box(-200, -250, 200, 50)]),
                               buildable=MultiPolygon(list(shapely.get_parts(
                                   box(-HALF, -HALF, HALF, HALF).difference(site.water.channels)))))
    lay = build_layout(site, profile, _slope(), (1, 2, 3))
    households = 600 // 5
    assert not lay.fill_footprint and lay.walls is None
    assert 0.6 * households <= len(lay.blocks) <= households + 1
    assert all(len(b.lots) == 1 for b in lay.blocks)
    tofts = [b.polygon for b in lay.blocks]
    assert shapely.union_all(tofts).area == pytest.approx(sum(t.area for t in tofts), rel=1e-6)
    streets = shapely.union_all(lay.arterials).buffer(0.5 * lay.arterial_m)
    assert all(t.intersection(streets).area < 1.0 for t in tofts)
    assert not lay.precinct.is_empty
    assert str(describe_at(lay, site, np.asarray(tofts[0].representative_point().coords[0]), 5)).startswith("Farm")


def test_hover_names_what_is_there(layout):
    site = _site()
    block = next(b for b in layout.blocks if len(b.lots))
    lot = block.lots[0]
    assert "plot" in str(describe_at(layout, site, np.asarray(lot.representative_point().coords[0]), 2))
    assert describe_at(layout, site, np.array([200.0, 100.0]), 2) == "River"
    assert describe_at(layout, site, np.array([900.0, -900.0]), 2) is None


def test_a_fishing_village_packs_rows_along_the_shore_about_its_jetty():
    site = _sea_site()
    frame_land = MultiPolygon(list(shapely.get_parts(box(-HALF, -HALF, HALF, HALF).difference(site.water.sea))))
    site = dataclasses.replace(site, footprint=MultiPolygon([box(-200, -150, 200, 100)]), buildable=frame_land)
    profile = _profile(tier=5, tier_name="Village", kind="fishing", walls=False, wall_strength=0.0,
                       population=900.0, polity=PolityRelation(), industries=[Industry("fishing", 0.8)])
    lay = build_layout(site, profile, _slope(), (1, 2, 3))
    assert len(lay.blocks) >= 0.85 * 180
    assert len(lay.piers) == 1
    jetty = np.asarray(lay.piers[0].centroid.coords[0])
    # Rows parallel to the straight shore: streets that run along it keep one distance from the water.
    rows = [g for g in lay.arterials if g.length > 100.0
            and np.ptp([site.water.sea.distance(shapely.Point(p)) for p in g.coords]) < 2.0]
    assert len(rows) >= 2
    # Compact about the jetty: no farmstead far beyond the rows' reach (no 2 km shore street).
    far = max(np.linalg.norm(np.asarray(b.polygon.centroid.coords[0]) - jetty) for b in lay.blocks)
    assert far < 500.0


def test_back_lanes_of_a_bent_street_do_not_cross():
    # Two arms meeting at an angle at the middle, plus the router's short spur there: the
    # lanes behind them run on round the bend as one each side, crossing nothing.
    arms = [LineString([(0, 0), (-300, -300)]), LineString([(0, 0), (60, 80), (260, 300)]),
            LineString([(0, 0), (-8, 8)])]
    lanes = back_lanes(arms, 25.0, 6.0)
    long = [g for g in lanes if g.length > 100.0]
    assert len(long) == 2
    every = [*arms[:2], *lanes]
    assert not any(a.crosses(b) for i, a in enumerate(every) for b in every[i + 1:])


def test_back_lanes_stay_off_the_green():
    green = shapely.Point(0, 0).buffer(60.0)
    # An arm leaving the green and bending as it goes: its lanes must not hook round onto it.
    arm = LineString([(60, 0), (90, 40), (300, 120)])
    lanes = back_lanes([arm], 25.0, 6.0, keep_out=green)
    assert lanes
    assert not any(g.intersects(green) for g in lanes)


def test_walls_take_in_sparse_ground_not_round_it():
    # A garden patch straddling the core's edge: the wall runs on through it, not in round its edge.
    patch = box(150, -330, 330, -130)
    site = dataclasses.replace(_site(), sparse=MultiPolygon([patch]))
    lay = build_layout(site, _profile(), _slope(), (1, 2, 3))
    assert lay.walls is not None
    assert lay.core.intersection(patch).area > 0.1 * patch.area
    assert lay.walls.line.intersection(patch.boundary.buffer(2.0)).length < 20.0


def test_walls_span_a_notch_in_the_edge_and_take_in_its_ground():
    # A slot cut into the footprint's west edge: the wall runs across its mouth, and the
    # ground it takes in joins the layout's own footprint (the site's is left as it was).
    site = _site()
    notch = box(-400, -250, -280, -150)
    site = dataclasses.replace(site, footprint=MultiPolygon(list(shapely.get_parts(site.footprint.difference(notch)))))
    lay = build_layout(site, _profile(), _slope(), (1, 2, 3))
    assert lay.walls is not None and lay.footprint is not None
    assert lay.footprint.contains(site.footprint.buffer(-1e-6))
    assert lay.footprint.intersection(notch).area > 0.8 * notch.area
    assert lay.walls.line.intersection(notch.buffer(-15.0)).length < 1.0     # across its mouth, not round it
    off = build_layout(site, _profile(), _slope(), (1, 2, 3), SettlementLayoutParams(wall_closing=0.0))
    assert off.footprint is None and off.walls is not None
    assert off.walls.line.intersection(notch.buffer(1.0)).length > 100.0
