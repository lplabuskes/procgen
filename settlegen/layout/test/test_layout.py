"""The layout on a hand-built site: a square town split by a river, two roads in."""
import dataclasses
import math

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box

from settlegen.site.anchors import Anchor, AnchorCandidates
from settlegen.frame import Frame
from settlegen.layout.blocks import lanes_of, split_blocks, strip_lots
from settlegen.layout.builder import build_layout
from settlegen.layout.describe import describe_at, summary
from settlegen.layout.districts import OPEN_DISTRICTS, TRADE_DISTRICTS
from settlegen.layout.geometry import chaikin, core_polygon
from settlegen.layout.households import TRADES, farm_toft_m2, households
from settlegen.layout.lots import derive_lots
from settlegen.layout.params import SettlementLayoutParams
from settlegen.layout.streets import radial_targets, route_arterials
from settlegen.profile.builder import Industry, PolityRelation, SettlementProfile
from settlegen.site.types import Gate, RoadLine, SiteRoads
from settlegen.site.builder import SettlementSite
from settlegen.layout.village import back_lanes
from settlegen.site.types import RiverLine, SiteWater

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


def test_a_wall_street_runs_just_inside_the_walls(layout):
    params = SettlementLayoutParams()
    ring = [g for g, k in zip(layout.arterials, layout.arterial_kinds) if k == "wall"]
    assert ring and sum(g.length for g in ring) > 0.5 * layout.walls.line.length
    offset = 0.5 * params.wall_m + 0.5 * params.secondary_m
    for line in ring:
        pts = [shapely.Point(p) for p in line.coords]
        assert all(abs(layout.walls.line.distance(p) - offset) < 1.5 for p in pts[1:-1])
        assert all(layout.core.buffer(1e-6).contains(p) for p in pts)
    # Blocks keep off it.
    street = shapely.union_all(ring).buffer(0.5 * params.secondary_m - 0.5)
    assert all(b.polygon.intersection(street).area < 1.0 for b in layout.blocks)


def test_secondary_arterials_fill_the_gaps_through_the_gates_only():
    params = SettlementLayoutParams(arterial_spacing_m=200.0)
    lay = build_layout(_site(), _profile(), _slope(), (1, 2, 3), params)
    kinds = lay.arterial_kinds
    extra = [g for g, k in zip(lay.arterials, kinds) if k == "secondary"]
    assert len(extra) >= 3
    net = shapely.union_all(lay.arterials)
    # Every dense block lies within about half the spacing of an arterial.
    for b in lay.blocks:
        assert net.distance(b.polygon.centroid) < 0.5 * params.arterial_spacing_m + 2.0 * params.route_step_m
    # Each starts on an earlier arterial; most run through to another or out to the footprint's
    # edge (a few dead-end, against the river).
    edge = _site().footprint.boundary
    dead = 0
    for i, line in enumerate(lay.arterials):
        if kinds[i] != "secondary":
            continue
        earlier = shapely.union_all(lay.arterials[:i])
        a, b = shapely.Point(line.coords[0]), shapely.Point(line.coords[-1])
        assert earlier.distance(a) < 1.0
        dead += min(earlier.distance(b), edge.distance(b)) >= 1.0
    assert dead <= 0.25 * len(extra)
    # None makes a gate of its own.
    walls = lay.walls
    assert walls is not None
    for p in shapely.get_parts(shapely.union_all(extra).intersection(walls.line)):
        if p.is_empty:
            continue
        xy = np.asarray(p.coords[0])
        assert np.min(np.linalg.norm(walls.gates - xy, axis=1)) < 2.0 * params.route_step_m


def test_arterials_reach_every_gate_and_bridge_the_river(layout):
    net = shapely.union_all(layout.arterials)
    for g in _site().roads.gates:
        assert net.distance(shapely.Point(g.xy)) < 1.0
    assert net.distance(shapely.Point(layout.anchors["market"])) < 1.0
    # The waterfront lies across the river from the market.
    assert len(layout.bridges) >= 1


def test_closes_and_alleys_open_up_deep_blocks(layout):
    assert len(layout.closes) >= 5 and len(layout.alleys) >= 1
    blocks = shapely.union_all([b.polygon for b in layout.blocks])
    half = 0.5 * layout.lane_m
    for lane in [*layout.closes, *layout.alleys]:
        # Off the blocks, open to a street at its start.
        assert blocks.intersection(lane.buffer(half - 0.25, cap_style="flat")).area < 1.0
        assert blocks.buffer(half + 1.0).contains(shapely.Point(lane.coords[0]))
    # A close stops inside its block: blocks lie beyond its head.
    heads = [np.asarray(c.coords[-1]) for c in layout.closes]
    assert sum(blocks.distance(shapely.Point(h)) < half + 1.0 for h in heads) == len(heads)
    # Lots fronting the lanes never overlap the block's other lots.
    for b in layout.blocks:
        if len(b.lots) > 1:
            # On a fixed grid: lots sharing near-collinear edges can trip the plain union.
            assert shapely.union_all(list(b.lots), grid_size=1e-6).area == pytest.approx(
                sum(g.area for g in b.lots), rel=1e-3)
    pt = np.asarray(layout.closes[0].interpolate(0.5, normalized=True).coords[0])
    assert describe_at(layout, _site(), pt, 2) == "Close"


def test_lanes_cut_in_from_a_side_or_down_the_middle():
    deep = box(0, 0, 120, 70)
    params = SettlementLayoutParams()
    pieces, closes, alleys = lanes_of(deep, 22.0, np.random.default_rng(1), params, planned=False)
    assert closes or alleys
    assert sum(g.area for g in pieces) < deep.area
    for c in closes:                       # across the block, from one long side, a lot short of the other
        y0, y1 = c.coords[0][1], c.coords[-1][1]
        assert min(y0, y1) == pytest.approx(0.0, abs=1.0) or max(y0, y1) == pytest.approx(70.0, abs=1.0)
        assert c.length < 70.0 - 20.0
    pieces, closes, alleys = lanes_of(deep, 22.0, np.random.default_rng(1), params, planned=True)
    assert not closes and len(alleys) == 1 and len(pieces) == 2
    assert alleys[0].length == pytest.approx(120.0, abs=1.0)
    # A block two lots deep keeps itself.
    shallow = box(0, 0, 120, 45)
    assert lanes_of(shallow, 22.0, np.random.default_rng(1), params, planned=False) == ([shallow], [], [])


def test_lanes_open_onto_the_street_on_a_skewed_block():
    # A parallelogram: the cut runs across the long axis, meeting the sheared sides 30° off square.
    shear = math.tan(math.radians(30.0))
    block = Polygon([(0, 0), (140, 0), (140 + 75 * shear, 75), (75 * shear, 75)])
    params = SettlementLayoutParams(alley_share=0.5)
    half = 0.5 * params.lane_m
    for seed in range(6):
        pieces, closes, alleys = lanes_of(block, 22.0, np.random.default_rng(seed), params, planned=False)
        left = shapely.union_all(pieces)
        for lane, ends in [*((c, (0,)) for c in closes), *((a, (0, -1)) for a in alleys)]:
            xy = np.asarray(lane.coords)
            for k in ends:
                end, inner = xy[k], xy[1 if k == 0 else -2]
                outward = LineString([end, 2.0 * end - inner])     # the lane carried on into the street
                gap = shapely.union_all([lane, outward]).buffer(half - 0.1, cap_style="flat")
                mouth = shapely.Point(end).buffer(2.0 * params.lane_m)
                assert left.intersection(gap).intersection(mouth).area < 0.5


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
    lots, _ = strip_lots(block, 8.0, 20.0, np.random.default_rng(0), True)
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


def test_radials_leave_by_the_landward_edge_spread_along_it():
    # A port: the market on the shore, sea to the east; the land front is the far
    # (west) side, narrow as seen from the market.
    part = box(-3000, -800, 0, 800)
    sea = MultiPolygon([box(0, -5000, 5000, 5000)])
    market = np.array([-50.0, 0.0])
    out = radial_targets(part, market, [], 8, sea, 300.0)
    assert len(out) >= 4
    assert all(sea.distance(shapely.Point(*p)) > 100.0 for p in out)
    gaps = [np.linalg.norm(a - b) for i, a in enumerate(out) for b in out[i + 1:]]
    assert min(gaps) >= 300.0


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
    assert 0.6 * households <= lay.n_lots <= households + 1
    tofts = [lot for b in lay.blocks for lot in b.lots]
    assert shapely.union_all(tofts).area == pytest.approx(sum(t.area for t in tofts), rel=1e-6)
    streets = shapely.union_all(lay.arterials).buffer(0.5 * lay.arterial_m)
    assert all(t.intersection(streets).area < 1.0 for t in tofts)
    assert not lay.precinct.is_empty and all(t.intersection(lay.precinct).area < 1.0 for t in tofts)
    farm = next(lot for b in lay.blocks for i, lot in enumerate(b.lots) if b.use(i) == "farm")
    assert str(describe_at(lay, site, np.asarray(farm.representative_point().coords[0]), 5)).startswith("Farm")


def _village(population: float, kind: str = "farming", **overrides):  # type: ignore[no-untyped-def]
    profile = _profile(tier=5, tier_name="Village", kind=kind, walls=False, wall_strength=0.0,
                       population=population, polity=PolityRelation(), industries=[], **overrides)
    site = _site()
    site = dataclasses.replace(site, footprint=MultiPolygon([box(-200, -250, 200, 50)]),
                               buildable=MultiPolygon(list(shapely.get_parts(
                                   box(-HALF, -HALF, HALF, HALF).difference(site.water.channels)))))
    return site, profile


def test_village_craftsmen_live_on_small_plots_near_the_middle():
    site, profile = _village(600.0)
    lay = build_layout(site, profile, _slope(), (1, 2, 3))
    centre = lay.anchors["market"]
    plots = [(b.use(i), lot) for b in lay.blocks for i, lot in enumerate(b.lots)]
    trades = [lot for use, lot in plots if use in TRADES and use != "miller"]
    farms = [lot for use, lot in plots if use == "farm"]
    assert 0.05 * len(plots) <= len(trades) <= 0.35 * len(plots)
    assert np.median([t.area for t in trades]) < 0.6 * np.median([f.area for f in farms])

    def dist(polys):  # type: ignore[no-untyped-def]
        return np.median([np.linalg.norm(np.asarray(p.centroid.coords[0]) - centre) for p in polys])
    assert dist(trades) < dist(farms)
    use, lot = next((use, lot) for use, lot in plots if use in TRADES)
    what = describe_at(lay, site, np.asarray(lot.representative_point().coords[0]), 5)
    assert TRADES[use].label in str(what)          # the plot, or "Workshop · Smithy" on a building
    assert "craftsmen" in summary(lay, profile)


def test_households_by_what_they_do():
    rng = np.random.default_rng(0)
    params = SettlementLayoutParams()
    _, small = _village(80.0)
    assert not any(h.trade for h in households(small, params, rng))          # too small for a smith
    _, fishing = _village(1000.0, kind="fishing", farming_share=0.4)
    homes = households(fishing, params, rng)
    assert len(homes) == 200
    trades = sum(h.trade for h in homes)
    assert 0.1 * 200 <= trades <= 0.35 * 200
    fishers = sum(h.use == "fisher" for h in homes)
    assert fishers == round(0.6 * (200 - trades))
    # The plots keep the village's ground between them.
    ground = 1e5
    farm = farm_toft_m2(homes, ground)
    assert sum(h.scale for h in homes) * farm == pytest.approx(ground)


def test_a_city_spreads_its_everyday_trades():
    lay = build_layout(_site(), _profile(), _slope(), (1, 2, 3))
    spots: dict[str, list] = {}
    for b in lay.blocks:
        for i, lot in enumerate(b.lots):
            if b.use(i):
                assert b.district in TRADE_DISTRICTS
                spots.setdefault(b.use(i), []).append(np.asarray(lot.centroid.coords[0]))
    assert {"smith", "baker", "alehouse"} <= set(spots)
    assert not {"miller", "weaver", "butcher"} & set(spots)          # quarters' trades, not spread
    for name, pts in spots.items():
        if len(pts) < 2:
            continue
        xy = np.asarray(pts)
        d = np.linalg.norm(xy[:, None] - xy[None], axis=2) + np.eye(len(xy)) * 1e9
        assert d.min() > 15.0, name                                   # never next door to its own


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
    assert lay.n_lots >= 0.85 * 180
    assert len(lay.piers) == 1
    jetty = np.asarray(lay.piers[0].centroid.coords[0])
    # Rows parallel to the straight shore: streets that run along it keep one distance from the water.
    rows = [g for g in lay.arterials if g.length > 100.0
            and np.ptp([site.water.sea.distance(shapely.Point(p)) for p in g.coords]) < 2.0]
    assert len(rows) >= 2
    # Compact about the jetty: no farmstead far beyond the rows' reach (no 2 km shore street).
    far = max(np.linalg.norm(np.asarray(lot.centroid.coords[0]) - jetty) for b in lay.blocks for lot in b.lots)
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


def test_quay_and_market_lots_run_back_to_fill_their_blocks(layout):
    deep = [b for b in layout.blocks if b.district in ("quay", "market", "elite") and len(b.lots)]
    assert deep
    cover = [sum(lot.area for lot in b.lots) / b.polygon.area for b in deep]
    assert np.median(cover) > 0.97          # ordinary 22 m rows leave ~0.9 (yards in the middle)


def test_lots_are_what_their_blocks_dividers_derive(layout):
    # Blocks keep the lots they were built with; the dividers alone must give them back, tiling.
    for b in layout.blocks:
        if not len(b.lots):
            continue
        derived = derive_lots(b.polygon, b.dividers, b.seeds)
        for kept, again in zip(b.lots, derived):
            assert kept.symmetric_difference(again).area < 1e-3 * kept.area
        assert shapely.union_all(list(derived)).area == pytest.approx(sum(g.area for g in derived), rel=1e-6)
