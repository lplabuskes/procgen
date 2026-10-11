"""A site's roads: chained across seams, landed on the shore, crossing rivers, leaving by gates."""
import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, box

from localmap.test.test_store import REGION_ID, small_store
from region.geometry import ecef_to_cell
from region.rivers import RegionRivers
from region.roads import RegionRoads
from region.test.test_builder import land_towns
from region.test.test_resources import globe  # noqa: F401  (fixture)
from settlegen.builder import build_settlement_map
from settlegen.chain import SettlementRef
from settlegen.frame import Frame
from settlegen.geometry import SettlementGeometry
from settlegen.params import SettlementMapParams
from settlegen.roads import RoadLine, _chain, _landings, _Piece, _split_at_coast, crossings, gates
from settlegen.test.conftest import lake_hex, lake_local, region  # noqa: F401
from settlegen.upsample import SettlementUpsampleParams
from settlegen.water import RiverLine, SiteWater


def _piece(points, map_index, cut_start=False, cut_end=False):
    pos = np.asarray(points, dtype=np.float64)
    return _Piece(pos=pos, flows=np.arange(1.0, len(pos)), cut_start=cut_start, cut_end=cut_end,
                  map_index=map_index)


def test_roads_cut_at_a_seam_join_whichever_way_they_run():
    a = _piece([[0, 0, 1], [0, 0.001, 1]], 0, cut_end=True)
    # Runs the other way: its cut is at its end too.
    b = _piece([[0, 0.002, 1], [0, 0.0012, 1]], 1, cut_end=True)
    out = _chain([a, b], gap=0.001)
    assert len(out) == 1
    p = out[0]
    np.testing.assert_allclose(p.pos[:, 1], [0, 0.001, 0.0012, 0.002])
    assert len(p.flows) == 3 and not p.cut_start and not p.cut_end


def test_roads_on_one_map_or_too_far_apart_stay_apart():
    a = _piece([[0, 0, 1], [0, 0.001, 1]], 0, cut_end=True)
    same_map = _piece([[0, 0.0011, 1], [0, 0.002, 1]], 0, cut_start=True)
    far = _piece([[0, 0.01, 1], [0, 0.02, 1]], 1, cut_start=True)
    uncut = _piece([[0, 0.0011, 1], [0, 0.003, 1]], 1)
    assert len(_chain([a, same_map, far, uncut], gap=0.001)) == 4


@pytest.fixture(scope="module")
def frame(lake_local, lake_hex):  # noqa: F811
    c = lake_local.grid.centers[lake_hex] if len(lake_local.grid.centers) > lake_hex else lake_local.grid.centers[0]
    geometry = SettlementGeometry(lake_local.geometry, np.asarray(c), 0.05)
    return Frame(geometry, 4000.0)


def _water(sea=MultiPolygon(), shore=MultiLineString(), lines=()):
    return SiteWater(sea=sea, lakes=MultiPolygon(), shorelines=shore, lines=list(lines), channels=MultiPolygon(),
                     rivers=RegionRivers(region_id=0, paths=[]))


def test_a_port_hop_split_at_its_coast_point_is_one_road(frame):
    city, coast = frame.to_ecef(np.array([[0.0, 0.0], [3000.0, 0.0]]))
    roads = RegionRoads(region_id=0, segments=[], polylines=[np.stack([city, coast, city])],
                        polyline_flows=[np.array([5.0, 7.0])], sea=[(coast, coast + 0.01, 5.0)])
    polylines, flows = _split_at_coast(roads, reach_m=10.0)
    assert len(polylines) == 1 and len(polylines[0]) == 2
    np.testing.assert_allclose(flows[0], [12.0])


def test_a_sea_leg_lands_where_its_road_meets_the_shore(frame):
    sea = MultiPolygon([box(1000.0, -2000.0, 2000.0, 2000.0)])
    water = _water(sea=sea, shore=MultiLineString([[(1000.0, -2000.0), (1000.0, 2000.0)]]))
    # The road runs from the city to the region's coast point, out past the shore.
    city, coast, far = frame.to_ecef(np.array([[0.0, 0.0], [1500.0, 300.0], [9000.0, 300.0]]))
    roads = RegionRoads(
        region_id=0, segments=[], polylines=[np.stack([city, frame.to_ecef(np.array([[700.0, 0.0]]))[0], coast])],
        polyline_flows=[np.array([3.0, 3.0])], sea=[(coast, far, 3.0), (coast, far, 2.0)])
    landings, legs, polylines = _landings(roads, water, frame, reach_m=2000.0, merge_m=45.0)
    assert len(landings) == 1
    landing = landings[0]
    assert abs(landing.xy[0] - 1000.0) < 1e-6 and landing.water == "sea" and landing.flow == 5.0
    assert landing.direction[0] > 0.9
    end = frame.to_xy(polylines[0][-1:])[0]
    np.testing.assert_allclose(end, landing.xy, atol=1e-3)
    assert all(np.allclose(a, landing.xy) for a, _, _ in legs)


def test_gates_lie_on_the_footprint_edge_and_face_out():
    footprint = MultiPolygon([Point(0.0, 0.0).buffer(500.0, quad_segs=32)])
    through = RoadLine(xy=np.array([[-1000.0, 10.0], [1000.0, 10.0]]), flows=np.array([4.0]))
    ending = RoadLine(xy=np.array([[0.0, 0.0], [0.0, -1000.0]]), flows=np.array([2.0]))
    twin = RoadLine(xy=np.array([[0.0, 1.0], [0.0, -1000.0]]), flows=np.array([1.0]))
    out = gates([through, ending, twin], footprint, merge_m=45.0)
    assert len(out) == 3
    for g in out:
        assert footprint.boundary.distance(Point(g.xy)) < 1e-6
        assert not footprint.contains(Point(g.xy + 1.0 * g.direction))
    south = [g for g in out if g.xy[1] < -400.0]
    assert len(south) == 1 and south[0].flow == 3.0


def test_crossings_are_bridged_by_width_or_for_cities():
    river = RiverLine(xy=np.array([[0.0, -1000.0], [0.0, 0.0], [0.0, 1000.0]]), flows=np.ones(2),
                      width_m=np.array([10.0, 30.0]), in_water=np.array([False, False]), minor=False)
    roads = [RoadLine(xy=np.array([[-500.0, -500.0], [500.0, -500.0]]), flows=np.array([1.0])),
             RoadLine(xy=np.array([[-500.0, 500.0], [500.0, 500.0]]), flows=np.array([1.0]))]
    out = crossings(roads, _water(lines=[river]), bridge_width_m=20.0, always_bridge=False, merge_m=45.0)
    assert sorted((c.xy[1], c.bridge) for c in out) == [(-500.0, False), (500.0, True)]
    city = crossings(roads, _water(lines=[river]), bridge_width_m=20.0, always_bridge=True, merge_m=45.0)
    assert all(c.bridge for c in city)


def test_no_crossing_under_a_lake():
    river = RiverLine(xy=np.array([[0.0, -1000.0], [0.0, 1000.0]]), flows=np.ones(1), width_m=np.array([50.0]),
                      in_water=np.array([True]), minor=False)
    road = RoadLine(xy=np.array([[-500.0, 0.0], [500.0, 0.0]]), flows=np.array([1.0]))
    assert crossings([road], _water(lines=[river]), 20.0, False, 45.0) == []


@pytest.fixture
def towns(globe):  # noqa: F811
    towns = land_towns(globe)
    globe.add_feature("towns", towns)
    yield towns
    globe.remove_feature("towns")


def test_town_sites_have_valid_roads(globe, towns):  # noqa: F811
    store = small_store(globe)
    here = np.flatnonzero(ecef_to_cell(towns.site_positions, globe.antimeridian) == REGION_ID)
    params = SettlementMapParams(upsample=SettlementUpsampleParams(texel_m=200.0))
    for k in here[:4]:
        site = build_settlement_map(store, SettlementRef("town", int(k)), params).site
        r = site.roads
        assert site.footprint.is_valid
        for line in r.lines:
            assert len(line.xy) == len(line.flows) + 1 and LineString(line.xy).is_valid
        for g in r.gates:
            assert site.footprint.boundary.distance(Point(g.xy)) < 1e-6
            assert min(LineString(r.lines[g.road].xy).distance(Point(g.xy)), 1.0) < 1e-6
        assert len(r.roads.polylines) == len(r.lines)
