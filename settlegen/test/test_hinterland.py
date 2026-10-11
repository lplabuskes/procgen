"""A site's hinterland and anchor candidates, on the synthetic lake frame."""
import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Point

from localmap.farmland import FarmlandParams
from region.test.test_resources import globe  # noqa: F401  (fixture)
from settlegen.anchors import SettlementAnchorParams, build_anchor_candidates
from settlegen.chain import SettlementRef
from settlegen.hinterland import Neighbour, build_hinterland
from settlegen.roads import RoadLine, SiteRoads
from settlegen.test.conftest import inner_center, lake_hex, lake_local, region, side_km  # noqa: F401
from settlegen.test.test_footprint import site  # noqa: F401  (fixture)
from settlegen.water import RiverLine, SiteWater, _as_multipolygon
from region.roads import RegionRoads


def _neighbour(xy, people):
    return Neighbour(xy=np.asarray(xy, dtype=float), kind="village", index=0, map=(0, 0), population=people,
                     mining=False, fields=MultiPolygon())


def _dry_spot(site, frac):  # noqa: F811
    """A dry point ``frac`` of the way from the frame's centre to its corner."""
    fields, frame, water = site
    still = shapely.union_all([water.sea, water.lakes])
    for f in np.linspace(frac, 0.95, 20):
        for sign in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            p = np.array(sign) * f * frame.half
            if not still.contains(Point(p)):
                return p
    raise AssertionError("no dry ground")


def _quota_ha(people, p=FarmlandParams()):
    return people * p.ha_per_farmer


def test_fields_are_shared_never_overlap_and_keep_off_water_and_the_footprint(site, globe):  # noqa: F811
    fields, frame, water = site
    px_ha = (frame.side_m / fields.tex_size) ** 2 / 1e4
    a, b = _dry_spot(site, 0.5), _dry_spot(site, 0.7)
    people = 60 * px_ha / FarmlandParams().ha_per_farmer
    footprint = _as_multipolygon(Point(_dry_spot(site, 0.3)).buffer(2.0 * np.sqrt(px_ha * 1e4)).difference(water.lakes))
    assert not footprint.is_empty
    hint = build_hinterland(globe, fields, frame, water, footprint, [], [], SettlementRef("village", 0, map=(0, 0)),
                            people, neighbours=[_neighbour(a, people), _neighbour(b, people)])
    polys = [hint.own] + [n.fields for n in hint.neighbours]
    assert all(p.is_valid and not p.is_empty for p in polys)
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            assert polys[i].intersection(polys[j]).area < 0.02 * min(polys[i].area, polys[j].area)
    still = shapely.union_all([water.sea, water.lakes])
    for p in polys:
        assert p.intersection(still).area < 1e-6 * p.area
        assert p.intersection(footprint).area < 1e-6 * p.area
        # Each claims about its farmers' hectares (the frame has the room).
        assert abs(p.area / 1e4 - _quota_ha(people)) < 0.25 * _quota_ha(people)


def test_a_village_farms_only_its_own_bank_of_a_wide_river(site, globe):  # noqa: F811
    fields, frame, water = site
    h = frame.half
    river = RiverLine(xy=np.array([[0.0, -h], [0.0, h]]), flows=np.ones(1), width_m=np.array([120.0]),
                      in_water=np.array([False]), minor=False)
    channels = MultiPolygon([shapely.box(-60.0, -h, 60.0, h)])
    wet = SiteWater(sea=water.sea, lakes=water.lakes, shorelines=water.shorelines, lines=[river],
                    channels=channels, rivers=water.rivers)
    px_ha = (frame.side_m / fields.tex_size) ** 2 / 1e4
    spot = _dry_spot(site, 0.4)
    spot[0] = abs(spot[0]) if spot[0] != 0 else 0.4 * h
    people = 40 * px_ha / FarmlandParams().ha_per_farmer
    village = _neighbour([max(spot[0], 3.0 * np.sqrt(px_ha * 1e4)), spot[1]], people)
    hint = build_hinterland(globe, fields, frame, wet, MultiPolygon(), [], [], SettlementRef("city", 0), 0.0,
                            neighbours=[village],
                            params=FarmlandParams(city_garden_ring_km=0.0, river_barrier_width_m=50.0))
    got = hint.neighbours[0].fields
    assert not got.is_empty
    assert got.intersection(shapely.box(-h, -h, 0.0, h)).area < 1e-6 * got.area


def test_anchor_candidates_lie_where_their_kind_belongs(site):  # noqa: F811
    fields, frame, water = site
    h = frame.half
    footprint = _as_multipolygon(shapely.box(-0.6 * h, -0.6 * h, 0.6 * h, 0.6 * h).difference(water.lakes))
    assert not footprint.is_empty
    busy = RoadLine(xy=np.array([[-h, 0.3 * h], [h, 0.3 * h]]), flows=np.array([10.0]))
    quiet = RoadLine(xy=np.array([[-h, -0.3 * h], [h, -0.3 * h]]), flows=np.array([1.0]))
    roads = SiteRoads(lines=[busy, quiet], water=[], sea=[], landings=[], crossings=[], gates=[],
                      roads=RegionRoads(region_id=0, segments=[]))
    params = SettlementAnchorParams(separation_m=0.1 * h, sample_m=0.02 * h, market_reach_m=0.01 * h,
                                    waterfront_reach_m=0.02 * h, smoothing_m=0.0, surround_m=0.2 * h)
    cand = build_anchor_candidates(fields, frame, water, roads, footprint, params)
    grown = footprint.buffer(1.0)
    for kind in ("citadel", "market", "sacred"):
        assert cand[kind], kind
        assert all(grown.contains(Point(a.xy)) for a in cand[kind])
    for kind in ("citadel", "market", "waterfront", "sacred"):
        scores = [a.score for a in cand[kind]]
        assert scores == sorted(scores, reverse=True) and all(0.0 <= s <= 1.0 for s in scores)
        pts = [a.xy for a in cand[kind]]
        assert all(np.linalg.norm(p - q) >= params.separation_m for i, p in enumerate(pts) for q in pts[i + 1:])
    # High ground: above the footprint's typical height above its surroundings.
    assert cand["citadel"][0].detail["height_m"] >= np.median([a.detail["height_m"] for a in cand["citadel"]])
    # The market's best is on the busy road.
    assert abs(cand["market"][0].xy[1] - 0.3 * h) < 1e-6 * h + params.market_reach_m
    # The lake meets the footprint: waterfront on its shore.
    if footprint.intersection(water.lakes.buffer(params.waterfront_reach_m)).area > 0:
        assert cand["waterfront"] and all(a.detail["water"] == "lake" for a in cand["waterfront"])


def test_only_a_villages_farmers_grow_its_hinterland(site, globe):  # noqa: F811
    fields, frame, water = site
    px_ha = (frame.side_m / fields.tex_size) ** 2 / 1e4
    people = 40 * px_ha / FarmlandParams().ha_per_farmer
    ref = SettlementRef("village", 0, map=(0, 0))
    spot = _dry_spot(site, 0.5)
    whole = build_hinterland(globe, fields, frame, water, MultiPolygon(), [], [], ref, people, neighbours=[
        _neighbour(spot, people)])
    half = Neighbour(xy=spot, kind="village", index=0, map=(0, 0), population=people, mining=False,
                     fields=MultiPolygon(), farming_share=0.5)
    fishing = build_hinterland(globe, fields, frame, water, MultiPolygon(), [], [], ref, people, neighbours=[half],
                               farming_share=0.25)
    assert fishing.own_target_ha == 0.25 * whole.own_target_ha
    assert fishing.neighbours[0].fields.area < 0.7 * whole.neighbours[0].fields.area
