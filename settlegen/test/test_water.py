"""A site's water: polygons from the refined masks, rivers chained across map seams."""
import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, box

from localmap.test.test_store import REGION_ID, small_store
from region.geometry import ecef_to_cell
from region.test.test_builder import land_towns
from region.test.test_resources import globe  # noqa: F401  (fixture)
from settlegen.test.conftest import inner_center, lake_hex, lake_local, region, side_km  # noqa: F401
from settlegen.builder import build_settlement_map
from settlegen.chain import SettlementRef
from settlegen.params import SettlementMapParams
from settlegen.upsample import SettlementUpsampleParams
from settlegen.water import (
    RiverLine, SettlementRiverParams, _chain, _Piece, _tidy_ends, natural_channels, shorelines,
)


def _piece(points, map_index, ids, starts=True, ends=True):
    pos = np.asarray(points, dtype=np.float64)
    return _Piece(pos=pos, flows=np.ones(len(pos) - 1), in_water=np.zeros(len(pos) - 1, bool),
                  ids=np.full(len(pos), ids), starts=starts, ends=ends, map_index=map_index)


def test_a_path_cut_at_a_seam_is_rejoined():
    a = _piece([[0, 0, 1], [0, 0.001, 1]], 0, 7, ends=False)
    b = _piece([[0, 0.0012, 1], [0, 0.002, 1]], 1, 7, starts=False)
    other_river = _piece([[0, 0.0011, 1], [0, 0.003, 1]], 1, 8, starts=False)
    same_map = _piece([[0, 0.00105, 1], [0, 0.004, 1]], 0, 7, starts=False)
    out = _chain([a, b, other_river, same_map], gap=0.001)
    joined = [p for p in out if len(p.pos) == 4]
    assert len(out) == 3 and len(joined) == 1 and joined[0].ends
    np.testing.assert_array_equal(joined[0].pos[2], b.pos[0])


def test_a_piece_too_far_is_left_alone():
    a = _piece([[0, 0, 1], [0, 0.001, 1]], 0, 7, ends=False)
    b = _piece([[0, 0.01, 1], [0, 0.02, 1]], 1, 7, starts=False)
    assert len(_chain([a, b], gap=0.001)) == 2


@pytest.fixture
def towns(globe):  # noqa: F811
    towns = land_towns(globe)
    globe.add_feature("towns", towns)
    yield towns
    globe.remove_feature("towns")


def test_town_sites_have_valid_water(globe, towns):  # noqa: F811
    store = small_store(globe)
    here = np.flatnonzero(ecef_to_cell(towns.site_positions, globe.antimeridian) == REGION_ID)
    params = SettlementMapParams(upsample=SettlementUpsampleParams(texel_m=200.0))
    checked = 0
    for k in here[:6]:
        built = build_settlement_map(store, SettlementRef("town", int(k)), params)
        f, site = built.fields, built.site
        w = site.water
        px = (site.frame.side_m / f.tex_size) ** 2
        for poly, key in ((w.sea, "ocean_mask"), (w.lakes, "lake_mask")):
            assert poly.is_valid and (poly.is_empty or poly.within(site.frame.box().buffer(1e-3)))
            mask_area = float((f.textures[key][0, ..., 0] > 0.5).sum()) * px

            # Smooth shores, not texel stairs: the same water to within a texel's width of shore.
            assert abs(poly.area - mask_area) <= 0.05 * mask_area + 2.0 * np.sqrt(px) * (
                w.shorelines.length + 4 * site.frame.side_m) * 0.5
        assert w.channels.is_valid
        for line in w.lines:
            assert len(line.xy) == len(line.width_m) + 1 and (line.width_m > 0).all()
        checked += 1
    assert checked > 0


def test_lake_polygons_follow_the_refined_mask(globe, lake_local, inner_center, side_km):  # noqa: F811
    """A frame on the synthetic lake: the polygon is the mask's water, with shores off the texel grid."""
    from settlegen.frame import Frame
    from settlegen.test.conftest import settlement_params
    from settlegen.upsample import build_settlement_fields
    from settlegen.water import shorelines, water_regions

    fields = build_settlement_fields(globe, [lake_local], inner_center, side_km, settlement_params(side_km))
    frame = Frame(fields.geometry, side_km * 1000.0)  # type: ignore[arg-type]
    _, lakes = water_regions(fields, frame)
    lake = fields.textures["lake_mask"][0, ..., 0] > 0.5
    assert lake.any() and not lakes.is_empty and lakes.is_valid
    px = (frame.side_m / fields.tex_size) ** 2
    assert abs(lakes.area - lake.sum() * px) < 0.1 * lake.sum() * px
    # Every lake texel's centre is in the polygon, every dry one's out (but along the shore).
    xy = frame.texel_xy(fields.tex_size).reshape(-1, 2)
    import shapely
    inside = shapely.contains_xy(lakes, xy[:, 0], xy[:, 1]).reshape(lake.shape)
    assert (inside == lake).mean() > 0.995
    shore = shorelines(lakes, frame)
    assert shore.length > 0.0


def _river(width, length=6000.0, wet_tail=0):
    xy = np.column_stack([np.linspace(0.0, length, 61), np.zeros(61)])
    in_water = np.zeros(60, dtype=bool)
    if wet_tail:
        in_water[-wet_tail:] = True
    return RiverLine(xy=xy, flows=np.ones(60), width_m=np.full(60, float(width)), in_water=in_water, minor=False)


def test_a_wide_river_gets_islands_inside_a_channel_widened_round_them():
    params = SettlementRiverParams(islands_per_km=2.0, width_wander=0.0, bank_wander=0.0)
    line = _river(200.0)
    channel = natural_channels([line], MultiPolygon(), params)
    assert channel.is_valid
    islands = [ring for poly in channel.geoms for ring in poly.interiors]
    assert islands
    # Beside each island both arms keep about the river's width.
    for ring in islands:
        isle = shapely.Polygon(ring)
        c = isle.centroid
        across = shapely.LineString([(c.x, -1000.0), (c.x, 1000.0)]).intersection(channel)
        assert across.length > 0.8 * 200.0
    assert natural_channels([line], MultiPolygon(), params).equals(channel)      # the same every time
    assert not any(p.interiors for p in natural_channels([_river(30.0)], MultiPolygon(), params).geoms)


class _Frame:
    """Stands in for a frame: its box."""

    def box(self):
        return box(-9000.0, -9000.0, 9000.0, 9000.0)


def test_a_channel_runs_into_its_sea_with_no_shore_across_its_mouth():
    frame = _Frame()
    # The river's last stretch is under the sea in the parent, which starts at x = 5000 here.
    line = _river(100.0, wet_tail=12)
    sea = MultiPolygon([box(5000.0, -3000.0, 9000.0, 3000.0)])
    params = SettlementRiverParams(islands_per_km=0.0)
    channel = natural_channels([line], sea, params)
    assert channel.intersection(sea).area < 1e-6 * channel.area
    assert channel.distance(sea) < 1e-6
    shore = shorelines(sea, frame, channel)  # type: ignore[arg-type]
    mouth = box(4990.0, -30.0, 5010.0, 30.0)
    assert shore.intersection(mouth).length < 1e-6
    assert shorelines(sea, frame).intersection(mouth).length > 0  # type: ignore[arg-type]


def test_islands_are_long_narrow_and_along_the_river():
    params = SettlementRiverParams(islands_per_km=2.0, width_wander=0.0, bank_wander=0.0)
    channel = natural_channels([_river(200.0)], MultiPolygon(), params)
    for ring in (r for poly in channel.geoms for r in poly.interiors):
        x0, y0, x1, y1 = shapely.Polygon(ring).bounds
        assert x1 - x0 > 3.0 * (y1 - y0)         # the river runs along x
        assert abs(0.5 * (y0 + y1)) < 0.1 * 200.0  # in the channel's middle


def test_islands_keep_clear_of_a_confluence_partway_along():
    params = SettlementRiverParams(islands_per_km=5.0, width_wander=0.0, bank_wander=0.0)
    main = _river(200.0)
    alone = [shapely.Polygon(r) for p in natural_channels([main], MultiPolygon(), params).geoms for r in p.interiors]
    assert len(alone) >= 2
    # A tributary joining from the side right where the river alone has an island.
    x = alone[0].centroid.x
    trib = RiverLine(xy=np.array([[x, 3000.0], [x, 0.0]]), flows=np.ones(1), width_m=np.full(1, 80.0),
                     in_water=np.zeros(1, bool), minor=False)
    channel = natural_channels([main, trib], MultiPolygon(), params)
    rings = [shapely.Polygon(r) for p in channel.geoms for r in p.interiors]
    assert rings                                   # islands elsewhere along it still
    junction = shapely.Point(x, 0.0)
    assert all(isle.distance(junction) > 200.0 for isle in rings)


def _line(xy):
    xy = np.asarray(xy, dtype=float)
    k = len(xy) - 1
    return RiverLine(xy=xy, flows=np.ones(k), width_m=np.full(k, 50.0), in_water=np.zeros(k, bool), minor=False)


def test_an_end_hooking_back_in_at_the_frames_edge_is_cut_where_it_leaves():
    # Starts inside, runs out past x = 1000 and turns back in.
    hooked = _line([[900.0, 0.0], [980.0, 0.0], [1050.0, 30.0], [980.0, 80.0], [500.0, 100.0], [-1200.0, 100.0]])
    out = _tidy_ends(hooked, 1000.0)
    assert np.allclose(out.xy[0], [1050.0, 30.0]) and len(out.flows) == len(out.xy) - 1
    assert np.allclose(out.xy[-1], [-1200.0, 100.0])


def test_an_end_left_just_inside_is_carried_out_past_the_edge():
    short = _line([[-200.0, 0.0], [700.0, 0.0], [950.0, 0.0]])
    out = _tidy_ends(short, 1000.0)
    assert out.xy[-1][0] > 1000.0 and len(out.width_m) == len(out.xy) - 1
    assert np.allclose(out.xy[0], [-200.0, 0.0])                 # far from the edge: left alone


def test_a_river_ending_short_of_the_sea_runs_on_into_it_and_a_source_does_not():
    # Ends 60 m short of a sea whose shore runs aslant (x = 5000 + y / 2).
    line = _river(100.0, length=4940.0)
    sea = MultiPolygon([shapely.Polygon([(5000.0, 0.0), (6500.0, 3000.0), (9000.0, 3000.0), (9000.0, -3000.0),
                                         (3500.0, -3000.0)])])
    params = SettlementRiverParams(islands_per_km=0.0, width_wander=0.0, bank_wander=0.0)
    channel = natural_channels([line], sea, params)
    # No land left between the channel and the sea across its mouth.
    mouth = shapely.box(4800.0, -30.0, 5100.0, 30.0).difference(sea)
    assert mouth.difference(channel).area < 0.01 * mouth.area
    # The source (x = 0, far from anything) stops there.
    assert channel.bounds[0] > -60.0


def test_a_wide_rivers_course_is_smoothed_over_its_width():
    # A sharp zig-zag in a 300 m wide river: the smoothed channel has no ballooned lobe.
    xy = np.array([[0.0, 0.0], [1000.0, 0.0], [1150.0, 300.0], [1300.0, 0.0], [3000.0, 0.0]])
    line = RiverLine(xy=xy, flows=np.ones(4), width_m=np.full(4, 300.0), in_water=np.zeros(4, bool), minor=False)
    params = SettlementRiverParams(islands_per_km=0.0, width_wander=0.0, bank_wander=0.0)
    smooth = natural_channels([line], MultiPolygon(), params)
    raw = natural_channels([line], MultiPolygon(), SettlementRiverParams(
        islands_per_km=0.0, width_wander=0.0, bank_wander=0.0, course_smoothing=0.0))
    assert smooth.bounds[3] < raw.bounds[3] - 50.0
