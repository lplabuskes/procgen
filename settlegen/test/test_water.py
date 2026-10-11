"""A site's water: polygons from the refined masks, rivers chained across map seams."""
import numpy as np
import pytest

from localmap.test.test_store import REGION_ID, small_store
from region.geometry import ecef_to_cell
from region.test.test_builder import land_towns
from region.test.test_resources import globe  # noqa: F401  (fixture)
from settlegen.test.conftest import inner_center, lake_hex, lake_local, region, side_km  # noqa: F401
from settlegen.builder import build_settlement_map
from settlegen.chain import SettlementRef
from settlegen.params import SettlementMapParams
from settlegen.upsample import SettlementUpsampleParams
from settlegen.water import _chain, _Piece


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
