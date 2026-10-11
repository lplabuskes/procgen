"""A site's density, footprint and buildable ground."""
import numpy as np
import pytest
import shapely
from shapely.geometry import MultiLineString, MultiPolygon

from region.rivers import RegionRivers
from region.test.test_resources import globe  # noqa: F401  (fixture)
from settlegen.chain import SettlementRef
from realmgen.towns.types import Towns
from settlegen.footprint import (
    SettlementFootprintParams, _claim, _crossing_extra, _holes, _notches, _small_islands, base_density_per_ha,
    build_footprint, buildable_ground, site_slope, wet_texels,
)
from settlegen.frame import Frame
from settlegen.test.conftest import inner_center, lake_hex, lake_local, region, settlement_params, side_km  # noqa: F401
from settlegen.upsample import build_settlement_fields
from settlegen.water import SiteWater, shorelines, water_regions


def test_the_claim_takes_the_best_allowed_texels():
    priority = np.arange(16, dtype=float).reshape(4, 4)
    allowed = np.ones((4, 4), dtype=bool)
    allowed[3, 3] = False
    claimed = _claim(priority, allowed, 3)
    assert claimed.sum() == 3 and not claimed[3, 3]
    assert claimed[3, :3].all()
    assert _claim(priority, allowed, 99).sum() == 15


def test_holes_are_the_dry_ground_the_claim_encloses_not_islands_out_in_the_water():
    claimed = np.zeros((9, 9), dtype=bool)
    claimed[1:6, 1:6] = True
    claimed[3, 3] = False                     # a hole in the claim
    wet = np.zeros((9, 9), dtype=bool)
    wet[6:9, 0:9] = True
    wet[7, 7] = False                         # an island, enclosed by water alone
    claimed[5, 2] = False                     # a notch on the shore: claim and water enclose it
    holes = _holes(claimed, wet, 0.25)
    assert holes[3, 3] and holes[5, 2] and not holes[7, 7]
    assert holes.sum() == 2


def test_ground_across_water_is_farther_by_the_crossing():
    wet = np.zeros((60, 60), dtype=bool)
    wet[:, 30:36] = True                                # a channel 6 texels (60 m) wide
    extra = _crossing_extra(wet, 10.0, (30.0, 15.0), 3.0)
    assert np.abs(extra[:, :25]).max() < 15.0           # this side: nothing to cross
    # The far side: the channel's 60 m counted three times, give or take the coarse grid.
    assert abs(float(np.median(extra[:, 40:])) - 120.0) < 40.0


def test_only_islands_big_enough_are_built_on():
    wet = np.zeros((40, 40), dtype=bool)
    wet[:, 10:30] = True
    wet[5:8, 15:18] = False                             # 9 texels
    wet[20:30, 15:25] = False                           # 100
    small = _small_islands(wet, 50.0)
    assert small[6, 16] and not small[25, 20] and not small[0, 0] and not small[0, 35]


def test_notches_narrower_than_the_disc_are_filled():
    claimed = np.zeros((40, 40), dtype=bool)
    claimed[5:35, 5:35] = True
    claimed[5:25, 18:21] = False                        # a 3-texel notch from the top edge
    wet = np.zeros_like(claimed)
    filled = _notches(claimed, wet, 3)
    assert filled[10:20, 18:21].all() and not filled[:5].any()


@pytest.fixture(scope="module")
def site(globe, lake_local, inner_center, side_km):  # noqa: F811
    """A frame on the synthetic lake, with its water."""
    fields = build_settlement_fields(globe, [lake_local], inner_center, side_km, settlement_params(side_km))
    frame = Frame(fields.geometry, side_km * 1000.0)  # type: ignore[arg-type]
    sea, lakes = water_regions(fields, frame)
    water = SiteWater(sea=sea, lakes=lakes, shorelines=shorelines(lakes, frame), lines=[], channels=MultiPolygon(),
                      rivers=RegionRivers(region_id=0, paths=[]))
    return fields, frame, water


class _Globe:
    """Stands in for the world: its seed and the walls of its towns."""

    seed = 4

    def __init__(self, walls=(False, True)):
        n = len(walls)
        self._towns = Towns(positions=np.zeros((n, 3)), population=np.ones(n), face_id=np.zeros(n, dtype=np.int64),
                            parent=np.zeros(n, dtype=np.int64), kind=np.zeros(n, dtype=np.int8),
                            has_walls=np.asarray(walls, dtype=bool))

    def get_feature(self, name):
        return self._towns if name == "towns" else None


def test_density_grows_with_size_and_walls_and_varies_by_settlement():
    world = _Globe()
    flat = SettlementFootprintParams(density_jitter=0.0)
    small = base_density_per_ha(world, SettlementRef("town", 0), 500.0, flat)  # type: ignore[arg-type]
    big = base_density_per_ha(world, SettlementRef("town", 0), 50_000.0, flat)  # type: ignore[arg-type]
    walled = base_density_per_ha(world, SettlementRef("town", 1), 500.0, flat)  # type: ignore[arg-type]
    assert small < big and walled == pytest.approx(small * flat.walled_density_mult)
    params = SettlementFootprintParams()
    ref = SettlementRef("village", 7, map=(1, 2))
    one = base_density_per_ha(world, ref, 300.0, params)  # type: ignore[arg-type]
    assert one == base_density_per_ha(world, ref, 300.0, params)  # type: ignore[arg-type]
    spread = [base_density_per_ha(world, SettlementRef("village", k, map=(1, 2)), 300.0, params)  # type: ignore
              for k in range(40)]
    assert np.std(np.log(spread)) > 0.15
    assert min(spread) >= params.min_density_per_ha and max(spread) <= params.max_density_per_ha


def _footprint(site, people, density, roads=(), params=None, k=3):
    fields, frame, water = site
    return build_footprint(fields, frame, water, SettlementRef("town", k), people, density, 4, roads, params)


def _people(site, texels):
    """People enough to claim ``texels`` fine texels at 100/ha."""
    fields, frame, _ = site
    return texels * (frame.side_m / fields.tex_size) ** 2 / 10_000.0 * 100.0


def test_a_footprint_claims_its_people_at_its_density_off_the_water_around_its_centre(site):
    fields, frame, water = site
    fp = _footprint(site, _people(site, 60), 100.0)
    assert fp.density_per_ha == pytest.approx(100.0 * fp.constraint)
    px = (frame.side_m / fields.tex_size) ** 2
    assert fp.target == round(_people(site, 60) / fp.density_per_ha * 10_000.0 / px)
    wet = wet_texels(fields, frame, water)
    # The people: densely claimed, and the filled holes at their lower density.
    held = fp.claimed.sum() + SettlementFootprintParams().hole_density_mult * fp.holes.sum()
    assert abs(held - fp.target) <= 1 and not ((fp.claimed | fp.holes) & wet).any()
    assert fp.polygon.is_valid and not fp.polygon.is_empty
    built = (fp.claimed | fp.holes).sum() * px
    # (A ~60-texel claim on a 32-texel frame: the tracing rounds its small pieces at the shore.)
    assert abs(fp.polygon.area - built) < 0.1 * built
    assert fp.sparse.area <= fp.polygon.area * (1 + 1e-9)
    still = shapely.union_all([water.sea, water.lakes])
    assert fp.polygon.intersection(still).area < 1e-6 * fp.polygon.area
    # Nearest first: what it claimed is nearer its centre than the dry ground it left.
    d = np.hypot(*np.moveaxis(frame.texel_xy(fields.tex_size), -1, 0))
    assert d[fp.claimed].mean() < 0.5 * d[~fp.claimed & ~wet].mean()
    again = _footprint(site, _people(site, 60), 100.0)
    assert np.array_equal(fp.claimed, again.claimed)


def test_a_hemmed_in_site_packs_denser(site):
    fields, frame, _ = site
    slope = site_slope(fields, frame)
    open_ground = _footprint(site, _people(site, 60), 100.0, params=SettlementFootprintParams(buildable_max_slope=10.0))
    steep = SettlementFootprintParams(buildable_max_slope=float(np.percentile(slope, 30)))
    hemmed = _footprint(site, _people(site, 60), 100.0, params=steep)
    assert hemmed.constraint > open_ground.constraint >= 1.0
    assert hemmed.target < open_ground.target


def test_roads_pull_growth_along_them(site):
    fields, frame, water = site
    # A road from the centre out to the frame's edge.
    road = np.array([[0.0, 0.0], [frame.half, 0.0]])
    plain = _footprint(site, _people(site, 120), 100.0)
    pulled = _footprint(site, _people(site, 120), 100.0, roads=[road],
                        params=SettlementFootprintParams(road_pull=1.0))
    xy = frame.texel_xy(fields.tex_size)
    along = np.abs(xy[..., 1])          # distance to the road's line
    assert along[pulled.claimed].mean() < along[plain.claimed].mean()


def test_buildable_ground_is_the_dry_ground_not_too_steep(site):
    fields, frame, water = site
    ground = buildable_ground(fields, frame, water)
    assert ground.is_valid and not ground.is_empty
    still = shapely.union_all([water.sea, water.lakes])
    assert ground.intersection(still).area < 1e-6 * frame.box().area
    slope = site_slope(fields, frame)
    steep = float(np.percentile(slope, 50))
    flatter = buildable_ground(fields, frame, water, SettlementFootprintParams(buildable_max_slope=steep))
    assert flatter.area < ground.area


def test_no_water_means_no_wet_texels(site):
    fields, frame, _ = site
    dry = SiteWater(sea=MultiPolygon(), lakes=MultiPolygon(), shorelines=MultiLineString(), lines=[],
                    channels=MultiPolygon(), rivers=RegionRivers(region_id=0, paths=[]))
    wet = wet_texels(fields, frame, dry)
    expected = fields.textures["lake_mask"][0, ..., 0] > 0.5
    assert np.array_equal(wet, expected)
