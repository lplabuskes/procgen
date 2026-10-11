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
    SettlementFootprintParams, _claim, base_density_per_ha, build_footprint, buildable_ground, site_slope, wet_texels,
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
    assert fp.claimed.sum() == fp.target and not (fp.claimed & wet).any()
    assert fp.polygon.is_valid and not fp.polygon.is_empty
    assert abs(fp.polygon.area - fp.target * px) < 0.05 * fp.target * px
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
