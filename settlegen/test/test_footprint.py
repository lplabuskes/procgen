"""A site's footprint and buildable ground: the local footprint's area, off the water, compact."""
import numpy as np
import pytest
import shapely
from shapely.geometry import MultiLineString, MultiPolygon

import settlegen.footprint as footprint
from region.rivers import RegionRivers
from region.test.test_resources import globe  # noqa: F401  (fixture)
from settlegen.chain import SettlementRef
from settlegen.footprint import (
    SettlementFootprintParams, _claim, build_footprint, buildable_ground, site_slope, wet_texels,
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


class _Owner:
    """Stands in for the owning local map: only its fields are read."""

    def __init__(self, fields):
        self.fields = fields


def _check(fp, fields, frame, water):
    px = (frame.side_m / fields.tex_size) ** 2
    wet = wet_texels(fields, frame, water)
    assert fp.claimed.sum() == fp.target and not (fp.claimed & wet).any()
    assert fp.polygon.is_valid and not fp.polygon.is_empty
    assert abs(fp.polygon.area - fp.target * px) < 0.05 * fp.target * px
    still = shapely.union_all([water.sea, water.lakes])
    assert fp.polygon.intersection(still).area < 1e-6 * fp.polygon.area


def test_a_city_footprint_has_its_local_area_off_the_water_and_follows_its_shares(
        site, lake_local, monkeypatch):  # noqa: F811
    fields, frame, water = site
    t = fields.tex_size
    # A local footprint over the frame's middle third, in 4 × 4 "local texels".
    share = np.zeros((t, t))
    share[t // 3:2 * t // 3, t // 3:2 * t // 3] = 1.0
    texel = (np.arange(t)[:, None] // 4) * 1000 + np.arange(t)[None, :] // 4
    monkeypatch.setattr(footprint, "city_shares", lambda *a: (share, texel))
    px_km2 = (frame.side_m / t) ** 2 / 1e6
    area = 0.8 * float(share.sum()) * px_km2
    fp = build_footprint(fields, frame, [_Owner(lake_local)], water, SettlementRef("city", 0), area, 4)  # type: ignore
    _check(fp, fields, frame, water)
    # Where it stands is the local footprint's: little outside it.
    assert (fp.claimed & (share == 0)).sum() < 0.2 * fp.target
    mean, _ = fp.share_error()
    assert mean < 0.25


def test_a_town_footprint_is_claimed_around_its_centre(site, lake_local):  # noqa: F811
    fields, frame, water = site
    px_km2 = (frame.side_m / fields.tex_size) ** 2 / 1e6
    area = 60 * px_km2
    fp = build_footprint(fields, frame, [_Owner(lake_local)], water, SettlementRef("town", 3), area, 4)  # type: ignore
    _check(fp, fields, frame, water)
    # Nearest first: what it claimed is nearer its centre than the dry ground it left.
    d = np.hypot(*np.moveaxis(frame.texel_xy(fields.tex_size), -1, 0))
    left = ~fp.claimed & ~wet_texels(fields, frame, water)
    assert d[fp.claimed].mean() < 0.5 * d[left].mean()
    # Seeded per settlement: the same twice, another town differs only by its noise.
    owner = [_Owner(lake_local)]
    again = build_footprint(fields, frame, owner, water, SettlementRef("town", 3), area, 4)  # type: ignore[arg-type]
    assert np.array_equal(fp.claimed, again.claimed)


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
