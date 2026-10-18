"""Shared settlement-map fixtures, over the local tests' synthetic globe and
region: the lake hex's local map and its neighbour's, and frames inside one
and across the seam between them."""
import numpy as np
import pytest
from shapely.geometry import MultiPolygon

from localmap.test.fixtures import local_params
from localmap.upsample import build_local_fields
from region.geometry import ecef_to_cell
from region.rivers import RegionRivers
from settlegen.frame import Frame
from settlegen.site.types import SiteWater
from settlegen.site.water import shorelines, water_regions
from settlegen.terrain.builder import (
    SettlementUpsampleParams, build_settlement_fields, km_per_local_unit, with_local_detail,
)

# Texels across a test frame.
FRAME_TEXELS = 32


def settlement_params(side_km: float, **overrides) -> SettlementUpsampleParams:
    base: dict = dict(texel_m=side_km * 1000.0 / FRAME_TEXELS, water_spread_texels=4, biome_smoothing=3)
    base.update(overrides)
    return with_local_detail(SettlementUpsampleParams(**base), local_params())


@pytest.fixture(scope="module")
def neighbour_hex(region, lake_hex, globe) -> int:
    """The lake hex's nearest neighbour with its centre in the region's own cell."""
    centers = region.grid.centers
    here = ecef_to_cell(region.geometry.local_to_ecef(centers[:, 0], centers[:, 1]), globe.antimeridian)
    d = np.hypot(*(centers - centers[lake_hex]).T)
    d[(here != region.region_id) | (np.arange(len(centers)) == lake_hex)] = np.inf
    return int(np.argmin(d))


@pytest.fixture(scope="module")
def lake_local(globe, region, lake_hex):
    return build_local_fields(globe, region, lake_hex, local_params())


@pytest.fixture(scope="module")
def neighbour_local(globe, region, neighbour_hex):
    return build_local_fields(globe, region, neighbour_hex, local_params())


@pytest.fixture(scope="module")
def side_km(lake_local) -> float:
    """A fifth of a local map across: inside the hex from its centre, two maps at the seam."""
    return 0.2 * km_per_local_unit(lake_local)


@pytest.fixture(scope="module")
def inner_center(region, lake_hex) -> np.ndarray:
    c = region.grid.centers[lake_hex]
    return region.geometry.local_to_ecef(c[:1], c[1:])[0]


@pytest.fixture(scope="module")
def seam_center(region, lake_hex, neighbour_hex) -> np.ndarray:
    """The midpoint of the edge the two hexes share."""
    c = 0.5 * (region.grid.centers[lake_hex] + region.grid.centers[neighbour_hex])
    return region.geometry.local_to_ecef(c[:1], c[1:])[0]


@pytest.fixture(scope="module")
def site(globe, lake_local, inner_center, side_km):
    """A frame on the synthetic lake, with its water."""
    fields = build_settlement_fields(globe, [lake_local], inner_center, side_km, settlement_params(side_km))
    frame = Frame(fields.geometry, side_km * 1000.0)  # type: ignore[arg-type]
    sea, lakes = water_regions(fields, frame)
    water = SiteWater(sea=sea, lakes=lakes, shorelines=shorelines(lakes, frame), lines=[], channels=MultiPolygon(),
                      rivers=RegionRivers(region_id=0, paths=[]))
    return fields, frame, water
