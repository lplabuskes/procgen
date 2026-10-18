"""Shared local-map fixtures, over the region tests' synthetic globe (with its
resource layers — ``globe``) and a coarse test-sized region of it.

``sites`` needs a ``local`` fixture from the requesting module."""
import numpy as np
import pytest

from localmap.upsample import LocalUpsampleParams
from region.geometry import ecef_to_cell
from region.rivers import build_region_rivers
from region.upsample import RegionUpsampleParams, build_region_fields

REGION_ID = 40      # the synthetic globe's lake region (see region/test/conftest.py)
TEX_SIZE = 16


def local_params(**overrides) -> LocalUpsampleParams:
    base: dict = dict(n_cols=6, tex_size=TEX_SIZE, water_spread_texels=4, biome_smoothing=3)
    base.update(overrides)
    return LocalUpsampleParams(**base)


@pytest.fixture(scope="module")
def region(globe):
    return build_region_fields(globe, REGION_ID, RegionUpsampleParams(n_cols=6, tex_size=TEX_SIZE))


@pytest.fixture(scope="module")
def region_rivers(region, globe):
    return build_region_rivers(region, globe)


@pytest.fixture(scope="module")
def lake_hex(region, globe) -> int:
    """The regional hex with the most lake, among those whose centre lies in the region's own cell."""
    counts = (region.textures["lake_mask"][..., 0] > 0.5).sum(axis=(1, 2))
    centers = region.geometry.local_to_ecef(region.grid.centers[:, 0], region.grid.centers[:, 1])
    here = ecef_to_cell(centers, globe.antimeridian) == region.region_id
    return int(np.argmax(np.where(here, counts, -1)))


@pytest.fixture(scope="module")
def river_hex(region, region_rivers) -> int:
    """The regional hex the most river vertices fall in."""
    counts = np.zeros(region.n_hexes)
    for positions, _, _ in region_rivers.paths:
        s, t, valid = region.geometry.ecef_to_local(np.asarray(positions, dtype=np.float64))
        d2 = (region.grid.centers[None, :, 0] - s[:, None]) ** 2 + (region.grid.centers[None, :, 1] - t[:, None]) ** 2
        np.add.at(counts, np.argmin(d2, axis=1)[valid], 1)
    return int(np.argmax(counts))


def water_mask(local) -> np.ndarray:
    """Ocean or lake, per local texel."""
    return (local.textures["ocean_mask"][..., 0] > 0.5) | (local.textures["lake_mask"][..., 0] > 0.5)


@pytest.fixture(scope="module")
def sites(local):
    """Three dry texels with plenty of dry ground around them, well apart."""
    dry = local.positions[local.in_hex & ~water_mask(local)]
    spacing = local.grid.texel_spacing_rad(local.tex_size)
    d = np.linalg.norm(dry[:, None, :] - dry[None, :, :], axis=2)
    roomy = (d < 6.0 * spacing).sum(axis=1)
    first = int(np.argmax(roomy))
    second = int(np.argmin(np.abs(d[first] - 8.0 * spacing) - 0.01 * roomy))
    third = int(np.argmax(np.minimum(d[first], d[second]) * (roomy > 0.5 * roomy.max())))
    return dry[[first, second, third]]
