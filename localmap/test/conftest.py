"""Shared local-map fixtures, over the region tests' synthetic globe (with its
resource layers — ``globe``) and a coarse test-sized region of it."""
import numpy as np
import pytest

from localmap.upsample import LocalUpsampleParams
from region.rivers import build_region_rivers
from region.test.conftest import synthetic_globe  # noqa: F401
from region.test.test_resources import globe  # noqa: F401
from region.upsample import RegionUpsampleParams, build_region_fields

REGION_ID = 40      # the synthetic globe's lake region (see region/test/conftest.py)
TEX_SIZE = 16


def local_params(**overrides) -> LocalUpsampleParams:
    base: dict = dict(n_cols=6, tex_size=TEX_SIZE, water_spread_texels=4, biome_smoothing=3)
    base.update(overrides)
    return LocalUpsampleParams(**base)


@pytest.fixture(scope="module")
def region(globe):  # noqa: F811
    return build_region_fields(globe, REGION_ID, RegionUpsampleParams(n_cols=6, tex_size=TEX_SIZE))


@pytest.fixture(scope="module")
def region_rivers(region, globe):  # noqa: F811
    return build_region_rivers(region, globe)


@pytest.fixture(scope="module")
def lake_hex(region) -> int:
    return int(np.argmax((region.textures["lake_mask"][..., 0] > 0.5).sum(axis=(1, 2))))


@pytest.fixture(scope="module")
def river_hex(region, region_rivers) -> int:
    """The regional hex the most river vertices fall in."""
    counts = np.zeros(region.n_hexes)
    for positions, _, _ in region_rivers.paths:
        s, t, valid = region.geometry.ecef_to_local(np.asarray(positions, dtype=np.float64))
        d2 = (region.grid.centers[None, :, 0] - s[:, None]) ** 2 + (region.grid.centers[None, :, 1] - t[:, None]) ** 2
        np.add.at(counts, np.argmin(d2, axis=1)[valid], 1)
    return int(np.argmax(counts))
