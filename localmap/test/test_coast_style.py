"""A local map styles and reshapes its shores from the ruggedness its region passes down."""
import numpy as np
import pytest

from localmap.test.conftest import REGION_ID, TEX_SIZE, local_params
from localmap.upsample import build_local_fields, local_conservation_report
from region.upsample import SEA_LEVEL_KM, RegionUpsampleParams, build_region_fields


@pytest.fixture(scope="module")
def styled_region(globe):
    tex = globe.mesh.face_textures
    tex["ruggedness"] = np.full(tex["elevation"].shape, 0.8, dtype=np.float32)
    try:
        return build_region_fields(globe, REGION_ID, RegionUpsampleParams(n_cols=6, tex_size=TEX_SIZE))
    finally:
        tex.pop("ruggedness")


@pytest.fixture(scope="module")
def shore_hex(styled_region) -> int:
    water = (styled_region.textures["ocean_mask"][..., 0] > 0.5) | (styled_region.textures["lake_mask"][..., 0] > 0.5)
    frac = (water & styled_region.in_hex).sum(axis=(1, 2)) / styled_region.in_hex.sum(axis=(1, 2))
    return int(np.argmin(np.abs(frac - 0.5)))


def test_reshaped_local_map_still_refines_its_region(globe, styled_region, shore_hex):
    on = build_local_fields(globe, styled_region, shore_hex, local_params())
    off = build_local_fields(globe, styled_region, shore_hex, local_params(reshape_coast=False))
    assert np.allclose(on.textures["ruggedness"], 0.8)
    assert not np.array_equal(on.textures["elevation"], off.textures["elevation"])
    assert local_conservation_report(on, styled_region)["elevation"]["max"] < 1e-3
    elev = on.textures["elevation"].reshape(-1)
    ocean = on.textures["ocean_mask"].reshape(-1) > 0.5
    assert np.all(elev[ocean] <= SEA_LEVEL_KM)
