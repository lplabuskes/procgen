"""build_region_map: the region tier end to end, Qt-free."""
import dataclasses

import numpy as np
import pytest

from realmgen.towns.types import TOWN_MARKET, Towns
from region.builder import build_region_map
from region.geometry import ecef_to_cell
from region.params import RegionParams
from region.streams import RegionStreamParams
from region.test.conftest import synthetic_globe  # noqa: F401  (fixture)
from region.upsample import RegionUpsampleParams

REGION_ID = 40


def small_params() -> RegionParams:
    return RegionParams(upsample=RegionUpsampleParams(n_cols=6, tex_size=16))


def land_towns(globe) -> Towns:
    """A market town on every land face's centroid."""
    centroids = np.asarray(globe.mesh.face_centroids, dtype=np.float64)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    land = np.flatnonzero(globe.mesh.face_textures["ocean_mask"][:, 3, 3, 0] < 0.5)
    return Towns(
        positions=centroids[land].copy(), population=np.full(land.size, 1_000.0),
        face_id=land.astype(np.int32), parent=np.full(land.size, -1, dtype=np.int32),
        kind=np.full(land.size, TOWN_MARKET, dtype=np.int8),
    )


@pytest.fixture
def towns(synthetic_globe):  # noqa: F811
    towns = land_towns(synthetic_globe)
    synthetic_globe.add_feature("towns", towns)
    yield towns
    synthetic_globe.remove_feature("towns")


def test_builds_the_region_and_refines_its_towns(synthetic_globe, towns):  # noqa: F811
    region = build_region_map(synthetic_globe, REGION_ID, small_params())
    assert region.region_id == REGION_ID and region.fields.grid.n_cols == 6
    assert region.rivers.paths
    here = ecef_to_cell(towns.site_positions, synthetic_globe.antimeridian) == REGION_ID
    assert here.any() and towns.refined[here].all() and not towns.refined[~here].any()


def test_rebuilds_identically(synthetic_globe, towns):  # noqa: F811
    first = build_region_map(synthetic_globe, REGION_ID, small_params())
    positions = towns.positions.copy()
    towns.reset_refinement()
    again = build_region_map(synthetic_globe, REGION_ID, small_params())
    np.testing.assert_array_equal(towns.positions, positions)
    assert len(again.rivers.paths) == len(first.rivers.paths)
    for (a, fa, _), (b, fb, _) in zip(first.rivers.paths, again.rivers.paths):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(fa, fb)
    assert len(again.roads.segments) == len(first.roads.segments)
    for key, tex in first.fields.textures.items():
        np.testing.assert_array_equal(tex, again.fields.textures[key])


def test_streams_are_cached_per_params(synthetic_globe, towns):  # noqa: F811
    region = build_region_map(synthetic_globe, REGION_ID, small_params())
    params = RegionStreamParams()
    first = region.streams(synthetic_globe, params)
    assert region.streams(synthetic_globe, dataclasses.replace(params)) is first
    assert region.streams(synthetic_globe, dataclasses.replace(params, min_nodes=params.min_nodes + 1)) is not first
