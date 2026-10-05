"""Shared fixtures: a tiny synthetic globe with smooth analytic fields.

A depth-3 icosphere dual with face textures on the pipeline's ``(N, H, W, C)`` grid.

The water fields are self-consistent (the refinement re-floods them): the ocean
is exactly the below-sea-level set, and the lake a level set of a carved bowl on
a highland, with ``filled_height`` holding its surface.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.mesh import Mesh
from core.world import World
from region.geometry import ecef_to_cell
from worldgen.hydrology.rivers import RiverGenerator

# Coarse, so each global texel collects many samples from a small test region grid.
GLOBAL_TEX_SIZE = 8

# The lake is carved into this region's highest ground (it has a coastline and a
# decent peak), so it has a rim on every side inside the refined grid.
LAKE_REGION = 40
# A flat-bottomed pan (sixth power), as real lakes are shallow (median ~40 m).
LAKE_DEPTH_KM = 0.6
LAKE_RADIUS_RAD = 0.08
# The water stands this far below the rim it is walled in by, leaving ~50 m of it.
LAKE_FREEBOARD_KM = 0.55
# Keeps the level set to the crater (far beyond any shoreline travel).
LAKE_CAP_RAD = 0.15
# Upstream texels at mean rain: a few dozen channels in the test region.
STREAM_THRESHOLD = 8.0


@pytest.fixture(scope="session")
def synthetic_globe() -> World:
    dual = Mesh.icosphere(3).mesh_dual()
    positions = dual.face_texture_positions(GLOBAL_TEX_SIZE, GLOBAL_TEX_SIZE)
    x, y, z = positions[..., 0], positions[..., 1], positions[..., 2]
    base = 1.5 * np.sin(3.0 * x) * np.cos(2.0 * y) + 0.8 * z

    inside = ecef_to_cell(positions.reshape(-1, 3), float(np.pi)) == LAKE_REGION
    peak = np.where(inside)[0][np.argmax(base.reshape(-1)[inside])]
    centre = positions.reshape(-1, 3)[peak].astype(np.float64)
    centre /= np.linalg.norm(centre)
    d = np.arccos(np.clip(positions @ centre, -1.0, 1.0))

    elevation = (base - LAKE_DEPTH_KM * np.exp(-(d / LAKE_RADIUS_RAD) ** 6)).astype(np.float32)
    dual.set_face_textures("elevation", elevation[..., None], n_channels=1)

    ocean = (elevation <= 0.0).astype(np.float32)
    dual.set_face_textures("ocean_mask", ocean[..., None], n_channels=1)

    lake_surface = float(base.reshape(-1)[peak]) - LAKE_FREEBOARD_KM
    lake = (d < LAKE_CAP_RAD) & (elevation <= lake_surface)
    dual.set_face_textures("lake_mask", lake[..., None].astype(np.float32), n_channels=1)
    filled = np.where(lake, lake_surface, elevation).astype(np.float32)
    dual.set_face_textures("filled_height", filled[..., None], n_channels=1)

    season = np.arange(4, dtype=np.float32)
    temperature = (
        25.0 - 40.0 * np.abs(z)[..., None] + 6.0 * np.cos(season * np.pi / 2.0)
    ).astype(np.float32)
    dual.set_face_textures("temperature", temperature, n_channels=4)

    precipitation = (
        200.0 + 150.0 * (1.0 + np.sin(4.0 * x + y))[..., None] * np.ones(4, dtype=np.float32)
    ).astype(np.float32)
    dual.set_face_textures("precipitation", precipitation, n_channels=4)

    globe = World.from_mesh(dual)
    globe.seed = 7
    globe.antimeridian = float(np.pi)

    # The stream network, from the real generator over these fields.
    rivers = RiverGenerator(
        dual, elevation[..., None], min_flow=STREAM_THRESHOLD, sea_level=0.0,
        precipitation_textures=precipitation, temperature_textures=temperature,
        ocean_mask_textures=ocean[..., None],
    )
    rivers.generate()
    globe.add_feature("streams", rivers.stream_network(STREAM_THRESHOLD))
    return globe
