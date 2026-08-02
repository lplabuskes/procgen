"""Shared fixtures: a tiny synthetic globe with smooth analytic fields.

Small enough to build in a test (a depth-3 icosphere dual), but structurally
identical to the real thing — face textures on the same ``(N, H, W, C)`` grid the
pipeline writes — so the upsampler is exercised on its real contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.mesh import Mesh
from core.world import World

# Deliberately coarse: the tests need a *global* grid that a small test-sized
# region grid can genuinely refine, so each global texel collects many regional
# samples (the real pipeline gets there with 10k faces instead).
GLOBAL_TEX_SIZE = 8


@pytest.fixture(scope="session")
def synthetic_globe() -> World:
    dual = Mesh.icosphere(3).mesh_dual()
    positions = dual.face_texture_positions(GLOBAL_TEX_SIZE, GLOBAL_TEX_SIZE)
    x, y, z = positions[..., 0], positions[..., 1], positions[..., 2]

    elevation = (1.5 * np.sin(3.0 * x) * np.cos(2.0 * y) + 0.8 * z).astype(np.float32)
    dual.set_face_textures("elevation", elevation[..., None], n_channels=1)

    ocean = (elevation < -0.5).astype(np.float32)
    dual.set_face_textures("ocean_mask", ocean[..., None], n_channels=1)

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
    return globe
