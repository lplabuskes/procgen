"""A local map's frame and hex grid over one regional hexagon."""
import math

import numpy as np
import pytest

from localmap.geometry import LocalGeometry
from localmap.grid import LocalHexGrid, in_flat_hexagon
from region.geometry import RegionGeometry
from region.grid import RegionHexGrid

REGION_ID = 40


@pytest.fixture(scope="module")
def region_grid() -> RegionHexGrid:
    return RegionHexGrid(RegionGeometry(REGION_ID, math.pi), 14)


def _hexes(grid: RegionHexGrid) -> list[int]:
    # Interior, edge and corner hexes: the frame must work past the cell too.
    return [0, len(grid) // 2, len(grid) - 1]


def test_local_frame_round_trips_and_centres_on_the_parent(region_grid):
    rng = np.random.default_rng(0)
    for h in _hexes(region_grid):
        geom = LocalGeometry(region_grid.geometry, region_grid, h)
        a, b = rng.random(200), rng.random(200)
        a2, b2, valid = geom.ecef_to_local(geom.local_to_ecef(a, b))
        assert valid.all()
        assert np.allclose(a2, a, atol=1e-9) and np.allclose(b2, b, atol=1e-9)
        assert np.allclose(geom.center_ecef(), region_grid.centers_ecef()[h], atol=1e-12)


def test_local_frame_is_right_handed(region_grid):
    eps, c = 1e-4, 0.5
    for h in _hexes(region_grid):
        geom = LocalGeometry(region_grid.geometry, region_grid, h)
        f = geom.local_to_ecef
        ds = f(np.array([c + eps]), np.array([c]))[0] - f(np.array([c - eps]), np.array([c]))[0]
        dt = f(np.array([c]), np.array([c + eps]))[0] - f(np.array([c]), np.array([c - eps]))[0]
        assert np.dot(np.cross(ds, dt), geom.center_ecef()) > 0


def test_local_hexes_cover_the_parent_hexagon_and_little_else(region_grid):
    geom = LocalGeometry(region_grid.geometry, region_grid, len(region_grid) // 2)
    grid = LocalHexGrid(geom, 16)
    full = RegionHexGrid(geom, 16)
    assert len(grid) < 0.8 * len(full)
    # Every point of the parent hexagon lies in some kept local hexagon.
    rng = np.random.default_rng(1)
    pts = rng.random((20000, 2))
    pts = pts[in_flat_hexagon(pts - 0.5, 0.5)]
    d = np.linalg.norm(pts[:, None, :] - grid.centers[None, :, :], axis=-1).min(axis=1)
    assert d.max() <= grid.radius + 1e-12


def test_in_map_is_the_parent_hexagon(region_grid):
    geom = LocalGeometry(region_grid.geometry, region_grid, 0)
    grid = LocalHexGrid(geom, 16)
    t = 32
    domain = grid.texture_in_hexagon(t) & grid.texture_in_map(t)
    # Texel counts per unit area: the hexagon covers 3√3/8 of its square (up to
    # the texel-quantised rims).
    texel_area = (2.0 * grid.radius / t) ** 2
    assert domain.sum() * texel_area == pytest.approx(3.0 * math.sqrt(3.0) / 8.0, rel=0.02)
