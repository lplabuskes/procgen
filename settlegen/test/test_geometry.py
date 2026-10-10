"""A settlement map's frame and its one-cell grid."""
import numpy as np
import pytest

from localmap.geometry import LocalGeometry
from settlegen.geometry import SettlementGeometry
from settlegen.grid import SettlementGrid


@pytest.fixture(scope="module")
def geometry(region, lake_hex):
    local = LocalGeometry(region.geometry, region.grid, lake_hex)
    return SettlementGeometry(local, np.array([0.4, 0.55]), 0.05)


def test_frame_round_trips_and_centres_on_its_point(geometry):
    rng = np.random.default_rng(0)
    a, b = rng.random(200), rng.random(200)
    a2, b2, valid = geometry.ecef_to_local(geometry.local_to_ecef(a, b))
    assert valid.all()
    assert np.allclose(a2, a, atol=1e-8) and np.allclose(b2, b, atol=1e-8)
    expected = geometry.parent.local_to_ecef(np.array([0.4]), np.array([0.55]))[0]
    assert np.allclose(geometry.center_ecef(), expected, atol=1e-12)


def test_frame_is_right_handed(geometry):
    eps, c = 1e-3, 0.5
    f = geometry.local_to_ecef
    ds = f(np.array([c + eps]), np.array([c]))[0] - f(np.array([c - eps]), np.array([c]))[0]
    dt = f(np.array([c]), np.array([c + eps]))[0] - f(np.array([c]), np.array([c - eps]))[0]
    assert np.dot(np.cross(ds, dt), geometry.center_ecef()) > 0


def test_one_cell_covers_the_frame(geometry):
    grid = SettlementGrid(geometry)
    assert len(grid) == 1 and grid.radius == pytest.approx(0.5)
    local = grid.texture_local(8)[0]
    assert np.allclose(local[0, 0], [1 / 16, 15 / 16]) and np.allclose(local[-1, -1], [15 / 16, 1 / 16])
    assert grid.texture_in_hexagon(8).all()


def test_neighbours_stop_only_at_the_frame_edge(geometry):
    t = 8
    nb = SettlementGrid(geometry).texel_neighbours(t).reshape(t, t, 4)
    edge = np.zeros((t, t), dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    assert (nb[~edge] >= 0).all()
    assert ((nb == -1).any(axis=-1) == edge).all()
    # up / down / left / right are the texels above, below, left and right.
    idx = np.arange(t * t).reshape(t, t)
    assert nb[3, 3, 0] == idx[2, 3] and nb[3, 3, 1] == idx[4, 3]
    assert nb[3, 3, 2] == idx[3, 2] and nb[3, 3, 3] == idx[3, 4]
