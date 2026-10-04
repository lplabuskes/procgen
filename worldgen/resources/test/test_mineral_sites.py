"""Mineral workings: each deposit split into workings on its face's own land."""
import numpy as np
from scipy.spatial import KDTree

from core.world import World
from worldgen.resources.minerals import mineral_workings, place_mineral_sites, site_workers


def _sphere_grid(n_faces: int = 6, size: int = 8):
    """Six cube faces as (N, H, W, 3) texel centres, with their centroids."""
    t = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    u, v = np.meshgrid(t, t, indexing="ij")
    one = np.ones_like(u)
    planes = [(one, u, v), (-one, u, v), (u, one, v), (u, -one, v), (u, v, one), (u, v, -one)][:n_faces]
    grids = np.stack([np.stack(p, axis=-1) for p in planes])
    grids /= np.linalg.norm(grids, axis=-1, keepdims=True)
    centroids = np.eye(3)[[0, 0, 1, 1, 2, 2]] * np.array([1, -1, 1, -1, 1, -1])[:, None]
    return grids, centroids.astype(np.float64)


def _place(land, faces):
    grids, centroids = _sphere_grid()
    sites = place_mineral_sites(np.asarray(faces), grids, land, centroids,
                                min_sites=2, max_sites=5, share_concentration=1.0, seed=3)
    return sites, grids, centroids


def test_workings_lie_on_their_faces_land_and_share_the_deposit():
    land = np.ones((6, 8, 8), dtype=bool)
    land[0, :, :4] = False
    (pos, dep, share), grids, centroids = _place(land, [0, 2, 2, 5])
    counts = np.bincount(dep, minlength=4)
    assert ((counts >= 2) & (counts <= 5)).all()
    np.testing.assert_allclose(np.bincount(dep, weights=share), 1.0, rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(pos, axis=1), 1.0, rtol=1e-6)
    faces = np.array([0, 2, 2, 5])
    assert (KDTree(centroids).query(pos)[1] == faces[dep]).all()
    # Face 0's workings are on its dry half (each within its own texel).
    on0 = pos[dep == 0]
    dry = grids[0][land[0]]
    spacing = np.linalg.norm(np.diff(grids[0], axis=1), axis=-1).max()
    assert (KDTree(dry).query(on0)[0] <= 0.75 * spacing).all()
    assert len({tuple(np.round(p, 6)) for p in pos}) == len(pos)


def test_a_face_without_land_keeps_one_working_at_its_centre():
    land = np.ones((6, 8, 8), dtype=bool)
    land[1] = False
    (pos, dep, share), _, centroids = _place(land, [1])
    assert len(pos) == 1 and share[0] == 1.0
    np.testing.assert_allclose(pos[0], centroids[1], atol=1e-6)


def test_workings_are_deterministic():
    land = np.ones((6, 8, 8), dtype=bool)
    (a, _, sa), _, _ = _place(land, [0, 3])
    (b, _, sb), _, _ = _place(land, [0, 3])
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(sa, sb)


def test_site_workers_split_each_deposits_miners_by_share():
    out = site_workers(np.array([0, 0, 1]), np.array([0.25, 0.75, 1.0]), np.array([100.0, 40.0]))
    np.testing.assert_allclose(out, [25.0, 75.0, 40.0])


def test_mineral_workings_without_sites_is_one_working_per_deposit():
    world = World.__new__(World)
    feats = {"minerals": (np.array([[0.0, 0.0, 2.0]]), np.array([1], np.int8), np.array([0.5], np.float32))}
    world.get_feature = feats.get  # type: ignore[method-assign]
    workings = mineral_workings(world)
    assert workings is not None
    pos, dep, share = workings
    np.testing.assert_allclose(pos, [[0.0, 0.0, 1.0]])
    assert dep.tolist() == [0] and share.tolist() == [1.0]
