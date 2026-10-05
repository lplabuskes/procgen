"""The continuous, mean-preserving surface through per-face values."""
import numpy as np
import pytest

from core.mesh import Mesh
from region.smooth import FaceFanField


@pytest.fixture(scope="module")
def mesh():
    return Mesh.icosphere(3).mesh_dual()


@pytest.fixture(scope="module")
def blobby(mesh):
    """Sparse non-negative values like danger: a few blocks of faces, 0 elsewhere."""
    rng = np.random.default_rng(3)
    values = np.zeros(len(mesh.faces))
    for seed in rng.choice(len(mesh.faces), 6, replace=False):
        values[seed] = rng.uniform(0.4, 1.0)
        for j in mesh.face_adjacency[seed]:
            values[j] = max(values[j], rng.uniform(0.1, 0.6))
    return values


def _face_samples(mesh, face: int, n: int, rng) -> np.ndarray:
    """Uniform points over one face's fan triangles (area-weighted)."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    c = np.asarray(mesh.face_centroids[face], dtype=np.float64)
    c /= np.linalg.norm(c)
    ring = list(mesh.faces[face])
    tris = [(c, verts[a], verts[b]) for a, b in zip(ring, ring[1:] + ring[:1])]
    areas = np.array([0.5 * np.linalg.norm(np.cross(a - o, b - o)) for o, a, b in tris])
    pick = rng.choice(len(tris), n, p=areas / areas.sum())
    r1, r2 = rng.random(n), rng.random(n)
    flip = r1 + r2 > 1
    r1, r2 = np.where(flip, 1 - r1, r1), np.where(flip, 1 - r2, r2)
    o = np.array([tris[k][0] for k in pick])
    a = np.array([tris[k][1] for k in pick])
    b = np.array([tris[k][2] for k in pick])
    p = o + r1[:, None] * (a - o) + r2[:, None] * (b - o)
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def test_face_means_are_exact(mesh, blobby):
    field = FaceFanField(mesh, blobby)
    rng = np.random.default_rng(0)
    for face in [*np.flatnonzero(blobby > 0)[:12], *np.flatnonzero(blobby == 0)[:4]]:
        mean = field(_face_samples(mesh, int(face), 40_000, rng)).mean()
        assert mean == pytest.approx(blobby[face], abs=0.02)


def test_non_negative_and_zero_on_safe_faces(mesh, blobby):
    field = FaceFanField(mesh, blobby)
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(50_000, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    assert field(pts).min() >= -1e-12
    safe = int(np.flatnonzero(blobby == 0)[0])
    assert np.abs(field(_face_samples(mesh, safe, 2_000, rng))).max() < 1e-12


def test_continuous_across_face_edges(mesh, blobby):
    """No steps: nearby points agree however the faces are assigned."""
    field = FaceFanField(mesh, blobby)
    rng = np.random.default_rng(2)
    p = rng.normal(size=(100_000, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    q = p + 1e-6 * rng.normal(size=p.shape)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    # The steepest slope is ~3·value over a face radius (~0.1 rad here).
    assert np.abs(field(p) - field(q)).max() < 1e-3


def test_a_uniform_block_has_a_dome_in_the_middle(mesh):
    """Rim faces rise to keep their means; the middle is a dome, not a dip, well below a 3× cone."""
    values = np.zeros(len(mesh.faces))
    core = 100
    ring = list(mesh.face_adjacency[core])
    for f in [core, *ring]:
        values[f] = 0.5
        for g in mesh.face_adjacency[f]:
            values[g] = 0.5
    field = FaceFanField(mesh, values)
    assert field.centre_values[core] >= field.centre_values[ring].max() - 1e-9
    assert field.centre_values[values > 0].max() < 2.0 * 0.5


def test_signed_values_keep_their_means(mesh):
    rng = np.random.default_rng(4)
    values = rng.normal(size=len(mesh.faces))
    field = FaceFanField(mesh, values)
    for face in range(0, len(values), 97):
        mean = field(_face_samples(mesh, face, 20_000, rng)).mean()
        assert mean == pytest.approx(values[face], abs=0.03)


def test_wrong_length_is_an_error(mesh):
    with pytest.raises(ValueError):
        FaceFanField(mesh, np.zeros(3))
