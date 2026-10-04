import numpy as np
import pytest

from region.sampling import GlobalSampler, anchor_means, settle_owners


def test_locate_returns_nearest_face(synthetic_globe):
    mesh = synthetic_globe.mesh
    sampler = GlobalSampler(mesh)
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((200, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    faces, u, v = sampler.locate(pts)
    brute = np.argmax(pts @ np.asarray(mesh.face_centroids).T, axis=1)
    assert np.array_equal(faces, brute)
    # Points inside their own face land in the face's UV square.
    assert u.min() > -0.25 and u.max() < 1.25
    assert v.min() > -0.25 and v.max() < 1.25


def test_face_centroid_maps_near_uv_centre(synthetic_globe):
    sampler = GlobalSampler(synthetic_globe.mesh)
    centroids = np.asarray(synthetic_globe.mesh.face_centroids)
    faces, u, v = sampler.locate(centroids)
    assert np.array_equal(faces, np.arange(len(centroids)))
    assert np.allclose(u, 0.5, atol=0.1)
    assert np.allclose(v, 0.5, atol=0.1)


def test_sample_reproduces_a_constant_texture(synthetic_globe):
    mesh = synthetic_globe.mesh
    sampler = GlobalSampler(mesh)
    tex = np.full((len(mesh.faces), 8, 8, 2), [3.0, -1.0], dtype=np.float32)
    faces, u, v = sampler.locate(np.asarray(mesh.face_centroids))
    values = sampler.sample(tex, faces, u, v)
    assert values.shape == (len(mesh.faces), 2)
    assert np.allclose(values, [3.0, -1.0])


def test_sample_interpolates_linearly(synthetic_globe):
    """A texture that is linear in u must be read back linearly."""
    mesh = synthetic_globe.mesh
    sampler = GlobalSampler(mesh)
    n, w = len(mesh.faces), 8
    ramp = np.linspace(0.0, 1.0, w, dtype=np.float32)
    tex = np.broadcast_to(ramp, (n, w, w)).copy()[..., None]
    faces = np.zeros(5, dtype=np.int64)
    u = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    v = np.full(5, 0.5)
    assert np.allclose(sampler.sample(tex, faces, u, v)[:, 0], u, atol=1e-6)


def test_anchors_are_in_range_and_stable(synthetic_globe):
    mesh = synthetic_globe.mesh
    sampler = GlobalSampler(mesh)
    h, w = mesh.face_textures["elevation"].shape[1:3]
    rng = np.random.default_rng(3)
    pts = rng.standard_normal((500, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    faces, u, v = sampler.locate(pts)
    anchors = sampler.anchors(faces, u, v, h, w)
    assert anchors.min() >= 0
    assert anchors.max() < len(mesh.faces) * h * w
    # The anchor's value is its own texel's — usually in the located face, but
    # a point by a seam may belong to the neighbour's texel there.
    tex = mesh.face_textures["elevation"]
    values = sampler.anchor_values(tex, anchors)
    a_faces = anchors // (h * w)
    rows = (anchors // w) % h
    cols = anchors % w
    assert np.allclose(values[:, 0], tex[a_faces, rows, cols, 0])
    assert np.mean(a_faces == faces) > 0.8
    # Every anchor owns itself, and its texel centre is near the point: within
    # a texel normally, two hops of under a texel each for a seam's remap.
    assert np.array_equal(sampler.owners(h, w)[anchors], anchors)
    centres = mesh.face_texture_positions(h, w).reshape(-1, 3)[anchors]
    spacing = np.sqrt(np.mean(mesh.face_areas)) / (w - 1)
    assert np.max(np.linalg.norm(centres - pts, axis=1)) < 2.0 * spacing


def test_owners_are_idempotent_and_mostly_self(synthetic_globe):
    mesh = synthetic_globe.mesh
    sampler = GlobalSampler(mesh)
    h, w = mesh.face_textures["elevation"].shape[1:3]
    owner = sampler.owners(h, w)
    assert np.array_equal(owner[owner], owner)
    # Only the seams and the over-hang past each polygon belong to a neighbour.
    assert 0.5 < np.mean(owner == np.arange(owner.size)) < 1.0


def test_anchor_means_ignores_zero_weight_samples():
    values = np.array([[1.0], [3.0], [100.0]])
    anchors = np.array([0, 0, 0])
    weights = np.array([1.0, 1.0, 0.0])
    means, counts = anchor_means(values, anchors, weights, 2)
    assert means[0, 0] == pytest.approx(2.0)
    assert counts[0] == 2.0
    assert counts[1] == 0.0


def test_settle_owners_resolves_chains_and_cycles():
    # 0 self; 1→0 chain; 2→3→4→2 cycle with 5→2 feeding it; 6↔7 two-cycle.
    owner = np.array([0, 0, 3, 4, 2, 2, 7, 6])
    settled = settle_owners(owner)
    assert np.array_equal(settled[settled], settled)
    assert np.array_equal(settled, [0, 0, 2, 2, 2, 2, 6, 6])
