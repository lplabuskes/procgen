import numpy as np
import pytest

from region.sampling import GlobalSampler, anchor_means


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
    # The anchor's texel value is what a nearest-neighbour read would give.
    tex = mesh.face_textures["elevation"]
    values = sampler.anchor_values(tex, anchors)
    rows = (anchors // w) % h
    cols = anchors % w
    assert np.allclose(values[:, 0], tex[faces, rows, cols, 0])


def test_anchor_means_ignores_zero_weight_samples():
    values = np.array([[1.0], [3.0], [100.0]])
    anchors = np.array([0, 0, 0])
    weights = np.array([1.0, 1.0, 0.0])
    means, counts = anchor_means(values, anchors, weights, 2)
    assert means[0, 0] == pytest.approx(2.0)
    assert counts[0] == 2.0
    assert counts[1] == 0.0
