import os

import numpy as np
import pytest

from core import mesh_cache
from core.mesh import Mesh, PolygonMesh


@pytest.fixture
def cached_mesh(tmp_path):
    path = tmp_path / "mesh_0.npz"
    mesh_cache._save(Mesh.icosphere(1).mesh_dual(), path)
    return path


def test_derived_geometry_is_saved_and_reused(cached_mesh, monkeypatch):
    first = mesh_cache._load(cached_mesh)
    positions = np.array(first.face_texture_positions(4, 4))
    pairs = first.boundary_pixel_pairs(4)
    assert cached_mesh.with_name("mesh_0.texture_positions_4x4.npz").exists()
    assert cached_mesh.with_name("mesh_0.boundary_pairs_4.npz").exists()

    # A fresh load of the same mesh reads them back rather than recomputing.
    def boom(*_a, **_k):
        raise AssertionError("recomputed")
    monkeypatch.setattr(PolygonMesh, "_face_texture_positions", boom)
    monkeypatch.setattr(PolygonMesh, "_compute_boundary_pairs", boom)
    second = mesh_cache._load(cached_mesh)
    reread = second.face_texture_positions(4, 4)
    assert np.array_equal(reread, positions) and not reread.flags.writeable
    assert all(np.array_equal(a, b) and a.dtype == b.dtype for a, b in zip(second.boundary_pixel_pairs(4), pairs))


def test_rewritten_mesh_invalidates_saved_geometry(cached_mesh):
    mesh_cache._load(cached_mesh).boundary_pixel_pairs(4)
    st = cached_mesh.stat()
    os.utime(cached_mesh, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    store = mesh_cache._load(cached_mesh).geometry_store
    assert store is not None and store.load("boundary_pairs_4") is None


def test_plain_meshes_keep_geometry_in_memory_only():
    mesh = Mesh.icosphere(1).mesh_dual()
    assert mesh.geometry_store is None
    assert mesh.face_texture_positions(4, 4).shape == (len(mesh.faces), 4, 4, 3)
