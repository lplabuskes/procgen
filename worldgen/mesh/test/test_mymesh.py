import numpy as np
import pytest
from mesh.mymesh import Mesh, PolygonMesh


class TestIcosphere:
    def test_depth0_counts(self):
        mesh = Mesh.icosphere(0)
        assert len(mesh.vertices) == 12
        assert len(mesh.faces) == 20

    def test_depth1_counts(self):
        # Each subdivision multiplies faces by 4 and adds 1 vertex per unique edge.
        # Icosahedron: 12 vertices, 20 faces, 30 edges → depth 1: 42 vertices, 80 faces.
        mesh = Mesh.icosphere(1)
        assert len(mesh.vertices) == 42
        assert len(mesh.faces) == 80

    def test_depth2_counts(self):
        # depth 2: 162 vertices, 320 faces
        mesh = Mesh.icosphere(2)
        assert len(mesh.vertices) == 162
        assert len(mesh.faces) == 320

    def test_vertices_on_unit_sphere(self):
        for depth in range(4):
            mesh = Mesh.icosphere(depth)
            norms = np.linalg.norm(mesh.vertices, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-6,
                                       err_msg=f"Failed at depth {depth}")

    def test_depth0_each_vertex_in_5_faces(self):
        mesh = Mesh.icosphere(0)
        face_count = {}
        for face in mesh.faces:
            for v in face:
                face_count[v] = face_count.get(v, 0) + 1
        assert all(c == 5 for c in face_count.values())

    def test_faces_reference_valid_vertices(self):
        mesh = Mesh.icosphere(2)
        n = len(mesh.vertices)
        for face in mesh.faces:
            assert all(0 <= v < n for v in face)

    def test_triangle_faces_have_outward_normals(self):
        """Every triangle normal should point away from the origin."""
        mesh = Mesh.icosphere(2)
        for face in mesh.faces:
            v0, v1, v2 = (mesh.vertices[i] for i in face)
            normal = np.cross(v1 - v0, v2 - v0)
            centroid = (v0 + v1 + v2) / 3
            assert np.dot(normal, centroid) > 0, (
                f"Face {face} has inward-pointing normal"
            )


class TestPolytopeMeshDual:
    def test_dual_icosphere0_counts(self):
        # Dual of icosahedron (12 verts, 20 faces) → dodecahedron (20 verts, 12 faces)
        mesh = Mesh.icosphere(0)
        dual = mesh.mesh_dual()
        assert len(dual.vertices) == 20
        assert len(dual.faces) == 12

    def test_dual_icosphere0_all_pentagons(self):
        mesh = Mesh.icosphere(0)
        dual = mesh.mesh_dual()
        assert all(len(f) == 5 for f in dual.faces)

    def test_dual_icosphere1_counts(self):
        # Dual of (42 verts, 80 faces) → (80 verts, 42 faces)
        mesh = Mesh.icosphere(1)
        dual = mesh.mesh_dual()
        assert len(dual.vertices) == 80
        assert len(dual.faces) == 42

    def test_dual_icosphere1_pentagon_and_hexagon_faces(self):
        # depth-1 icosphere dual: 12 pentagons + 30 hexagons = 42 faces
        mesh = Mesh.icosphere(1)
        dual = mesh.mesh_dual()
        sizes = [len(f) for f in dual.faces]
        assert sizes.count(5) == 12
        assert sizes.count(6) == 30

    def test_dual_vertices_on_unit_sphere(self):
        for depth in range(3):
            mesh = Mesh.icosphere(depth)
            dual = mesh.mesh_dual()
            norms = np.linalg.norm(dual.vertices, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-6,
                                       err_msg=f"Failed at depth {depth}")

    def test_dual_face_winding_outward(self):
        """Fan triangles of each dual face must have normals pointing outward."""
        for depth in range(3):
            mesh = Mesh.icosphere(depth)
            dual = mesh.mesh_dual()
            for face_idx, face in enumerate(dual.faces):
                verts = dual.vertices[np.array(face)]
                centroid = np.mean(verts, axis=0)
                centroid /= np.linalg.norm(centroid)
                v0 = verts[0]
                for i in range(1, len(face) - 1):
                    vi = verts[i]
                    vi1 = verts[i + 1]
                    normal = np.cross(vi - v0, vi1 - v0)
                    assert np.dot(normal, centroid) > 0, (
                        f"depth={depth} face_idx={face_idx} face={face} "
                        f"triangle {i} has inward-pointing normal"
                    )

    def test_dual_face_vertices_are_convex_on_sphere(self):
        """Consecutive edges of each dual face should turn the same direction."""
        for depth in range(3):
            mesh = Mesh.icosphere(depth)
            dual = mesh.mesh_dual()
            for face_idx, face in enumerate(dual.faces):
                verts = dual.vertices[np.array(face)]
                centroid = np.mean(verts, axis=0)
                n = centroid / np.linalg.norm(centroid)
                n_verts = len(face)
                for i in range(n_verts):
                    a = verts[i]
                    b = verts[(i + 1) % n_verts]
                    c = verts[(i + 2) % n_verts]
                    cross = np.cross(b - a, c - b)
                    assert np.dot(cross, n) > 0, (
                        f"depth={depth} face_idx={face_idx}: non-convex turn at vertex {i}"
                    )
