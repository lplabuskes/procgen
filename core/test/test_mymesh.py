import numpy as np
from core.mesh import Mesh


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


class TestFaceTexturePositions:
    def test_shape(self):
        mesh = Mesh.icosphere(1)
        dual = mesh.mesh_dual()
        pos = dual.face_texture_positions(8, 16)
        assert pos.shape == (len(dual.faces), 8, 16, 3)

    def test_on_unit_sphere(self):
        mesh = Mesh.icosphere(2)
        dual = mesh.mesh_dual()
        pos = dual.face_texture_positions(16, 16)
        norms = np.linalg.norm(pos, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_dtype_is_float32(self):
        mesh = Mesh.icosphere(1)
        pos = mesh.face_texture_positions(4, 4)
        assert pos.dtype == np.float32

    def test_pixels_near_face_centroid(self):
        # Every pixel for face f should be in the same hemisphere as that face's centroid.
        mesh = Mesh.icosphere(2)
        dual = mesh.mesh_dual()
        pos = dual.face_texture_positions(8, 8)  # (N, H, W, 3)
        centroids = dual.face_centroids  # (N, 3)
        dots = np.einsum('nhwc,nc->nhw', pos, centroids)
        assert np.all(dots > 0), "Some texture pixels ended up in the wrong hemisphere"

    def test_vertex_uv_roundtrip(self):
        # For each face vertex, compute its UV coordinate using the stored tangent frame,
        # snap to the nearest pixel in a dense grid, and verify that pixel's 3D position
        # is within 1° of the original vertex on the sphere.
        H, W = 101, 101
        mesh = Mesh.icosphere(0)
        pos = mesh.face_texture_positions(H, W)

        for face_idx, face in enumerate(mesh.faces):
            verts = mesh.vertices[np.array(face)]
            centroid = mesh._tangent_centroids[face_idx]
            u_axis = mesh._u_axes[face_idx]
            v_axis = mesh._v_axes[face_idx]
            u_offset = float(mesh._u_offsets[face_idx])
            v_offset = float(mesh._v_offsets[face_idx])
            u_range = float(mesh._u_ranges[face_idx])
            v_range = float(mesh._v_ranges[face_idx])

            for vert in verts:
                offset = vert - centroid
                u_norm = (np.dot(offset, u_axis) - u_offset) / u_range
                v_norm = (np.dot(offset, v_axis) - v_offset) / v_range
                j = int(np.clip(round(u_norm * (W - 1)), 0, W - 1))
                i = int(np.clip(round(v_norm * (H - 1)), 0, H - 1))
                pixel_pos = pos[face_idx, i, j]
                vert_sphere = vert / np.linalg.norm(vert)
                dot = float(np.dot(pixel_pos, vert_sphere))
                angle_deg = float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))
                assert angle_deg < 1.0, (
                    f"face {face_idx}: vertex roundtrip error {angle_deg:.3f}°"
                )

    def test_triangular_mesh_directly(self):
        mesh = Mesh.icosphere(1)
        pos = mesh.face_texture_positions(4, 4)
        assert pos.shape == (len(mesh.faces), 4, 4, 3)
        norms = np.linalg.norm(pos, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


class TestFaceTextureInPolygon:
    def test_shape_and_dtype(self):
        dual = Mesh.icosphere(2).mesh_dual()
        mask = dual.face_texture_in_polygon(16, 16)
        assert mask.shape == (len(dual.faces), 16, 16)
        assert mask.dtype == bool

    def test_centre_inside_corners_outside(self):
        # A hexagon/pentagon inscribed in its square texture leaves the square's
        # corners outside the polygon; the central texels are always inside.
        dual = Mesh.icosphere(2).mesh_dual()
        mask = dual.face_texture_in_polygon(16, 16)
        assert mask[:, 7:9, 7:9].all()                 # centre 2×2 always inside
        corners = mask[:, [0, 0, -1, -1], [0, -1, 0, -1]]
        assert not corners.any()                       # all four corners outside
        # The polygon is strictly smaller than the square it is drawn in.
        assert 0.4 < mask.mean() < 0.95

    def test_inside_pixels_match_positions(self):
        # Inside texels are exactly the fan-triangle interior the renderer uses, so
        # an inside texel must be closer to its own face centroid than the texture
        # extrapolation pushes the over-hanging corners.
        dual = Mesh.icosphere(1).mesh_dual()
        pos = dual.face_texture_positions(16, 16)
        mask = dual.face_texture_in_polygon(16, 16)
        centroids = np.asarray(dual.face_centroids)
        dots = np.einsum('nhwc,nc->nhw', pos, centroids / np.linalg.norm(
            centroids, axis=1, keepdims=True))
        # Inside texels sit nearer their centroid than the worst outside texel.
        assert dots[mask].min() >= dots[~mask].min()
