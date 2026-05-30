import random
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np


def _edge_pixels(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    uv0: np.ndarray,
    uv1: np.ndarray,
    tol_sq: float,
) -> List[Tuple[float, int, int]]:
    """Return (t, row, col) for grid pixels within tol of the UV segment uv0->uv1."""
    edge = uv1 - uv0
    edge_len_sq = float(edge @ edge)
    if edge_len_sq < 1e-12:
        return []
    du = u_grid - uv0[0]
    dv = v_grid - uv0[1]
    t = (du * edge[0] + dv * edge[1]) / edge_len_sq
    proj_u = uv0[0] + t * edge[0]
    proj_v = uv0[1] + t * edge[1]
    dist_sq = (u_grid - proj_u) ** 2 + (v_grid - proj_v) ** 2
    mask = (t >= 0.0) & (t <= 1.0) & (dist_sq < tol_sq)
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return []
    ts = t[rows, cols]
    order = np.argsort(ts)
    return [(float(ts[i]), int(rows[i]), int(cols[i])) for i in order]


class PolygonMesh:
    def __init__(self, vertices: np.ndarray, faces: List[Tuple[int, ...]]):
        self.vertices = vertices.astype(np.float32)
        self.faces = faces

        n_faces = len(faces)
        self.face_colors = np.random.rand(n_faces, 4).astype(np.float32)
        self.face_colors[:, 3] = 1.0
        self._rebuild_render_arrays()

        self.face_textures: Dict[str, np.ndarray] = {}
        self.face_metadata: List[Dict[str, Any]] = [{} for _ in range(n_faces)]
        self.face_adjacency: List[List[int]] = self._build_face_adjacency()
        self.face_centroids: np.ndarray = self._compute_face_centroids()
        self.face_areas: np.ndarray = self._compute_face_areas()

    def _rebuild_render_arrays(self) -> None:
        triangles = []
        triangle_face_indices = []
        render_uv_list = []

        n_faces = len(self.faces)
        tangent_centroids = np.zeros((n_faces, 3), dtype=np.float32)
        u_axes = np.zeros((n_faces, 3), dtype=np.float32)
        v_axes = np.zeros((n_faces, 3), dtype=np.float32)
        u_offsets = np.zeros(n_faces, dtype=np.float32)
        v_offsets = np.zeros(n_faces, dtype=np.float32)
        u_ranges = np.zeros(n_faces, dtype=np.float32)
        v_ranges = np.zeros(n_faces, dtype=np.float32)
        face_vertex_uvs: List[np.ndarray] = []

        for face_idx, face in enumerate(self.faces):
            verts = self.vertices[np.array(face)]
            centroid = np.mean(verts, axis=0)
            n = centroid / np.linalg.norm(centroid)
            ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            u_axis = np.cross(ref, n)
            u_axis /= np.linalg.norm(u_axis)
            v_axis = np.cross(n, u_axis)
            offsets = verts - centroid
            us = offsets @ u_axis
            vs = offsets @ v_axis
            u_r = float(us.max() - us.min()) or 1.0
            v_r = float(vs.max() - vs.min()) or 1.0
            face_uv = np.stack(
                [(us - us.min()) / u_r, (vs - vs.min()) / v_r], axis=1
            ).astype(np.float32)

            tangent_centroids[face_idx] = centroid
            u_axes[face_idx] = u_axis
            v_axes[face_idx] = v_axis
            u_offsets[face_idx] = us.min()
            v_offsets[face_idx] = vs.min()
            u_ranges[face_idx] = u_r
            v_ranges[face_idx] = v_r
            face_vertex_uvs.append(face_uv)

            for i in range(1, len(face) - 1):
                triangles.append((face[0], face[i], face[i + 1]))
                triangle_face_indices.append(face_idx)
                render_uv_list.extend([face_uv[0], face_uv[i], face_uv[i + 1]])

        self.index_array = np.array(triangles, dtype=np.uint32)
        flat_indices = self.index_array.flatten()
        self.render_vertices = self.vertices[flat_indices]
        tfi = np.array(triangle_face_indices, dtype=np.int32)
        self.render_colors = np.repeat(self.face_colors[tfi], 3, axis=0)
        self.render_face_indices: np.ndarray = np.repeat(tfi, 3)
        self.render_local_uvs: np.ndarray = np.array(render_uv_list, dtype=np.float32).reshape(-1, 2)

        self._tangent_centroids = tangent_centroids
        self._u_axes = u_axes
        self._v_axes = v_axes
        self._u_offsets = u_offsets
        self._v_offsets = v_offsets
        self._u_ranges = u_ranges
        self._v_ranges = v_ranges
        self._face_vertex_uvs = face_vertex_uvs

    def _compute_face_areas(self) -> np.ndarray:
        areas = np.zeros(len(self.faces), dtype=np.float64)
        for face_idx, face in enumerate(self.faces):
            verts = self.vertices[np.array(face)]
            v0 = verts[0]
            edges1 = verts[1:-1] - v0
            edges2 = verts[2:] - v0
            crosses = np.cross(edges1, edges2)
            areas[face_idx] = 0.5 * np.linalg.norm(crosses, axis=1).sum()
        return areas

    def _compute_face_centroids(self) -> np.ndarray:
        n_faces = len(self.faces)
        centroids = np.zeros((n_faces, 3), dtype=np.float64)
        for i, face in enumerate(self.faces):
            pts = self.vertices[np.array(face)]
            c = np.mean(pts, axis=0)
            centroids[i] = c / np.linalg.norm(c)
        return centroids

    def _build_face_adjacency(self) -> List[List[int]]:
        edge_to_faces: Dict[FrozenSet, List[int]] = {}
        for face_idx, face in enumerate(self.faces):
            n = len(face)
            for i in range(n):
                edge = frozenset([face[i], face[(i + 1) % n]])
                edge_to_faces.setdefault(edge, []).append(face_idx)
        adjacency: List[List[int]] = [[] for _ in range(len(self.faces))]
        for neighbors in edge_to_faces.values():
            if len(neighbors) == 2:
                a, b = neighbors
                adjacency[a].append(b)
                adjacency[b].append(a)
        return adjacency

    def set_face_colors(self, colors: np.ndarray) -> None:
        self.face_colors = colors.astype(np.float32)
        self._rebuild_render_arrays()

    def set_face_metadata(self, face_idx: int, **metadata: Any) -> None:
        self.face_metadata[face_idx].update(metadata)

    def set_face_textures(self, name: str, textures: np.ndarray, n_channels: int = 4) -> None:
        """Add or replace a named texture layer.

        Shape must be (N_faces, H, W, n_channels). Dtype must be float32 or uint8.
        Use n_channels=4 for display textures (RGBA); use n_channels=1 for
        single-channel data layers (elevation, lake mask, biome index, etc.).
        Dtype is preserved as-is — pass uint8 for integer-index layers, float32
        for everything else.
        """
        if textures.ndim != 4:
            raise ValueError(f"textures must be 4-D (N, H, W, C), got shape {textures.shape}")
        if textures.shape[0] != len(self.faces):
            raise ValueError(f"textures count {textures.shape[0]} != face count {len(self.faces)}")
        if textures.shape[3] != n_channels:
            raise ValueError(
                f"textures channel count {textures.shape[3]} != n_channels={n_channels}"
            )
        if textures.dtype not in (np.float32, np.uint8):
            raise ValueError(
                f"textures dtype must be float32 or uint8, got {textures.dtype}"
            )
        self.face_textures[name] = textures

    def apply_metadata_channel(self, key: str, values: List[Any]) -> None:
        if len(values) != len(self.faces):
            raise ValueError(
                f"values length {len(values)} != face count {len(self.faces)}"
            )
        for face_idx, value in enumerate(values):
            self.face_metadata[face_idx][key] = value

    def face_tangent_frames(self):
        """Return pre-computed per-face tangent-plane data.

        Returns (u_axes, v_axes, u_offsets, v_offsets, u_ranges, v_ranges, vertex_uvs):
        - u_axes, v_axes: float32 (n_faces, 3) — orthonormal tangent-plane axes
        - u_offsets, v_offsets: float32 (n_faces,) — min projected coord per axis
        - u_ranges, v_ranges: float32 (n_faces,) — coord span per axis
        - vertex_uvs: list of (n_v, 2) float32 — per-face vertex UV coords in [0, 1]²

        To project a 3D point p onto face i's UV plane:
            offset = p - _tangent_centroids[i]
            u_norm = (dot(offset, u_axes[i]) - u_offsets[i]) / u_ranges[i]
            v_norm = (dot(offset, v_axes[i]) - v_offsets[i]) / v_ranges[i]
        The face centroid maps to UV (-u_offsets[i]/u_ranges[i], -v_offsets[i]/v_ranges[i]).
        """
        return (
            self._u_axes,
            self._v_axes,
            self._u_offsets,
            self._v_offsets,
            self._u_ranges,
            self._v_ranges,
            self._face_vertex_uvs,
        )

    def face_uv_to_3d(self, face: int, u: float, v: float) -> np.ndarray:
        """Map face-local UV in [0, 1]² to a 3D tangent-plane position."""
        u_local = u * float(self._u_ranges[face]) + float(self._u_offsets[face])
        v_local = v * float(self._v_ranges[face]) + float(self._v_offsets[face])
        return (
            self._tangent_centroids[face]
            + u_local * self._u_axes[face]
            + v_local * self._v_axes[face]
        )

    def face_3d_to_uv(self, face: int, pos: np.ndarray) -> Tuple[float, float]:
        """Project a 3D point into face-local UV coordinates."""
        offset = pos - self._tangent_centroids[face]
        u_local = float(offset @ self._u_axes[face])
        v_local = float(offset @ self._v_axes[face])
        u = (u_local - float(self._u_offsets[face])) / float(self._u_ranges[face])
        v = (v_local - float(self._v_offsets[face])) / float(self._v_ranges[face])
        return u, v

    def iter_boundary_pixel_pairs(self, tex_size: int):
        """Yield (face_a, face_b, r_a, c_a, r_b, c_b) for every matched seam pixel pair.

        For each pair of adjacent faces sharing exactly 2 vertices, finds the
        pixels in each face's UV grid that lie within ~0.75 pixel of the shared
        edge, then matches them by their parameter t along that edge.  Yields
        each matched pair exactly once (face_a < face_b always).
        """
        coords = np.linspace(0.0, 1.0, tex_size)
        u_grid, v_grid = np.meshgrid(coords, coords)
        tol_sq = (0.75 / (tex_size - 1)) ** 2
        t_tol = 1.5 / (tex_size - 1)
        _, _, _, _, _, _, vertex_uvs = self.face_tangent_frames()

        for face_a, neighbors in enumerate(self.face_adjacency):
            verts_a = self.faces[face_a]
            verts_a_set = set(verts_a)
            for face_b in neighbors:
                if face_b <= face_a:
                    continue
                verts_b = self.faces[face_b]
                shared = list(verts_a_set & set(verts_b))
                if len(shared) != 2:
                    continue
                v0, v1 = shared
                uv_a0 = np.asarray(vertex_uvs[face_a][verts_a.index(v0)], dtype=np.float64)
                uv_a1 = np.asarray(vertex_uvs[face_a][verts_a.index(v1)], dtype=np.float64)
                uv_b0 = np.asarray(vertex_uvs[face_b][verts_b.index(v0)], dtype=np.float64)
                uv_b1 = np.asarray(vertex_uvs[face_b][verts_b.index(v1)], dtype=np.float64)
                pixels_a = _edge_pixels(u_grid, v_grid, uv_a0, uv_a1, tol_sq)
                pixels_b = _edge_pixels(u_grid, v_grid, uv_b0, uv_b1, tol_sq)
                if not pixels_a or not pixels_b:
                    continue
                tb = np.array([p[0] for p in pixels_b])
                for t_a, r_a, c_a in pixels_a:
                    diffs = np.abs(tb - t_a)
                    k_b = int(np.argmin(diffs))
                    if diffs[k_b] > t_tol:
                        continue
                    _, r_b, c_b = pixels_b[k_b]
                    yield face_a, face_b, r_a, c_a, r_b, c_b

    def measure_boundary_texture_jumps(self, texture: np.ndarray) -> Dict[str, float]:
        """Measure discontinuities in a (N, H, W, C) texture at face boundaries.

        For each matched seam pixel pair the mean absolute difference across all
        C channels is computed; summary statistics are returned over all pairs.

        Returns:
            dict with keys mean, median, max, rms, n_pairs (in texture units).
        """
        tex_size = texture.shape[1]
        all_jumps: List[float] = []
        for face_a, face_b, r_a, c_a, r_b, c_b in self.iter_boundary_pixel_pairs(tex_size):
            diff = np.abs(
                texture[face_a, r_a, c_a, :].astype(np.float64)
                - texture[face_b, r_b, c_b, :].astype(np.float64)
            )
            all_jumps.append(float(diff.mean()))
        if not all_jumps:
            return {"mean": 0.0, "median": 0.0, "max": 0.0, "rms": 0.0, "n_pairs": 0}
        jumps = np.array(all_jumps)
        return {
            "mean":    float(np.mean(jumps)),
            "median":  float(np.median(jumps)),
            "max":     float(np.max(jumps)),
            "rms":     float(np.sqrt(np.mean(jumps ** 2))),
            "n_pairs": int(len(jumps)),
        }

    def face_texture_positions(self, h: int, w: int) -> np.ndarray:
        """Return unit-sphere 3D positions for an (H, W) texture grid over every face.

        For each pixel UV, finds the fan triangle (same fan triangulation the renderer
        uses) containing that UV, computes barycentric coordinates in UV space, and
        interpolates the triangle's actual 3D mesh-vertex positions before normalizing
        to the unit sphere. Because adjacent faces share their edge vertices, pixels
        sitting on a shared edge map to the same great-circle 3D point in both
        neighboring faces — so per-pixel sphere-coordinate computations (e.g. 3D
        simplex noise) are naturally continuous across face boundaries.

        Pixels whose UV falls outside the polygon (in the four corners of the [0,1]²
        bounding box) are assigned to whichever fan triangle they fit "least badly"
        (largest minimum barycentric); the resulting linear extrapolation keeps edge-
        adjacent pixels well-behaved for bilinear texture filtering.

        Returns:
            float32 array of shape (N_faces, H, W, 3).
        """
        u_coords = np.linspace(0.0, 1.0, w, dtype=np.float64)
        v_coords = np.linspace(0.0, 1.0, h, dtype=np.float64)
        uu, vv = np.meshgrid(u_coords, v_coords)
        pixel_uv = np.stack([uu.ravel(), vv.ravel()], axis=1)  # (h*w, 2)
        n_pix = pixel_uv.shape[0]
        pix_idx = np.arange(n_pix)

        n_faces = len(self.faces)
        positions = np.zeros((n_faces, h, w, 3), dtype=np.float32)

        for face_idx, face in enumerate(self.faces):
            face_arr = np.array(face)
            verts_3d = self.vertices[face_arr].astype(np.float64)         # (n_v, 3)
            verts_uv = self._face_vertex_uvs[face_idx].astype(np.float64)  # (n_v, 2)

            # Fan triangles (V0, Vi, Vi+1) — same triangulation as _rebuild_render_arrays.
            v0_uv = verts_uv[0]
            v0_3d = verts_3d[0]
            e1_uv = verts_uv[1:-1] - v0_uv   # (n_tris, 2)
            e2_uv = verts_uv[2:] - v0_uv     # (n_tris, 2)
            e1_3d = verts_3d[1:-1] - v0_3d   # (n_tris, 3)
            e2_3d = verts_3d[2:] - v0_3d     # (n_tris, 3)

            # 2D signed-area determinant per triangle.
            det = e1_uv[:, 0] * e2_uv[:, 1] - e1_uv[:, 1] * e2_uv[:, 0]
            det = np.where(np.abs(det) < 1e-12, 1.0, det)

            pv = pixel_uv - v0_uv  # (n_pix, 2)
            pv_cross_e2 = (pv[:, 0:1] * e2_uv[None, :, 1]
                           - pv[:, 1:2] * e2_uv[None, :, 0])               # (n_pix, n_tris)
            e1_cross_pv = (e1_uv[None, :, 0] * pv[:, 1:2]
                           - e1_uv[None, :, 1] * pv[:, 0:1])               # (n_pix, n_tris)
            beta = pv_cross_e2 / det[None, :]
            gamma = e1_cross_pv / det[None, :]
            alpha = 1.0 - beta - gamma

            # Pick the triangle a pixel lies in (all bary >= 0). Outside-the-polygon
            # pixels get the triangle they're "closest to" — i.e. the one with the
            # largest minimum barycentric coordinate. This extrapolates linearly.
            min_bary = np.minimum(alpha, np.minimum(beta, gamma))
            best_tri = np.argmax(min_bary, axis=1)
            b_sel = beta[pix_idx, best_tri]
            g_sel = gamma[pix_idx, best_tri]

            pts = (v0_3d[None, :]
                   + b_sel[:, None] * e1_3d[best_tri]
                   + g_sel[:, None] * e2_3d[best_tri])
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            positions[face_idx] = (pts / norms).astype(np.float32).reshape(h, w, 3)

        return positions


class Mesh(PolygonMesh):
    def __init__(self, vertices: np.ndarray, faces: List[Tuple[int, int, int]]):
        tmp = np.argsort(vertices[:, 2])  # Sort along z axis
        mapping = np.argsort(tmp)
        sorted_vertices = vertices[tmp].astype(np.float32)
        remapped_faces = sorted([(mapping[x], mapping[y], mapping[z]) for x, y, z in faces])
        super().__init__(sorted_vertices, remapped_faces)

    def mesh_dual(self) -> PolygonMesh:
        # First compute the centroids of each face
        dual_vertices = []
        for face in self.faces:
            pts = self.vertices[np.array(face)]
            centroid = np.mean(pts, axis=0)
            centroid /= np.linalg.norm(centroid)
            dual_vertices.append(centroid)
        dual_vertices = np.array(dual_vertices)

        # Now, for each vertex in the original mesh, find all faces that include it
        dual_faces_dict: Dict[int, list[int]] = {}
        for face_idx, face in enumerate(self.faces):
            for v in face:
                if v not in dual_faces_dict:
                    dual_faces_dict[v] = [face_idx]
                else:
                    dual_faces_dict[v].append(face_idx)

        # Reorder the face indices in each dual face to be CCW when viewed from outside.
        # We project each face centroid onto the tangent plane at the vertex and
        # sort by angle in that local 2D frame. Using the global XY projection
        # (arctan2 of the raw x/y components) fails for equatorial vertices because
        # it ignores the z-component of the displacement vectors.
        dual_faces = []
        for v_idx, face_indices in dual_faces_dict.items():
            v = self.vertices[v_idx]
            n = v  # unit outward normal (v is already on the unit sphere)
            ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            u_axis = np.cross(ref, n)
            u_axis /= np.linalg.norm(u_axis)
            w_axis = np.cross(n, u_axis)  # u_axis × w_axis = n (right-hand frame)

            def angle_from_center(face_idx, u=u_axis, w=w_axis, center=v):
                vec = dual_vertices[face_idx] - center
                return np.arctan2(np.dot(vec, w), np.dot(vec, u))

            face_indices.sort(key=angle_from_center)
            dual_faces.append(tuple(face_indices))

        return PolygonMesh(dual_vertices, dual_faces)

    def deregularize(self, n_swaps: int = 1000, n_relax: int = 10,
                     stiffness: float = 0.1, outer_iters: int = 3,
                     seed: Optional[int] = None) -> 'Mesh':
        """Edge-swap + spring-relax loop to break icosphere regularity."""
        rng = random.Random(seed)
        verts = self.vertices.astype(np.float64)
        faces: List[Tuple[int, int, int]] = list(self.faces)
        n_verts = len(verts)

        # adj[v] = set of neighboring vertex indices
        adj: List[set] = [set() for _ in range(n_verts)]
        # vertex_faces[v] = list of face indices that contain v
        vertex_faces: List[List[int]] = [[] for _ in range(n_verts)]
        # edge_faces[frozenset{a,b}] = list of face indices sharing that edge
        edge_faces: Dict[FrozenSet, List[int]] = {}

        for fi, (a, b, c) in enumerate(faces):
            vertex_faces[a].append(fi)
            vertex_faces[b].append(fi)
            vertex_faces[c].append(fi)
            adj[a].update([b, c])
            adj[b].update([a, c])
            adj[c].update([a, b])
            for e in (frozenset([a, b]), frozenset([a, c]), frozenset([b, c])):
                edge_faces.setdefault(e, []).append(fi)

        def ensure_ccw(p0: int, p1: int, p2: int) -> Tuple[int, int, int]:
            centroid = (verts[p0] + verts[p1] + verts[p2]) / 3.0
            normal = np.cross(verts[p1] - verts[p0], verts[p2] - verts[p0])
            return (p0, p1, p2) if np.dot(normal, centroid) >= 0 else (p0, p2, p1)

        def remove_face(fi: int) -> None:
            f = faces[fi]
            a, b, c = f
            for v in f:
                vertex_faces[v].remove(fi)
            for e in (frozenset([a, b]), frozenset([a, c]), frozenset([b, c])):
                edge_faces[e].remove(fi)
                if not edge_faces[e]:
                    del edge_faces[e]

        def add_face(fi: int, f: Tuple[int, int, int]) -> None:
            faces[fi] = f
            a, b, c = f
            for v in f:
                vertex_faces[v].append(fi)
            for e in (frozenset([a, b]), frozenset([a, c]), frozenset([b, c])):
                edge_faces.setdefault(e, []).append(fi)

        for _ in range(outer_iters):
            # --- Edge swap phase ---
            for _ in range(n_swaps):
                fi_pick = rng.randrange(len(faces))
                f_pick = faces[fi_pick]
                ei = rng.randrange(3)
                bv, cv = f_pick[ei], f_pick[(ei + 1) % 3]
                edge = frozenset([bv, cv])
                ef = edge_faces.get(edge)
                if ef is None or len(ef) != 2:
                    continue

                fi1, fi2 = ef[0], ef[1]
                f1, f2 = faces[fi1], faces[fi2]
                av = next(v for v in f1 if v != bv and v != cv)
                dv = next(v for v in f2 if v != bv and v != cv)

                # av and dv gain a degree. avoid degree >=7 to avoid octagons
                if len(adj[av]) >= 7 or len(adj[dv]) >= 7:
                    continue
                # bv and cv lose a degree. avoid degree <=5 to avoid squares
                if len(adj[bv]) <= 5 or len(adj[cv]) <= 5:
                    continue

                # ensure_ccw only swaps p1/p2, so result is (av, bv, dv) or (av, dv, bv)
                new_f1 = ensure_ccw(av, bv, dv)
                p0, p1, p2 = new_f1
                # Derive new_f2 so the shared AD edge runs opposite to new_f1
                if (p0 == av and p1 == dv) or (p1 == av and p2 == dv) or (p2 == av and p0 == dv):
                    new_f2 = (dv, av, cv)  # a→d in new_f1 → d→a in new_f2
                else:
                    new_f2 = (av, dv, cv)  # d→a in new_f1 → a→d in new_f2

                remove_face(fi1)
                remove_face(fi2)
                adj[bv].discard(cv)
                adj[cv].discard(bv)
                adj[av].add(dv)
                adj[dv].add(av)
                add_face(fi1, new_f1)
                add_face(fi2, new_f2)

            # --- Spring relaxation phase ---
            # Build directed edge arrays for vectorized mean-neighbor computation
            src_list, dst_list = [], []
            for v in range(n_verts):
                for u in adj[v]:
                    src_list.append(v)
                    dst_list.append(u)
            src_arr = np.array(src_list)
            dst_arr = np.array(dst_list)
            degree = np.bincount(src_arr, minlength=n_verts).astype(np.float64)

            for _ in range(n_relax):
                neighbor_sum = np.zeros((n_verts, 3), dtype=np.float64)
                np.add.at(neighbor_sum, src_arr, verts[dst_arr])
                target = neighbor_sum / degree[:, np.newaxis]
                verts += stiffness * (target - verts)
                verts /= np.linalg.norm(verts, axis=1, keepdims=True)

        # Commit results
        self.vertices = verts.astype(np.float32)
        self.faces = faces
        n_faces = len(faces)
        self.face_colors = np.random.rand(n_faces, 4).astype(np.float32)
        self.face_colors[:, 3] = 1.0
        self._rebuild_render_arrays()
        self.face_adjacency = self._build_face_adjacency()
        self.face_centroids = self._compute_face_centroids()
        self.face_areas = self._compute_face_areas()
        return self

    # Vertices from:
    # https://github.com/lazysquirrellabs/sphere_generator/blob/main/Assets/Lazy%20Squirrel%20Labs/SphereGenerator/Runtime/Generators/IcosphereGenerator.cs
    @staticmethod
    def icosphere(depth: int = 0):
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [phi, 1, 0],
            [0, phi, -1],
            [0, phi, 1],
            [1, 0, -phi],
            [1, 0, phi],
            [phi, -1, 0],
            [-1, 0, -phi],
            [-phi, 1, 0],
            [-1, 0, phi],
            [0, -phi, -1],
            [0, -phi, 1],
            [-phi, -1, 0]
        ])
        faces = [
            (0, 1, 2),
            (0, 3, 1),
            (0, 2, 4),
            (3, 0, 5),
            (0, 4, 5),
            (1, 3, 6),
            (1, 7, 2),
            (7, 1, 6),
            (4, 2, 8),
            (7, 8, 2),
            (9, 3, 5),
            (6, 3, 9),
            (5, 4, 10),
            (4, 8, 10),
            (9, 5, 10),
            (7, 6, 11),
            (7, 11, 8),
            (11, 6, 9),
            (8, 11, 10),
            (10, 11, 9)
        ]

        for _ in range(depth):
            # Subdivide each triangle into 4 smaller triangles
            new_faces = []
            new_vertices = []  # temporary cache of added vertices to cut down on numpy appends
            edge_indices = {}  # given two vertices, what's the index of their midpoint

            for face in faces:
                e0 = (face[0], face[1]) if face[0] < face[1] else (face[1], face[0])
                e1 = (face[0], face[2]) if face[0] < face[2] else (face[2], face[0])
                e2 = (face[1], face[2]) if face[1] < face[2] else (face[2], face[1])

                if e0 in edge_indices:
                    v3 = edge_indices[e0]
                else:
                    pt3 = (vertices[face[0]] + vertices[face[1]]) / 2
                    v3 = vertices.shape[0] + len(new_vertices)
                    edge_indices[e0] = v3
                    new_vertices.append(pt3)

                if e1 in edge_indices:
                    v4 = edge_indices[e1]
                else:
                    pt4 = (vertices[face[0]] + vertices[face[2]]) / 2
                    v4 = vertices.shape[0] + len(new_vertices)
                    edge_indices[e1] = v4
                    new_vertices.append(pt4)

                if e2 in edge_indices:
                    v5 = edge_indices[e2]
                else:
                    pt5 = (vertices[face[1]] + vertices[face[2]]) / 2
                    v5 = vertices.shape[0] + len(new_vertices)
                    edge_indices[e2] = v5
                    new_vertices.append(pt5)

                new_faces.append((face[0], v3, v4))
                new_faces.append((v3, face[1], v5))
                new_faces.append((v3, v5, v4))
                new_faces.append((v4, v5, face[2]))

            faces = new_faces
            vertices = np.vstack([vertices] + new_vertices)

        # We have a subdivided icosahedron, normalize vertices to get icosphere
        vertices /= np.linalg.norm(vertices, axis=1)[..., np.newaxis]

        return Mesh(vertices, faces)

    @staticmethod
    def triangle():
        v = np.array([[-1.0, -1.0, 0.0],
                      [1.0, -1.0, 0.0],
                      [0.0, 1.0, 0.0]])
        f = [(0, 1, 2)]
        return Mesh(v, f)
