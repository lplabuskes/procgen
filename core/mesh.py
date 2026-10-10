import random
from typing import Any, Dict, FrozenSet, List, Optional, Protocol, Sequence, Tuple

import numpy as np


class GeometryStore(Protocol):
    """Persistent home for a mesh's derived geometry (see ``core.mesh_cache``).

    ``load`` returns ``None`` when absent or stale; ``save`` may silently fail.
    """

    def load(self, key: str) -> Optional[Tuple[np.ndarray, ...]]: ...

    def save(self, key: str, arrays: Tuple[np.ndarray, ...]) -> None: ...


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


# Element budget for one (chunk, tex_size²) band array: ~32 MB per float64 array.
_BAND_BATCH_ELEMS = 4_000_000


def _edge_band_batch(
    p0: np.ndarray,
    p1: np.ndarray,
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    tol_sq: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised _edge_pixels over many edges; returns (t, mask) each (E, P)."""
    edge = p1 - p0                                       # (E, 2)
    elen_sq = (edge * edge).sum(axis=1)                  # (E,)
    safe = elen_sq >= 1e-12
    elen_safe = np.where(safe, elen_sq, 1.0)
    du = grid_u[None, :] - p0[:, 0:1]                    # (E, P)
    dv = grid_v[None, :] - p0[:, 1:2]
    t = (du * edge[:, 0:1] + dv * edge[:, 1:2]) / elen_safe[:, None]
    proj_u = p0[:, 0:1] + t * edge[:, 0:1]
    proj_v = p0[:, 1:2] + t * edge[:, 1:2]
    dist_sq = (grid_u[None, :] - proj_u) ** 2 + (grid_v[None, :] - proj_v) ** 2
    mask = (t >= 0.0) & (t <= 1.0) & (dist_sq < tol_sq) & safe[:, None]
    return t, mask


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
        # Seam pixel pairs per tex_size, texel positions and in-polygon masks per (h, w):
        # geometry-only, read-only.
        self._boundary_pair_cache: Dict[int, Tuple[np.ndarray, ...]] = {}
        self._texture_position_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._in_polygon_cache: Dict[Tuple[int, int], np.ndarray] = {}
        # Persists them across loads (set by ``core.mesh_cache``; bump its GEOMETRY_VERSION on change).
        self.geometry_store: Optional[GeometryStore] = None
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
        """Add or replace a named texture layer; shape must be (N, H, W, n_channels)."""
        if textures.ndim != 4:
            raise ValueError(f"textures must be 4-D (N, H, W, C), got shape {textures.shape}")
        if textures.shape[0] != len(self.faces):
            raise ValueError(f"textures count {textures.shape[0]} != face count {len(self.faces)}")
        if textures.shape[3] != n_channels:
            raise ValueError(
                f"textures channel count {textures.shape[3]} != n_channels={n_channels}"
            )
        if textures.dtype not in (np.float32, np.float16, np.uint8):
            raise ValueError(
                f"textures dtype must be float32, float16, or uint8, got {textures.dtype}"
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
        """Return (u_axes, v_axes, u_offsets, v_offsets, u_ranges, v_ranges, vertex_uvs).

        Project 3D point p onto face i: offset = p - _tangent_centroids[i];
        u = (dot(offset, u_axes[i]) - u_offsets[i]) / u_ranges[i], similarly for v.
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
        """Yield (face_a, face_b, r_a, c_a, r_b, c_b) for matched cross-edge pixel pairs."""
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

    def _compute_boundary_pairs(self, tex_size: int) -> Tuple[np.ndarray, ...]:
        """Batched vectorised equivalent of iter_boundary_pixel_pairs."""
        coords = np.linspace(0.0, 1.0, tex_size)
        u_grid, v_grid = np.meshgrid(coords, coords)
        grid_u = u_grid.reshape(-1)                       # flat index = row*tex_size + col
        grid_v = v_grid.reshape(-1)
        tol_sq = (0.75 / (tex_size - 1)) ** 2
        t_tol = 1.5 / (tex_size - 1)
        _, _, _, _, _, _, vertex_uvs = self.face_tangent_frames()

        # Enumerate qualifying face pairs and collect both sides' edge endpoints.
        a0: Any = []
        a1: Any = []
        b0: Any = []
        b1: Any = []
        pair_faces: List[Tuple[int, int]] = []
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
                a0.append(vertex_uvs[face_a][verts_a.index(v0)])
                a1.append(vertex_uvs[face_a][verts_a.index(v1)])
                b0.append(vertex_uvs[face_b][verts_b.index(v0)])
                b1.append(vertex_uvs[face_b][verts_b.index(v1)])
                pair_faces.append((face_a, face_b))

        fa_out: List[int] = []
        fb_out: List[int] = []
        ra_out: List[int] = []
        ca_out: List[int] = []
        rb_out: List[int] = []
        cb_out: List[int] = []
        if pair_faces:
            a0 = np.asarray(a0, dtype=np.float64)
            a1 = np.asarray(a1, dtype=np.float64)
            b0 = np.asarray(b0, dtype=np.float64)
            b1 = np.asarray(b1, dtype=np.float64)
            # Chunk over pairs so the (chunk, tex_size²) band arrays stay bounded
            # regardless of resolution (tex_size can be up to 128).
            chunk = max(1, _BAND_BATCH_ELEMS // (tex_size * tex_size))
            for c0 in range(0, len(pair_faces), chunk):
                c1 = min(c0 + chunk, len(pair_faces))
                ta, mask_a = _edge_band_batch(a0[c0:c1], a1[c0:c1], grid_u, grid_v, tol_sq)
                tb, mask_b = _edge_band_batch(b0[c0:c1], b1[c0:c1], grid_u, grid_v, tol_sq)
                for lp in range(c1 - c0):
                    a_idx = np.where(mask_a[lp])[0]
                    b_idx = np.where(mask_b[lp])[0]
                    if a_idx.size == 0 or b_idx.size == 0:
                        continue
                    face_a, face_b = pair_faces[c0 + lp]
                    ta_p = ta[lp, a_idx]
                    oa = np.argsort(ta_p)
                    a_idx = a_idx[oa]
                    ta_p = ta_p[oa]
                    tb_p = tb[lp, b_idx]
                    ob = np.argsort(tb_p)
                    b_idx = b_idx[ob]
                    tb_p = tb_p[ob]
                    for k in range(a_idx.size):
                        diffs = np.abs(tb_p - ta_p[k])
                        k_b = int(np.argmin(diffs))
                        if diffs[k_b] > t_tol:
                            continue
                        ia = int(a_idx[k])
                        ib = int(b_idx[k_b])
                        fa_out.append(face_a)
                        fb_out.append(face_b)
                        ra_out.append(ia // tex_size)
                        ca_out.append(ia % tex_size)
                        rb_out.append(ib // tex_size)
                        cb_out.append(ib % tex_size)

        return tuple(
            np.asarray(c, dtype=np.int32)
            for c in (fa_out, fb_out, ra_out, ca_out, rb_out, cb_out)
        )

    def boundary_pixel_pairs(self, tex_size: int) -> Tuple[np.ndarray, ...]:
        """Cached, read-only (face_a, face_b, row_a, col_a, row_b, col_b) matched seam pixel pairs."""
        cached = self._boundary_pair_cache.get(tex_size)
        if cached is not None:
            return cached

        key = f"boundary_pairs_{tex_size}"
        result = self.geometry_store.load(key) if self.geometry_store is not None else None
        if result is None:
            result = self._compute_boundary_pairs(tex_size)
            if self.geometry_store is not None:
                self.geometry_store.save(key, result)
        self._boundary_pair_cache[tex_size] = result
        return result

    def measure_boundary_texture_jumps(self, texture: np.ndarray) -> Dict[str, float]:
        """Mean absolute difference per matched seam pair; returns {mean, median, max, rms, n_pairs}."""
        tex_size = texture.shape[1]
        fa, fb, ra, ca, rb, cb = self.boundary_pixel_pairs(tex_size)
        if fa.size == 0:
            return {"mean": 0.0, "median": 0.0, "max": 0.0, "rms": 0.0, "n_pairs": 0}
        diff = np.abs(
            texture[fa, ra, ca, :].astype(np.float64)
            - texture[fb, rb, cb, :].astype(np.float64)
        )                                                    # (n_pairs, C)
        jumps = diff.mean(axis=1)                            # (n_pairs,)
        return {
            "mean":    float(np.mean(jumps)),
            "median":  float(np.median(jumps)),
            "max":     float(np.max(jumps)),
            "rms":     float(np.sqrt(np.mean(jumps ** 2))),
            "n_pairs": int(len(jumps)),
        }

    def face_texture_positions(self, h: int, w: int, faces: Optional[Sequence[int]] = None) -> np.ndarray:
        """Return float32 (N, H, W, 3) unit-sphere positions for each face's pixel grid.

        Shared-edge pixels map to the same point in both faces; corner pixels outside
        the polygon extrapolate from the fan triangle with the largest minimum barycentric.
        Given ``faces``, only those, in that order.  The full set is cached read-only.
        """
        cached = self._texture_position_cache.get((h, w))
        if cached is not None:
            return cached if faces is None else cached[np.asarray(faces, dtype=np.int64)]
        if faces is None:
            key = f"texture_positions_{h}x{w}"
            stored = self.geometry_store.load(key) if self.geometry_store is not None else None
            if stored is not None:
                positions = stored[0]
            else:
                positions = self._face_texture_positions(h, w, None)
                if self.geometry_store is not None:
                    self.geometry_store.save(key, (positions,))
            positions.flags.writeable = False
            self._texture_position_cache[(h, w)] = positions
            return positions
        return self._face_texture_positions(h, w, faces)

    def _face_texture_positions(self, h: int, w: int, faces: Optional[Sequence[int]]) -> np.ndarray:
        u_coords = np.linspace(0.0, 1.0, w, dtype=np.float64)
        v_coords = np.linspace(0.0, 1.0, h, dtype=np.float64)
        uu, vv = np.meshgrid(u_coords, v_coords)
        pixel_uv = np.stack([uu.ravel(), vv.ravel()], axis=1)  # (h*w, 2)
        n_pix = pixel_uv.shape[0]
        pix_idx = np.arange(n_pix)

        which = range(len(self.faces)) if faces is None else [int(f) for f in faces]
        positions = np.zeros((len(which), h, w, 3), dtype=np.float32)

        for slot, face_idx in enumerate(which):
            face_arr = np.array(self.faces[face_idx])
            verts_3d = self.vertices[face_arr].astype(np.float64)         # (n_v, 3)
            verts_uv = self._face_vertex_uvs[face_idx].astype(np.float64)  # (n_v, 2)

            # Fan triangles (V0, Vi, Vi+1) — same triangulation as _rebuild_render_arrays.
            v0_uv = verts_uv[0]
            v0_3d = verts_3d[0]
            e1_uv = verts_uv[1:-1] - v0_uv   # (n_tris, 2)
            e2_uv = verts_uv[2:] - v0_uv     # (n_tris, 2)
            e1_3d = verts_3d[1:-1] - v0_3d   # (n_tris, 3)
            e2_3d = verts_3d[2:] - v0_3d     # (n_tris, 3)

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

            # The containing triangle, or for outside pixels the one with the largest
            # minimum barycentric (linear extrapolation).
            min_bary = np.minimum(alpha, np.minimum(beta, gamma))
            best_tri = np.argmax(min_bary, axis=1)
            b_sel = beta[pix_idx, best_tri]
            g_sel = gamma[pix_idx, best_tri]

            pts = (v0_3d[None, :]
                   + b_sel[:, None] * e1_3d[best_tri]
                   + g_sel[:, None] * e2_3d[best_tri])
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            positions[slot] = (pts / norms).astype(np.float32).reshape(h, w, 3)

        return positions

    def face_texture_in_polygon(
        self, h: int, w: int, *, eps: float = 1e-9, faces: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Bool ``(N, H, W)``: texels inside each face's rendered polygon (vs. the
        square texture's over-hanging corners).

        Inside iff the centre or any corner is in a fan triangle (barycentrics >= -eps),
        so straddling texels count as inside.  Given ``faces``, only those, in that order.
        The full set at the default ``eps`` is cached read-only.
        """
        default = eps == 1e-9
        cached = self._in_polygon_cache.get((h, w)) if default else None
        if cached is not None:
            return cached if faces is None else cached[np.asarray(faces, dtype=np.int64)]
        if faces is None and default:
            key = f"in_polygon_{h}x{w}"
            stored = self.geometry_store.load(key) if self.geometry_store is not None else None
            if stored is not None:
                mask = stored[0]
            else:
                mask = self._face_texture_in_polygon(h, w, eps, None)
                if self.geometry_store is not None:
                    self.geometry_store.save(key, (mask,))
            mask.flags.writeable = False
            self._in_polygon_cache[(h, w)] = mask
            return mask
        return self._face_texture_in_polygon(h, w, eps, faces)

    def _face_texture_in_polygon(self, h: int, w: int, eps: float, faces: Optional[Sequence[int]]) -> np.ndarray:
        u_coords = np.linspace(0.0, 1.0, w, dtype=np.float64)
        v_coords = np.linspace(0.0, 1.0, h, dtype=np.float64)
        uu, vv = np.meshgrid(u_coords, v_coords)
        pixel_uv = np.stack([uu.ravel(), vv.ravel()], axis=1)  # (h*w, 2)

        du = 0.5 / (w - 1) if w > 1 else 0.0
        dv = 0.5 / (h - 1) if h > 1 else 0.0
        offsets = np.array([[0.0, 0.0], [-du, -dv], [du, -dv], [-du, dv], [du, dv]])
        # (5*h*w, 2): center then 4 corners, all pixels stacked
        sample_uvs = (pixel_uv[None, :, :] + offsets[:, None, :]).reshape(-1, 2)

        which = range(len(self.faces)) if faces is None else [int(f) for f in faces]
        n_pix = h * w
        mask = np.zeros((len(which), h, w), dtype=bool)

        for slot, face_idx in enumerate(which):
            verts_uv = self._face_vertex_uvs[face_idx].astype(np.float64)  # (n_v, 2)
            v0_uv = verts_uv[0]
            e1_uv = verts_uv[1:-1] - v0_uv   # (n_tris, 2)
            e2_uv = verts_uv[2:] - v0_uv     # (n_tris, 2)

            det = e1_uv[:, 0] * e2_uv[:, 1] - e1_uv[:, 1] * e2_uv[:, 0]
            det = np.where(np.abs(det) < 1e-12, 1.0, det)

            pv = sample_uvs - v0_uv  # (5*n_pix, 2)
            pv_cross_e2 = (pv[:, 0:1] * e2_uv[None, :, 1]
                           - pv[:, 1:2] * e2_uv[None, :, 0])
            e1_cross_pv = (e1_uv[None, :, 0] * pv[:, 1:2]
                           - e1_uv[None, :, 1] * pv[:, 0:1])
            beta = pv_cross_e2 / det[None, :]
            gamma = e1_cross_pv / det[None, :]
            alpha = 1.0 - beta - gamma
            min_bary = np.minimum(alpha, np.minimum(beta, gamma))  # (5*n_pix, n_tris)
            inside = (min_bary.max(axis=1) >= -eps)  # (5*n_pix,)
            inside_any = inside.reshape(5, n_pix).any(axis=0)
            mask[slot] = inside_any.reshape(h, w)

        return mask


class Mesh(PolygonMesh):
    def __init__(self, vertices: np.ndarray, faces: List[Tuple[int, int, int]]):
        tmp = np.argsort(vertices[:, 2])  # Sort along z axis
        mapping = np.argsort(tmp)
        sorted_vertices = vertices[tmp].astype(np.float32)
        remapped_faces = sorted([(mapping[x], mapping[y], mapping[z]) for x, y, z in faces])
        super().__init__(sorted_vertices, remapped_faces)

    def mesh_dual(self) -> PolygonMesh:
        dual_vertices = []
        for face in self.faces:
            pts = self.vertices[np.array(face)]
            centroid = np.mean(pts, axis=0)
            centroid /= np.linalg.norm(centroid)
            dual_vertices.append(centroid)
        dual_vertices = np.array(dual_vertices)

        dual_faces_dict: Dict[int, list[int]] = {}
        for face_idx, face in enumerate(self.faces):
            for v in face:
                if v not in dual_faces_dict:
                    dual_faces_dict[v] = [face_idx]
                else:
                    dual_faces_dict[v].append(face_idx)

        # Order each dual face CCW from outside by angle in the vertex's tangent plane
        # (a global XY arctan2 fails for equatorial vertices).
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
