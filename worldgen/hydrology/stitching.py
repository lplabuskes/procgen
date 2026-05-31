"""Inter-face grid stitching and connected-component search.

Public API:
    pixels_on_edge(uv0, uv1, resolution) -> list[tuple[float, int, int]]
    stitch_face_boundaries(mesh, vertex_uvs, resolution) -> dict[int, list[int]]
    find_connected_components(mask, neighbors_fn, n_nodes, min_size=1) -> list[list[int]]
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from worldgen.mesh.mymesh import _BAND_BATCH_ELEMS, PolygonMesh


def pixels_on_edge(
    uv0: np.ndarray,
    uv1: np.ndarray,
    resolution: int,
) -> List[Tuple[float, int, int]]:
    """Pixels within ~1 pixel of UV segment uv0→uv1. Returns [(t, row, col)] sorted by t."""
    res = resolution
    edge = uv1 - uv0
    edge_len_sq = float(edge @ edge)
    if edge_len_sq < 1e-12:
        return []
    # Half-pixel tolerance keeps interior pixels out while catching the
    # boundary band on each side of the polygon edge.
    tol = 0.75 / (res - 1)
    tol_sq = tol * tol

    coords = np.arange(res) / (res - 1)
    u_grid = np.broadcast_to(coords[np.newaxis, :], (res, res))   # cols
    v_grid = np.broadcast_to(coords[:, np.newaxis], (res, res))   # rows
    du = u_grid - uv0[0]
    dv = v_grid - uv0[1]
    t = (du * edge[0] + dv * edge[1]) / edge_len_sq
    proj_u = uv0[0] + t * edge[0]
    proj_v = uv0[1] + t * edge[1]
    perp_sq = (u_grid - proj_u) ** 2 + (v_grid - proj_v) ** 2

    mask = (t >= -1e-3) & (t <= 1.0 + 1e-3) & (perp_sq < tol_sq)
    rows, cols = np.where(mask)
    if rows.size == 0:
        return []
    ts = t[rows, cols]
    order = np.argsort(ts)
    return [
        (float(ts[k]), int(rows[k]), int(cols[k]))
        for k in order
    ]


def _edge_pixels_batch(
    p0: np.ndarray,
    p1: np.ndarray,
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    tol_sq: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised :func:`pixels_on_edge` over many edges at once.

    Args:
        p0, p1: (E, 2) edge endpoints in UV.
        grid_u, grid_v: (P,) flattened pixel-centre UV coords (P = res²).
        tol_sq: squared perpendicular tolerance.

    Returns:
        (t, mask) each (E, P) — the projection parameter and the
        within-band mask, computed identically to ``pixels_on_edge`` so the
        downstream matching reproduces it bit-for-bit.
    """
    edge = p1 - p0                                       # (E, 2)
    elen_sq = (edge * edge).sum(axis=1)                  # (E,)
    safe = elen_sq >= 1e-12
    elen_safe = np.where(safe, elen_sq, 1.0)
    du = grid_u[None, :] - p0[:, 0:1]                    # (E, P)
    dv = grid_v[None, :] - p0[:, 1:2]
    t = (du * edge[:, 0:1] + dv * edge[:, 1:2]) / elen_safe[:, None]
    proj_u = p0[:, 0:1] + t * edge[:, 0:1]
    proj_v = p0[:, 1:2] + t * edge[:, 1:2]
    perp_sq = (grid_u[None, :] - proj_u) ** 2 + (grid_v[None, :] - proj_v) ** 2
    mask = (t >= -1e-3) & (t <= 1.0 + 1e-3) & (perp_sq < tol_sq) & safe[:, None]
    return t, mask


def stitch_face_boundaries(
    mesh: PolygonMesh,
    vertex_uvs: list,
    resolution: int,
) -> Dict[int, List[int]]:
    """Build inter-face node adjacency along shared polygon edges.

    Matches pixels within ~½ pixel of each shared edge by parameter t.
    Returns sparse dict: node_idx -> list[neighbour_node_idx].

    The per-edge band geometry is computed in two batched passes (one for each
    side of every shared edge) rather than per edge; the matching that follows
    is identical to the scalar :func:`pixels_on_edge` path, so the resulting
    adjacency is unchanged.
    """
    res = resolution
    inter_adj: Dict[int, List[int]] = {}
    pixels_per_face = res * res
    t_tol = 1.5 / (res - 1)
    tol = 0.75 / (res - 1)
    tol_sq = tol * tol

    coords = np.arange(res) / (res - 1)
    grid_u = np.broadcast_to(coords[np.newaxis, :], (res, res)).reshape(-1)  # cols
    grid_v = np.broadcast_to(coords[:, np.newaxis], (res, res)).reshape(-1)  # rows

    # Enumerate qualifying face pairs (shared exactly two vertices), collecting
    # both sides' edge endpoints so the band geometry can be computed in bulk.
    a0: List[np.ndarray] = []
    a1: List[np.ndarray] = []
    b0: List[np.ndarray] = []
    b1: List[np.ndarray] = []
    pair_faces: List[Tuple[int, int]] = []
    for face_a, neighbors in enumerate(mesh.face_adjacency):
        verts_a = mesh.faces[face_a]
        verts_a_set = set(verts_a)
        for face_b in neighbors:
            if face_b <= face_a:
                continue
            verts_b = mesh.faces[face_b]
            shared = list(verts_a_set & set(verts_b))
            if len(shared) != 2:
                continue
            v0, v1 = shared
            a0.append(vertex_uvs[face_a][verts_a.index(v0)])
            a1.append(vertex_uvs[face_a][verts_a.index(v1)])
            b0.append(vertex_uvs[face_b][verts_b.index(v0)])
            b1.append(vertex_uvs[face_b][verts_b.index(v1)])
            pair_faces.append((face_a, face_b))

    if not pair_faces:
        return inter_adj

    a0 = np.asarray(a0, dtype=np.float64)
    a1 = np.asarray(a1, dtype=np.float64)
    b0 = np.asarray(b0, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)

    def _add_edge(a: int, b: int) -> None:
        la = inter_adj.setdefault(a, [])
        if b not in la:
            la.append(b)
        lb = inter_adj.setdefault(b, [])
        if a not in lb:
            lb.append(a)

    # Chunk over pairs so the (chunk, res²) band arrays stay bounded regardless
    # of resolution.
    chunk = max(1, _BAND_BATCH_ELEMS // (res * res))
    for c0 in range(0, len(pair_faces), chunk):
        c1 = min(c0 + chunk, len(pair_faces))
        ta, mask_a = _edge_pixels_batch(a0[c0:c1], a1[c0:c1], grid_u, grid_v, tol_sq)
        tb, mask_b = _edge_pixels_batch(b0[c0:c1], b1[c0:c1], grid_u, grid_v, tol_sq)
        for lp in range(c1 - c0):
            a_idx = np.where(mask_a[lp])[0]
            b_idx = np.where(mask_b[lp])[0]
            if a_idx.size == 0 or b_idx.size == 0:
                continue
            face_a, face_b = pair_faces[c0 + lp]
            # Sort each side by t (np.where already gives row-major order, matching
            # the scalar path's grid traversal before its own argsort).
            ta_p = ta[lp, a_idx]
            oa = np.argsort(ta_p)
            a_idx = a_idx[oa]
            ta_p = ta_p[oa]
            tb_p = tb[lp, b_idx]
            ob = np.argsort(tb_p)
            b_idx = b_idx[ob]
            tb_p = tb_p[ob]

            face_a_base = face_a * pixels_per_face
            face_b_base = face_b * pixels_per_face
            for k in range(a_idx.size):
                diffs = np.abs(tb_p - ta_p[k])
                k_b = int(np.argmin(diffs))
                if diffs[k_b] > t_tol:
                    continue
                node_a = face_a_base + int(a_idx[k])
                node_b = face_b_base + int(b_idx[k_b])
                _add_edge(node_a, node_b)

    return inter_adj


def find_connected_components(
    mask: np.ndarray,
    neighbors_fn: Callable[[int], List[int]],
    n_nodes: int,
    min_size: int = 1,
) -> List[List[int]]:
    """Return groups of node indices where mask is True, connected via neighbors_fn."""
    visited = np.zeros(n_nodes, dtype=bool)
    components: List[List[int]] = []
    for start in np.where(mask)[0].tolist():
        if visited[start]:
            continue
        comp: List[int] = []
        stack = [start]
        visited[start] = True
        while stack:
            node = stack.pop()
            comp.append(node)
            for nb in neighbors_fn(node):
                if not visited[nb] and mask[nb]:
                    visited[nb] = True
                    stack.append(nb)
        if len(comp) >= min_size:
            components.append(comp)
    return components
