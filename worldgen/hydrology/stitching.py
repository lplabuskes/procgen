"""Inter-face grid stitching and connected-component search.

Public API:
    pixels_on_edge(uv0, uv1, resolution) -> list[tuple[float, int, int]]
    stitch_face_boundaries(mesh, vertex_uvs, resolution) -> dict[int, list[int]]
    find_connected_components(mask, neighbors_fn, n_nodes, min_size=1) -> list[list[int]]
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from worldgen.mesh.mymesh import PolygonMesh


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


def stitch_face_boundaries(
    mesh: PolygonMesh,
    vertex_uvs: list,
    resolution: int,
) -> Dict[int, List[int]]:
    """Build inter-face node adjacency along shared polygon edges.

    Matches pixels within ~½ pixel of each shared edge by parameter t.
    Returns sparse dict: node_idx -> list[neighbour_node_idx].
    """
    inter_adj: Dict[int, List[int]] = {}
    pixels_per_face = resolution * resolution
    t_tol = 1.5 / (resolution - 1)

    def _add_edge(a: int, b: int) -> None:
        la = inter_adj.setdefault(a, [])
        if b not in la:
            la.append(b)
        lb = inter_adj.setdefault(b, [])
        if a not in lb:
            lb.append(a)

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
            uv_a0 = np.asarray(vertex_uvs[face_a][verts_a.index(v0)], dtype=np.float64)
            uv_a1 = np.asarray(vertex_uvs[face_a][verts_a.index(v1)], dtype=np.float64)
            uv_b0 = np.asarray(vertex_uvs[face_b][verts_b.index(v0)], dtype=np.float64)
            uv_b1 = np.asarray(vertex_uvs[face_b][verts_b.index(v1)], dtype=np.float64)

            p_a = pixels_on_edge(uv_a0, uv_a1, resolution)
            p_b = pixels_on_edge(uv_b0, uv_b1, resolution)
            if not p_a or not p_b:
                continue

            tb = np.array([p[0] for p in p_b])
            for t_a, r_a, c_a in p_a:
                diffs = np.abs(tb - t_a)
                k_b = int(np.argmin(diffs))
                if diffs[k_b] > t_tol:
                    continue
                _, r_b, c_b = p_b[k_b]
                node_a = face_a * pixels_per_face + r_a * resolution + c_a
                node_b = face_b * pixels_per_face + r_b * resolution + c_b
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
