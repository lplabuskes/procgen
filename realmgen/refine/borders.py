"""Texel-resolution, control-biased polity border refinement.

The face-level border between two polities is the shared hexagon edge — a single
straight segment.  Because the cost surface is face-level, the boundary cannot be
recomputed from cost; instead it is refined *geometrically* as a control-biased
(multiplicatively-weighted) Voronoi boundary between the two faces' centroids.

For each pair of adjacent faces governed by different polities, the shared edge is
sampled at texel resolution.  At each sample the boundary is offset across the
edge to the point where ``dist/control`` equalises between the two centroids, so
the higher-control polity bulges into the weaker one.  The result replaces the
single straight segment with a finer, organic polyline, keyed by the same
hierarchy depth used by ``PolityStage._compute_polity_borders`` (depth 0 =
sovereign borders, depth d > 0 = internal subtree borders).
"""
from __future__ import annotations

from typing import List

import numpy as np


def _weighted_voronoi_offset(
    s: np.ndarray, d: np.ndarray, a: np.ndarray, b: np.ndarray,
    w_a: float, w_b: float, max_off: float,
) -> float:
    """Offset ``λ`` along unit direction ``d`` from edge point ``s`` to the point
    where ``|x-a|/w_a == |x-b|/w_b`` (multiplicatively-weighted bisector).

    Solves ``w_b²|x-a|² == w_a²|x-b|²`` with ``x = s + λ d`` (a quadratic in λ),
    picking the root of smallest magnitude and clamping to ``±max_off``.
    """
    u = s - a
    v = s - b
    wa2, wb2 = w_a * w_a, w_b * w_b
    qa = wb2 - wa2                                   # λ² coefficient
    qb = 2.0 * (wb2 * float(d @ u) - wa2 * float(d @ v))
    qc = wb2 * float(u @ u) - wa2 * float(v @ v)

    if abs(qa) < 1e-12:
        lam = 0.0 if abs(qb) < 1e-12 else -qc / qb
    else:
        disc = qb * qb - 4.0 * qa * qc
        if disc < 0.0:
            lam = 0.0
        else:
            root = np.sqrt(disc)
            r1 = (-qb + root) / (2.0 * qa)
            r2 = (-qb - root) / (2.0 * qa)
            lam = r1 if abs(r1) <= abs(r2) else r2
    return float(np.clip(lam, -max_off, max_off))


def _ancestor_table(polities: List, polity_of_face: np.ndarray,
                    is_water: np.ndarray) -> tuple[np.ndarray, int]:
    """``ancestor_at[d, face]`` = polity id of the face's depth-d ancestor (-1 if
    none).  Mirrors ``PolityStage._compute_polity_borders``."""
    pol_by_id = {p.polity_id: p for p in polities}
    max_depth = max(p.depth for p in polities)
    n = len(polity_of_face)

    pol_chain: dict[int, list[int]] = {}
    for pol in polities:
        chain: list[int] = []
        cur = pol
        while cur is not None:
            chain.append(cur.polity_id)
            cur = pol_by_id.get(cur.liege) if cur.liege is not None else None
        chain.reverse()
        pol_chain[pol.polity_id] = chain

    ancestor_at = np.full((max_depth + 1, n), -1, dtype=np.int32)
    for face_idx in range(n):
        if is_water[face_idx]:
            continue
        pid = int(polity_of_face[face_idx])
        if pid < 0:
            continue
        chain = pol_chain.get(pid, [pid])
        for d in range(max_depth + 1):
            ancestor_at[d, face_idx] = chain[min(d, len(chain) - 1)]
    return ancestor_at, max_depth


def refine_polity_borders(
    mesh,
    polities: List,
    polity_of_face: np.ndarray,
    is_water: np.ndarray,
    authority: np.ndarray,
    *,
    n_samples: int = 5,
    control_floor: float = 1e-3,
) -> dict:
    """Return ``{depth: (M, 6)}`` refined border segment endpoints (xyz, xyz)."""
    if not polities:
        return {}

    ancestor_at, max_depth = _ancestor_table(polities, polity_of_face, is_water)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = mesh.faces
    centroids = np.asarray(mesh.face_centroids, dtype=np.float64)
    control = np.maximum(np.asarray(authority, dtype=np.float64), control_floor)

    ts = np.linspace(0.0, 1.0, max(n_samples, 2))
    by_depth: dict[int, list] = {}

    for face_a, neighbors in enumerate(mesh.face_adjacency):
        if is_water[face_a]:
            continue
        verts_a = set(faces[face_a])
        for face_b in neighbors:
            if face_b <= face_a or is_water[face_b]:
                continue
            if ancestor_at[0, face_a] < 0 and ancestor_at[0, face_b] < 0:
                continue
            border_depth = -1
            for d in range(max_depth + 1):
                if ancestor_at[d, face_a] != ancestor_at[d, face_b]:
                    border_depth = d
                    break
            if border_depth < 0:
                continue
            shared = list(verts_a & set(faces[face_b]))
            if len(shared) != 2:
                continue

            v0 = vertices[shared[0]]
            v1 = vertices[shared[1]]
            a = centroids[face_a]
            b = centroids[face_b]
            d_hat = b - a
            nrm = np.linalg.norm(d_hat)
            if nrm < 1e-12:
                by_depth.setdefault(border_depth, []).append(np.concatenate([v0, v1]))
                continue
            d_hat = d_hat / nrm
            max_off = float(0.4 * nrm)
            w_a = float(control[face_a])
            w_b = float(control[face_b])

            pts = []
            for t in ts:
                s = (1.0 - t) * v0 + t * v1
                lam = _weighted_voronoi_offset(s, d_hat, a, b, w_a, w_b, max_off)
                p = s + lam * d_hat
                p = p / np.linalg.norm(p)  # back onto the unit sphere
                pts.append(p)

            segs = by_depth.setdefault(border_depth, [])
            for i in range(len(pts) - 1):
                segs.append(np.concatenate([pts[i], pts[i + 1]]))

    return {d: np.array(segs, dtype=np.float32) for d, segs in by_depth.items()}
