"""A continuous, mean-preserving, non-negative surface through per-face values.

Some world fields exist only per face (danger), and a region drawing them flat
shows the ~50,000 km² faces as blocks with hard steps between them.
:class:`FaceFanField` replaces the step function with a piecewise-linear surface
over each face's *fan*: the triangles (centroid, vertex i, vertex i+1).  A vertex
value is shared by the three faces that meet there, so the surface is continuous,
and each face's area mean is exactly its own value:

    mean_f = (c_f + 2·W_f) / 3,   W_f = area-weighted mean of the fan's rim values

For a non-negative field that also forces the surface to 0 on every face whose
value is 0 — edges and vertices included — so danger must fade out *inside* the
faces that carry it.  Within those constraints the node values are chosen to make
the surface as smooth as possible: the least-squares solution minimising the
squared difference along every fan-triangle edge, with each face's mean an exact
equality constraint (one sparse KKT solve, repeated with any node that came out
negative pinned to 0).  Inside a uniform block of faces that gives a gentle
dome; the faces at the block's edge, whose outer rim is pinned to 0, rise above
their value to keep their means, and an isolated face — its whole rim pinned — is
a cone reaching 3× its value.

Points are located on the sphere with gnomonic barycentrics (a 3×3 solve per fan
triangle), which agree across a shared edge whichever face a point is assigned
to, so the continuity is exact rather than up to the texture seams.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import KDTree

if TYPE_CHECKING:
    from core.mesh import PolygonMesh

# Barycentric slack for "inside a triangle": points on a shared edge or vertex
# may round either way.
_INSIDE_EPS = -1e-9
# Points processed per batch (bounds the (P, K, 3) scratch arrays).
_CHUNK = 200_000
# Rounds of pinning negative nodes to 0 and re-solving.
_MAX_PIN_ROUNDS = 12


def _smoothest_nodes(
    vals: np.ndarray, n_verts: int, tri_face: np.ndarray, tri_a: np.ndarray, tri_b: np.ndarray,
    rim_weight: np.ndarray,
) -> np.ndarray:
    """Node values ``[centroids (N) | vertices (V)]`` of the smoothest surface with
    face means ``vals``, non-negative when ``vals`` is.

    ``rim_weight[t]`` is fan triangle ``t``'s share of its face's area, so the
    mean constraint reads ``c_f + Σ_t rim_weight[t]·(v_a + v_b) = 3·vals[f]``.
    """
    n = len(vals)
    size = n + n_verts
    # Constraint matrix B (one row per face) over all nodes.
    rows = np.concatenate([np.arange(n), tri_face, tri_face])
    cols = np.concatenate([np.arange(n), n + tri_a, n + tri_b])
    data = np.concatenate([np.ones(n), rim_weight, rim_weight])
    B = csr_matrix((data, (rows, cols)), shape=(n, size))
    rhs = 3.0 * vals
    # Graph Laplacian over the fan-triangle edges.
    src = np.concatenate([tri_face, tri_face, n + tri_a])
    dst = np.concatenate([n + tri_a, n + tri_b, n + tri_b])
    ones = np.ones(len(src))
    adj = coo_matrix((np.concatenate([ones, ones]), (np.concatenate([src, dst]), np.concatenate([dst, src]))),
                     shape=(size, size)).tocsr()
    lap = csr_matrix((np.asarray(adj.sum(axis=1)).ravel(), (np.arange(size), np.arange(size))),
                     shape=(size, size)) - adj

    x = np.zeros(size)
    fixed = np.zeros(size, dtype=bool)
    if vals.min() >= 0.0:
        # A face at 0 with a non-negative surface is 0 everywhere on it.
        zero = vals <= 0.0
        fixed[:n] = zero
        fixed[n + tri_a[zero[tri_face]]] = True
    active = ~(vals <= 0.0) if vals.min() >= 0.0 else np.ones(n, dtype=bool)
    for _ in range(_MAX_PIN_ROUNDS):
        free = np.flatnonzero(~fixed)
        faces = np.flatnonzero(active)
        if free.size == 0 or faces.size == 0:
            break
        Bf = B[faces][:, free]
        Lf = lap[free][:, free]
        kkt = bmat([[2.0 * Lf, Bf.T], [Bf, None]], format="csc")
        sol = spsolve(kkt, np.concatenate([np.zeros(free.size), rhs[faces]]))
        x[:] = 0.0
        x[free] = sol[:free.size]
        if vals.min() < 0.0:
            break
        negative = free[x[free] < -1e-12]
        # A centroid is its face's last free node; it never needs pinning while
        # any rim node is free, and pinning it with none free would be infeasible.
        negative = negative[negative >= n]
        if negative.size == 0:
            break
        fixed[negative] = True
    return np.maximum(x, 0.0) if vals.min() >= 0.0 else x


class FaceFanField:
    """Evaluate a continuous surface with exact per-face means at sphere points."""

    def __init__(self, mesh: "PolygonMesh", values: np.ndarray) -> None:
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        n = len(mesh.faces)
        if vals.shape[0] != n:
            raise ValueError(f"expected {n} face values, got {vals.shape[0]}")
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        cents = np.asarray(mesh.face_centroids, dtype=np.float64)
        cents = cents / np.linalg.norm(cents, axis=1, keepdims=True)

        # Fan triangles, flattened: face, rim vertices a and b.
        sizes = np.array([len(f) for f in mesh.faces], dtype=np.int64)
        tri_face = np.repeat(np.arange(n), sizes)
        tri_a = np.concatenate([np.asarray(f, dtype=np.int64) for f in mesh.faces])
        tri_b = np.concatenate([np.roll(np.asarray(f, dtype=np.int64), -1) for f in mesh.faces])
        area = 0.5 * np.linalg.norm(
            np.cross(verts[tri_a] - cents[tri_face], verts[tri_b] - cents[tri_face]), axis=1
        )
        face_area = np.bincount(tri_face, weights=area, minlength=n)
        nodes = _smoothest_nodes(vals, len(verts), tri_face, tri_a, tri_b, area / face_area[tri_face])
        centre, vertex = nodes[:n], nodes[n:]

        # Per-triangle inverse of [centroid | a | b], so bary = inv @ p.
        mats = np.stack([cents[tri_face], verts[tri_a], verts[tri_b]], axis=2)   # (T, 3, 3) columns
        self._inv = np.linalg.inv(mats)
        self._corner_values = np.stack([centre[tri_face], vertex[tri_a], vertex[tri_b]], axis=1)
        # Each face's fan triangles, padded to the largest degree.
        start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        width = int(sizes.max())
        slots = start[:, None] + np.arange(width)[None, :]
        self._face_tris = np.where(np.arange(width)[None, :] < sizes[:, None], slots, -1)
        self._tree = KDTree(cents)
        self.centre_values = centre
        self.vertex_values = vertex
        # Faces whose surface — and every neighbour's — is 0 throughout: a point
        # nearest one of them is 0 without locating it (it lies in that face or
        # an adjacent one).  Most of a sparse field like danger.
        flat = np.bincount(tri_face, weights=np.abs(self._corner_values).max(axis=1), minlength=n) == 0.0
        self._quiet = flat & np.array([all(flat[j] for j in nbrs) for nbrs in mesh.face_adjacency])

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """``(P,)`` field values at ``(P, 3)`` sphere points."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        out = np.empty(len(pts))
        for lo in range(0, len(pts), _CHUNK):
            out[lo:lo + _CHUNK] = self._evaluate(pts[lo:lo + _CHUNK])
        return out

    def _evaluate(self, pts: np.ndarray) -> np.ndarray:
        _, faces = self._tree.query(pts, k=1, workers=-1)
        faces = np.asarray(faces, dtype=np.int64)
        value = np.zeros(len(pts))
        live = np.flatnonzero(~self._quiet[faces])
        if live.size:
            value[live] = self._locate(pts[live], faces[live])
        return value

    def _locate(self, pts: np.ndarray, faces: np.ndarray) -> np.ndarray:
        value, inside = self._in_faces(pts, faces[:, None])
        # The mesh is deregularised, so ~10% of points sit in a neighbour's
        # polygon rather than their nearest centroid's; widen the search for those.
        for k in (3, 8):
            miss = np.flatnonzero(~inside)
            if miss.size == 0:
                break
            _, near = self._tree.query(pts[miss], k=k, workers=-1)
            value[miss], inside[miss] = self._in_faces(pts[miss], np.asarray(near, dtype=np.int64))
        return value

    def _in_faces(self, pts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Value in the best-fitting fan triangle among ``faces`` ``(P, K)``, and
        whether that triangle actually contains the point."""
        tris = self._face_tris[faces].reshape(len(pts), -1)          # (P, K·width)
        valid = tris >= 0
        safe = np.where(valid, tris, 0)
        bary = np.einsum("pkij,pj->pki", self._inv[safe], pts)          # (P, K·width, 3)
        # Gnomonic barycentrics: normalise so they sum to 1.
        bary = bary / np.sum(bary, axis=2, keepdims=True)
        fit = np.where(valid, bary.min(axis=2), -np.inf)
        best = np.argmax(fit, axis=1)
        rows = np.arange(len(pts))
        b = bary[rows, best]
        value = np.einsum("pi,pi->p", b, self._corner_values[safe[rows, best]])
        return value, fit[rows, best] >= _INSIDE_EPS
