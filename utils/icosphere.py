"""The subdivided icosphere as a triangle lattice, and the edges of a triangle array."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

# Angle subtended by an icosahedron edge; each subdivision halves it.
ICOSA_EDGE_RAD = 1.1071487177940904


def unique_edges(tris: np.ndarray, n_verts: int) -> tuple[np.ndarray, np.ndarray]:
    """``(edges (E, 2), tri_edges (F, 3))`` for a triangle array.

    ``tri_edges[f, k]`` indexes the edge from corner ``k`` to corner ``(k + 1) % 3``.
    """
    a = tris
    b = np.roll(tris, -1, axis=1)
    lo = np.minimum(a, b).astype(np.int64)
    hi = np.maximum(a, b).astype(np.int64)
    keys = (lo * n_verts + hi).ravel()
    uniq, inverse = np.unique(keys, return_inverse=True)
    edges = np.stack([uniq // n_verts, uniq % n_verts], axis=1)
    return edges, inverse.reshape(tris.shape)


@lru_cache(maxsize=2)
def icosphere_lattice(depth: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(verts, tris, edges, tri_edges)`` of an icosphere subdivided ``depth`` times.

    Vectorised because ``Mesh.icosphere`` is far too slow at ~10⁶ vertices.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    tris = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    for _ in range(depth):
        n = len(verts)
        edges, te = unique_edges(tris, n)
        mids = verts[edges[:, 0]] + verts[edges[:, 1]]
        mids /= np.linalg.norm(mids, axis=1, keepdims=True)
        m = te + n                     # midpoint vertex of edge (k, k+1)
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        mab, mbc, mca = m[:, 0], m[:, 1], m[:, 2]
        tris = np.concatenate([
            np.stack([a, mab, mca], axis=1),
            np.stack([mab, b, mbc], axis=1),
            np.stack([mca, mbc, c], axis=1),
            np.stack([mab, mbc, mca], axis=1),
        ])
        verts = np.vstack([verts, mids])
    edges, te = unique_edges(tris, len(verts))
    return verts, tris, edges, te
