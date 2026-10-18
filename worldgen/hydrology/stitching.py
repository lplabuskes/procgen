"""Connected-component search over the per-face pixel graph.

Seam adjacency comes from the mesh's boundary pixel pairs
(``hydrology.rivers.inter_face_adjacency``).
"""
from __future__ import annotations

from typing import List

import numpy as np

from utils.jit import njit


def find_connected_components_csr(
    mask: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    min_size: int = 1,
) -> List[List[int]]:
    """Groups of node indices where ``mask`` is True, connected over a CSR neighbour graph;
    components of fewer than ``min_size`` nodes are dropped."""
    mask = np.asarray(mask, dtype=bool)
    nodes, bounds = _components_kernel(mask, indptr, indices, int(min_size))
    return [nodes[bounds[i]:bounds[i + 1]].tolist() for i in range(len(bounds) - 1)]


@njit(cache=True)
def _components_kernel(mask, indptr, indices, min_size):  # pragma: no cover - compiled
    n = mask.shape[0]
    visited = np.zeros(n, dtype=np.bool_)
    total = 0
    for i in range(n):
        if mask[i]:
            total += 1
    nodes = np.empty(total, dtype=np.int64)
    stack = np.empty(total, dtype=np.int64)
    bounds = [0]
    pos = 0
    for start in range(n):
        if not mask[start] or visited[start]:
            continue
        begin = pos
        stack[0] = start
        top = 1
        visited[start] = True
        while top > 0:
            top -= 1
            node = stack[top]
            nodes[pos] = node
            pos += 1
            for e in range(indptr[node], indptr[node + 1]):
                nb = indices[e]
                if not visited[nb] and mask[nb]:
                    visited[nb] = True
                    stack[top] = nb
                    top += 1
        if pos - begin >= min_size:
            bounds.append(pos)
        else:
            pos = begin
    return nodes[:pos], np.array(bounds, dtype=np.int64)
