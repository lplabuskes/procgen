"""Connected-component search over the per-face pixel graph.

Seam adjacency comes from the mesh's boundary pixel pairs
(``hydrology.rivers.inter_face_adjacency``).
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is an optional accelerator
    def njit(func=None, **_):  # type: ignore[misc]
        return func if func is not None else (lambda f: f)


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


def find_connected_components_csr(
    mask: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    min_size: int = 1,
) -> List[List[int]]:
    """:func:`find_connected_components` over a CSR neighbour graph.

    Identical output (component and node order) to the callback version.
    """
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
