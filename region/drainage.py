"""Priority-flood drainage on a square raster, shared by small lakes and streams.

numba-accelerated when available, pure Python otherwise.
"""
from __future__ import annotations

import heapq

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is an optional accelerator
    def njit(func=None, **_):  # type: ignore[misc]
        return func if func is not None else (lambda f: f)


@njit(cache=True)
def priority_flood(
    elevation: np.ndarray, outlet: np.ndarray, n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Priority-flood an ``n × n`` grid (8-connected) from the ``outlet`` nodes.

    Returns ``(parent, order, level)``: the node each was flooded from (-1 at an
    outlet), the flood order (reversed: upstream first) and the filled surface
    (depressions raised to their spill level; ``elevation`` at outlets and
    nodes no outlet reaches).
    """
    size = n * n
    parent = np.full(size, -1, dtype=np.int64)
    order = np.empty(size, dtype=np.int64)
    level = elevation.copy()
    visited = outlet.copy()
    heap = [(0.0, np.int64(0)) for _ in range(0)]
    for i in range(size):
        if outlet[i]:
            heap.append((elevation[i], np.int64(i)))
    heapq.heapify(heap)
    k = 0
    while heap:
        surface, key = heapq.heappop(heap)
        order[k] = key
        k += 1
        r, c = key // n, key % n
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr, cc = r + dr, c + dc
                if (dr == 0 and dc == 0) or rr < 0 or rr >= n or cc < 0 or cc >= n:
                    continue
                nb = rr * n + cc
                if visited[nb]:
                    continue
                visited[nb] = True
                parent[nb] = key
                level[nb] = max(elevation[nb], surface)
                heapq.heappush(heap, (level[nb], np.int64(nb)))
    return parent, order[:k], level


@njit(cache=True)
def accumulate(parent: np.ndarray, order: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Tree accumulation of ``weight`` down ``parent``, upstream first."""
    acc = weight.copy()
    for i in range(order.shape[0] - 1, -1, -1):
        key = order[i]
        if parent[key] >= 0:
            acc[parent[key]] += acc[key]
    return acc
