"""Connected-component search over the per-face pixel graph.

Inter-face seam adjacency is now derived from the mesh's cached boundary pixel
pairs (see ``RiverGenerator._build_inter_adj_from_pairs``); the former
per-edge stitching helpers were retired in favour of that shared cache.

Public API:
    find_connected_components(mask, neighbors_fn, n_nodes, min_size=1) -> list[list[int]]
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np


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
