"""Graphs given as edge lists or adjacency lists: CSR conversion, connected components, edge chaining."""
from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def adjacency_csr(adjacency: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    """``(indptr, indices)`` int64 CSR of an adjacency list, neighbour order kept."""
    counts = np.fromiter((len(a) for a in adjacency), dtype=np.int64, count=len(adjacency))
    indptr = np.zeros(len(adjacency) + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    indices = np.fromiter((j for a in adjacency for j in a), dtype=np.int64, count=int(indptr[-1]))
    return indptr, indices


def edge_components(
    src: np.ndarray, dst: np.ndarray, n: int, keep: Optional[np.ndarray] = None,
) -> tuple[int, np.ndarray]:
    """``(n_components, (n,) label)``: connected components of ``n`` nodes over the
    undirected edges ``src[k]–dst[k]`` (only those where ``keep``, if given).
    Nodes on no kept edge are components of their own."""
    src, dst = np.asarray(src), np.asarray(dst)
    if keep is not None:
        src, dst = src[keep], dst[keep]
    graph = csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    n_comp, label = connected_components(graph, directed=False)
    return int(n_comp), label


def chain_edges(edges, stops=frozenset()) -> List[list]:
    """Chain undirected ``(a, b)`` edges into vertex polylines.

    Walks stop at vertices of degree != 2 and in ``stops``; a closed loop comes
    back as ``[v, ..., v]`` and junctions appear in every incident chain.
    """
    adj: dict = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    visited: set = set()

    def ekey(a, b):
        return (a, b) if a < b else (b, a)

    def is_stop(v) -> bool:
        return len(adj[v]) != 2 or v in stops

    def walk(start, nxt) -> list:
        chain = [start]
        prev, cur = start, nxt
        while True:
            chain.append(cur)
            visited.add(ekey(prev, cur))
            if is_stop(cur):
                break
            nbrs = [w for w in adj[cur] if ekey(cur, w) not in visited]
            if not nbrs:
                break
            prev, cur = cur, nbrs[0]
        return chain

    chains: List[list] = []
    for v in [v for v in adj if is_stop(v)]:
        for w in adj[v]:
            if ekey(v, w) not in visited:
                chains.append(walk(v, w))
    # Remaining edges are pure cycles.
    for a, b in edges:
        if ekey(a, b) not in visited:
            chains.append(walk(a, b))
    return chains
