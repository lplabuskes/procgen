"""Settlement seeder — hierarchical greedy placement on the land cost graph.

``seed_settlements`` runs tiers from widest exclusion to narrowest, greedily
placing at the best non-forbidden face and forbidding everything within the
tier's exclusion radius.  The forbidden mask is cumulative, so a capital stays
the sole urban anchor of its region; lower tiers fill the gaps.
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

from realmgen.placement.types import Settlement


def placement_score(
    suitability: np.ndarray,
    situational_value: np.ndarray,
    *,
    situational_weight: float = 0.40,
) -> np.ndarray:
    """(N,) float32 blend ``(1 − w)·suitability + w·situational_value``, clipped to [0, 1]."""
    w = float(np.clip(situational_weight, 0.0, 1.0))
    suit = np.asarray(suitability, dtype=np.float64)
    sit = np.asarray(situational_value, dtype=np.float64)
    return np.clip((1.0 - w) * suit + w * sit, 0.0, 1.0).astype(np.float32)


def build_cost_graph(
    face_adjacency: List[List[int]],
    edge_costs: List[np.ndarray],
) -> csr_matrix:
    """Directed (N, N) float64 CSR of land traversal costs; infinite edges omitted.

    ``edge_costs[i]`` is aligned with ``face_adjacency[i]`` (as from
    ``worldgen.cost.surfaces.build_land_edge_costs``).
    """
    rows: list = []
    cols: list = []
    data: list = []
    for i, (adj, costs) in enumerate(zip(face_adjacency, edge_costs)):
        for j, c in zip(adj, costs):
            c_f = float(c)
            if np.isfinite(c_f):
                rows.append(i)
                cols.append(j)
                data.append(c_f)
    n = len(face_adjacency)
    if not data:
        return csr_matrix((n, n), dtype=np.float64)
    return csr_matrix(
        (
            np.array(data, dtype=np.float64),
            (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)),
        ),
        shape=(n, n),
    )


def seed_settlements(
    score: np.ndarray,
    is_water: np.ndarray,
    face_centroids: np.ndarray,
    cost_graph: csr_matrix,
    tier_configs: list[tuple[float, float, float]],
) -> list[Settlement]:
    """Hierarchical greedy settlement seeder (see module docstring).

    Args:
        score: (N,) placement score in [0, 1].
        face_centroids: (N, 3) unit-sphere centroids.
        tier_configs: ``(exclusion_radius, score_threshold, initial_population)``
            per tier, widest exclusion first.

    Returns:
        Settlements in placement order.
    """
    forbidden = np.asarray(is_water, dtype=bool).copy()
    score_arr = np.asarray(score, dtype=np.float64)
    centroids = np.asarray(face_centroids, dtype=np.float64)
    settlements: list[Settlement] = []

    for tier_idx, (excl_radius, threshold, initial_pop) in enumerate(tier_configs):
        while True:
            available = ~forbidden
            if not available.any():
                break
            masked = np.where(available, score_arr, -np.inf)
            best = int(np.argmax(masked))
            if not available[best] or score_arr[best] < threshold:
                break

            settlements.append(Settlement(
                face_id=best,
                position=centroids[best].copy(),
                tier=tier_idx,
                score=float(score_arr[best]),
                population=float(initial_pop),
            ))

            dists = sp_dijkstra(
                cost_graph,
                directed=True,
                indices=best,
                limit=excl_radius,
                return_predecessors=False,
            )
            forbidden |= dists <= excl_radius

    return settlements
