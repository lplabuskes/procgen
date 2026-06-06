"""Settlement seeder — hierarchical greedy placement on the land cost graph.

Placement score
---------------
``placement_score`` blends suitability and static situational value into one
[0, 1] scalar.  The weight is a single tunable so callers don't have to think
about the two individually; the default (0.40) gives situational value a
significant but not dominant role.

Hierarchical seeding
--------------------
``seed_settlements`` processes tiers from highest (widest exclusion) to lowest.
Within each tier it greedily picks the highest-scoring non-forbidden land face,
places a settlement, then runs Dijkstra from that face over the land cost graph
and forbids all faces within the tier's exclusion radius.

The forbidden mask is *cumulative*: a tier-0 capital's large exclusion zone
prevents any later settlement — city, town, or village — from being placed
within it.  This keeps the capital as the sole urban anchor for its region,
which is the intended emergent behaviour.  Lower-tier settlements fill the gaps
between capitals and cities on subsequent passes.

Cost graph
----------
``build_cost_graph`` converts the per-face adjacency list + per-edge float
arrays (as returned by ``worldgen.cost.surfaces.build_land_edge_costs``) into a
directed scipy CSR matrix.  Impassable edges (``np.inf``) are omitted; scipy's
Dijkstra treats missing entries as infinite automatically.
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
    """Weighted blend of suitability and situational value, clipped to [0, 1].

    Args:
        suitability:       (N,) float in [0, 1] from the suitability scorer.
        situational_value: (N,) float in [0, 1] from the situational-value scorer.
        situational_weight: fraction given to situational value (default 0.40).

    Returns:
        (N,) float32 in [0, 1].
    """
    w = float(np.clip(situational_weight, 0.0, 1.0))
    suit = np.asarray(suitability, dtype=np.float64)
    sit = np.asarray(situational_value, dtype=np.float64)
    return np.clip((1.0 - w) * suit + w * sit, 0.0, 1.0).astype(np.float32)


def build_cost_graph(
    face_adjacency: List[List[int]],
    edge_costs: List[np.ndarray],
) -> csr_matrix:
    """Directed CSR sparse matrix of land traversal costs.

    Converts the per-face adjacency list and per-edge cost arrays (as produced
    by ``worldgen.cost.surfaces.build_land_edge_costs``) into a scipy CSR
    matrix.  Infinite-cost edges are omitted — scipy's Dijkstra treats missing
    entries as unreachable.

    Args:
        face_adjacency: list of length N; each element is the list of neighbor
            face indices for face i.
        edge_costs: list of length N; each element is a float64 array of costs
            aligned with ``face_adjacency[i]``.

    Returns:
        Directed CSR sparse matrix of shape (N, N) with dtype float64.
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
    tier_configs: list[tuple[float, float, float, float]],
) -> list[Settlement]:
    """Hierarchical greedy settlement seeder.

    For each tier (from highest order to lowest) the algorithm iterates:

    1. Find the highest-scoring non-forbidden face with score ≥ threshold.
    2. Place a settlement there.
    3. Run Dijkstra from that face; forbid all faces within exclusion_radius.
    4. Repeat until no qualifying face remains for this tier.

    The forbidden mask is cumulative across tiers: a capital's wide exclusion
    zone (typically the largest) prevents any subsequent settlement of any tier
    from being placed within it.

    Args:
        score:        (N,) float32 combined placement score in [0, 1].
        is_water:     (N,) bool — always forbidden.
        face_centroids: (N, 3) float64 unit-sphere centroids.
        cost_graph:   directed CSR sparse matrix from :func:`build_cost_graph`.
        tier_configs: list of ``(exclusion_radius, score_threshold, influence,
            demand)`` per tier, ordered from highest (widest exclusion) to
            lowest.  ``influence`` is the hinterland Dijkstra offset; ``demand``
            is the initial trade demand weight.

    Returns:
        Ordered list of :class:`~realmgen.placement.types.Settlement` objects in
        placement order (highest tier first).
    """
    forbidden = np.asarray(is_water, dtype=bool).copy()
    score_arr = np.asarray(score, dtype=np.float64)
    centroids = np.asarray(face_centroids, dtype=np.float64)
    settlements: list[Settlement] = []

    for tier_idx, (excl_radius, threshold, influence, demand) in enumerate(tier_configs):
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
                influence=float(influence),
                demand=float(demand),
            ))

            # Forbid everything reachable within the exclusion radius.
            dists = sp_dijkstra(
                cost_graph,
                directed=True,
                indices=best,
                limit=excl_radius,
                return_predecessors=False,
            )
            forbidden |= dists <= excl_radius

    return settlements
