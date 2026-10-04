"""Trade-driven settlement spawning.

Three mechanisms add settlements during the population relaxation phase:

1. **Hub spawning** — faces with very high trade flow and no nearby settlement.
   Represents natural chokepoints or junctions where commerce concentrates.
   Always placed as towns (tier 2).

2. **Waypoint spawning** — intermediate stops on long inter-settlement routes.
   When the cost-distance between two settlements exceeds ``waypoint_min_cost``,
   the highest-scoring eligible face in the middle 60% of that route gets a
   village (tier 3).

3. **Port spawning** — goods move between land and water only at settlements,
   but while the settlements relax, trade may land anywhere at a steep cost.
   Where a lot of it still does, a port town (tier 2) is founded, and on the
   next pass that landing is an ordinary port.

Returned settlements are added to the settlement list *before* the relaxation
loop runs, so they participate in all subsequent trade and population passes.
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

from realmgen.placement.types import Settlement

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is an optional accelerator
    def njit(func=None, **_):  # type: ignore[misc]
        return func if func is not None else (lambda f: f)


@njit(cache=True)
def _next_waypoint(
    start: int, order: np.ndarray, k_src: np.ndarray, k_dst: np.ndarray, d_pair: np.ndarray,
    nodes: np.ndarray, pred: np.ndarray, dist: np.ndarray, forbidden: np.ndarray,
    suit: np.ndarray, flow: np.ndarray, max_flow: float, min_suit: float, max_steps: int = 10_000,
) -> tuple[int, int]:
    """``(position in order, face)`` of the next pair from ``order[start:]`` that
    gets a waypoint, or ``(len(order), -1)`` when none does.

    Each pair's path is walked back from its destination over the source's
    predecessor row (scipy's -9999 sentinel, or more than ``max_steps`` faces,
    means no path).  Its waypoint is the face of the path's middle 60% by cost
    from the source — not forbidden, at least ``min_suit`` suitable — with the
    highest ``suit × flow / max_flow``; the first along the path on ties.
    """
    for pos in range(start, len(order)):
        i = order[pos]
        ks, kd = k_src[i], k_dst[i]
        src, cur = nodes[ks], nodes[kd]
        lo, hi = 0.2 * d_pair[i], 0.8 * d_pair[i]
        best, best_score = -1, -np.inf
        for _ in range(max_steps):
            dc = dist[ks, cur]
            if lo <= dc <= hi and not forbidden[cur] and suit[cur] >= min_suit:
                score = suit[cur] * flow[cur] / max_flow
                if score >= best_score:        # walking backwards: ties go to the earlier face
                    best, best_score = cur, score
            if cur == src:
                break
            p = pred[ks, cur]
            if p < 0:
                best = -1
                break
            cur = p
        else:
            best = -1
        if best >= 0:
            return pos, int(best)
    return len(order), -1


def spawn_trade_settlements(
    n_faces: int,
    settlements: List[Settlement],
    face_flow: np.ndarray,
    suitability: np.ndarray,
    is_water: np.ndarray,
    cost_graph: csr_matrix,
    centroids: np.ndarray,
    tier_configs: list,
    *,
    hub_spawn_quantile: float = 0.97,
    hub_min_suitability: float = 0.15,
    hub_exclusion: float = 5.0,
    waypoint_min_cost: float = 30.0,
    waypoint_min_suitability: float = 0.10,
    waypoint_exclusion: float = 3.0,
    landing_flow: np.ndarray | None = None,
    port_spawn_quantile: float = 0.80,
    port_min_suitability: float = 0.05,
    port_exclusion: float = 2.0,
) -> List[Settlement]:
    """Return newly spawned trade-driven settlements (caller merges the list).

    Args:
        n_faces: mesh face count N.
        settlements: existing settlement list — not mutated.
        face_flow: ``(N,)`` float — cumulative trade flow per face from the
            initial trade pass.
        suitability: ``(N,)`` float — per-face suitability score in [0, 1].
        is_water: ``(N,)`` bool — ocean ∪ lake mask; no settlement placed here.
        cost_graph: ``(N, N)`` land-cost CSR matrix (used undirected).
        centroids: ``(N, 3)`` unit-sphere face centroids.
        tier_configs: ``[(exclusion, threshold, initial_population), ...]`` —
            index 2 = town (used for hubs), index 3 = village (used for
            waypoints).
        hub_spawn_quantile: percentile of positive face flows above which a
            face is a hub candidate.
        hub_min_suitability: minimum suitability for hub placement.
        hub_exclusion: minimum cost-distance from any settlement to spawn a hub.
        waypoint_min_cost: minimum inter-settlement cost-distance to insert a
            waypoint on the route.
        waypoint_min_suitability: minimum suitability for waypoint placement.
        waypoint_exclusion: minimum cost-distance from any settlement for a
            waypoint.
        landing_flow: ``(N,)`` float — trade flow over fallback landings (land
            faces that are not ports); ``None`` spawns no ports.
        port_spawn_quantile: a landing founds a port when its flow reaches this
            quantile of the positive face flows.
        port_min_suitability: minimum suitability for a port.
        port_exclusion: minimum cost-distance from any settlement for a port.

    Returns:
        New :class:`~realmgen.placement.types.Settlement` objects to merge.
    """
    if not settlements:
        return []

    flow = np.asarray(face_flow, dtype=np.float64)
    suit = np.asarray(suitability, dtype=np.float64)
    water = np.asarray(is_water, dtype=bool)

    s_nodes = np.array([s.face_id for s in settlements], dtype=np.int32)

    # Multi-source Dijkstra from all existing settlements.
    dist, pred = sp_dijkstra(
        cost_graph, directed=False, indices=s_nodes, return_predecessors=True,
    )
    # dist: (S, N), pred: (S, N)
    min_dist = dist.min(axis=0)  # (N,) — distance to nearest existing settlement

    spawned: List[Settlement] = []

    # ── Hub spawning ──────────────────────────────────────────────────────────

    hub_initial_pop = float(tier_configs[2][2])

    positive_flow = flow[flow > 0.0]
    if positive_flow.size > 0:
        hub_threshold = float(np.quantile(positive_flow, hub_spawn_quantile))

        # Forbidden: water, or too close to any existing settlement.
        forbidden = water | (min_dist < hub_exclusion)

        candidates = np.where(
            (flow >= hub_threshold) &
            (suit >= hub_min_suitability) &
            ~forbidden
        )[0]
        candidates = candidates[np.argsort(-flow[candidates])]   # highest flow first

        for face_id in candidates:
            if forbidden[int(face_id)]:
                continue
            spawned.append(Settlement(
                face_id=int(face_id),
                position=centroids[int(face_id)].copy(),
                tier=2,
                score=float(flow[int(face_id)]),
                population=hub_initial_pop,
            ))
            # Expand forbidden zone to prevent clustering of new hubs.
            d_from_hub = sp_dijkstra(
                cost_graph, directed=False,
                indices=[int(face_id)], return_predecessors=False, limit=hub_exclusion,
            )[0]
            forbidden |= d_from_hub < hub_exclusion

    # ── Port spawning ─────────────────────────────────────────────────────────

    if landing_flow is not None and positive_flow.size > 0:
        landing = np.asarray(landing_flow, dtype=np.float64)
        port_threshold = max(float(np.quantile(positive_flow, port_spawn_quantile)), 1e-12)
        forbidden = water | (min_dist < port_exclusion)
        for s in spawned:
            forbidden[s.face_id] = True
        candidates = np.where(
            (landing >= port_threshold) & (suit >= port_min_suitability) & ~forbidden
        )[0]
        for face_id in candidates[np.argsort(-landing[candidates])]:
            if forbidden[int(face_id)]:
                continue
            spawned.append(Settlement(
                face_id=int(face_id),
                position=centroids[int(face_id)].copy(),
                tier=2,
                score=float(landing[int(face_id)]),
                population=hub_initial_pop,
            ))
            d_from_port = sp_dijkstra(
                cost_graph, directed=False,
                indices=[int(face_id)], return_predecessors=False, limit=port_exclusion,
            )[0]
            forbidden |= d_from_port < port_exclusion

    # ── Waypoint spawning ─────────────────────────────────────────────────────

    wp_initial_pop = float(tier_configs[3][2])

    combined = settlements + spawned
    if len(combined) < 2:
        return spawned

    all_nodes = np.array([s.face_id for s in combined], dtype=np.int32)

    # Full Dijkstra from combined settlement list for inter-settlement distances.
    dist2, pred2 = sp_dijkstra(
        cost_graph, directed=False, indices=all_nodes, return_predecessors=True,
    )

    # Forbidden zone for waypoints: water or too close to any settlement.
    min_dist2 = dist2.min(axis=0)
    forbidden_wp = water | (min_dist2 < waypoint_exclusion)

    max_flow = float(flow.max()) if flow.max() > 0 else 1.0

    # Collect long pairs, process longest first so the most important routes
    # get first pick of candidate faces.
    n_combined = len(combined)
    pair_dist = dist2[:, all_nodes]                                   # (S, S)
    k_src, k_dst = np.triu_indices(n_combined, k=1)
    d_pair = pair_dist[k_src, k_dst]
    long_pair = np.isfinite(d_pair) & (d_pair >= waypoint_min_cost)
    k_src, k_dst, d_pair = k_src[long_pair], k_dst[long_pair], d_pair[long_pair]
    order = np.lexsort((k_dst, k_src, d_pair))[::-1]                  # longest first (then src, dst descending)

    pred2 = np.ascontiguousarray(pred2, dtype=np.int64)
    pos = 0
    while True:
        pos, best = _next_waypoint(
            pos, order, k_src, k_dst, d_pair, all_nodes.astype(np.int64), pred2, dist2, forbidden_wp,
            suit, flow, max_flow, float(waypoint_min_suitability),
        )
        if best < 0:
            break
        pos += 1
        spawned.append(Settlement(
            face_id=int(best),
            position=centroids[int(best)].copy(),
            tier=3,
            score=float(suit[int(best)]),
            population=wp_initial_pop,
        ))
        # Mark vicinity of the new waypoint as forbidden for future waypoints.
        d_from_wp = sp_dijkstra(
            cost_graph, directed=False,
            indices=[int(best)], return_predecessors=False, limit=waypoint_exclusion,
        )[0]
        forbidden_wp |= d_from_wp < waypoint_exclusion

    return spawned
