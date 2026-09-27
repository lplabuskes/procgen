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


def _trace_path(pred_row: np.ndarray, src: int, dst: int, max_steps: int = 10_000) -> list[int]:
    """Return face sequence from *src* to *dst* via the predecessor map.

    Args:
        pred_row: predecessor row for source *src*, shape (N,) int32.
        src: source face id.
        dst: destination face id.
        max_steps: loop-break limit for safety.

    Returns:
        List ``[src, ..., dst]``, or ``[]`` if the path is broken.
    """
    path: list[int] = []
    cur = dst
    for _ in range(max_steps):
        path.append(cur)
        if cur == src:
            break
        p = int(pred_row[cur])
        if p < 0:   # scipy sentinel -9999 = no predecessor
            return []
        cur = p
    else:
        return []
    path.reverse()
    return path


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
                indices=[int(face_id)], return_predecessors=False,
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
                indices=[int(face_id)], return_predecessors=False,
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
    pair_costs: list[tuple[float, int, int]] = []
    n_combined = len(combined)
    for k_src in range(n_combined):
        for k_dst in range(k_src + 1, n_combined):
            d = float(dist2[k_src, int(all_nodes[k_dst])])
            if np.isfinite(d) and d >= waypoint_min_cost:
                pair_costs.append((d, k_src, k_dst))

    pair_costs.sort(reverse=True)

    for total_cost, k_src, k_dst in pair_costs:
        src_node = int(all_nodes[k_src])
        dst_node = int(all_nodes[k_dst])

        path = _trace_path(pred2[k_src], src_node, dst_node)
        if not path:
            continue

        # Keep only the middle 60% of the path by cost-distance from the source,
        # excluding faces too close to existing or newly placed settlements.
        lo = 0.2 * total_cost
        hi = 0.8 * total_cost
        eligible = [
            f for f in path
            if (lo <= float(dist2[k_src, f]) <= hi)
            and not forbidden_wp[f]
            and float(suit[f]) >= waypoint_min_suitability
        ]
        if not eligible:
            continue

        # Pick the face with the highest combined suitability × normalised flow.
        best = max(eligible, key=lambda f: float(suit[f]) * float(flow[f]) / max_flow)

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
            indices=[int(best)], return_predecessors=False,
        )[0]
        forbidden_wp |= d_from_wp < waypoint_exclusion

    return spawned
