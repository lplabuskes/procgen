"""Trade network — flow accumulation on the hybrid land/sea cost graph.

Goods flow from sources (mineral deposits, rare biological resources) to
settlements along the least-cost path on a *hybrid* graph that allows travel
on either the land or sea surface, switching between them at navigable-river
faces (amphibious) and coastal port transitions.

The algorithm
-------------
1. Build a ``(2N + 1)``-node hybrid graph (land nodes ``0..N-1``, sea nodes
   ``N..2N-1``, virtual source at ``2N``).
2. Connect the virtual source to each settlement's land node with cost
   ``demand_penalty × (1 - normalised_demand)``, so high-demand (high-tier)
   settlements are effectively closer and attract more routing.
3. Single-source Dijkstra from the virtual source — finds the nearest
   settlement (by hybrid cost) for every reachable node.
4. For each good source (face with abundance > 0), follow the predecessor tree
   back to the virtual source, adding flow ``abundance × exp(−flow_decay × dist)``
   to every edge on the path.
5. Aggregate per-edge flow into per-face flow and extract high-flow route
   segments for display.

Surface-switching rules
-----------------------
- **Navigable river faces** (``land_passable AND sea_passable``): free switch in
  both directions (cost ``river_switch_cost``, default 0).
- **Port transitions**: coastal land face *i* adjacent to sea face *j* →
  directed edges ``(i → sea_j)`` and ``(sea_j → i)`` each with cost
  ``port_cost``.  No other land↔sea crossing is permitted.

Node indexing
-------------
- Land node for face ``i``: index ``i``
- Sea node for face ``i``: index ``i + N``
- Virtual source: index ``2N``
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

from realmgen.placement.types import Settlement

DEFAULT_TIER_DEMANDS = (1.0, 0.6, 0.3, 0.12)


@dataclass
class TradeResult:
    """Output of :func:`compute_trade_network`."""

    face_flow: np.ndarray   # (N,) float32 — cumulative trade flow touching each face
    hub_mask: np.ndarray    # (N,) bool — high-flow junction faces
    routes: list            # [(start_3d, end_3d, flow)] segments above display threshold


def build_hybrid_cost_graph(
    n: int,
    face_adjacency: List[List[int]],
    land_edge_costs: List[np.ndarray],
    land_passable: np.ndarray,
    sea_edge_costs: List[np.ndarray],
    sea_passable: np.ndarray,
    settlement_land_nodes: List[int],
    settlement_demands: np.ndarray,
    *,
    port_cost: float = 5.0,
    river_switch_cost: float = 0.0,
    demand_penalty: float = 5.0,
) -> csr_matrix:
    """Build a ``(2N+1, 2N+1)`` hybrid land/sea cost graph.

    Args:
        n: number of mesh faces.
        face_adjacency: ``list[list[int]]`` of length N.
        land_edge_costs: per-edge costs from ``build_land_edge_costs``, len N.
        land_passable: ``(N,)`` bool — False on open water (ocean ∪ lake).
        sea_edge_costs: per-edge costs from ``build_sea_edge_costs``, len N.
        sea_passable: ``(N,)`` bool — True on sailable water ∪ navigable river.
        settlement_land_nodes: face indices of placed settlements.
        settlement_demands: ``(n_settlements,)`` float in [0, 1] — normalised
            demand weights; index-aligned with ``settlement_land_nodes``.
        port_cost: one-time loading/unloading cost at a coastal port.
        river_switch_cost: cost to switch surfaces at a navigable river face.
        demand_penalty: maximum extra cost added to the virtual-source edge for
            the lowest-demand settlement; 0 = all equally attractive.

    Returns:
        Directed ``(2N+1, 2N+1)`` CSR matrix.
    """
    virtual = 2 * n
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    land_p = np.asarray(land_passable, dtype=bool)
    sea_p = np.asarray(sea_passable, dtype=bool)
    amphibious = land_p & sea_p  # navigable river faces

    # Land-surface edges (nodes 0..N-1)
    for i, (adj, costs) in enumerate(zip(face_adjacency, land_edge_costs)):
        for j, c in zip(adj, costs):
            if np.isfinite(c):
                rows.append(i); cols.append(j); data.append(float(c))

    # Sea-surface edges (nodes N..2N-1)
    for i, (adj, costs) in enumerate(zip(face_adjacency, sea_edge_costs)):
        for j, c in zip(adj, costs):
            if np.isfinite(c):
                rows.append(n + i); cols.append(n + j); data.append(float(c))

    # Navigable river surface switches (amphibious faces)
    sc = max(float(river_switch_cost), 0.0)
    for raw_i in np.where(amphibious)[0]:
        fi = int(raw_i)
        rows.append(fi);     cols.append(n + fi); data.append(sc)
        rows.append(n + fi); cols.append(fi);     data.append(sc)

    # Port transitions: pure-land face i ↔ adjacent pure-sea face j
    pc = max(float(port_cost), 0.0)
    for i, adj in enumerate(face_adjacency):
        if not land_p[i] or sea_p[i]:
            continue  # only from pure land faces
        for j in adj:
            if sea_p[j] and not land_p[j]:
                rows.append(i);     cols.append(n + j); data.append(pc)
                rows.append(n + j); cols.append(i);     data.append(pc)

    # Virtual source → settlements (weighted by demand)
    dp = max(float(demand_penalty), 0.0)
    demands = np.asarray(settlement_demands, dtype=np.float64)
    for s_node, demand in zip(settlement_land_nodes, demands):
        virtual_cost = dp * (1.0 - float(demand))
        rows.append(virtual); cols.append(int(s_node)); data.append(virtual_cost)

    shape = (virtual + 1, virtual + 1)
    if not data:
        return csr_matrix(shape, dtype=np.float64)
    return csr_matrix(
        (
            np.array(data, dtype=np.float64),
            (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)),
        ),
        shape=shape,
    )


def compute_trade_network(
    mesh,
    settlements: List[Settlement],
    good_sources: List[tuple],
    land_edge_costs: List[np.ndarray],
    land_passable: np.ndarray,
    sea_edge_costs: List[np.ndarray],
    sea_passable: np.ndarray,
    *,
    port_cost: float = 5.0,
    river_switch_cost: float = 0.0,
    tier_demands: tuple = DEFAULT_TIER_DEMANDS,
    demand_penalty: float = 5.0,
    flow_decay: float = 0.05,
    hub_flow_quantile: float = 0.90,
    min_display_flow: float | None = None,
) -> TradeResult:
    """Accumulate trade flow from good sources to settlements on the hybrid graph.

    For each good source, the predecessor tree from the single-source Dijkstra
    (originating at the virtual source) gives the route from that source to its
    nearest settlement.  Flow along the path is:

        flow = abundance × demand(settlement) × exp(−flow_decay × path_cost)

    where ``demand`` is the settlement's tier weight recovered by walking to the
    settlement node on the predecessor path.

    Args:
        mesh: ``PolygonMesh`` — provides ``face_adjacency`` and ``face_centroids``.
        settlements: list of :class:`~realmgen.placement.types.Settlement`.
        good_sources: list of ``(face_id, abundance)`` tuples — one entry per
            deposit. ``face_id`` is a land-face index; ``abundance`` is a
            non-negative float.
        land_edge_costs: from ``build_land_edge_costs``, len N.
        land_passable: ``(N,)`` bool.
        sea_edge_costs: from ``build_sea_edge_costs``, len N.
        sea_passable: ``(N,)`` bool.
        port_cost: one-time loading/unloading cost at a coastal transition.
        river_switch_cost: cost to switch surfaces at a navigable river face.
        tier_demands: demand weights indexed by ``Settlement.tier``.
        demand_penalty: extra cost added to the virtual-source edge for the
            lowest-demand settlement (0 = all equally attractive).
        flow_decay: exponential decay factor for flow with path cost; larger
            values suppress long-distance trade.
        hub_flow_quantile: percentile of positive face flows above which a face
            is flagged as a trade hub.
        min_display_flow: minimum edge flow to include in the route list for
            display; defaults to the median of all positive edge flows.

    Returns:
        :class:`TradeResult`.
    """
    n = len(mesh.faces)
    virtual = 2 * n

    empty = TradeResult(
        face_flow=np.zeros(n, dtype=np.float32),
        hub_mask=np.zeros(n, dtype=bool),
        routes=[],
    )
    if not settlements or not good_sources:
        return empty

    centroids = np.asarray(mesh.face_centroids, dtype=np.float64)

    # Normalised demand weights per settlement (capitals = 1.0)
    raw_demands = np.array(
        [float(tier_demands[min(s.tier, len(tier_demands) - 1)]) for s in settlements],
        dtype=np.float64,
    )
    max_d = raw_demands.max()
    demands = raw_demands / max_d if max_d > 0 else raw_demands

    settlement_land_nodes = [s.face_id for s in settlements]
    settlement_demand_map = {
        int(s.face_id): float(d) for s, d in zip(settlements, demands)
    }

    # Build hybrid graph
    graph = build_hybrid_cost_graph(
        n, mesh.face_adjacency,
        land_edge_costs, land_passable,
        sea_edge_costs, sea_passable,
        settlement_land_nodes, demands,
        port_cost=port_cost,
        river_switch_cost=river_switch_cost,
        demand_penalty=demand_penalty,
    )

    # Single-source Dijkstra from the virtual node
    dist, pred = sp_dijkstra(
        graph,
        directed=True,
        indices=virtual,
        return_predecessors=True,
    )

    # Flow accumulation along predecessor trees
    decay = float(flow_decay)
    edge_flow: dict[tuple[int, int], float] = {}

    for face_id, abundance in good_sources:
        src_node = int(face_id)
        d = dist[src_node]
        if not np.isfinite(d):
            continue

        # Walk predecessor tree to recover demand of the nearest settlement
        path_edges: List[tuple[int, int]] = []
        cur = src_node
        nearest_demand = 1.0
        while True:
            p = int(pred[cur])
            if p < 0 or p == virtual:
                break
            path_edges.append((p, cur))
            cur = p
        # cur is now the settlement land node
        if cur in settlement_demand_map:
            nearest_demand = settlement_demand_map[cur]

        flow = float(abundance) * nearest_demand * np.exp(-decay * d)
        if flow < 1e-9 or not path_edges:
            continue

        for edge in path_edges:
            edge_flow[edge] = edge_flow.get(edge, 0.0) + flow

    if not edge_flow:
        return empty

    # Per-face flow: sum of all incident edge flows
    face_flow = np.zeros(n, dtype=np.float64)
    for (src_node, dst_node), flow in edge_flow.items():
        src_face = src_node % n
        dst_face = dst_node % n
        face_flow[src_face] += flow
        face_flow[dst_face] += flow

    # Hub detection
    positive = face_flow[face_flow > 0.0]
    if positive.size > 0:
        hub_threshold = float(np.quantile(positive, hub_flow_quantile))
    else:
        hub_threshold = np.inf
    hub_mask = face_flow >= hub_threshold

    # Route segments for display
    all_flows = np.array(list(edge_flow.values()), dtype=np.float64)
    if min_display_flow is None:
        min_display_flow = float(np.median(all_flows)) if all_flows.size > 0 else 0.0

    routes = []
    for (src_node, dst_node), flow in edge_flow.items():
        if flow < min_display_flow:
            continue
        src_face = src_node % n
        dst_face = dst_node % n
        routes.append((centroids[src_face].copy(), centroids[dst_face].copy(), float(flow)))

    return TradeResult(
        face_flow=face_flow.astype(np.float32),
        hub_mask=hub_mask,
        routes=routes,
    )
