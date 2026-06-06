"""Trade network — two-phase flow accumulation on the hybrid land/sea cost graph.

Phase 1 — Resources to settlements
    Each deposit routes to its nearest settlement (by hybrid cost).  Short,
    local routes.  The settlement accumulates the raw abundance as "goods."

Phase 2 — Inter-settlement trade
    Each settlement with goods routes to every other settlement.  Flow is:

        flow = goods × demand(dst) × exp(−flow_decay × inter_settlement_cost)

    These are the longer, interesting routes shown on the map.

The underlying graph is (2N, 2N): land nodes 0..N-1, sea nodes N..2N-1.
Surface switching happens at navigable-river faces (amphibious, zero cost) and
at coastal port transitions (land face ↔ adjacent sea face, port_cost).

A single undirected multi-source Dijkstra from all settlement nodes provides
both the nearest-settlement assignment (Phase 1) and inter-settlement distances
(Phase 2) in one pass.

Node indexing
-------------
- Land node for face i: index i
- Sea node for face i: index i + N
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


# ── Internal helpers ──────────────────────────────────────────────────────────


def _collect_hybrid_edges(
    n: int,
    face_adjacency: List[List[int]],
    land_edge_costs: List[np.ndarray],
    land_passable: np.ndarray,
    sea_edge_costs: List[np.ndarray],
    sea_passable: np.ndarray,
    port_cost: float,
    river_switch_cost: float,
) -> tuple[list, list, list]:
    """Return (rows, cols, data) for all hybrid graph edges.

    Builds the base (2N, 2N) graph — land surface, sea surface, amphibious
    river switches, and port transitions.  Virtual-source edges are NOT included.
    """
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    land_p = np.asarray(land_passable, dtype=bool)
    sea_p = np.asarray(sea_passable, dtype=bool)
    amphibious = land_p & sea_p

    for i, (adj, costs) in enumerate(zip(face_adjacency, land_edge_costs)):
        for j, c in zip(adj, costs):
            if np.isfinite(c):
                rows.append(i); cols.append(j); data.append(float(c))

    for i, (adj, costs) in enumerate(zip(face_adjacency, sea_edge_costs)):
        for j, c in zip(adj, costs):
            if np.isfinite(c):
                rows.append(n + i); cols.append(n + j); data.append(float(c))

    sc = max(float(river_switch_cost), 0.0)
    for raw_i in np.where(amphibious)[0]:
        fi = int(raw_i)
        rows.append(fi);     cols.append(n + fi); data.append(sc)
        rows.append(n + fi); cols.append(fi);     data.append(sc)

    pc = max(float(port_cost), 0.0)
    for i, adj in enumerate(face_adjacency):
        if not land_p[i] or sea_p[i]:
            continue
        for j in adj:
            if sea_p[j] and not land_p[j]:
                rows.append(i);     cols.append(n + j); data.append(pc)
                rows.append(n + j); cols.append(i);     data.append(pc)

    return rows, cols, data


def _accumulate_path(
    pred_row: np.ndarray,
    from_node: int,
    to_node: int,
    flow: float,
    edge_flow: dict,
    max_steps: int = 100_000,
) -> None:
    """Walk predecessor array from from_node back to to_node, adding flow to each edge."""
    cur = from_node
    for _ in range(max_steps):
        p = int(pred_row[cur])
        if p < 0:
            break
        key = (min(p, cur), max(p, cur))
        edge_flow[key] = edge_flow.get(key, 0.0) + flow
        cur = p
        if cur == to_node:
            break


# ── Public API ────────────────────────────────────────────────────────────────


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
    """Build a ``(2N+1, 2N+1)`` hybrid land/sea cost graph with virtual source.

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
    rows, cols, data = _collect_hybrid_edges(
        n, face_adjacency, land_edge_costs, land_passable,
        sea_edge_costs, sea_passable, port_cost, river_switch_cost,
    )

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
    """Accumulate trade flow in two phases.

    **Phase 1** — each deposit routes to its nearest settlement (by hybrid cost).
    Flow on each path segment = ``abundance × exp(−flow_decay × dist_to_settlement)``.
    The raw abundance is credited to that settlement as "goods."

    **Phase 2** — each settlement with goods routes to every other settlement.
    Flow = ``goods × demand(dst) × exp(−flow_decay × inter_settlement_cost)``.
    This produces the inter-settlement trade network shown on the map.

    Args:
        mesh: ``PolygonMesh`` — provides ``face_adjacency`` and ``face_centroids``.
        settlements: list of :class:`~realmgen.placement.types.Settlement`.
        good_sources: list of ``(face_id, abundance)`` tuples — one entry per
            deposit.  ``face_id`` is a land-face index; ``abundance`` is a
            non-negative float.
        land_edge_costs: from ``build_land_edge_costs``, len N.
        land_passable: ``(N,)`` bool.
        sea_edge_costs: from ``build_sea_edge_costs``, len N.
        sea_passable: ``(N,)`` bool.
        port_cost: one-time loading/unloading cost at a coastal transition.
        river_switch_cost: cost to switch surfaces at a navigable river face.
        tier_demands: demand weights indexed by ``Settlement.tier``.
        demand_penalty: unused in two-phase model; kept for API compatibility.
        flow_decay: exponential decay factor applied to both phase-1 and
            phase-2 routes with increasing cost distance.
        hub_flow_quantile: percentile of positive face flows above which a
            face is flagged as a trade hub.
        min_display_flow: minimum edge flow to include in the route list for
            display; defaults to the median of all positive edge flows.

    Returns:
        :class:`TradeResult`.
    """
    n = len(mesh.faces)
    empty = TradeResult(
        face_flow=np.zeros(n, dtype=np.float32),
        hub_mask=np.zeros(n, dtype=bool),
        routes=[],
    )
    if not settlements or not good_sources:
        return empty

    centroids = np.asarray(mesh.face_centroids, dtype=np.float64)

    # Normalised demand weights per settlement (tier 0 capitals = 1.0)
    raw_demands = np.array(
        [float(tier_demands[min(s.tier, len(tier_demands) - 1)]) for s in settlements],
        dtype=np.float64,
    )
    max_d = raw_demands.max()
    norm_demands = raw_demands / max_d if max_d > 0 else raw_demands.copy()
    settlement_nodes = [s.face_id for s in settlements]

    # Build (2N, 2N) hybrid graph (no virtual source)
    size = 2 * n
    rows, cols, data = _collect_hybrid_edges(
        n, mesh.face_adjacency,
        land_edge_costs, np.asarray(land_passable, dtype=bool),
        sea_edge_costs, np.asarray(sea_passable, dtype=bool),
        port_cost, river_switch_cost,
    )
    if data:
        graph = csr_matrix(
            (
                np.array(data, dtype=np.float64),
                (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)),
            ),
            shape=(size, size),
        )
    else:
        graph = csr_matrix((size, size), dtype=np.float64)

    # Single undirected multi-source Dijkstra from all settlement land nodes.
    # dist[k, node] = shortest (undirected) distance from settlement k to node.
    # pred[k, node] = predecessor of node on shortest path from settlement k.
    dist, pred = sp_dijkstra(
        graph, directed=False, indices=settlement_nodes, return_predecessors=True,
    )

    decay = float(flow_decay)
    edge_flow: dict[tuple[int, int], float] = {}
    deposit_goods = np.zeros(len(settlements), dtype=np.float64)

    # Phase 1: route each deposit to its nearest settlement
    for face_id, abundance in good_sources:
        src_node = int(face_id)
        if src_node >= size:
            continue
        dists_to_src = dist[:, src_node]
        k = int(np.argmin(dists_to_src))
        d = float(dists_to_src[k])
        if not np.isfinite(d):
            continue
        deposit_goods[k] += float(abundance)
        phase1_flow = float(abundance) * np.exp(-decay * d)
        if phase1_flow < 1e-9:
            continue
        _accumulate_path(pred[k], src_node, settlement_nodes[k], phase1_flow, edge_flow)

    # Phase 2: inter-settlement trade
    for k_src in range(len(settlements)):
        goods = deposit_goods[k_src]
        if goods <= 0.0:
            continue
        src_node = settlement_nodes[k_src]
        for k_dst in range(len(settlements)):
            if k_src == k_dst:
                continue
            dst_node = settlement_nodes[k_dst]
            d = float(dist[k_src, dst_node])
            if not np.isfinite(d):
                continue
            inter_flow = goods * float(norm_demands[k_dst]) * np.exp(-decay * d)
            if inter_flow < 1e-9:
                continue
            _accumulate_path(pred[k_src], dst_node, src_node, inter_flow, edge_flow)

    if not edge_flow:
        return empty

    # Per-face flow: sum contributions from both endpoints of each edge
    face_flow = np.zeros(n, dtype=np.float64)
    for (a, b), flow in edge_flow.items():
        face_flow[a % n] += flow
        face_flow[b % n] += flow

    # Hub detection
    positive = face_flow[face_flow > 0.0]
    hub_threshold = float(np.quantile(positive, hub_flow_quantile)) if positive.size > 0 else np.inf
    hub_mask = face_flow >= hub_threshold

    # Route segments for display
    all_flows = np.array(list(edge_flow.values()), dtype=np.float64)
    if min_display_flow is None:
        min_display_flow = float(np.median(all_flows)) if all_flows.size > 0 else 0.0

    routes = []
    for (a, b), flow in edge_flow.items():
        if flow < min_display_flow:
            continue
        routes.append((centroids[a % n].copy(), centroids[b % n].copy(), float(flow)))

    return TradeResult(
        face_flow=face_flow.astype(np.float32),
        hub_mask=hub_mask,
        routes=routes,
    )
