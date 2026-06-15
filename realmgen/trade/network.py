"""Trade network — two-phase flow accumulation on the hybrid land/sea cost graph.

Phase 1: each resource deposit routes to its nearest settlement; settlement accumulates goods.
Phase 2: each settlement routes goods to all others, modulated by rarity and destination supply:

    demand_modifier[dst, g] = 1 − local_supply_fraction[dst, g]
    flow = Σ_g(goods[src,g] × rarity[g] × demand_modifier[dst,g])
           × settlement_demand[dst] × exp(−flow_decay × cost)

A settlement rich in silk has low silk demand but high spice demand, producing realistic
long-distance routes for rare goods. One undirected multi-source Dijkstra from all
settlements covers both phases. Land node i = i; sea node i = i + N.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

from realmgen.placement.types import Settlement


# ── Good-type namespace ───────────────────────────────────────────────────────

# Minerals occupy type IDs 0..N_MINERAL_TYPES-1.
# Rare bio occupy N_MINERAL_TYPES..N_MINERAL_TYPES+N_RARE_BIO_TYPES-1.
# These must match the offsets used in realmgen.builder when tagging good_sources.
N_MINERAL_TYPES = 9
RARE_BIO_TYPE_OFFSET = N_MINERAL_TYPES

# Per-type rarity multiplier.  Higher → more inter-settlement flow and longer routes.
# Values reflect historical long-distance trade importance, not physical abundance.
GOOD_RARITY_WEIGHTS: dict[int, float] = {
    0: 1.0,   # Iron        — utilitarian, widely distributed
    1: 1.5,   # Copper
    2: 3.0,   # Gold        — precious metal
    3: 4.0,   # Gems        — luxury
    4: 1.0,   # Coal        — utilitarian
    5: 1.5,   # Tin
    6: 3.5,   # Rare Metals — strategic
    7: 2.5,   # Salt        — historically essential across cultures
    8: 1.5,   # Marble
    9: 5.0,   # Spices      — the premier long-distance trade good historically
    10: 3.0,  # Tea
    11: 4.5,  # Silk        — luxury, source regions strongly bounded
    12: 3.5,  # Ivory
    13: 2.5,  # Furs
    14: 3.0,  # Incense
    15: 2.5,  # Dyes
    16: 2.0,  # Amber
    17: 2.0,  # Rubber
    18: 3.5,  # Pearls
}


@dataclass
class TradeResult:
    """Output of :func:`compute_trade_network`."""

    face_flow: np.ndarray   # (N,) float32 — aggregate trade flow across all good types
    hub_mask: np.ndarray    # (N,) bool — high-flow junction faces
    routes: list            # [(start_3d, end_3d, flow)] segments above display threshold
    # (S, S) float32 — Phase-2 flow from settlement i (exporter) to j (importer); index-aligned with settlements arg
    pair_flow: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))


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
                rows.append(i)
                cols.append(j)
                data.append(float(c))

    for i, (adj, costs) in enumerate(zip(face_adjacency, sea_edge_costs)):
        for j, c in zip(adj, costs):
            if np.isfinite(c):
                rows.append(n + i)
                cols.append(n + j)
                data.append(float(c))

    sc = max(float(river_switch_cost), 0.0)
    for raw_i in np.where(amphibious)[0]:
        fi = int(raw_i)
        rows.append(fi)
        cols.append(n + fi)
        data.append(sc)
        rows.append(n + fi)
        cols.append(fi)
        data.append(sc)

    pc = max(float(port_cost), 0.0)
    for i, adj in enumerate(face_adjacency):
        if not land_p[i] or sea_p[i]:
            continue
        for j in adj:
            if sea_p[j] and not land_p[j]:
                rows.append(i)
                cols.append(n + j)
                data.append(pc)
                rows.append(n + j)
                cols.append(i)
                data.append(pc)

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


def build_base_hybrid_graph(
    n: int,
    face_adjacency: List[List[int]],
    land_edge_costs: List[np.ndarray],
    land_passable: np.ndarray,
    sea_edge_costs: List[np.ndarray],
    sea_passable: np.ndarray,
    *,
    port_cost: float = 5.0,
    river_switch_cost: float = 0.0,
) -> csr_matrix:
    """Build the base ``(2N, 2N)`` hybrid land/sea cost graph (no virtual source).

    Land nodes ``0..N-1``, sea nodes ``N..2N-1``; port edges bridge coastal
    land↔adjacent-sea, amphibious switches at navigable-river faces.  This is the
    surface the polity stage projects sea authority over (Dijkstra from capital
    land nodes, read back the land-node slice).  Mirrors the graph
    :func:`compute_trade_network` builds internally.
    """
    size = 2 * n
    rows, cols, data = _collect_hybrid_edges(
        n, face_adjacency,
        land_edge_costs, np.asarray(land_passable, dtype=bool),
        sea_edge_costs, np.asarray(sea_passable, dtype=bool),
        port_cost, river_switch_cost,
    )
    if not data:
        return csr_matrix((size, size), dtype=np.float64)
    return csr_matrix(
        (
            np.array(data, dtype=np.float64),
            (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)),
        ),
        shape=(size, size),
    )


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
        rows.append(virtual)
        cols.append(int(s_node))
        data.append(virtual_cost)

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
    settlement_demands: np.ndarray | None = None,
    flow_decay: float = 0.05,
    hub_display_quantile: float = 0.90,
    min_display_flow: float | None = None,
    rarity_weights: dict | None = None,
) -> TradeResult:
    """Accumulate trade flow in two phases.

    **Phase 1** — each deposit routes to its nearest settlement (by hybrid cost).
    Flow on each path = ``abundance × rarity[g] × exp(−flow_decay × dist)``.
    Raw abundance is credited to that settlement per good type.

    **Phase 2** — each settlement with goods routes to every other settlement.
    Demand for good type *g* at the destination is modulated by how well the
    destination already supplies *g* locally::

        demand_modifier[dst, g] = 1 − deposit_goods[dst, g] / global_max[g]
        inter_flow = Σ_g(goods[src,g] × rarity[g] × demand_modifier[dst,g])
                     × settlement_demand[dst] × exp(−flow_decay × cost)

    This produces long-distance routes for rare goods that destinations lack,
    while keeping short-distance routes for common goods.  The directional
    ``inter_flow`` between every settlement pair is retained in
    :attr:`TradeResult.pair_flow` so downstream consumers can attribute a
    settlement's trade wealth to specific partners.

    Args:
        mesh: ``PolygonMesh`` — provides ``face_adjacency`` and ``face_centroids``.
        settlements: list of :class:`~realmgen.placement.types.Settlement`.
        good_sources: list of ``(face_id, abundance, good_type_id)`` tuples.
            ``face_id`` is a land-face index; ``abundance`` is a non-negative
            float; ``good_type_id`` is an integer key into ``rarity_weights``
            (use :data:`RARE_BIO_TYPE_OFFSET` to separate mineral and bio IDs).
        land_edge_costs: from ``build_land_edge_costs``, len N.
        land_passable: ``(N,)`` bool.
        sea_edge_costs: from ``build_sea_edge_costs``, len N.
        sea_passable: ``(N,)`` bool.
        port_cost: one-time loading/unloading cost at a coastal transition.
        river_switch_cost: cost to switch surfaces at a navigable river face.
        settlement_demands: optional ``(S,)`` per-settlement demand weights.
            When provided, overrides each settlement's ``demand`` field — used
            by the population relaxation loop to substitute population-
            proportional demands.
        flow_decay: exponential decay factor applied to both phase-1 and
            phase-2 routes with increasing cost distance.
        hub_display_quantile: percentile of positive face flows above which a
            face is flagged as a trade hub for display.
        min_display_flow: minimum edge flow to include in the route list for
            display; defaults to 0.0 (all positive-flow edges included — the
            renderer's width calculation handles visual thinning).
        rarity_weights: per-type rarity multipliers; defaults to
            :data:`GOOD_RARITY_WEIGHTS`.  Unknown type IDs get weight 1.0.

    Returns:
        :class:`TradeResult`, including ``pair_flow`` — the ``(S, S)`` directional
        inter-settlement trade matrix index-aligned with ``settlements``.
    """
    n = len(mesh.faces)
    s = len(settlements)
    empty = TradeResult(
        face_flow=np.zeros(n, dtype=np.float32),
        hub_mask=np.zeros(n, dtype=bool),
        routes=[],
        pair_flow=np.zeros((s, s), dtype=np.float32),
    )
    if not settlements or not good_sources:
        return empty

    weights = rarity_weights if rarity_weights is not None else GOOD_RARITY_WEIGHTS

    centroids = np.asarray(mesh.face_centroids, dtype=np.float64)

    # Normalised demand weights per settlement.
    # settlement_demands (population-proportional, from relaxation passes) takes
    # priority; without it, all settlements are treated as equally attractive.
    if settlement_demands is not None:
        raw_demands = np.asarray(settlement_demands, dtype=np.float64)
    else:
        raw_demands = np.ones(len(settlements), dtype=np.float64)
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

    # Determine good-type array size from sources and weights.
    max_type = max(int(gs[2]) for gs in good_sources)
    if weights:
        max_type = max(max_type, max(weights.keys()))
    n_types = max_type + 1

    # Rarity array indexed by type ID; unknown types default to 1.0.
    rarity_arr = np.array(
        [float(weights.get(g, 1.0)) for g in range(n_types)], dtype=np.float64
    )

    decay = float(flow_decay)
    edge_flow: dict[tuple[int, int], float] = {}
    # deposit_goods[k, g] = raw abundance of type g collected at settlement k.
    deposit_goods = np.zeros((len(settlements), n_types), dtype=np.float64)

    # Phase 1: route each deposit to its nearest settlement
    for face_id, abundance, good_type in good_sources:
        src_node = int(face_id)
        if src_node >= size:
            continue
        dists_to_src = dist[:, src_node]
        k = int(np.argmin(dists_to_src))
        d = float(dists_to_src[k])
        if not np.isfinite(d):
            continue
        g = int(good_type)
        deposit_goods[k, g] += float(abundance)
        phase1_flow = float(abundance) * float(rarity_arr[g]) * np.exp(-decay * d)
        if phase1_flow < 1e-9:
            continue
        _accumulate_path(pred[k], src_node, settlement_nodes[k], phase1_flow, edge_flow)

    # Phase 2: inter-settlement trade, modulated by local supply per good type.
    # demand_modifier[k, g] = 1 − how saturated settlement k already is with type g.
    global_max = deposit_goods.max(axis=0)  # (n_types,)
    global_max = np.where(global_max > 0.0, global_max, 1.0)
    local_frac = deposit_goods / global_max  # (S, n_types) in [0, 1]
    demand_modifier = 1.0 - local_frac      # (S, n_types): 0 = saturated, 1 = none locally

    # Rarity-weighted goods: how much value each settlement can export per type.
    weighted_goods = deposit_goods * rarity_arr  # (S, n_types)

    # Per-pair attribution: pair_flow[src, dst] = trade volume src exports to dst.
    pair_flow = np.zeros((len(settlements), len(settlements)), dtype=np.float64)

    for k_src in range(len(settlements)):
        if weighted_goods[k_src].sum() <= 0.0:
            continue
        src_node = settlement_nodes[k_src]
        for k_dst in range(len(settlements)):
            if k_src == k_dst:
                continue
            dst_node = settlement_nodes[k_dst]
            d = float(dist[k_src, dst_node])
            if not np.isfinite(d):
                continue
            # Sum over types: export value weighted by destination's unmet need.
            total_goods = float((weighted_goods[k_src] * demand_modifier[k_dst]).sum())
            if total_goods <= 0.0:
                continue
            inter_flow = total_goods * float(norm_demands[k_dst]) * np.exp(-decay * d)
            if inter_flow < 1e-9:
                continue
            pair_flow[k_src, k_dst] = inter_flow
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
    hub_threshold = float(np.quantile(positive, hub_display_quantile)) if positive.size > 0 else np.inf
    hub_mask = face_flow >= hub_threshold

    # Route segments for display.
    # min_display_flow=None includes all positive-flow edges; the renderer's
    # per-segment width calculation already provides visual filtering.
    threshold = float(min_display_flow) if min_display_flow is not None else 0.0

    routes = []
    for (a, b), flow in edge_flow.items():
        if flow < threshold:
            continue
        routes.append((centroids[a % n].copy(), centroids[b % n].copy(), float(flow)))

    return TradeResult(
        face_flow=face_flow.astype(np.float32),
        hub_mask=hub_mask,
        routes=routes,
        pair_flow=pair_flow.astype(np.float32),
    )
