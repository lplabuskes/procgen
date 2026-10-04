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

import dataclasses
from dataclasses import dataclass, field
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
    # (N,) float32 — flow crossing between land and water at each land face that
    # is not a port (the fallback landings); all zero without landings.
    landing_flow: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    # (D,) per good source (aligned with the ``good_sources`` arg): the settlement
    # that works it (-1 = none can reach it), the route cost there, and the
    # abundance credited to that settlement after ``deposit_decay``.
    deposit_owner: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    deposit_cost: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    deposit_credit: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    # (S,) flow of each settlement's own goods at its face: deposit hauls arriving
    # plus everything it exports (the rest of its face flow is imports and transit).
    goods_flow: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))


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
    ports: np.ndarray | None = None,
    landing_cost: float = np.inf,
) -> tuple[list, list, list]:
    """Return (rows, cols, data) for all hybrid graph edges.

    Builds the base (2N, 2N) graph — land surface, sea surface, amphibious
    river switches, and port transitions.  Virtual-source edges are NOT included.

    ``ports`` (``(N,)`` bool) limits the land↔water transitions — port edges and
    river switches — to those faces at their usual cost; every other coastal or
    river face gets them at ``landing_cost`` instead, or not at all when it is
    infinite.  ``None`` allows them everywhere.
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

    port = np.ones(n, dtype=bool) if ports is None else np.asarray(ports, dtype=bool)
    landing = max(float(landing_cost), 0.0)

    def transition(i: int, j: int, cost: float) -> None:
        c = cost if port[i] else landing
        if not np.isfinite(c):
            return
        rows.append(i)
        cols.append(n + j)
        data.append(c)
        rows.append(n + j)
        cols.append(i)
        data.append(c)

    sc = max(float(river_switch_cost), 0.0)
    for raw_i in np.where(amphibious)[0]:
        transition(int(raw_i), int(raw_i), sc)

    pc = max(float(port_cost), 0.0)
    for i, adj in enumerate(face_adjacency):
        if not land_p[i] or sea_p[i]:
            continue
        for j in adj:
            if sea_p[j] and not land_p[j]:
                transition(i, j, pc)

    return rows, cols, data


@njit(cache=True)
def _accumulate_paths(
    pred: np.ndarray,
    job_row: np.ndarray,
    job_from: np.ndarray,
    job_to: np.ndarray,
    job_flow: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    max_steps: int = 100_000,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Walk each job's predecessor row from ``job_from`` back to ``job_to``,
    adding its flow to every undirected edge on the way.

    Edges are slots of the symmetric graph's CSR (``indptr``/``indices``, the
    row being the lower node).  Returns ``(slot_flow, order, n_edges)``:
    ``order[:n_edges]`` lists the slots in the order they were first touched.
    Jobs run in order, so each edge sums its flows in the order they arrive.
    """
    slot_flow = np.zeros(len(indices), dtype=np.float64)
    seen = np.zeros(len(indices), dtype=np.bool_)
    order = np.empty(len(indices), dtype=np.int64)
    n_edges = 0
    for j in range(len(job_row)):
        row = pred[job_row[j]]
        flow = job_flow[j]
        cur = job_from[j]
        for _ in range(max_steps):
            p = row[cur]
            if p < 0:
                break
            a, b = (p, cur) if p < cur else (cur, p)
            slot = -1
            for q in range(indptr[a], indptr[a + 1]):
                if indices[q] == b:
                    slot = q
                    break
            if not seen[slot]:
                seen[slot] = True
                order[n_edges] = slot
                n_edges += 1
            slot_flow[slot] += flow
            cur = p
            if cur == job_to[j]:
                break
    return slot_flow, order, n_edges


@njit(cache=True)
def _sum_edge_ends(
    edge_a: np.ndarray, edge_b: np.ndarray, flow: np.ndarray, n: int, ports: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``(face_flow, landing_flow)`` from per-edge flows, summed in edge order."""
    face_flow = np.zeros(n, dtype=np.float64)
    landing_flow = np.zeros(n, dtype=np.float64)
    for e in range(len(edge_a)):
        a, b = edge_a[e], edge_b[e]
        face_flow[a % n] += flow[e]
        face_flow[b % n] += flow[e]
        if len(ports) and a < n <= b and not ports[a]:
            landing_flow[a] += flow[e]
    return face_flow, landing_flow


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
    settlement_ports: bool = True,
    landing_cost: float = np.inf,
    settlement_demands: np.ndarray | None = None,
    flow_decay: float = 0.05,
    deposit_decay: float = 0.0,
    hub_display_quantile: float = 0.90,
    min_display_flow: float | None = None,
    rarity_weights: dict | None = None,
) -> TradeResult:
    """Accumulate trade flow in two phases.

    **Phase 1** — each deposit routes to its nearest settlement (by hybrid cost).
    That settlement is credited ``abundance × exp(−deposit_decay × dist)`` of
    good *g* — a far deposit yields little, since working it means hauling
    supplies out and ore back — and the haul on the path is that credit
    ``× rarity[g] × exp(−flow_decay × dist)``.

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
        settlement_ports: goods move between land and water only at a
            settlement's face — a coastal settlement is a port, one on a
            navigable river a river port.  Rivers are water all the way to their
            mouths, so an inland river city ships straight out to sea.
            ``False`` allows the transition at every coastal and river face.
        landing_cost: with ``settlement_ports``, the cost of a fallback landing
            anywhere else (``inf`` = none).  The relaxation loop uses it to find
            where trade wants a port that does not exist yet.
        settlement_demands: optional ``(S,)`` per-settlement demand weights.
            When provided, overrides each settlement's ``demand`` field — used
            by the population relaxation loop to substitute population-
            proportional demands.
        flow_decay: exponential decay factor applied to both phase-1 and
            phase-2 routes with increasing cost distance.
        deposit_decay: exponential decay of a deposit's credited yield with its
            cost distance from the settlement that works it (0 = full yield at
            any distance).
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
        landing_flow=np.zeros(n, dtype=np.float32),
        deposit_owner=np.full(len(good_sources), -1, dtype=np.int32),
        deposit_cost=np.full(len(good_sources), np.inf),
        deposit_credit=np.zeros(len(good_sources)),
        goods_flow=np.zeros(s),
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
    ports = None
    if settlement_ports:
        ports = np.zeros(n, dtype=bool)
        ports[settlement_nodes] = True
    rows, cols, data = _collect_hybrid_edges(
        n, mesh.face_adjacency,
        land_edge_costs, np.asarray(land_passable, dtype=bool),
        sea_edge_costs, np.asarray(sea_passable, dtype=bool),
        port_cost, river_switch_cost, ports, landing_cost,
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
    # Path walks to run, in order: (pred row, from node, to node, flow).
    jobs_row: list = []
    jobs_from: list = []
    jobs_to: list = []
    jobs_flow: list = []
    # deposit_goods[k, g] = credited abundance of type g collected at settlement k.
    deposit_goods = np.zeros((len(settlements), n_types), dtype=np.float64)

    # Phase 1: route each deposit to its nearest settlement
    n_src = len(good_sources)
    deposit_owner = np.full(n_src, -1, dtype=np.int32)
    deposit_cost = np.full(n_src, np.inf)
    deposit_credit = np.zeros(n_src)
    goods_flow = np.zeros(len(settlements))
    for i, (face_id, abundance, good_type) in enumerate(good_sources):
        src_node = int(face_id)
        if src_node >= size:
            continue
        dists_to_src = dist[:, src_node]
        k = int(np.argmin(dists_to_src))
        d = float(dists_to_src[k])
        if not np.isfinite(d):
            continue
        g = int(good_type)
        credit = float(abundance) * np.exp(-float(deposit_decay) * d)
        deposit_owner[i], deposit_cost[i], deposit_credit[i] = k, d, credit
        deposit_goods[k, g] += credit
        phase1_flow = credit * float(rarity_arr[g]) * np.exp(-decay * d)
        if phase1_flow < 1e-9:
            continue
        goods_flow[k] += phase1_flow
        jobs_row.append(k)
        jobs_from.append(src_node)
        jobs_to.append(settlement_nodes[k])
        jobs_flow.append(phase1_flow)

    # Phase 2: inter-settlement trade, modulated by local supply per good type.
    # demand_modifier[k, g] = 1 − how saturated settlement k already is with type g.
    global_max = deposit_goods.max(axis=0)  # (n_types,)
    global_max = np.where(global_max > 0.0, global_max, 1.0)
    local_frac = deposit_goods / global_max  # (S, n_types) in [0, 1]
    demand_modifier = 1.0 - local_frac      # (S, n_types): 0 = saturated, 1 = none locally

    # Rarity-weighted goods: how much value each settlement can export per type.
    weighted_goods = deposit_goods * rarity_arr  # (S, n_types)

    # Per-pair attribution: pair_flow[src, dst] = trade volume src exports to dst.
    # Sum over types: export value weighted by destination's unmet need.
    nodes = np.asarray(settlement_nodes, dtype=np.int64)
    pair_dist = dist[:, nodes]                                                   # (S, S)
    total_goods = (weighted_goods[:, None, :] * demand_modifier[None, :, :]).sum(axis=2)
    with np.errstate(invalid="ignore"):
        pair_flow = total_goods * norm_demands[None, :] * np.exp(-decay * pair_dist)
    trades = (
        (weighted_goods.sum(axis=1) > 0.0)[:, None]
        & ~np.eye(len(settlements), dtype=bool)
        & np.isfinite(pair_dist)
        & (total_goods > 0.0)
        & (pair_flow >= 1e-9)
    )
    pair_flow = np.where(trades, pair_flow, 0.0)
    k_src, k_dst = np.nonzero(trades)                                            # row-major: src, then dst
    np.add.at(goods_flow, k_src, pair_flow[k_src, k_dst])
    jobs_row.extend(k_src.tolist())
    jobs_from.extend(nodes[k_dst].tolist())
    jobs_to.extend(nodes[k_src].tolist())
    jobs_flow.extend(pair_flow[k_src, k_dst].tolist())

    if not jobs_row:
        return dataclasses.replace(
            empty, deposit_owner=deposit_owner, deposit_cost=deposit_cost, deposit_credit=deposit_credit)

    # Walk every path over the graph's undirected edges (the lower node's CSR row).
    # (From the pattern: zero-cost edges are explicit zeros a sum would drop.)
    pattern = csr_matrix((np.ones(graph.nnz), graph.indices, graph.indptr), shape=graph.shape)
    sym = (pattern + pattern.T).tocsr()
    sym.sort_indices()
    pred_rows = np.ascontiguousarray(pred, dtype=np.int64)
    slot_flow, order, n_edges = _accumulate_paths(
        pred_rows,
        np.asarray(jobs_row, dtype=np.int64), np.asarray(jobs_from, dtype=np.int64),
        np.asarray(jobs_to, dtype=np.int64), np.asarray(jobs_flow, dtype=np.float64),
        sym.indptr.astype(np.int64), sym.indices.astype(np.int64),
    )
    if n_edges == 0:
        return dataclasses.replace(
            empty, deposit_owner=deposit_owner, deposit_cost=deposit_cost, deposit_credit=deposit_credit)
    slots = order[:n_edges]
    edge_a = np.repeat(np.arange(size, dtype=np.int64), np.diff(sym.indptr))[slots]
    edge_b = sym.indices.astype(np.int64)[slots]
    edge_flow = slot_flow[slots]

    # Per-face flow: sum contributions from both endpoints of each edge, and the
    # flow over the fallback landings (land node ↔ water node at a non-port).
    face_flow, landing_flow = _sum_edge_ends(
        edge_a, edge_b, edge_flow, n, ports if ports is not None else np.zeros(0, dtype=bool))

    # Hub detection
    positive = face_flow[face_flow > 0.0]
    hub_threshold = float(np.quantile(positive, hub_display_quantile)) if positive.size > 0 else np.inf
    hub_mask = face_flow >= hub_threshold

    # Route segments for display.
    # min_display_flow=None includes all positive-flow edges; the renderer's
    # per-segment width calculation already provides visual filtering.
    threshold = float(min_display_flow) if min_display_flow is not None else 0.0

    routes = []
    for a, b, flow in zip(edge_a.tolist(), edge_b.tolist(), edge_flow.tolist()):
        if flow < threshold:
            continue
        routes.append((centroids[a % n].copy(), centroids[b % n].copy(), float(flow)))

    return TradeResult(
        face_flow=face_flow.astype(np.float32),
        hub_mask=hub_mask,
        routes=routes,
        pair_flow=pair_flow.astype(np.float32),
        landing_flow=landing_flow.astype(np.float32),
        deposit_owner=deposit_owner,
        deposit_cost=deposit_cost,
        deposit_credit=deposit_credit,
        goods_flow=goods_flow,
    )
