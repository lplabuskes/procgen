"""Trade network — two-phase flow accumulation on the hybrid land/sea cost graph.

Phase 1: each resource deposit routes to its nearest settlement; settlement accumulates goods.
Phase 2: each settlement routes goods to all others, modulated by rarity and destination supply:

    demand_modifier[dst, g] = 1 − local_supply_fraction[dst, g]
    flow = Σ_g(goods[src,g] × rarity[g] × demand_modifier[dst,g])
           × settlement_demand[dst] × exp(−flow_decay × cost)

One undirected multi-source Dijkstra from all settlements covers both phases.
Land node i = i; sea node i = i + N.
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

# Minerals are type IDs 0..N_MINERAL_TYPES-1, rare bio follow.  Must match the
# offsets used when tagging good_sources.
N_MINERAL_TYPES = 9
RARE_BIO_TYPE_OFFSET = N_MINERAL_TYPES

# Per-type rarity multiplier: historical long-distance trade importance, not abundance.
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
    # (S, S) float32 — Phase-2 flow from settlement i (exporter) to j (importer)
    pair_flow: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    # (N,) float32 — land↔water flow at each non-port land face (fallback landings).
    landing_flow: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    # (D,) per good source: working settlement (-1 = unreachable), route cost, and
    # abundance credited after ``deposit_decay``.
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(rows, cols, data)`` arrays for the base (2N, 2N) hybrid graph (no virtual source).

    ``ports`` (``(N,)`` bool) limits land↔water transitions (port edges and river
    switches) to those faces; elsewhere they cost ``landing_cost`` (``inf`` = none).
    ``None`` allows them everywhere.  Edges come land, sea, river switches, then port
    edges, each by face then adjacency order.
    """
    land_p = np.asarray(land_passable, dtype=bool)
    sea_p = np.asarray(sea_passable, dtype=bool)
    amphibious = land_p & sea_p

    deg = np.fromiter((len(a) for a in face_adjacency), dtype=np.int64, count=len(face_adjacency))
    src = np.repeat(np.arange(len(face_adjacency), dtype=np.int64), deg)
    dst = np.fromiter((j for a in face_adjacency for j in a), dtype=np.int64, count=int(deg.sum()))

    def surface(costs: List[np.ndarray], offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = np.concatenate([np.asarray(x, dtype=np.float64) for x in costs]) if costs else np.zeros(0)
        ok = np.isfinite(c)
        return src[ok] + offset, dst[ok] + offset, c[ok]

    port = np.ones(n, dtype=bool) if ports is None else np.asarray(ports, dtype=bool)
    landing = max(float(landing_cost), 0.0)

    # Transitions i (land) ↔ n + j (sea), both ways: river switches at amphibious
    # faces, then ports from land-only faces onto sea-only neighbours.
    amph = np.flatnonzero(amphibious)
    coastal = land_p[src] & ~sea_p[src] & sea_p[dst] & ~land_p[dst]
    ti = np.concatenate([amph, src[coastal]])
    tj = np.concatenate([amph, dst[coastal]])
    base = np.concatenate([np.full(amph.size, max(float(river_switch_cost), 0.0)),
                           np.full(int(coastal.sum()), max(float(port_cost), 0.0))])
    tc = np.where(port[ti], base, landing)
    keep = np.isfinite(tc)
    ti, tj, tc = ti[keep], tj[keep], tc[keep]

    land, sea = surface(land_edge_costs, 0), surface(sea_edge_costs, n)
    rows = np.concatenate([land[0], sea[0], np.stack([ti, n + tj], axis=1).ravel()])
    cols = np.concatenate([land[1], sea[1], np.stack([n + tj, ti], axis=1).ravel()])
    data = np.concatenate([land[2], sea[2], np.repeat(tc, 2)])
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

    Edges are slots of the symmetric graph's CSR (row = lower node).  Returns
    ``(slot_flow, order, n_edges)``; ``order[:n_edges]`` lists slots in first-touch order.
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

    Land nodes ``0..N-1``, sea nodes ``N..2N-1``.  The polity stage projects sea
    authority over it.  Mirrors the graph :func:`compute_trade_network` builds.
    """
    size = 2 * n
    rows, cols, data = _collect_hybrid_edges(
        n, face_adjacency,
        land_edge_costs, np.asarray(land_passable, dtype=bool),
        sea_edge_costs, np.asarray(sea_passable, dtype=bool),
        port_cost, river_switch_cost,
    )
    if data.size == 0:
        return csr_matrix((size, size), dtype=np.float64)
    return csr_matrix((data, (rows.astype(np.int32), cols.astype(np.int32))), shape=(size, size))


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
    k = min(len(settlement_land_nodes), len(demands))
    rows = np.concatenate([rows, np.full(k, virtual, dtype=np.int64)])
    cols = np.concatenate([cols, np.asarray(settlement_land_nodes[:k], dtype=np.int64)])
    data = np.concatenate([data, dp * (1.0 - demands[:k])])

    shape = (virtual + 1, virtual + 1)
    if data.size == 0:
        return csr_matrix(shape, dtype=np.float64)
    return csr_matrix((data, (rows.astype(np.int32), cols.astype(np.int32))), shape=shape)


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

    **Phase 1** — each deposit routes to its nearest settlement (by hybrid cost),
    which is credited ``abundance × exp(−deposit_decay × dist)`` of good *g* (a far
    deposit means hauling supplies out and ore back); the haul on the path is that
    credit ``× rarity[g] × exp(−flow_decay × dist)``.

    **Phase 2** — each settlement with goods routes to every other settlement::

        demand_modifier[dst, g] = 1 − deposit_goods[dst, g] / global_max[g]
        inter_flow = Σ_g(goods[src,g] × rarity[g] × demand_modifier[dst,g])
                     × settlement_demand[dst] × exp(−flow_decay × cost)

    The directional ``inter_flow`` is kept in :attr:`TradeResult.pair_flow`.

    Args:
        mesh: ``PolygonMesh`` — provides ``face_adjacency`` and ``face_centroids``.
        good_sources: ``(face_id, abundance, good_type_id)`` tuples; ``good_type_id``
            keys ``rarity_weights`` (bio IDs start at :data:`RARE_BIO_TYPE_OFFSET`).
        land_edge_costs / sea_edge_costs: from ``build_land_edge_costs`` /
            ``build_sea_edge_costs``, len N.
        settlement_ports: goods cross land↔water only at a settlement's face
            (coastal port or river port; rivers are water to their mouths, so an
            inland river city ships straight out to sea).  ``False`` allows the
            transition at every coastal and river face.
        landing_cost: with ``settlement_ports``, the cost of a fallback landing
            elsewhere (``inf`` = none); the relaxation loop uses it to find where
            trade wants a port that does not exist yet.
        settlement_demands: optional ``(S,)`` demand weights overriding equal demand
            (the relaxation loop passes population-proportional ones).
        deposit_decay: decay of a deposit's credited yield with cost distance
            (0 = full yield at any distance).
        hub_display_quantile: quantile of positive face flows above which a face
            is a display hub.
        min_display_flow: minimum edge flow for the display route list (default 0).
        rarity_weights: defaults to :data:`GOOD_RARITY_WEIGHTS`; unknown IDs get 1.0.

    Returns:
        :class:`TradeResult`; ``pair_flow`` is ``(S, S)``, index-aligned with ``settlements``.
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

    if settlement_demands is not None:
        raw_demands = np.asarray(settlement_demands, dtype=np.float64)
    else:
        raw_demands = np.ones(len(settlements), dtype=np.float64)
    max_d = raw_demands.max()
    norm_demands = raw_demands / max_d if max_d > 0 else raw_demands.copy()
    settlement_nodes = [s.face_id for s in settlements]

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
    if data.size:
        graph = csr_matrix((data, (rows.astype(np.int32), cols.astype(np.int32))), shape=(size, size))
    else:
        graph = csr_matrix((size, size), dtype=np.float64)

    # dist/pred[k, node]: shortest distance / predecessor from settlement k.
    dist, pred = sp_dijkstra(
        graph, directed=False, indices=settlement_nodes, return_predecessors=True,
    )

    max_type = max(int(gs[2]) for gs in good_sources)
    if weights:
        max_type = max(max_type, max(weights.keys()))
    n_types = max_type + 1

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

    # Phase 1
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

    # Phase 2
    global_max = deposit_goods.max(axis=0)  # (n_types,)
    global_max = np.where(global_max > 0.0, global_max, 1.0)
    local_frac = deposit_goods / global_max  # (S, n_types) in [0, 1]
    demand_modifier = 1.0 - local_frac      # (S, n_types): 0 = saturated, 1 = none locally

    weighted_goods = deposit_goods * rarity_arr  # (S, n_types)

    # pair_flow[src, dst] = trade volume src exports to dst.
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

    # Symmetrise from the sparsity pattern: zero-cost edges are explicit zeros a sum would drop.
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

    face_flow, landing_flow = _sum_edge_ends(
        edge_a, edge_b, edge_flow, n, ports if ports is not None else np.zeros(0, dtype=bool))

    positive = face_flow[face_flow > 0.0]
    hub_threshold = float(np.quantile(positive, hub_display_quantile)) if positive.size > 0 else np.inf
    hub_mask = face_flow >= hub_threshold

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
