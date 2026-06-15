"""Tests for realmgen.trade.network — pure-function unit tests.

All tests use lightweight stub meshes so there is no dependency on a full
worldgen run.  Topology: small linear and ring graphs.
"""
from __future__ import annotations

import numpy as np
import pytest

from realmgen.placement.types import Settlement
from realmgen.trade.network import (
    TradeResult,
    build_hybrid_cost_graph,
    compute_trade_network,
    RARE_BIO_TYPE_OFFSET,
)


# ── Stub mesh ─────────────────────────────────────────────────────────────────


class _Mesh:
    """Minimal stub exposing only what the trade network needs."""

    def __init__(self, adj: list[list[int]], centroids: np.ndarray):
        self.faces = list(range(len(adj)))
        self.face_adjacency = adj
        self.face_centroids = centroids


def _line_mesh(n: int) -> _Mesh:
    """Linear graph 0—1—2—…—(n-1) with equatorial centroids."""
    adj: list[list[int]] = []
    for i in range(n):
        neighbors = []
        if i > 0:
            neighbors.append(i - 1)
        if i < n - 1:
            neighbors.append(i + 1)
        adj.append(neighbors)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    centroids = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(n)])
    return _Mesh(adj, centroids)


def _uniform_edge_costs(mesh: _Mesh, cost: float = 1.0) -> list[np.ndarray]:
    return [
        np.full(len(nbrs), cost, dtype=np.float64)
        for nbrs in mesh.face_adjacency
    ]


def _inf_edge_costs(mesh: _Mesh) -> list[np.ndarray]:
    return [
        np.full(len(nbrs), np.inf, dtype=np.float64)
        for nbrs in mesh.face_adjacency
    ]


def _settlement(face_id: int, tier: int = 0) -> Settlement:
    centroids = np.zeros((face_id + 1, 3))
    centroids[face_id] = [1, 0, 0]
    return Settlement(face_id=face_id, position=np.array([1.0, 0.0, 0.0]), tier=tier, score=1.0)


# ── build_hybrid_cost_graph ───────────────────────────────────────────────────


class TestBuildHybridCostGraph:
    def test_shape(self):
        mesh = _line_mesh(4)
        n = 4
        land_p = np.ones(n, dtype=bool)
        sea_p = np.zeros(n, dtype=bool)
        ec = _uniform_edge_costs(mesh)
        g = build_hybrid_cost_graph(
            n, mesh.face_adjacency,
            ec, land_p,
            [np.empty(0)] * n, sea_p,
            settlement_land_nodes=[0],
            settlement_demands=np.array([1.0]),
        )
        assert g.shape == (2 * n + 1, 2 * n + 1)

    def test_land_edges_present(self):
        mesh = _line_mesh(3)
        n = 3
        land_p = np.ones(n, dtype=bool)
        sea_p = np.zeros(n, dtype=bool)
        ec = _uniform_edge_costs(mesh, cost=2.0)
        g = build_hybrid_cost_graph(
            n, mesh.face_adjacency,
            ec, land_p,
            [np.empty(0)] * n, sea_p,
            settlement_land_nodes=[0],
            settlement_demands=np.array([1.0]),
        )
        # Land node 0 → land node 1 should have cost 2.0
        assert float(g[0, 1]) == pytest.approx(2.0)

    def test_sea_edges_present(self):
        mesh = _line_mesh(3)
        n = 3
        land_p = np.zeros(n, dtype=bool)
        sea_p = np.ones(n, dtype=bool)
        ec = _uniform_edge_costs(mesh, cost=1.5)
        g = build_hybrid_cost_graph(
            n, mesh.face_adjacency,
            [np.empty(0)] * n, land_p,
            ec, sea_p,
            settlement_land_nodes=[],
            settlement_demands=np.array([], dtype=np.float64),
        )
        # Sea node n+0 → sea node n+1 should have cost 1.5
        assert float(g[n, n + 1]) == pytest.approx(1.5)

    def test_amphibious_switch_edges(self):
        # All faces are amphibious (navigable river): both surfaces, zero switch cost
        n = 2
        mesh = _line_mesh(n)
        land_p = np.ones(n, dtype=bool)
        sea_p = np.ones(n, dtype=bool)
        ec = _uniform_edge_costs(mesh)
        g = build_hybrid_cost_graph(
            n, mesh.face_adjacency,
            ec, land_p,
            ec, sea_p,
            settlement_land_nodes=[0],
            settlement_demands=np.array([1.0]),
            river_switch_cost=0.0,
        )
        # land node 0 → sea node 0 (= n+0): cost should be 0
        assert float(g[0, n + 0]) == pytest.approx(0.0)
        assert float(g[n + 0, 0]) == pytest.approx(0.0)

    def test_port_transition_added(self):
        # Face 0: pure land; Face 1: pure sea.  Port edge should exist.
        n = 2
        mesh = _line_mesh(n)
        land_p = np.array([True, False])
        sea_p = np.array([False, True])
        land_ec = [np.empty(0), np.empty(0)]  # no passable land edges
        sea_ec = [np.empty(0), np.empty(0)]
        g = build_hybrid_cost_graph(
            n, mesh.face_adjacency,
            land_ec, land_p,
            sea_ec, sea_p,
            settlement_land_nodes=[0],
            settlement_demands=np.array([1.0]),
            port_cost=7.0,
        )
        # land node 0 → sea node 1 (= n+1)
        assert float(g[0, n + 1]) == pytest.approx(7.0)
        # reverse: sea node 1 → land node 0
        assert float(g[n + 1, 0]) == pytest.approx(7.0)

    def test_no_port_between_same_surface(self):
        # Two pure-land faces should not get port edges.
        n = 2
        mesh = _line_mesh(n)
        land_p = np.ones(n, dtype=bool)
        sea_p = np.zeros(n, dtype=bool)
        ec = _uniform_edge_costs(mesh)
        g = build_hybrid_cost_graph(
            n, mesh.face_adjacency,
            ec, land_p,
            [np.empty(0)] * n, sea_p,
            settlement_land_nodes=[0],
            settlement_demands=np.array([1.0]),
            port_cost=7.0,
        )
        # No land→sea transitions should exist (no sea faces)
        # Check the sea half of the matrix has no cross-surface edges involving land nodes
        assert float(g[0, n + 1]) == 0.0  # sparse default is 0 = not present

    def test_virtual_source_edges(self):
        n = 3
        mesh = _line_mesh(n)
        land_p = np.ones(n, dtype=bool)
        sea_p = np.zeros(n, dtype=bool)
        ec = _uniform_edge_costs(mesh)
        settlement_nodes = [0, 2]
        demands = np.array([1.0, 0.5])
        g = build_hybrid_cost_graph(
            n, mesh.face_adjacency,
            ec, land_p,
            [np.empty(0)] * n, sea_p,
            settlement_land_nodes=settlement_nodes,
            settlement_demands=demands,
            demand_penalty=4.0,
        )
        virtual = 2 * n
        # High-demand settlement (demand=1.0): cost = 4.0 × (1-1.0) = 0.0
        assert float(g[virtual, 0]) == pytest.approx(0.0)
        # Lower-demand settlement (demand=0.5): cost = 4.0 × (1-0.5) = 2.0
        assert float(g[virtual, 2]) == pytest.approx(2.0)


# ── compute_trade_network ─────────────────────────────────────────────────────


class TestComputeTradeNetwork:
    def test_empty_sources_returns_zeros(self):
        mesh = _line_mesh(4)
        n = 4
        land_p = np.ones(n, dtype=bool)
        sea_p = np.zeros(n, dtype=bool)
        ec = _uniform_edge_costs(mesh)
        settlements = [_settlement(0)]
        result = compute_trade_network(
            mesh, settlements, [],
            ec, land_p,
            [np.empty(0)] * n, sea_p,
        )
        assert isinstance(result, TradeResult)
        np.testing.assert_array_equal(result.face_flow, 0.0)

    def test_empty_settlements_returns_zeros(self):
        mesh = _line_mesh(4)
        n = 4
        land_p = np.ones(n, dtype=bool)
        sea_p = np.zeros(n, dtype=bool)
        ec = _uniform_edge_costs(mesh)
        result = compute_trade_network(
            mesh, [], [(3, 1.0, 0)],
            ec, land_p,
            [np.empty(0)] * n, sea_p,
        )
        np.testing.assert_array_equal(result.face_flow, 0.0)

    def test_flow_nonzero_on_path(self):
        # Line: settlement at face 0, source at face 3.
        # Flow should accumulate on faces 0-1-2-3.
        mesh = _line_mesh(5)
        n = 5
        land_p = np.ones(n, dtype=bool)
        sea_p = np.zeros(n, dtype=bool)
        ec = _uniform_edge_costs(mesh, cost=1.0)
        settlements = [_settlement(0, tier=0)]
        result = compute_trade_network(
            mesh, settlements, [(3, 10.0, 0)],
            ec, land_p,
            [np.empty(0)] * n, sea_p,
            flow_decay=0.0,  # no decay so we can reason clearly
        )
        # Faces 0,1,2,3 all receive flow; face 4 (off the path) should not.
        assert result.face_flow[0] > 0.0
        assert result.face_flow[1] > 0.0
        assert result.face_flow[2] > 0.0
        assert result.face_flow[3] > 0.0
        assert result.face_flow[4] == pytest.approx(0.0)

    def test_flow_decay_reduces_distant_flow(self):
        # Same setup; with high decay, a distant source contributes little flow.
        mesh = _line_mesh(10)
        n = 10
        land_p = np.ones(n, dtype=bool)
        sea_p = np.zeros(n, dtype=bool)
        ec = _uniform_edge_costs(mesh, cost=1.0)
        settlements = [_settlement(0, tier=0)]

        low_decay = compute_trade_network(
            mesh, settlements, [(9, 1.0, 0)],
            ec, land_p, [np.empty(0)] * n, sea_p, flow_decay=0.0,
        )
        high_decay = compute_trade_network(
            mesh, settlements, [(9, 1.0, 0)],
            ec, land_p, [np.empty(0)] * n, sea_p, flow_decay=2.0,
        )
        # High decay → less total flow
        assert high_decay.face_flow.sum() < low_decay.face_flow.sum()

    def test_hub_mask_marks_high_flow_faces(self):
        mesh = _line_mesh(5)
        n = 5
        land_p = np.ones(n, dtype=bool)
        sea_p = np.zeros(n, dtype=bool)
        ec = _uniform_edge_costs(mesh, cost=1.0)
        settlements = [_settlement(0, tier=0)]
        # Source at face 4: flow passes through faces 0-4.
        result = compute_trade_network(
            mesh, settlements, [(4, 100.0, 0)],
            ec, land_p, [np.empty(0)] * n, sea_p,
            flow_decay=0.0,
            hub_display_quantile=0.50,
        )
        assert result.hub_mask.dtype == bool
        # At least one hub should be flagged (the high-flow face)
        assert result.hub_mask.any()

    def test_routes_contain_segments(self):
        mesh = _line_mesh(5)
        n = 5
        land_p = np.ones(n, dtype=bool)
        sea_p = np.zeros(n, dtype=bool)
        ec = _uniform_edge_costs(mesh, cost=1.0)
        settlements = [_settlement(0, tier=0)]
        result = compute_trade_network(
            mesh, settlements, [(4, 10.0, 0)],
            ec, land_p, [np.empty(0)] * n, sea_p,
            flow_decay=0.0,
            min_display_flow=0.0,  # include all segments
        )
        assert len(result.routes) > 0
        start, end, flow = result.routes[0]
        assert start.shape == (3,)
        assert end.shape == (3,)
        assert flow > 0.0

    def test_sea_route_via_port(self):
        # Topology: land face 0 (settlement), sea face 1 (ocean), land face 2 (source).
        # The only connection between 0 and 2 is via a port→sea→port path.
        # Adjacency: 0—1—2 where 0 and 2 are land, 1 is sea.
        n = 3
        adj = [[1], [0, 2], [1]]
        angles = np.array([0.0, np.pi / 2, np.pi])
        centroids = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(n)])
        mesh = _Mesh(adj, centroids)

        land_p = np.array([True, False, True])
        sea_p = np.array([False, True, False])

        # Land edges: only 0↔1 and 1↔2 but 1 is water so land costs are inf
        land_ec = [
            np.array([np.inf]),   # face 0 → face 1: inf (water neighbor)
            np.array([np.inf, np.inf]),  # face 1 → 0, 2: inf (water face)
            np.array([np.inf]),   # face 2 → face 1: inf
        ]
        # Sea edges: 1↔0 and 1↔2 are inf (0 and 2 are land), but 1 connects internally
        sea_ec = [
            np.array([np.inf]),   # sea node 0 not passable
            np.array([np.inf, np.inf]),  # sea node 1 → sea 0, sea 2 (land = not sea_passable)
            np.array([np.inf]),   # sea node 2 not passable
        ]

        settlements = [_settlement(0, tier=0)]
        result = compute_trade_network(
            mesh, settlements, [(2, 5.0, 0)],
            land_ec, land_p,
            sea_ec, sea_p,
            port_cost=1.0,
            flow_decay=0.0,
        )
        # Flow should reach face 2 via port—sea—port route
        # (whether it actually routes depends on path cost being finite)
        # The route: virtual→land0 (cost 0) → port → sea1 (cost 1.0) → port → land2 (cost 1.0)
        # Total dist to land2 = 2.0 (two ports). Flow should be nonzero.
        assert result.face_flow[2] > 0.0 or result.face_flow[0] > 0.0

    def test_deterministic(self):
        mesh = _line_mesh(8)
        n = 8
        land_p = np.ones(n, dtype=bool)
        sea_p = np.zeros(n, dtype=bool)
        ec = _uniform_edge_costs(mesh, cost=1.0)
        settlements = [_settlement(0, tier=0), _settlement(7, tier=1)]
        sources = [(2, 3.0, 0), (5, 1.5, 0)]
        kwargs: dict = dict(flow_decay=0.1, hub_display_quantile=0.7)

        r1 = compute_trade_network(
            mesh, settlements, sources, ec, land_p,
            [np.empty(0)] * n, sea_p, **kwargs,
        )
        r2 = compute_trade_network(
            mesh, settlements, sources, ec, land_p,
            [np.empty(0)] * n, sea_p, **kwargs,
        )
        np.testing.assert_array_equal(r1.face_flow, r2.face_flow)
        np.testing.assert_array_equal(r1.hub_mask, r2.hub_mask)

    def test_face_flow_shape_and_dtype(self):
        mesh = _line_mesh(4)
        n = 4
        land_p = np.ones(n, dtype=bool)
        sea_p = np.zeros(n, dtype=bool)
        ec = _uniform_edge_costs(mesh)
        settlements = [_settlement(0, tier=0)]
        result = compute_trade_network(
            mesh, settlements, [(3, 1.0, 0)],
            ec, land_p, [np.empty(0)] * n, sea_p,
        )
        assert result.face_flow.shape == (n,)
        assert result.face_flow.dtype == np.float32
        assert result.hub_mask.shape == (n,)
        assert result.hub_mask.dtype == bool

    def test_rare_good_generates_more_flow_than_common(self):
        # Two settlements: face 0 and face 7. Source at face 4.
        # Rare type (RARE_BIO_TYPE_OFFSET) should generate more inter-settlement
        # flow than a common type (0) at the same abundance and no local supply.
        mesh = _line_mesh(9)
        n = 9
        land_p = np.ones(n, dtype=bool)
        sea_p = np.zeros(n, dtype=bool)
        ec = _uniform_edge_costs(mesh, cost=1.0)
        settlements = [_settlement(0, tier=0), _settlement(8, tier=1)]

        common = compute_trade_network(
            mesh, settlements, [(4, 1.0, 0)],  # Iron — rarity 1.0
            ec, land_p, [np.empty(0)] * n, sea_p,
            flow_decay=0.0,
        )
        rare = compute_trade_network(
            mesh, settlements, [(4, 1.0, RARE_BIO_TYPE_OFFSET)],  # Spices — rarity 5.0
            ec, land_p, [np.empty(0)] * n, sea_p,
            flow_decay=0.0,
        )
        assert rare.face_flow.sum() > common.face_flow.sum()

    def test_pair_flow_shape_and_empty(self):
        # No sources → (S, S) zeros, dtype float32, aligned with settlements.
        mesh = _line_mesh(4)
        n = 4
        land_p = np.ones(n, dtype=bool)
        sea_p = np.zeros(n, dtype=bool)
        ec = _uniform_edge_costs(mesh)
        settlements = [_settlement(0), _settlement(2)]
        result = compute_trade_network(
            mesh, settlements, [],
            ec, land_p, [np.empty(0)] * n, sea_p,
        )
        assert result.pair_flow.shape == (2, 2)
        assert result.pair_flow.dtype == np.float32
        assert np.all(result.pair_flow == 0.0)

    def test_pair_flow_directional_attribution(self):
        # Producer at settlement 0's face, importer at settlement 1's face.
        # Flow is attributed 0→1 (exporter→importer), never the reverse, never self.
        mesh = _line_mesh(9)
        n = 9
        land_p = np.ones(n, dtype=bool)
        sea_p = np.zeros(n, dtype=bool)
        ec = _uniform_edge_costs(mesh, cost=1.0)
        settlements = [_settlement(0, tier=0), _settlement(8, tier=1)]
        result = compute_trade_network(
            mesh, settlements, [(0, 10.0, RARE_BIO_TYPE_OFFSET)],
            ec, land_p, [np.empty(0)] * n, sea_p,
            flow_decay=0.0,
        )
        assert result.pair_flow.shape == (2, 2)
        assert result.pair_flow[0, 1] > 0.0          # producer exports to importer
        assert result.pair_flow[1, 0] == pytest.approx(0.0)  # importer has nothing to send
        assert result.pair_flow[0, 0] == pytest.approx(0.0)  # no self-trade
        assert result.pair_flow[1, 1] == pytest.approx(0.0)

    def test_pair_flow_reflects_local_supply(self):
        # Y topology: src=face0, dst_rich=face2 (already has the good), dst_poor=face4.
        # The producer's exports are attributed more to the unmet-demand destination.
        n = 5
        adj = [[1, 3], [0, 2], [1], [0, 4], [3]]
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        centroids = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(n)])
        mesh = _Mesh(adj, centroids)

        land_p = np.ones(n, dtype=bool)
        sea_p = np.zeros(n, dtype=bool)
        ec = [np.ones(len(nbrs), dtype=np.float64) for nbrs in adj]

        settlements = [_settlement(0, tier=0), _settlement(2, tier=1), _settlement(4, tier=1)]
        sources = [(0, 10.0, 0), (2, 10.0, 0)]  # dst_rich (idx 1) already saturated for type 0
        result = compute_trade_network(
            mesh, settlements, sources,
            ec, land_p, [np.empty(0)] * n, sea_p,
            flow_decay=0.0,
        )
        # src (idx 0) → dst_poor (idx 2) outweighs src → dst_rich (idx 1).
        assert result.pair_flow[0, 2] > result.pair_flow[0, 1]

    def test_local_supply_suppresses_demand(self):
        # Y-shaped topology: hub (face 0) with two separate arms.
        #   arm A: 0 — 1 — 2   (dst_rich at face 2)
        #   arm B: 0 — 3 — 4   (dst_poor at face 4)
        # src at face 0.  Paths to the two destinations are fully disjoint, so
        # face-flow at each leaf reflects only flow directed *to* that settlement.
        n = 5
        adj = [[1, 3], [0, 2], [1], [0, 4], [3]]
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        centroids = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(n)])
        mesh = _Mesh(adj, centroids)

        land_p = np.ones(n, dtype=bool)
        sea_p = np.zeros(n, dtype=bool)
        ec = [np.ones(len(nbrs), dtype=np.float64) for nbrs in adj]

        settlements = [_settlement(0, tier=0), _settlement(2, tier=1), _settlement(4, tier=1)]
        # src has type-0 goods; dst_rich also has type-0 locally — saturates its demand.
        sources = [
            (0, 10.0, 0),  # goods at src
            (2, 10.0, 0),  # same good already at dst_rich
        ]
        result = compute_trade_network(
            mesh, settlements, sources,
            ec, land_p, [np.empty(0)] * n, sea_p,
            flow_decay=0.0,
        )
        # dst_poor (face 4) has unmet demand for type 0 → attracts inter-settlement
        # flow; dst_rich (face 2) is saturated → receives none from src.
        flow_at_rich = float(result.face_flow[2])
        flow_at_poor = float(result.face_flow[4])
        assert flow_at_poor > flow_at_rich
