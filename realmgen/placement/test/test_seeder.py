"""Unit tests for realmgen.placement.seeder on stub graphs."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from realmgen.placement.seeder import (
    build_cost_graph,
    placement_score,
    seed_settlements,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _uniform_graph(n: int, cost: float = 1.0) -> csr_matrix:
    """Fully-connected directed graph with uniform edge cost."""
    rows, cols, data = [], [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                rows.append(i)
                cols.append(j)
                data.append(cost)
    return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)


def _line_graph(n: int, cost: float = 1.0) -> tuple:
    """Linear graph 0—1—2—…—(n-1) with uniform cost.

    Returns (csr_matrix, face_adjacency list, edge_costs list).
    """
    adj: list = []
    ec: list = []
    for i in range(n):
        neighbors = []
        if i > 0:
            neighbors.append(i - 1)
        if i < n - 1:
            neighbors.append(i + 1)
        adj.append(neighbors)
        ec.append(np.full(len(neighbors), cost, dtype=np.float64))
    rows, cols, data = [], [], []
    for i, neighbors in enumerate(adj):
        for j in neighbors:
            rows.append(i)
            cols.append(j)
            data.append(cost)
    g = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    return g, adj, ec


def _equatorial_centroids(n: int) -> np.ndarray:
    """n unit-sphere points spread along the equator."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles), np.zeros(n)])


# ── placement_score ───────────────────────────────────────────────────────────


class TestPlacementScore:
    def test_equal_weight_averages(self):
        s = np.array([0.8, 0.2], dtype=np.float32)
        sv = np.array([0.2, 0.8], dtype=np.float32)
        out = placement_score(s, sv, situational_weight=0.5)
        assert out[0] == pytest.approx(0.5, abs=1e-5)
        assert out[1] == pytest.approx(0.5, abs=1e-5)

    def test_zero_weight_returns_suitability(self):
        s = np.array([0.7], dtype=np.float32)
        sv = np.array([0.3], dtype=np.float32)
        out = placement_score(s, sv, situational_weight=0.0)
        assert out[0] == pytest.approx(0.7, abs=1e-5)

    def test_full_weight_returns_situational(self):
        s = np.array([0.7], dtype=np.float32)
        sv = np.array([0.3], dtype=np.float32)
        out = placement_score(s, sv, situational_weight=1.0)
        assert out[0] == pytest.approx(0.3, abs=1e-5)

    def test_output_clamped_to_unit(self):
        # Out-of-range inputs get clamped.
        s = np.array([1.5, -0.5], dtype=np.float32)
        sv = np.array([0.5, 0.5], dtype=np.float32)
        out = placement_score(s, sv, situational_weight=0.5)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_returns_float32(self):
        out = placement_score(
            np.array([0.5], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
        )
        assert out.dtype == np.float32


# ── build_cost_graph ──────────────────────────────────────────────────────────


class TestBuildCostGraph:
    def test_shape(self):
        adj = [[1, 2], [0], [0]]
        ec = [np.array([1.0, 2.0]), np.array([1.0]), np.array([2.0])]
        g = build_cost_graph(adj, ec)
        assert g.shape == (3, 3)

    def test_finite_edges_included(self):
        adj = [[1, 2], [0], [0]]
        ec = [np.array([1.0, 2.0]), np.array([1.0]), np.array([2.0])]
        g = build_cost_graph(adj, ec)
        assert g.nnz == 4  # four directed finite edges

    def test_infinite_edges_excluded(self):
        adj = [[1, 2], [0], [0]]
        ec = [np.array([1.0, np.inf]), np.array([1.0]), np.array([np.inf])]
        g = build_cost_graph(adj, ec)
        assert g.nnz == 2  # only the two finite edges

    def test_values_match(self):
        adj = [[1], [0]]
        ec = [np.array([3.7]), np.array([2.1])]
        g = build_cost_graph(adj, ec)
        assert float(g[0, 1]) == pytest.approx(3.7)
        assert float(g[1, 0]) == pytest.approx(2.1)

    def test_empty_adjacency(self):
        g = build_cost_graph([], [])
        assert g.shape == (0, 0)

    def test_all_infinite_gives_empty_graph(self):
        adj = [[1], [0]]
        ec = [np.array([np.inf]), np.array([np.inf])]
        g = build_cost_graph(adj, ec)
        assert g.nnz == 0


# ── seed_settlements ──────────────────────────────────────────────────────────


class TestSeedSettlements:
    def test_water_faces_always_skipped(self):
        n = 5
        g = _uniform_graph(n, cost=1.0)
        score = np.ones(n, dtype=np.float32)
        is_water = np.array([True, True, True, False, False])
        setts = seed_settlements(
            score, is_water, _equatorial_centroids(n), g,
            tier_configs=[(0.5, 0.0, 1.0)],
        )
        for s in setts:
            assert not is_water[s.face_id], f"face {s.face_id} is water"

    def test_highest_score_picked_first(self):
        n = 5
        # Large cost → cost-distance between any two nodes is large;
        # only one settlement per tier.
        g = _uniform_graph(n, cost=100.0)
        score = np.array([0.3, 0.9, 0.5, 0.7, 0.1], dtype=np.float32)
        is_water = np.zeros(n, dtype=bool)
        setts = seed_settlements(
            score, is_water, _equatorial_centroids(n), g,
            tier_configs=[(0.0, 0.0, 1.0)],  # exclusion_radius=0 → only the placed face is forbidden
        )
        assert setts[0].face_id == 1  # index 1 has score 0.9

    def test_exclusion_radius_enforced(self):
        # Line graph 0—1—2—3—4, cost=1.0, radius=1.5.
        # Face 2 scores highest. After placing it, faces 1, 2, 3
        # are within cost-distance 1.5 (direct neighbors cost 1.0 each).
        n = 5
        g, _, _ = _line_graph(n)
        score = np.array([0.4, 0.4, 1.0, 0.4, 0.4], dtype=np.float32)
        is_water = np.zeros(n, dtype=bool)
        setts = seed_settlements(
            score, is_water, _equatorial_centroids(n), g,
            tier_configs=[(1.5, 0.0, 1.0)],
        )
        placed = {s.face_id for s in setts}
        assert 2 in placed
        assert 1 not in placed
        assert 3 not in placed

    def test_tier_indices_assigned_correctly(self):
        # Two tiers; uniform graph with large cost → one settlement per tier.
        n = 6
        g = _uniform_graph(n, cost=100.0)
        score = np.linspace(0.1, 1.0, n, dtype=np.float32)
        is_water = np.zeros(n, dtype=bool)
        setts = seed_settlements(
            score, is_water, _equatorial_centroids(n), g,
            tier_configs=[(50.0, 0.5, 1.0), (50.0, 0.1, 0.5)],
        )
        tiers = [s.tier for s in setts]
        assert 0 in tiers
        # Tier 1 may not appear if tier 0's exclusion covers the whole graph.

    def test_score_threshold_respected(self):
        n = 6
        g = _uniform_graph(n, cost=0.01)  # tiny exclusion
        score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        is_water = np.zeros(n, dtype=bool)
        setts = seed_settlements(
            score, is_water, _equatorial_centroids(n), g,
            tier_configs=[(0.0, 0.45, 1.0)],
        )
        for s in setts:
            assert s.score >= 0.45, f"score {s.score} below threshold"

    def test_empty_when_all_water(self):
        n = 5
        g = _uniform_graph(n, cost=1.0)
        setts = seed_settlements(
            np.ones(n, dtype=np.float32),
            np.ones(n, dtype=bool),
            _equatorial_centroids(n),
            g,
            tier_configs=[(5.0, 0.0, 1.0)],
        )
        assert setts == []

    def test_no_tier_configs_returns_empty(self):
        n = 5
        g = _uniform_graph(n, cost=1.0)
        setts = seed_settlements(
            np.ones(n, dtype=np.float32),
            np.zeros(n, dtype=bool),
            _equatorial_centroids(n),
            g,
            tier_configs=[],
        )
        assert setts == []

    def test_deterministic(self):
        rng = np.random.default_rng(42)
        n = 20
        score = rng.random(n).astype(np.float32)
        is_water = rng.random(n) < 0.3
        g = _uniform_graph(n, cost=1.0)
        tier_configs = [(3.0, 0.3, 1.0), (1.5, 0.1, 0.5)]
        c = _equatorial_centroids(n)
        s1 = seed_settlements(score, is_water, c, g, tier_configs)
        s2 = seed_settlements(score, is_water, c, g, tier_configs)
        assert [s.face_id for s in s1] == [s.face_id for s in s2]
        assert [s.tier for s in s1] == [s.tier for s in s2]

    def test_cumulative_exclusion_blocks_lower_tiers(self):
        # Tier 0 places at node 0 with large radius; tier 1 should find nothing
        # because all nodes are within that radius.
        n = 4
        g = _uniform_graph(n, cost=1.0)
        score = np.array([1.0, 0.8, 0.6, 0.4], dtype=np.float32)
        is_water = np.zeros(n, dtype=bool)
        setts = seed_settlements(
            score, is_water, _equatorial_centroids(n), g,
            tier_configs=[(100.0, 0.5, 1.0), (100.0, 0.1, 0.5)],
        )
        # Only one settlement in tier 0; tier 1 finds the whole graph forbidden.
        tier0 = [s for s in setts if s.tier == 0]
        tier1 = [s for s in setts if s.tier == 1]
        assert len(tier0) == 1
        assert len(tier1) == 0

    def test_settlement_position_matches_centroid(self):
        n = 3
        g = _uniform_graph(n, cost=100.0)
        c = _equatorial_centroids(n)
        score = np.array([0.8, 0.5, 0.3], dtype=np.float32)
        setts = seed_settlements(
            score, np.zeros(n, dtype=bool), c, g,
            tier_configs=[(0.0, 0.0, 1.0)],
        )
        assert len(setts) >= 1
        fid = setts[0].face_id
        np.testing.assert_allclose(setts[0].position, c[fid])
