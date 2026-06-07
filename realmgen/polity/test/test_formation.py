"""Unit tests for realmgen.polity.formation — pure-function tests.

All tests use lightweight stub graphs with no dependency on a full worldgen run,
mirroring realmgen.population.test.test_calibration.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from realmgen.polity.formation import (
    compute_border_tension,
    compute_polity_power,
    grow_territory,
    seed_capitals,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _line_graph(n: int, cost: float = 1.0) -> csr_matrix:
    """Linear graph 0—1—2—…—(n-1) with uniform symmetric cost."""
    rows, cols, data = [], [], []
    for i in range(n - 1):
        rows += [i, i + 1]
        cols += [i + 1, i]
        data += [cost, cost]
    if data:
        return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    return csr_matrix((n, n), dtype=np.float64)


def _line_adjacency(n: int):
    """Adjacency list + uniform-cost edge arrays for an n-face line graph."""
    adj = []
    costs = []
    for i in range(n):
        neigh = []
        if i > 0:
            neigh.append(i - 1)
        if i < n - 1:
            neigh.append(i + 1)
        adj.append(neigh)
        costs.append(np.ones(len(neigh), dtype=np.float64))
    return adj, costs


# ── seed_capitals ───────────────────────────────────────────────────────────────


class TestSeedCapitals:
    def test_empty(self):
        g = _line_graph(3)
        assert seed_capitals([], np.array([]), g, min_separation=1.0, min_pop_fraction=0.0) == []

    def test_separation_excludes_close_settlements(self):
        # Line 0—1—2—3—4.  Two settlements at 0 and 1 (dist 1 apart).
        # With min_separation 2.0 the smaller one (face 1) is excluded.
        g = _line_graph(5)
        faces = [0, 1]
        pop = np.array([100.0, 90.0])
        chosen = seed_capitals(faces, pop, g, min_separation=2.0, min_pop_fraction=0.0)
        assert chosen == [0]

    def test_far_apart_both_chosen(self):
        g = _line_graph(5)
        faces = [0, 4]  # dist 4 apart
        pop = np.array([100.0, 80.0])
        chosen = seed_capitals(faces, pop, g, min_separation=2.0, min_pop_fraction=0.0)
        assert set(chosen) == {0, 1}

    def test_pop_fraction_filters_small(self):
        g = _line_graph(5)
        faces = [0, 4]
        pop = np.array([100.0, 5.0])  # second is 5% of peak
        chosen = seed_capitals(faces, pop, g, min_separation=1.0, min_pop_fraction=0.1)
        assert chosen == [0]  # second below 10% threshold

    def test_chosen_in_descending_population_order(self):
        g = _line_graph(7)
        faces = [0, 3, 6]
        pop = np.array([50.0, 100.0, 75.0])
        chosen = seed_capitals(faces, pop, g, min_separation=2.0, min_pop_fraction=0.0)
        # Largest (idx 1) first, then idx 2 (75), then idx 0 (50).
        assert chosen == [1, 2, 0]

    def test_deterministic(self):
        g = _line_graph(7)
        faces = [0, 3, 6]
        pop = np.array([50.0, 100.0, 75.0])
        a = seed_capitals(faces, pop, g, min_separation=2.0, min_pop_fraction=0.0)
        b = seed_capitals(faces, pop, g, min_separation=2.0, min_pop_fraction=0.0)
        assert a == b


# ── grow_territory ──────────────────────────────────────────────────────────────


class TestGrowTerritory:
    def test_no_capitals_all_unclaimed(self):
        g = _line_graph(5)
        out = grow_territory(5, [], np.array([]), g, power_bias=0.0, max_reach=100.0)
        assert (out == -1).all()

    def test_equal_power_splits_at_midpoint(self):
        # Line 0—1—2—3—4, capitals at 0 and 4, equal power, no bias.
        g = _line_graph(5)
        out = grow_territory(5, [0, 4], np.array([1.0, 1.0]), g,
                             power_bias=0.0, max_reach=100.0)
        assert out[0] == 0 and out[1] == 0
        assert out[3] == 1 and out[4] == 1
        # Face 2 is equidistant → argmin ties to the first capital (index 0).
        assert out[2] == 0

    def test_capital_owns_its_own_face(self):
        g = _line_graph(5)
        out = grow_territory(5, [0, 4], np.array([100.0, 0.001]), g,
                             power_bias=50.0, max_reach=100.0)
        assert out[0] == 0
        assert out[4] == 1  # forced, even though heavily out-powered

    def test_stronger_capital_claims_more(self):
        # Capital 0 far stronger than capital 6 → claims past the midpoint.
        g = _line_graph(7)
        out = grow_territory(7, [0, 6], np.array([100.0, 1.0]), g,
                             power_bias=5.0, max_reach=100.0)
        n0 = int((out == 0).sum())
        n1 = int((out == 1).sum())
        assert n0 > n1

    def test_max_reach_leaves_frontier_unclaimed(self):
        # Single capital at 0, max_reach 2.5 → faces 3, 4 out of range.
        g = _line_graph(5)
        out = grow_territory(5, [0], np.array([1.0]), g,
                             power_bias=0.0, max_reach=2.5)
        assert out[0] == 0 and out[1] == 0 and out[2] == 0
        assert out[3] == -1 and out[4] == -1

    def test_unreachable_faces_unclaimed(self):
        # Two disconnected components: {0,1} and {2,3}.  Capital only in first.
        rows, cols, data = [0, 1, 2, 3], [1, 0, 3, 2], [1.0, 1.0, 1.0, 1.0]
        g = csr_matrix((data, (rows, cols)), shape=(4, 4), dtype=np.float64)
        out = grow_territory(4, [0], np.array([1.0]), g,
                             power_bias=0.0, max_reach=100.0)
        assert out[0] == 0 and out[1] == 0
        assert out[2] == -1 and out[3] == -1

    def test_returns_int32(self):
        g = _line_graph(3)
        out = grow_territory(3, [0], np.array([1.0]), g, power_bias=0.0, max_reach=100.0)
        assert out.dtype == np.int32


# ── compute_polity_power ────────────────────────────────────────────────────────


class TestComputePolityPower:
    def test_sums_member_populations(self):
        polity_of_face = np.array([0, 0, 1, 1, -1], dtype=np.int32)
        faces = [0, 1, 2, 3]
        pop = np.array([10.0, 5.0, 20.0, 2.0])
        power = compute_polity_power(polity_of_face, faces, pop, 2)
        assert power[0] == pytest.approx(15.0)
        assert power[1] == pytest.approx(22.0)

    def test_settlement_on_unclaimed_face_ignored(self):
        polity_of_face = np.array([0, -1], dtype=np.int32)
        faces = [0, 1]
        pop = np.array([10.0, 99.0])
        power = compute_polity_power(polity_of_face, faces, pop, 1)
        assert power[0] == pytest.approx(10.0)

    def test_no_settlements(self):
        polity_of_face = np.array([0, 1], dtype=np.int32)
        power = compute_polity_power(polity_of_face, [], np.array([]), 2)
        assert power.shape == (2,)
        assert (power == 0.0).all()


# ── compute_border_tension ──────────────────────────────────────────────────────


class TestComputeBorderTension:
    def test_no_borders_when_single_polity(self):
        adj, costs = _line_adjacency(4)
        polity_of_face = np.zeros(4, dtype=np.int32)
        pair, face_t = compute_border_tension(
            adj, polity_of_face, np.array([1.0]), costs, np.zeros(4),
            barrier_scale=1.0, flow_ref=1.0,
        )
        assert pair == {}
        assert (face_t == 0.0).all()

    def test_single_border_pair_detected(self):
        # Line 0—1 | 2—3 : polities {0,1} and {2,3}, border on edge 1—2.
        adj, costs = _line_adjacency(4)
        polity_of_face = np.array([0, 0, 1, 1], dtype=np.int32)
        flow = np.array([0.0, 1.0, 1.0, 0.0])
        pair, face_t = compute_border_tension(
            adj, polity_of_face, np.array([1.0, 1.0]), costs, flow,
            barrier_scale=1.0, flow_ref=1.0,
        )
        assert set(pair.keys()) == {(0, 1)}
        assert pair[(0, 1)] > 0.0
        # Border faces 1 and 2 light up; interior faces 0, 3 stay calm.
        assert face_t[1] > 0.0 and face_t[2] > 0.0
        assert face_t[0] == 0.0 and face_t[3] == 0.0

    def test_strong_barrier_lowers_tension(self):
        adj, costs_low = _line_adjacency(4)
        adj2, costs_high = _line_adjacency(4)
        # Make the border edge expensive in the high-cost variant.
        for c in costs_high:
            c[:] = 10.0
        polity_of_face = np.array([0, 0, 1, 1], dtype=np.int32)
        flow = np.ones(4)
        power = np.array([1.0, 1.0])
        low, _ = compute_border_tension(
            adj, polity_of_face, power, costs_low, flow,
            barrier_scale=1.0, flow_ref=1.0,
        )
        high, _ = compute_border_tension(
            adj2, polity_of_face, power, costs_high, flow,
            barrier_scale=1.0, flow_ref=1.0,
        )
        assert high[(0, 1)] < low[(0, 1)]

    def test_power_imbalance_lowers_tension(self):
        adj, costs = _line_adjacency(4)
        polity_of_face = np.array([0, 0, 1, 1], dtype=np.int32)
        flow = np.ones(4)
        balanced, _ = compute_border_tension(
            adj, polity_of_face, np.array([1.0, 1.0]), costs, flow,
            barrier_scale=1.0, flow_ref=1.0,
        )
        lopsided, _ = compute_border_tension(
            adj, polity_of_face, np.array([100.0, 1.0]), costs, flow,
            barrier_scale=1.0, flow_ref=1.0,
        )
        assert lopsided[(0, 1)] < balanced[(0, 1)]

    def test_baseline_tension_without_trade(self):
        # A permeable border between matched powers is contested even with zero
        # trade crossing it (the trade-independent baseline).
        adj, costs = _line_adjacency(4)
        polity_of_face = np.array([0, 0, 1, 1], dtype=np.int32)
        pair, _ = compute_border_tension(
            adj, polity_of_face, np.array([1.0, 1.0]), costs, np.zeros(4),
            barrier_scale=1.0, flow_ref=1.0, artery_weight=0.6,
        )
        assert pair[(0, 1)] > 0.0  # baseline survives with no trade

    def test_artery_weight_one_zeros_quiet_border(self):
        # With all weight on trade, a zero-flow border has zero tension.
        adj, costs = _line_adjacency(4)
        polity_of_face = np.array([0, 0, 1, 1], dtype=np.int32)
        pair, face_t = compute_border_tension(
            adj, polity_of_face, np.array([1.0, 1.0]), costs, np.zeros(4),
            barrier_scale=1.0, flow_ref=1.0, artery_weight=1.0,
        )
        # Zero-tension crossings are simply not recorded (absent == zero).
        assert pair.get((0, 1), 0.0) == pytest.approx(0.0)
        assert (face_t == 0.0).all()

    def test_higher_flow_raises_tension(self):
        adj, costs = _line_adjacency(4)
        polity_of_face = np.array([0, 0, 1, 1], dtype=np.int32)
        power = np.array([1.0, 1.0])
        quiet, _ = compute_border_tension(
            adj, polity_of_face, power, costs, np.full(4, 0.1),
            barrier_scale=1.0, flow_ref=1.0,
        )
        busy, _ = compute_border_tension(
            adj, polity_of_face, power, costs, np.full(4, 5.0),
            barrier_scale=1.0, flow_ref=1.0,
        )
        assert busy[(0, 1)] > quiet[(0, 1)]

    def test_unclaimed_faces_create_no_border(self):
        # polity 0 and unclaimed (-1) → no tension pair.
        adj, costs = _line_adjacency(4)
        polity_of_face = np.array([0, 0, -1, -1], dtype=np.int32)
        pair, face_t = compute_border_tension(
            adj, polity_of_face, np.array([1.0]), costs, np.ones(4),
            barrier_scale=1.0, flow_ref=1.0,
        )
        assert pair == {}
