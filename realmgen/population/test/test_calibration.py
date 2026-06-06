"""Unit tests for realmgen.population.calibration — pure-function tests.

All tests use lightweight stub graphs with no dependency on a full worldgen run.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from realmgen.placement.types import Settlement
from realmgen.population.calibration import (
    apply_population_to_settlements,
    assign_hinterlands,
    compute_carrying_capacity,
    compute_population,
    compute_trade_wealth,
    cull_settlements,
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
    if data:
        return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    return csr_matrix((n, n), dtype=np.float64)


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


def _settlement(
    face_id: int, tier: int = 0, pop: float = 0.0,
    influence: float = 0.0, demand: float = 1.0,
) -> Settlement:
    pos = np.zeros(3, dtype=np.float64)
    pos[0] = 1.0
    return Settlement(
        face_id=face_id, position=pos, tier=tier, score=0.5,
        influence=influence, demand=demand, population=pop,
    )


# ── assign_hinterlands ────────────────────────────────────────────────────────


class TestAssignHinterlands:
    def test_empty_settlements_all_minus_one(self):
        g = _uniform_graph(5)
        result = assign_hinterlands(5, [], g)
        assert (result == -1).all()

    def test_single_settlement_claims_all_reachable(self):
        n = 5
        g = _uniform_graph(n)
        settlements = [_settlement(2, tier=0)]
        result = assign_hinterlands(n, settlements, g)
        # All faces reachable → all assigned to settlement 0.
        assert (result >= 0).all()
        assert (result == 0).all()

    def test_two_settlements_split_line(self):
        # Line graph: 0—1—2—3—4 with uniform cost.
        # Settlement at 0 and 4 should split roughly in the middle.
        n = 5
        g = _line_graph(n)
        settlements = [_settlement(0, tier=0), _settlement(4, tier=0)]
        result = assign_hinterlands(n, settlements, g)
        assert result[0] == 0  # closest to settlement at face 0
        assert result[4] == 1  # closest to settlement at face 4
        # The split is symmetric — every face is assigned.
        assert (result >= 0).all()

    def test_influence_widens_low_influence_territory(self):
        # Line graph 0—1—2—3—4.  Settlements at the two ends.
        # s0 at face 0 (influence 0.0), s1 at face 4 (influence 2.0).
        # Without influence the graph is symmetric; with influence s0 should
        # claim more faces since its effective cost is uniformly 2.0 lower.
        #
        # Effective costs to each face:
        #   face 0: s0=0+0=0, s1=4+2=6  → s0
        #   face 1: s0=1+0=1, s1=3+2=5  → s0
        #   face 2: s0=2+0=2, s1=2+2=4  → s0
        #   face 3: s0=3+0=3, s1=1+2=3  → tie (s0 wins via argmin with index 0)
        #   face 4: s0=4+0=4, s1=0+2=2  → s1
        n = 5
        g = _line_graph(n, cost=1.0)
        s0 = _settlement(0, tier=0, influence=0.0)
        s1 = _settlement(4, tier=1, influence=2.0)
        result = assign_hinterlands(n, [s0, s1], g)
        s0_count = int((result == 0).sum())
        s1_count = int((result == 1).sum())
        assert s0_count > s1_count, f"Expected s0 to claim more ({s0_count} vs {s1_count})"
        assert result[4] == 1  # s1 claims its own face

    def test_disjoint_graph_unreachable_faces_minus_one(self):
        # Two disconnected components: faces 0-1 and faces 2-3.
        # Settlement only in component 0—1.
        rows = [0, 1]
        cols = [1, 0]
        data = [1.0, 1.0]
        g = csr_matrix((data, (rows, cols)), shape=(4, 4), dtype=np.float64)
        settlements = [_settlement(0, tier=0)]
        result = assign_hinterlands(4, settlements, g)
        assert result[0] == 0
        assert result[1] == 0
        assert result[2] == -1
        assert result[3] == -1

    def test_returns_int32(self):
        n = 4
        g = _uniform_graph(n)
        result = assign_hinterlands(n, [_settlement(0)], g)
        assert result.dtype == np.int32


# ── compute_carrying_capacity ──────────────────────────────────────────────────


class TestComputeCarryingCapacity:
    def test_basic_sum(self):
        # Two settlements: s0 owns faces 0, 1; s1 owns faces 2, 3.
        hinterland = np.array([0, 0, 1, 1], dtype=np.int32)
        fertility = np.array([0.5, 0.5, 1.0, 1.0], dtype=np.float32)
        areas = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        cap = compute_carrying_capacity(hinterland, fertility, areas, 2)
        assert cap[0] == pytest.approx(1.0)
        assert cap[1] == pytest.approx(2.0)

    def test_unassigned_faces_excluded(self):
        hinterland = np.array([-1, 0, -1], dtype=np.int32)
        fertility = np.array([1.0, 0.8, 1.0], dtype=np.float32)
        areas = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        cap = compute_carrying_capacity(hinterland, fertility, areas, 1)
        assert cap[0] == pytest.approx(1.6)  # only face 1: 0.8 × 2.0

    def test_all_unassigned_returns_zeros(self):
        hinterland = np.array([-1, -1, -1], dtype=np.int32)
        fertility = np.ones(3, dtype=np.float32)
        areas = np.ones(3, dtype=np.float64)
        cap = compute_carrying_capacity(hinterland, fertility, areas, 2)
        assert (cap == 0.0).all()

    def test_area_weighted(self):
        hinterland = np.array([0, 0], dtype=np.int32)
        fertility = np.array([0.5, 0.5], dtype=np.float32)
        areas = np.array([1.0, 3.0], dtype=np.float64)
        cap = compute_carrying_capacity(hinterland, fertility, areas, 1)
        assert cap[0] == pytest.approx(2.0)  # 0.5×1 + 0.5×3


# ── compute_trade_wealth ──────────────────────────────────────────────────────


class TestComputeTradeWealth:
    def test_extracts_flow_at_settlement_face(self):
        settlements = [_settlement(2), _settlement(4)]
        flow = np.array([0.0, 0.0, 7.5, 0.0, 3.2], dtype=np.float32)
        wealth = compute_trade_wealth(settlements, flow)
        assert wealth[0] == pytest.approx(7.5)
        assert wealth[1] == pytest.approx(3.2, rel=1e-5)

    def test_empty_settlements(self):
        wealth = compute_trade_wealth([], np.zeros(5, dtype=np.float32))
        assert wealth.shape == (0,)

    def test_zero_flow(self):
        settlements = [_settlement(0), _settlement(1)]
        wealth = compute_trade_wealth(settlements, np.zeros(5, dtype=np.float32))
        assert (wealth == 0.0).all()


# ── compute_population ────────────────────────────────────────────────────────


class TestComputePopulation:
    def test_zero_carrying_cap_gives_zero(self):
        cap = np.array([0.0, 0.0], dtype=np.float64)
        wealth = np.array([10.0, 5.0], dtype=np.float64)
        pop = compute_population(cap, wealth)
        assert (pop == 0.0).all()

    def test_nonzero_cap_positive_pop(self):
        cap = np.array([1.0, 1.0], dtype=np.float64)
        wealth = np.array([1.0, 1.0], dtype=np.float64)
        pop = compute_population(cap, wealth)
        assert (pop > 0).all()

    def test_higher_trade_wealth_gives_higher_pop(self):
        cap = np.array([1.0, 1.0], dtype=np.float64)
        wealth = np.array([10.0, 1.0], dtype=np.float64)
        pop = compute_population(cap, wealth)
        assert pop[0] > pop[1]

    def test_trade_weight_zero_returns_carrying_cap(self):
        cap = np.array([2.0, 3.0], dtype=np.float64)
        wealth = np.array([100.0, 200.0], dtype=np.float64)
        pop = compute_population(cap, wealth, trade_weight=0.0)
        np.testing.assert_allclose(pop, cap)

    def test_all_zero_wealth_returns_cap(self):
        cap = np.array([1.0, 2.0], dtype=np.float64)
        wealth = np.zeros(2, dtype=np.float64)
        pop = compute_population(cap, wealth)
        # log1p(0) = 0 → trade bonus = 0 → pop = cap * 1.0
        np.testing.assert_allclose(pop, cap)


# ── cull_settlements ──────────────────────────────────────────────────────────


class TestCullSettlements:
    def test_all_above_threshold_unchanged(self):
        s = [_settlement(0, tier=1), _settlement(1, tier=2)]
        pop = np.array([100.0, 80.0], dtype=np.float64)
        kept_s, kept_p = cull_settlements(s, pop, min_pop_fraction=0.0)
        assert len(kept_s) == 2

    def test_below_threshold_culled_regardless_of_tier(self):
        # Both a low-tier and high-tier settlement below threshold get culled.
        s = [_settlement(0, tier=0), _settlement(1, tier=1), _settlement(2, tier=3)]
        pop = np.array([100.0, 0.001, 0.001], dtype=np.float64)
        kept_s, kept_p = cull_settlements(s, pop, min_pop_fraction=0.01)
        face_ids = [x.face_id for x in kept_s]
        assert 0 in face_ids
        assert 1 not in face_ids   # tier-1 below threshold → culled (no downgrade)
        assert 2 not in face_ids   # tier-3 below threshold → culled

    def test_tiers_unchanged_on_survivors(self):
        # Culling must not modify tiers on surviving settlements.
        s = [_settlement(0, tier=0), _settlement(1, tier=1)]
        pop = np.array([100.0, 80.0], dtype=np.float64)
        kept_s, kept_p = cull_settlements(s, pop, min_pop_fraction=0.0)
        assert kept_s[0].tier == 0
        assert kept_s[1].tier == 1

    def test_empty_input(self):
        kept_s, kept_p = cull_settlements([], np.array([], dtype=np.float64))
        assert kept_s == []
        assert kept_p.shape == (0,)

    def test_output_aligned(self):
        s = [_settlement(i, tier=min(i, 3)) for i in range(5)]
        pop = np.array([100.0, 80.0, 60.0, 40.0, 20.0], dtype=np.float64)
        kept_s, kept_p = cull_settlements(s, pop, min_pop_fraction=0.0)
        assert len(kept_s) == len(kept_p)

    def test_zero_max_pop_keeps_all(self):
        # All population zero: threshold is 0, so nothing is below it.
        s = [_settlement(0, tier=0), _settlement(1, tier=3)]
        pop = np.zeros(2, dtype=np.float64)
        kept_s, kept_p = cull_settlements(s, pop, min_pop_fraction=0.01)
        assert len(kept_s) == 2


# ── apply_population_to_settlements ──────────────────────────────────────────


class TestApplyPopulation:
    def test_population_stamped(self):
        s = [_settlement(0), _settlement(1)]
        pop = np.array([42.0, 7.0], dtype=np.float64)
        result = apply_population_to_settlements(s, pop)
        assert result[0].population == pytest.approx(42.0)
        assert result[1].population == pytest.approx(7.0)

    def test_original_unchanged(self):
        s = [_settlement(0, pop=0.0)]
        pop = np.array([99.0], dtype=np.float64)
        apply_population_to_settlements(s, pop)
        assert s[0].population == 0.0  # original untouched

    def test_empty(self):
        result = apply_population_to_settlements([], np.array([], dtype=np.float64))
        assert result == []
