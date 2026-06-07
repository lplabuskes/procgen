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
    compute_carrying_capacity,
    compute_hinterland_shares,
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


def _settlement(face_id: int, tier: int = 0, pop: float = 0.0) -> Settlement:
    pos = np.zeros(3, dtype=np.float64)
    pos[0] = 1.0
    return Settlement(face_id=face_id, position=pos, tier=tier, score=0.5, population=pop)


# ── compute_hinterland_shares ───────────────────────────────────────────────────


class TestHinterlandShares:
    def test_empty_settlements(self):
        g = _uniform_graph(5)
        shares, owner = compute_hinterland_shares(5, [], g, max_distance=10.0)
        assert shares.shape == (0, 5)
        assert (owner == -1).all()

    def test_single_settlement_claims_all_in_range(self):
        n = 5
        g = _uniform_graph(n)  # every face one hop from face 2 (dist ≤ 1)
        settlements = [_settlement(2, tier=0)]
        shares, owner = compute_hinterland_shares(n, settlements, g, max_distance=10.0)
        assert (owner == 0).all()
        # Sole bidder → it gets the full share of every in-range face.
        np.testing.assert_allclose(shares[0], 1.0)

    def test_shares_sum_to_one_in_range(self):
        n = 5
        g = _line_graph(n)
        settlements = [_settlement(0), _settlement(4)]
        shares, owner = compute_hinterland_shares(n, settlements, g, max_distance=10.0)
        col_sums = shares.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0)  # all faces within range

    def test_two_settlements_own_their_ends(self):
        n = 5
        g = _line_graph(n)
        settlements = [_settlement(0), _settlement(4)]
        _shares, owner = compute_hinterland_shares(n, settlements, g, max_distance=10.0)
        assert owner[0] == 0
        assert owner[4] == 1
        assert (owner >= 0).all()

    def test_higher_pull_claims_contested_middle(self):
        # Line 0—1—2—3—4.  s0 (high pop, high pull) vs s1 (low pop, low pull).
        # At the middle face 2 both are equidistant; the higher-pop s0 wins and
        # claims more faces.  s1 still keeps a positive share at its own face.
        n = 5
        g = _line_graph(n, cost=1.0)
        s0 = _settlement(0, pop=1.0)
        s1 = _settlement(4, pop=0.05)
        shares, owner = compute_hinterland_shares(
            n, [s0, s1], g, max_distance=10.0,
        )
        assert owner[2] == 0  # s0 wins the equidistant middle face
        assert int((owner == 0).sum()) > int((owner == 1).sum())
        assert shares[1, 4] > 0.0  # s1 retains a share even where out-bid

    def test_low_pull_settlement_keeps_a_share(self):
        # The key feature: a low-pop village beside a high-pop city still
        # captures a nonzero share of nearby land, even though the city is the
        # displayed owner of those faces.
        n = 3
        g = _line_graph(n)  # city — — village  (0 — 1 — 2)
        city = _settlement(0, pop=1.0)     # high pull
        village = _settlement(2, pop=0.05)  # low pull
        shares, owner = compute_hinterland_shares(
            n, [city, village], g, max_distance=10.0,
        )
        # City out-bids the village even at the village's own face …
        assert owner[2] == 0
        # … yet the village still holds a positive share there.
        assert shares[1, 2] > 0.0

    def test_max_distance_gates_reach(self):
        # Line 0—1—2—3—4, single settlement at 0.  With max_distance 2.5,
        # faces 3 and 4 (dist 3, 4) are out of range → owner -1, share 0.
        n = 5
        g = _line_graph(n, cost=1.0)
        shares, owner = compute_hinterland_shares(
            n, [_settlement(0)], g, max_distance=2.5,
        )
        assert owner[0] == 0 and owner[1] == 0 and owner[2] == 0
        assert owner[3] == -1 and owner[4] == -1
        assert shares[0, 3] == 0.0 and shares[0, 4] == 0.0

    def test_disjoint_graph_unreachable_faces_minus_one(self):
        rows = [0, 1]
        cols = [1, 0]
        data = [1.0, 1.0]
        g = csr_matrix((data, (rows, cols)), shape=(4, 4), dtype=np.float64)
        shares, owner = compute_hinterland_shares(
            4, [_settlement(0)], g, max_distance=10.0,
        )
        assert owner[0] == 0 and owner[1] == 0
        assert owner[2] == -1 and owner[3] == -1
        assert (shares[0, 2:] == 0.0).all()

    def test_owner_returns_int32(self):
        g = _uniform_graph(4)
        _shares, owner = compute_hinterland_shares(
            4, [_settlement(0)], g, max_distance=10.0,
        )
        assert owner.dtype == np.int32


# ── compute_carrying_capacity ──────────────────────────────────────────────────


class TestComputeCarryingCapacity:
    def test_basic_sum_from_shares(self):
        # Hard shares: s0 owns faces 0, 1; s1 owns faces 2, 3.
        shares = np.array([
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ], dtype=np.float64)
        fertility = np.array([0.5, 0.5, 1.0, 1.0], dtype=np.float32)
        areas = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        cap = compute_carrying_capacity(shares, fertility, areas)
        assert cap[0] == pytest.approx(1.0)
        assert cap[1] == pytest.approx(2.0)

    def test_fractional_shares_split_capacity(self):
        # One face split 0.25 / 0.75 between two settlements.
        shares = np.array([[0.25], [0.75]], dtype=np.float64)
        fertility = np.array([0.8], dtype=np.float32)
        areas = np.array([2.0], dtype=np.float64)
        cap = compute_carrying_capacity(shares, fertility, areas)
        assert cap[0] == pytest.approx(0.25 * 0.8 * 2.0)
        assert cap[1] == pytest.approx(0.75 * 0.8 * 2.0)

    def test_unclaimed_faces_excluded(self):
        # Middle face (column 0) claimed by nobody → contributes nothing.
        shares = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
        fertility = np.array([1.0, 0.8, 1.0], dtype=np.float32)
        areas = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        cap = compute_carrying_capacity(shares, fertility, areas)
        assert cap[0] == pytest.approx(1.6)  # only face 1: 0.8 × 2.0

    def test_no_settlements_returns_empty(self):
        shares = np.zeros((0, 3), dtype=np.float64)
        cap = compute_carrying_capacity(shares, np.ones(3), np.ones(3))
        assert cap.shape == (0,)

    def test_area_weighted(self):
        shares = np.array([[1.0, 1.0]], dtype=np.float64)
        fertility = np.array([0.5, 0.5], dtype=np.float32)
        areas = np.array([1.0, 3.0], dtype=np.float64)
        cap = compute_carrying_capacity(shares, fertility, areas)
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
