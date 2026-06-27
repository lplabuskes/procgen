"""Unit tests for realmgen.population.calibration — pure-function tests.

All tests use lightweight stub graphs with no dependency on a full worldgen run.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from realmgen.population.calibration import (
    KM2_PER_STERADIAN,
    apply_population_to_settlements,
    compute_carrying_capacity,
    compute_food_base,
    compute_hinterland_shares,
    compute_rural_population,
    compute_trade_wealth,
    compute_urban_population,
    cull_settlements,
)
from realmgen.polity.test.conftest import _line_graph
from realmgen.test.conftest import _settlement


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


# ── compute_food_base / compute_rural_population ──────────────────────────────


class TestComputeFoodBase:
    def test_linear_law_matches_density_times_area(self):
        fert = np.array([1.0, 0.5], dtype=np.float32)
        areas = np.array([2.0, 2.0], dtype=np.float64)  # steradians
        fb = compute_food_base(fert, areas, density_max=10.0, fertility_exponent=1.0)
        assert fb[0] == pytest.approx(10.0 * 2.0 * KM2_PER_STERADIAN)
        assert fb[1] == pytest.approx(5.0 * 2.0 * KM2_PER_STERADIAN)

    def test_exponent_restores_contrast(self):
        # γ=2 squeezes low-fertility faces much harder than high-fertility ones.
        fert = np.array([0.9, 0.3], dtype=np.float32)
        areas = np.ones(2, dtype=np.float64)
        lin = compute_food_base(fert, areas, density_max=1.0, fertility_exponent=1.0)
        quad = compute_food_base(fert, areas, density_max=1.0, fertility_exponent=2.0)
        assert lin[0] / lin[1] == pytest.approx(3.0)
        assert quad[0] / quad[1] == pytest.approx(9.0)

    def test_fertility_clipped(self):
        fert = np.array([-0.5, 1.5], dtype=np.float32)
        areas = np.ones(2, dtype=np.float64)
        fb = compute_food_base(fert, areas, density_max=1.0)
        assert fb[0] == 0.0
        assert fb[1] == pytest.approx(KM2_PER_STERADIAN)


class TestComputeRuralPopulation:
    def test_water_masked_to_zero(self):
        fb = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        is_water = np.array([False, True, False])
        rural = compute_rural_population(fb, is_water)
        np.testing.assert_allclose(rural, [100.0, 0.0, 300.0])


# ── compute_carrying_capacity ──────────────────────────────────────────────────


class TestComputeCarryingCapacity:
    def test_basic_sum_from_shares(self):
        # Hard shares: s0 owns faces 0, 1; s1 owns faces 2, 3.
        shares = np.array([
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ], dtype=np.float64)
        food_base = np.array([0.5, 0.5, 1.0, 1.0], dtype=np.float64)
        cap = compute_carrying_capacity(shares, food_base)
        assert cap[0] == pytest.approx(1.0)
        assert cap[1] == pytest.approx(2.0)

    def test_fractional_shares_split_capacity(self):
        # One face split 0.25 / 0.75 between two settlements.
        shares = np.array([[0.25], [0.75]], dtype=np.float64)
        food_base = np.array([1.6], dtype=np.float64)
        cap = compute_carrying_capacity(shares, food_base)
        assert cap[0] == pytest.approx(0.25 * 1.6)
        assert cap[1] == pytest.approx(0.75 * 1.6)

    def test_unclaimed_faces_excluded(self):
        # Faces claimed by nobody contribute nothing.
        shares = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
        food_base = np.array([2.0, 1.6, 2.0], dtype=np.float64)
        cap = compute_carrying_capacity(shares, food_base)
        assert cap[0] == pytest.approx(1.6)  # only face 1

    def test_no_settlements_returns_empty(self):
        shares = np.zeros((0, 3), dtype=np.float64)
        cap = compute_carrying_capacity(shares, np.ones(3))
        assert cap.shape == (0,)


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


# ── compute_urban_population ──────────────────────────────────────────────────


class TestComputeUrbanPopulation:
    def test_surplus_term_scales_with_cap(self):
        cap = np.array([100_000.0, 200_000.0], dtype=np.float64)
        wealth = np.zeros(2, dtype=np.float64)
        pop = compute_urban_population(cap, wealth, surplus_rate=0.15)
        np.testing.assert_allclose(pop, [15_000.0, 30_000.0])

    def test_trade_is_additive_not_multiplicative(self):
        # The Constantinople/Venice unlock: zero hinterland, big trade flow →
        # a real city fed entirely on imported grain.
        cap = np.array([0.0], dtype=np.float64)
        wealth = np.array([5.0], dtype=np.float64)
        pop = compute_urban_population(
            cap, wealth, surplus_rate=0.15, trade_food_weight=20_000.0,
        )
        assert pop[0] == pytest.approx(100_000.0)

    def test_higher_trade_wealth_gives_higher_pop(self):
        cap = np.array([1.0, 1.0], dtype=np.float64)
        wealth = np.array([10.0, 1.0], dtype=np.float64)
        pop = compute_urban_population(cap, wealth, trade_food_weight=1.0)
        assert pop[0] > pop[1]

    def test_trade_food_weight_zero_returns_surplus_only(self):
        cap = np.array([2.0, 3.0], dtype=np.float64)
        wealth = np.array([100.0, 200.0], dtype=np.float64)
        pop = compute_urban_population(
            cap, wealth, surplus_rate=0.1, trade_food_weight=0.0,
        )
        np.testing.assert_allclose(pop, 0.1 * cap)

    def test_negative_inputs_clamped(self):
        cap = np.array([-5.0], dtype=np.float64)
        wealth = np.array([-2.0], dtype=np.float64)
        pop = compute_urban_population(
            cap, wealth, surplus_rate=0.15, trade_food_weight=1.0,
        )
        assert pop[0] == 0.0


# ── cull_settlements ──────────────────────────────────────────────────────────


class TestCullSettlements:
    def test_all_above_floor_unchanged(self):
        s = [_settlement(0, tier=1), _settlement(1, tier=2)]
        pop = np.array([100.0, 80.0], dtype=np.float64)
        kept_s, kept_p, kept = cull_settlements(s, pop, min_population=50.0)
        assert len(kept_s) == 2
        assert kept.all()

    def test_below_floor_culled_regardless_of_tier(self):
        # The urban-formation floor is absolute: a rich farming region with no
        # trade node has villages but no city, whatever its seeding tier.
        s = [_settlement(0, tier=0), _settlement(1, tier=1), _settlement(2, tier=3)]
        pop = np.array([50_000.0, 4_000.0, 900.0], dtype=np.float64)
        kept_s, kept_p, kept = cull_settlements(s, pop, min_population=10_000.0)
        face_ids = [x.face_id for x in kept_s]
        assert face_ids == [0]
        np.testing.assert_array_equal(kept, [True, False, False])

    def test_keep_mask_filters_aligned_arrays(self):
        s = [_settlement(i) for i in range(4)]
        pop = np.array([20.0, 5.0, 30.0, 1.0], dtype=np.float64)
        aligned = np.array([1.0, 2.0, 3.0, 4.0])
        kept_s, kept_p, kept = cull_settlements(s, pop, min_population=10.0)
        np.testing.assert_allclose(aligned[kept], [1.0, 3.0])
        np.testing.assert_allclose(kept_p, [20.0, 30.0])

    def test_tiers_unchanged_on_survivors(self):
        # Culling must not modify tiers on surviving settlements.
        s = [_settlement(0, tier=0), _settlement(1, tier=1)]
        pop = np.array([100.0, 80.0], dtype=np.float64)
        kept_s, kept_p, _ = cull_settlements(s, pop, min_population=0.0)
        assert kept_s[0].tier == 0
        assert kept_s[1].tier == 1

    def test_empty_input(self):
        kept_s, kept_p, kept = cull_settlements([], np.array([], dtype=np.float64))
        assert kept_s == []
        assert kept_p.shape == (0,)
        assert kept.shape == (0,)

    def test_output_aligned(self):
        s = [_settlement(i, tier=min(i, 3)) for i in range(5)]
        pop = np.array([100.0, 80.0, 60.0, 40.0, 20.0], dtype=np.float64)
        kept_s, kept_p, _ = cull_settlements(s, pop, min_population=0.0)
        assert len(kept_s) == len(kept_p)

    def test_zero_floor_keeps_all(self):
        s = [_settlement(0, tier=0), _settlement(1, tier=3)]
        pop = np.zeros(2, dtype=np.float64)
        kept_s, kept_p, _ = cull_settlements(s, pop, min_population=0.0)
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
