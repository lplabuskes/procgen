"""Unit tests for realmgen.polity.seeding — defensibility, capitals, scalars.

Lightweight stub graphs, no worldgen run.
"""
from __future__ import annotations

import numpy as np
import pytest

from realmgen.polity.seeding import (
    cohesion_to_gain,
    compute_defensibility,
    compute_settlement_defensibility,
    roll_polity_scalars,
    seed_capitals,
)
from realmgen.polity.test.conftest import _line_graph


class TestDefensibility:
    def test_at_least_one_and_monotone_in_cost(self):
        cost = np.array([1.0, 2.0, 5.0, 10.0])
        d = compute_defensibility(cost, np.zeros(4), rugged_weight=1.0, situational_weight=0.0)
        assert (d >= 1.0).all()
        assert d[3] >= d[0]

    def test_situational_value_adds(self):
        cost = np.full(4, 2.0)
        d_no = compute_defensibility(cost, np.zeros(4), situational_weight=1.0)
        d_yes = compute_defensibility(cost, np.array([0.0, 1.0, 0.0, 0.0]), situational_weight=1.0)
        assert d_yes[1] > d_no[1]


class TestSeedCapitals:
    def test_empty(self):
        g = _line_graph(3)
        assert seed_capitals([], np.array([]), g, min_separation=1.0, min_pop_fraction=0.0) == []

    def test_separation_excludes_close_settlements(self):
        g = _line_graph(5)
        chosen = seed_capitals([0, 1], np.array([100.0, 90.0]), g,
                               min_separation=2.0, min_pop_fraction=0.0)
        assert chosen == [0]

    def test_pop_fraction_filters_small(self):
        g = _line_graph(5)
        chosen = seed_capitals([0, 4], np.array([100.0, 5.0]), g,
                               min_separation=1.0, min_pop_fraction=0.1)
        assert chosen == [0]

    def test_weights_override_population_order(self):
        # Same populations, but face 2's weight is highest → chosen first.
        g = _line_graph(7)
        chosen = seed_capitals([0, 3, 6], np.array([100.0, 100.0, 100.0]), g,
                               weights=np.array([1.0, 3.0, 2.0]),
                               min_separation=2.0, min_pop_fraction=0.0)
        assert chosen == [1, 2, 0]

    def test_weight_ranked_but_pop_gated(self):
        # Face 1 has huge weight but tiny population → gated out by pop fraction.
        g = _line_graph(7)
        chosen = seed_capitals([0, 3, 6], np.array([100.0, 2.0, 80.0]), g,
                               weights=np.array([1.0, 99.0, 2.0]),
                               min_separation=2.0, min_pop_fraction=0.1)
        assert 1 not in chosen
        assert set(chosen) == {0, 2}

    def test_deterministic(self):
        g = _line_graph(7)
        faces, pop = [0, 3, 6], np.array([50.0, 100.0, 75.0])
        a = seed_capitals(faces, pop, g, min_separation=2.0, min_pop_fraction=0.0)
        b = seed_capitals(faces, pop, g, min_separation=2.0, min_pop_fraction=0.0)
        assert a == b


def _settlement_defensibility_inputs(n: int = 10, s_faces: list | None = None):
    """Return (land_base_cost, sv, river_flow, trade_wealth, danger, land_mask, faces)."""
    if s_faces is None:
        s_faces = [0, 3, 7]
    rng = np.random.default_rng(0)
    land_base_cost = rng.uniform(1.0, 5.0, n)
    sv = rng.uniform(0.0, 1.0, n)
    river_flow = np.zeros(n)
    river_flow[3] = 10.0   # face 3 has a river
    trade_wealth = rng.uniform(0.0, 1000.0, len(s_faces))
    danger = rng.uniform(0.0, 0.5, n)
    land_mask = np.ones(n, dtype=bool)
    return land_base_cost, sv, river_flow, trade_wealth, danger, land_mask, s_faces


class TestSettlementDefensibility:
    def test_defensibility_at_least_one(self):
        lbc, sv, rf, tw, dg, lm, faces = _settlement_defensibility_inputs()
        defs, _, _ = compute_settlement_defensibility(faces, lbc, sv, rf, tw, dg, lm, base_seed=1)
        assert (defs >= 1.0).all()

    def test_empty_input(self):
        n = 8
        lbc = np.ones(n)
        sv = np.zeros(n)
        rf = np.zeros(n)
        tw = np.empty(0)
        dg = np.zeros(n)
        lm = np.ones(n, dtype=bool)
        defs, walls, wstr = compute_settlement_defensibility([], lbc, sv, rf, tw, dg, lm, base_seed=0)
        assert defs.size == 0 and walls.size == 0 and wstr.size == 0

    def test_river_bonus_adds_when_flow_present(self):
        n = 10
        lbc = np.full(n, 2.0)
        sv = np.zeros(n)
        dg = np.zeros(n)
        lm = np.ones(n, dtype=bool)
        # Face 0 has no river; face 1 has significant river flow.
        rf_none = np.zeros(n)
        rf_river = np.zeros(n)
        rf_river[1] = 50.0
        tw = np.zeros(2)
        d_none, _, _ = compute_settlement_defensibility([0, 1], lbc, sv, rf_none, tw, dg, lm, base_seed=0)
        d_riv, _, _ = compute_settlement_defensibility([0, 1], lbc, sv, rf_river, tw, dg, lm, base_seed=0)
        assert d_riv[1] > d_none[1]    # river face gets bonus
        assert d_riv[0] == pytest.approx(d_none[0])  # non-river face unchanged

    def test_walls_increase_defensibility(self):
        # Force wall_base_prob=1.0 so every settlement gets walls.
        lbc, sv, rf, tw, dg, lm, faces = _settlement_defensibility_inputs()
        d_wall, walls, wstr = compute_settlement_defensibility(
            faces, lbc, sv, rf, tw, dg, lm, base_seed=5,
            wall_base_prob=1.0, wall_min_strength=0.5, wall_max_strength=0.5,
        )
        d_none, _, _ = compute_settlement_defensibility(
            faces, lbc, sv, rf, tw, dg, lm, base_seed=5,
            wall_base_prob=0.0,
        )
        assert walls.all()
        assert (d_wall > d_none).all()
        assert np.allclose(wstr, 0.5)

    def test_deterministic(self):
        lbc, sv, rf, tw, dg, lm, faces = _settlement_defensibility_inputs()
        a = compute_settlement_defensibility(faces, lbc, sv, rf, tw, dg, lm, base_seed=42)
        b = compute_settlement_defensibility(faces, lbc, sv, rf, tw, dg, lm, base_seed=42)
        for x, y in zip(a, b):
            assert np.allclose(x, y)

    def test_different_seeds_may_differ(self):
        lbc, sv, rf, tw, dg, lm, faces = _settlement_defensibility_inputs()
        d1, w1, _ = compute_settlement_defensibility(
            faces, lbc, sv, rf, tw, dg, lm, base_seed=1, wall_base_prob=0.5,
        )
        d2, w2, _ = compute_settlement_defensibility(
            faces, lbc, sv, rf, tw, dg, lm, base_seed=99, wall_base_prob=0.5,
        )
        # Different seeds can produce different wall assignments.
        # At least one outcome (has_walls) or defensibility should differ.
        assert not (np.allclose(d1, d2) and np.array_equal(w1, w2))

    def test_danger_wealth_boost_wall_probability(self):
        # With zero boost, wall_base_prob=0 → no walls.
        n = 10
        faces = [2, 5]
        lbc = np.full(n, 2.0)
        sv = np.zeros(n)
        rf = np.zeros(n)
        tw = np.array([500.0, 500.0])
        dg = np.full(n, 0.8)
        lm = np.ones(n, dtype=bool)
        _, walls_none, _ = compute_settlement_defensibility(
            faces, lbc, sv, rf, tw, dg, lm, base_seed=7,
            wall_base_prob=0.0, wall_danger_boost=0.0, wall_wealth_boost=0.0,
        )
        assert not walls_none.any()
        # With full boost probability clamped to 1.0 → always walls.
        _, walls_all, _ = compute_settlement_defensibility(
            faces, lbc, sv, rf, tw, dg, lm, base_seed=7,
            wall_base_prob=0.0, wall_danger_boost=1.0, wall_wealth_boost=1.0,
        )
        assert walls_all.all()


class TestScalars:
    def test_deterministic_for_same_seed_and_faces(self):
        faces = [3, 9, 17]
        pn = np.array([0.5, 0.8, 0.2])
        tn = np.array([0.1, 0.9, 0.3])
        a = roll_polity_scalars(faces, pn, tn, base_seed=42)
        b = roll_polity_scalars(faces, pn, tn, base_seed=42)
        for x, y in zip(a, b):
            assert np.allclose(x, y)

    def test_keyed_on_face_not_order(self):
        # A capital's scalars depend only on its face index, not list position.
        pn = np.array([0.5, 0.5])
        tn = np.array([0.5, 0.5])
        a = roll_polity_scalars([3, 9], pn, tn, base_seed=7)
        b = roll_polity_scalars([9, 3], pn, tn, base_seed=7)
        for xi, yi in zip(a, b):  # face 3 is index 0 in a, index 1 in b → same scalars
            assert xi[0] == pytest.approx(yi[1])
            assert xi[1] == pytest.approx(yi[0])

    def test_ranges_respected(self):
        faces = list(range(50))
        pn = np.linspace(0, 1, 50)
        tn = np.linspace(0, 1, 50)
        cohesion, infra, legit, maritime = roll_polity_scalars(
            faces, pn, tn, base_seed=1,
            cohesion_min=0.4, infrastructure_min=0.4, legitimacy_spread=0.3, maritime_scale=1.5,
        )
        assert (cohesion >= 0.4).all() and (cohesion <= 1.0).all()
        assert (infra >= 0.4).all() and (infra <= 1.0).all()
        assert (legit >= 0.7).all() and (legit <= 1.3).all()
        assert (maritime >= 0.0).all() and (maritime <= 1.5).all()

    def test_cohesion_gain_monotone(self):
        g = cohesion_to_gain(np.array([0.0, 0.5, 1.0]), gain_floor=0.5, gain_span=1.0)
        assert g[0] == pytest.approx(0.5)
        assert g[2] == pytest.approx(1.5)
        assert g[0] < g[1] < g[2]
