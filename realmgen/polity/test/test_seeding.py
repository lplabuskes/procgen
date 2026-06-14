"""Unit tests for realmgen.polity.seeding — defensibility, capitals, scalars.

Lightweight stub graphs, no worldgen run.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from realmgen.polity.seeding import (
    cohesion_to_gain,
    compute_defensibility,
    roll_polity_scalars,
    seed_capitals,
)


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
