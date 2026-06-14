"""Unit tests for realmgen.polity.authority — projection, claiming, classify.

Lightweight stub graphs, no worldgen run.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from realmgen.polity.authority import (
    claim_faces,
    classify_settlements,
    compute_authority_fields,
)
from realmgen.polity.types import ControlClass


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


class TestAuthorityFields:
    def test_decays_with_distance(self):
        g = _line_graph(5, cost=1.0)
        field = compute_authority_fields(
            5, [0], np.array([1.0]), np.array([1.0]), np.array([0.0]), g,
            base_range=2.0, range_floor=0.5,
        )
        assert field.shape == (1, 5)
        assert field[0, 0] == pytest.approx(1.0)            # legitimacy at the seat
        assert field[0, 0] > field[0, 1] > field[0, 4]      # monotone falloff

    def test_cheap_terrain_extends_field(self):
        cheap = _line_graph(5, cost=1.0)
        dear = _line_graph(5, cost=5.0)
        kw: dict[str, Any] = dict(base_range=3.0, range_floor=0.5)
        f_cheap = compute_authority_fields(5, [0], np.array([1.0]), np.array([1.0]), np.array([0.0]), cheap, **kw)
        f_dear = compute_authority_fields(5, [0], np.array([1.0]), np.array([1.0]), np.array([0.0]), dear, **kw)
        assert f_cheap[0, 3] > f_dear[0, 3]                 # barrier pinches the field

    def test_infrastructure_widens_range(self):
        g = _line_graph(6, cost=1.0)
        lo = compute_authority_fields(6, [0], np.array([1.0]), np.array([0.4]), np.array([0.0]), g, base_range=3.0)
        hi = compute_authority_fields(6, [0], np.array([1.0]), np.array([1.0]), np.array([0.0]), g, base_range=3.0)
        assert hi[0, 4] > lo[0, 4]                          # more infra → reaches further

    def test_legitimacy_scales_magnitude(self):
        g = _line_graph(4, cost=1.0)
        lo = compute_authority_fields(4, [0], np.array([0.8]), np.array([1.0]), np.array([0.0]), g)
        hi = compute_authority_fields(4, [0], np.array([1.2]), np.array([1.0]), np.array([0.0]), g)
        assert hi[0, 0] == pytest.approx(1.2)
        assert (hi[0] > lo[0]).all()

    def test_sea_channel_beads_coastal_capital(self):
        # Land path 0→2 is expensive (20); a cheap sea route (≈3) bridges them.
        n = 3
        land = csr_matrix(([10.0, 10.0, 10.0, 10.0], ([0, 1, 1, 2], [1, 0, 2, 1])),
                          shape=(n, n), dtype=np.float64)
        # Hybrid (2N×2N): cheap sea 3—4—5, ports land0↔sea3 and land2↔sea5.
        r = [0, 1, 1, 2, 3, 4, 4, 5, 0, 3, 2, 5]
        c = [1, 0, 2, 1, 4, 3, 5, 4, 3, 0, 5, 2]
        d = [10, 10, 10, 10, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5]
        hybrid = csr_matrix((d, (r, c)), shape=(2 * n, 2 * n), dtype=np.float64)
        kw = dict(base_range=5.0, range_floor=0.5, sea_range_factor=3.0)
        land_only = compute_authority_fields(n, [0], np.array([1.0]), np.array([1.0]), np.array([0.0]),
                                             land, hybrid, **kw)
        sea_rich = compute_authority_fields(n, [0], np.array([1.0]), np.array([1.0]), np.array([1.0]),
                                            land, hybrid, **kw)
        assert sea_rich[0, 2] > land_only[0, 2] + 0.3       # maritime reach lights up the far coast

    def test_sea_weight_scales_down_the_sea_channel(self):
        # Same coastal bead as above, but a sub-1 sea weight must shrink the maritime
        # contribution at the far coast (where the sea channel decides the value).
        n = 3
        land = csr_matrix(([10.0, 10.0, 10.0, 10.0], ([0, 1, 1, 2], [1, 0, 2, 1])),
                          shape=(n, n), dtype=np.float64)
        r = [0, 1, 1, 2, 3, 4, 4, 5, 0, 3, 2, 5]
        c = [1, 0, 2, 1, 4, 3, 5, 4, 3, 0, 5, 2]
        d = [10, 10, 10, 10, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5]
        hybrid = csr_matrix((d, (r, c)), shape=(2 * n, 2 * n), dtype=np.float64)
        kw = dict(base_range=5.0, range_floor=0.5, sea_range_factor=3.0)
        full = compute_authority_fields(n, [0], np.array([1.0]), np.array([1.0]), np.array([1.0]),
                                        land, hybrid, sea_weight=1.0, **kw)
        half = compute_authority_fields(n, [0], np.array([1.0]), np.array([1.0]), np.array([1.0]),
                                        land, hybrid, sea_weight=0.5, **kw)
        assert half[0, 2] < full[0, 2]
        assert half[0, 2] == pytest.approx(0.5 * full[0, 2], rel=1e-6)


class TestClaimFaces:
    def _two_capital_field(self, n=5):
        # Capital 0 strongest on the left, capital 1 strongest on the right.
        g = _line_graph(n, cost=1.0)
        return compute_authority_fields(
            n, [0, n - 1], np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([0.0, 0.0]),
            g, base_range=2.0,
        )

    def test_argmax_partition_and_capitals_own_seat(self):
        field = self._two_capital_field(5)
        assign, _ = claim_faces(field, [0, 4], np.ones(5, dtype=bool), hysteresis=0.0)
        assert assign[0] == 0 and assign[4] == 1
        assert assign[1] == 0 and assign[3] == 1

    def test_water_ineligible_stays_unclaimed(self):
        field = self._two_capital_field(5)
        eligible = np.ones(5, dtype=bool)
        eligible[2] = False                                  # face 2 is water
        assign, _ = claim_faces(field, [0, 4], eligible, hysteresis=0.0)
        assert assign[2] == -1

    def test_hysteresis_keeps_incumbent_on_tie(self):
        # Symmetric field → face 2 is a tie; incumbent prev claimant keeps it.
        field = self._two_capital_field(5)
        prev = np.array([0, 0, 1, 1, 1], dtype=np.int32)     # face 2 currently held by polity 1
        assign, _ = claim_faces(field, [0, 4], np.ones(5, dtype=bool), hysteresis=0.5, prev=prev)
        assert assign[2] == 1                                # stickiness, not the argmin-tie default

    def test_convergence_bounded(self):
        field = self._two_capital_field(7)
        _, n_iters = claim_faces(field, [0, 6], np.ones(7, dtype=bool), hysteresis=0.15, max_iters=8)
        assert 1 <= n_iters <= 8

    def test_deterministic(self):
        field = self._two_capital_field(6)
        a, _ = claim_faces(field, [0, 5], np.ones(6, dtype=bool), hysteresis=0.15)
        b, _ = claim_faces(field, [0, 5], np.ones(6, dtype=bool), hysteresis=0.15)
        assert (a == b).all()

    def test_empty_field(self):
        assign, n = claim_faces(np.zeros((0, 4)), [], np.ones(4, dtype=bool))
        assert (assign == -1).all() and n == 0


class TestClassify:
    def _setup(self, field_val, pop, defn, domain) -> dict[str, Any]:
        # One polity (id 0), two settlements: capital at face 0, subject at face 1.
        # `domain` is the capital's projecting power.
        return dict(
            settlement_faces=[0, 1],
            populations=np.array([pop[0], pop[1]]),
            polity_of_face=np.array([0, 0], dtype=np.int32),
            field=np.array([[1.0, field_val]]),              # field[0, face]
            projecting_power=np.array([domain]),
            defensibility=np.array([1.0, defn]),
            authority_gain=np.array([1.0]),
            is_capital=np.array([True, False]),
        )

    def test_capital_always_direct(self):
        kw = self._setup(field_val=0.0, pop=(10.0, 10.0), defn=10.0, domain=0.1)
        control, _ = classify_settlements(**kw, direct_threshold=3.0, vassal_floor=0.6)
        assert control[0] == int(ControlClass.DIRECT)

    def test_high_ratio_direct(self):
        # eff = domain·field·gain = 100·0.8·1 = 80; loc = 5·1 = 5 → r = 16 ≥ 3.
        kw = self._setup(field_val=0.8, pop=(50.0, 5.0), defn=1.0, domain=100.0)
        control, ratio = classify_settlements(**kw, direct_threshold=3.0, vassal_floor=0.6)
        assert ratio[1] >= 3.0
        assert control[1] == int(ControlClass.DIRECT)

    def test_mid_ratio_vassal(self):
        # eff = 10·0.3·1 = 3; loc = 2·1 = 2 → r = 1.5 ∈ [0.6, 3).
        kw = self._setup(field_val=0.3, pop=(50.0, 2.0), defn=1.0, domain=10.0)
        control, ratio = classify_settlements(**kw, direct_threshold=3.0, vassal_floor=0.6)
        assert 0.6 <= ratio[1] < 3.0
        assert control[1] == int(ControlClass.VASSAL)

    def test_low_ratio_independent(self):
        # eff = 10·0.02·1 = 0.2; loc = 5·2 = 10 → r = 0.02 < 0.6.
        kw = self._setup(field_val=0.02, pop=(50.0, 5.0), defn=2.0, domain=10.0)
        control, ratio = classify_settlements(**kw, direct_threshold=3.0, vassal_floor=0.6)
        assert ratio[1] < 0.6
        assert control[1] == int(ControlClass.INDEPENDENT)

    def test_unclaimed_face_is_independent(self):
        control, _ = classify_settlements(
            [0], np.array([10.0]), np.array([-1], dtype=np.int32),
            np.zeros((1, 1)), np.array([1.0]), np.array([1.0]), np.array([1.0]),
            np.array([False]), direct_threshold=3.0, vassal_floor=0.6,
        )
        assert control[0] == int(ControlClass.INDEPENDENT)
