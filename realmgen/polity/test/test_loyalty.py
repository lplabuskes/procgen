"""Unit tests for realmgen.polity.loyalty — the Stage-4 trade-attribution levers.

Each lever is a pure function over the trade-attribution matrix + claim partition;
the last class checks they compose correctly through ``classify_settlements``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from realmgen.polity.authority import classify_settlements
from realmgen.polity.loyalty import (
    compute_internal_trade_fraction,
    economic_grip_multiplier,
    peripheral_wealth_breakaways,
)
from realmgen.polity.types import ControlClass


class TestInternalTradeFraction:
    def test_all_internal(self):
        # Two settlements in the same cell trade only with each other.
        faces = [0, 1]
        partition = np.array([5, 5], dtype=np.int32)        # both in cell 5
        attr = np.array([[0.0, 4.0], [4.0, 0.0]])
        frac = compute_internal_trade_fraction(faces, partition, attr)
        assert frac == pytest.approx([1.0, 1.0])

    def test_all_external(self):
        # Partners sit in different cells → nothing internal.
        faces = [0, 1]
        partition = np.array([0, 1], dtype=np.int32)
        attr = np.array([[0.0, 4.0], [4.0, 0.0]])
        frac = compute_internal_trade_fraction(faces, partition, attr)
        assert frac == pytest.approx([0.0, 0.0])

    def test_mixed_counts_both_directions(self):
        # Settlement 0 (cell 0): trades 3 with 1 (same cell), 1 with 2 (other cell).
        # Bilateral volume uses exports + imports.
        faces = [0, 1, 2]
        partition = np.array([0, 0, 9], dtype=np.int32)
        attr = np.array([
            [0.0, 2.0, 1.0],   # 0 exports: 2→1, 1→2
            [1.0, 0.0, 0.0],   # 1 exports: 1→0
            [0.0, 0.0, 0.0],
        ])
        frac = compute_internal_trade_fraction(faces, partition, attr)
        # Settlement 0: internal with 1 = (2+1)=3; external with 2 = (1+0)=1 → 3/4.
        assert frac[0] == pytest.approx(0.75)
        # Settlement 1: only trades with 0 (same cell) → fully internal.
        assert frac[1] == pytest.approx(1.0)

    def test_unclaimed_cell_has_no_internal(self):
        faces = [0, 1]
        partition = np.array([-1, -1], dtype=np.int32)
        attr = np.array([[0.0, 4.0], [4.0, 0.0]])
        frac = compute_internal_trade_fraction(faces, partition, attr)
        assert frac == pytest.approx([0.0, 0.0])

    def test_no_trade_is_zero(self):
        faces = [0, 1]
        partition = np.array([0, 0], dtype=np.int32)
        frac = compute_internal_trade_fraction(faces, partition, np.zeros((2, 2)))
        assert frac == pytest.approx([0.0, 0.0])

    def test_missing_attribution_returns_zeros(self):
        frac = compute_internal_trade_fraction([0, 1, 2], np.zeros(3), np.zeros((0, 0)))
        assert frac.shape == (3,)
        assert (frac == 0.0).all()


class TestEconomicGripMultiplier:
    def test_scales_with_internal_fraction(self):
        mult = economic_grip_multiplier(np.array([0.0, 0.5, 1.0]), grip_strength=0.6)
        assert mult == pytest.approx([1.0, 1.3, 1.6])

    def test_zero_strength_is_identity(self):
        mult = economic_grip_multiplier(np.array([0.0, 0.5, 1.0]), grip_strength=0.0)
        assert mult == pytest.approx([1.0, 1.0, 1.0])


class TestPeripheralWealthBreakaways:
    def _base(self) -> dict[str, Any]:
        # Cell 0: capital = settlement 0 (wealth 10), subject = settlement 1.
        return dict(
            settlement_faces=[0, 1],
            partition=np.array([0, 0], dtype=np.int32),
            capital_sidx=[0],
        )

    def test_rich_outward_subject_breaks_away(self):
        kw = self._base()
        out = peripheral_wealth_breakaways(
            **kw, trade_wealth=np.array([10.0, 30.0]),     # subject 3× richer
            internal_fraction=np.array([1.0, 0.1]),        # subject mostly external
            wealth_ratio=2.0, external_floor=0.5,
        )
        assert out[1] and not out[0]                       # subject flagged, capital never

    def test_internal_subject_holds(self):
        kw = self._base()
        out = peripheral_wealth_breakaways(
            **kw, trade_wealth=np.array([10.0, 30.0]),
            internal_fraction=np.array([1.0, 0.9]),         # trade faces inward
            wealth_ratio=2.0, external_floor=0.5,
        )
        assert not out.any()

    def test_poor_subject_holds(self):
        kw = self._base()
        out = peripheral_wealth_breakaways(
            **kw, trade_wealth=np.array([10.0, 12.0]),       # only 1.2× the capital
            internal_fraction=np.array([1.0, 0.1]),
            wealth_ratio=2.0, external_floor=0.5,
        )
        assert not out.any()

    def test_capital_never_flagged(self):
        # With wealth_ratio <= 1 a capital self-qualifies (tw >= 1·tw, outward);
        # the is_capital guard must still exempt it. Cell 0 is settlement 0's seat.
        out = peripheral_wealth_breakaways(
            settlement_faces=[0, 1], partition=np.array([0, 0], dtype=np.int32),
            capital_sidx=[0], trade_wealth=np.array([100.0, 5.0]),
            internal_fraction=np.array([0.0, 1.0]),
            wealth_ratio=1.0, external_floor=0.5,
        )
        assert not out[0]

    def test_unclaimed_not_flagged(self):
        out = peripheral_wealth_breakaways(
            settlement_faces=[0, 1], partition=np.array([-1, -1], dtype=np.int32),
            capital_sidx=[0], trade_wealth=np.array([10.0, 100.0]),
            internal_fraction=np.array([0.0, 0.0]),
            wealth_ratio=2.0, external_floor=0.5,
        )
        assert not out.any()

    def test_disabled_when_ratio_nonpositive(self):
        kw = self._base()
        out = peripheral_wealth_breakaways(
            **kw, trade_wealth=np.array([10.0, 100.0]),
            internal_fraction=np.array([1.0, 0.0]),
            wealth_ratio=0.0, external_floor=0.5,
        )
        assert not out.any()


class TestLeversThroughClassify:
    """Both levers reach the direct/vassal/independent label via classify_settlements."""

    def _kw(self, field_val, pop, defn, domain) -> dict[str, Any]:
        # One polity (id 0): capital at face 0, subject at face 1.
        return dict(
            settlement_faces=[0, 1],
            populations=np.array([pop[0], pop[1]]),
            polity_of_face=np.array([0, 0], dtype=np.int32),
            field=np.array([[1.0, field_val]]),
            projecting_power=np.array([domain]),
            defensibility=np.array([1.0, defn]),
            authority_gain=np.array([1.0]),
            is_capital=np.array([True, False]),
        )

    def test_economic_grip_rescues_from_independence(self):
        # Bare ratio r = 10·0.04·1 / (5·1) = 0.08 < vassal_floor (0.1) → INDEPENDENT.
        kw = self._kw(field_val=0.04, pop=(50.0, 5.0), defn=1.0, domain=10.0)
        base, _ = classify_settlements(**kw, direct_threshold=3.0, vassal_floor=0.1)
        assert base[1] == int(ControlClass.INDEPENDENT)
        # A strong intra-polity grip (×2) lifts r to 0.16, above the vassal floor.
        gripped, ratio = classify_settlements(
            **kw, direct_threshold=3.0, vassal_floor=0.1,
            loyalty_mult=np.array([1.0, 2.0]),
        )
        assert ratio[1] == pytest.approx(0.16)
        assert gripped[1] == int(ControlClass.VASSAL)

    def test_peripheral_wealth_forces_independence(self):
        # Bare ratio r = 100·0.8·1 / (5·1) = 16 ≥ 3 → would be DIRECT.
        kw = self._kw(field_val=0.8, pop=(50.0, 5.0), defn=1.0, domain=100.0)
        base, _ = classify_settlements(**kw, direct_threshold=3.0, vassal_floor=0.6)
        assert base[1] == int(ControlClass.DIRECT)
        forced, _ = classify_settlements(
            **kw, direct_threshold=3.0, vassal_floor=0.6,
            force_independent=np.array([False, True]),
        )
        assert forced[1] == int(ControlClass.INDEPENDENT)

    def test_force_independent_cannot_dislodge_capital(self):
        kw = self._kw(field_val=0.8, pop=(50.0, 5.0), defn=1.0, domain=100.0)
        forced, _ = classify_settlements(
            **kw, direct_threshold=3.0, vassal_floor=0.6,
            force_independent=np.array([True, False]),         # try to evict the capital
        )
        assert forced[0] == int(ControlClass.DIRECT)
