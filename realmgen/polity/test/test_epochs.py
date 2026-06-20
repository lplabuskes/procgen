"""Unit tests for realmgen.polity.epochs — Stage 5 fragmentation.

Pure selection logic is exercised over hand-built ``Polity`` forests; a small
line-graph smoke test confirms the runner threads through ``ForestBuilder``.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.sparse import csr_matrix

from realmgen.polity.builder import PolityParams
from realmgen.polity.epochs import (
    SADDLE_SPLIT,
    VASSAL_SECESSION,
    FragmentationParams,
    _breakaway_seat,
    run_fragmentation_epochs,
    select_fragmentations,
)
from realmgen.polity.forest import ForestBuilder, _Ctx
from realmgen.polity.types import Polity


def _seats(events):
    return [ev.breakaway_capital_face for ev in events]


def _pol(
    pid: int, face: int, *, cohesion: float = 1.0, liege: Optional[int] = None,
    autonomy: float = 0.0, vassals: Optional[List[int]] = None,
    members: Optional[List[int]] = None,
) -> Polity:
    return Polity(
        polity_id=pid, capital_face=face, capital_position=np.zeros(3),
        cohesion=cohesion, liege=liege, autonomy=autonomy,
        vassals=list(vassals or []),
        member_settlement_faces=list(members if members is not None else [face]),
    )


def _domain(of_face: dict, n: int) -> np.ndarray:
    """Sovereign-of-face array (faces not in ``of_face`` are unclaimed)."""
    sof = np.full(n, -1, dtype=np.int32)
    for f, root in of_face.items():
        sof[f] = root
    return sof


class TestBreakawaySeat:
    def test_prefers_weakest_vassal(self):
        by_id = {
            0: _pol(0, 0, vassals=[1, 2]),
            1: _pol(1, 10, liege=0, autonomy=0.2),
            2: _pol(2, 20, liege=0, autonomy=0.8),  # weakest held
        }
        assert _breakaway_seat(by_id[0], by_id, np.zeros(64)) == (20, VASSAL_SECESSION)

    def test_saddle_when_no_vassals(self):
        pol = _pol(0, 0, members=[0, 5, 6, 7])
        auth = np.zeros(8)
        auth[5], auth[6], auth[7] = 0.9, 0.1, 0.5  # face 6 is the saddle
        assert _breakaway_seat(pol, {0: pol}, auth) == (6, SADDLE_SPLIT)

    def test_none_when_capital_only(self):
        pol = _pol(0, 0, members=[0])
        assert _breakaway_seat(pol, {0: pol}, np.zeros(8)) is None


class TestSelectFragmentations:
    def test_tight_small_realm_holds(self):
        pol = _pol(0, 0, cohesion=0.9, members=[0, 1, 2])
        sof = _domain({0: 0, 1: 0, 2: 0}, 8)
        events = select_fragmentations(
            [pol], sof, np.zeros(8), base_seed=1, epoch=0,
            shock_scale=0.4, size_reference=400.0,
        )
        assert events == []

    def test_overextended_low_cohesion_sheds_weakest_vassal(self):
        pols = [
            _pol(0, 0, cohesion=0.5, vassals=[1, 2]),
            _pol(1, 10, liege=0, autonomy=0.2),
            _pol(2, 20, liege=0, autonomy=0.8),
        ]
        sof = _domain({f: 0 for f in range(500)}, 500)  # huge domain → demand = 1.0
        events = select_fragmentations(
            pols, sof, np.zeros(500), base_seed=1, epoch=0,
            shock_scale=0.4, size_reference=400.0,
        )
        assert _seats(events) == [20]
        ev = events[0]
        assert ev.kind == VASSAL_SECESSION
        assert ev.parent_capital_face == 0
        assert ev.parent_domain_faces == 500
        assert ev.demand == 1.0
        assert ev.parent_cohesion == 0.5

    def test_overextended_no_vassals_splits_at_saddle(self):
        pol = _pol(0, 0, cohesion=0.4, members=[0, 5, 6, 7])
        sof = _domain({f: 0 for f in range(500)}, 500)
        auth = np.zeros(500)
        auth[5], auth[6], auth[7] = 0.9, 0.1, 0.5
        events = select_fragmentations(
            [pol], sof, auth, base_seed=1, epoch=0,
            shock_scale=0.4, size_reference=400.0,
        )
        assert _seats(events) == [6]
        assert events[0].kind == SADDLE_SPLIT

    def test_vassals_are_not_iterated(self):
        # A lone vassal (liege set) never fragments on its own account.
        pols = [_pol(0, 0, cohesion=0.95, members=[0]), _pol(1, 9, liege=0, autonomy=0.9)]
        sof = _domain({0: 0}, 16)
        events = select_fragmentations(
            pols, sof, np.zeros(16), base_seed=1, epoch=0,
            shock_scale=0.4, size_reference=400.0,
        )
        assert events == []

    def test_shock_scale_zero_is_deterministic_threshold(self):
        # With no shock, fragment iff demand > cohesion (strictly).
        hold = _pol(0, 0, cohesion=0.6, members=[0, 1])
        sof_hold = _domain({f: 0 for f in range(200)}, 200)   # demand 0.5 < 0.6
        assert select_fragmentations(
            [hold], sof_hold, np.zeros(200), base_seed=2, epoch=0,
            shock_scale=0.0, size_reference=400.0,
        ) == []

        shed = _pol(0, 0, cohesion=0.6, vassals=[1])
        shed_v = _pol(1, 30, liege=0, autonomy=0.5)
        sof_shed = _domain({f: 0 for f in range(280)}, 280)   # demand 0.7 > 0.6
        assert _seats(select_fragmentations(
            [shed, shed_v], sof_shed, np.zeros(280), base_seed=2, epoch=0,
            shock_scale=0.0, size_reference=400.0,
        )) == [30]

    def test_deterministic(self):
        pols = [
            _pol(0, 0, cohesion=0.6, vassals=[1]),
            _pol(1, 10, liege=0, autonomy=0.5),
        ]
        sof = _domain({f: 0 for f in range(300)}, 300)
        auth = np.zeros(300)
        a = select_fragmentations(pols, sof, auth, 7, 1, shock_scale=0.4, size_reference=400.0)
        b = select_fragmentations(pols, sof, auth, 7, 1, shock_scale=0.4, size_reference=400.0)
        assert _seats(a) == _seats(b)

    def test_empty_forest(self):
        assert select_fragmentations(
            [], np.full(4, -1, dtype=np.int32), np.zeros(4),
            base_seed=0, epoch=0, shock_scale=0.4, size_reference=400.0,
        ) == []


def _line_graph(n: int, cost: float = 1.0) -> csr_matrix:
    rows, cols, data = [], [], []
    for i in range(n - 1):
        rows += [i, i + 1]
        cols += [i + 1, i]
        data += [cost, cost]
    return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)


def _line_ctx(n: int = 6) -> _Ctx:
    settlement_faces = np.array([0, n - 1], dtype=np.int64)
    populations = np.array([100.0, 50.0], dtype=np.float64)
    return _Ctx(
        n=n, centroids=np.zeros((n, 3), dtype=np.float64),
        settlement_faces=settlement_faces, populations=populations,
        pop_norm=populations / populations.max(), trade_norm=np.zeros(2),
        trade_wealth=np.zeros(2), trade_attribution=np.zeros((0, 0)),
        defensibility=np.ones(n), land_graph=_line_graph(n), hybrid_graph=None,
        base_seed=0, p=PolityParams(),
        face_to_settlement={0: 0, n - 1: 1},
    )


class TestRunner:
    def test_zero_epochs_matches_plain_build(self):
        ctx = _line_ctx()
        land = np.ones(ctx.n, dtype=bool)
        params = FragmentationParams()
        params.n_epochs = 0

        out, history = run_fragmentation_epochs(ctx, [0], land, lambda *_: None, params=params)
        ref = ForestBuilder(ctx)
        ref.build([0], land, lambda *_: None)
        assert np.array_equal(out.polity_of_face, ref.polity_of_face)
        assert history == []

    def test_runs_with_epochs(self):
        ctx = _line_ctx()
        land = np.ones(ctx.n, dtype=bool)
        params = FragmentationParams()
        params.n_epochs = 2
        out, history = run_fragmentation_epochs(ctx, [0], land, lambda *_: None, params=params)
        # Every land face is claimed by some sovereign and the result is a valid forest.
        assert out.polity_of_face.shape == (ctx.n,)
        assert all(p is not None for p in out.polities)
        # Any recorded secession references a real settlement face and is ordered by epoch.
        faces = set(ctx.face_to_settlement)
        assert all(ev.breakaway_capital_face in faces for ev in history)
        assert [ev.epoch for ev in history] == sorted(ev.epoch for ev in history)

    def test_history_records_a_secession_when_forced(self):
        # Force fragmentation: tiny size_reference + a two-settlement realm so the
        # single sovereign exceeds demand and sheds its peer as a breakaway.
        ctx = _line_ctx()
        land = np.ones(ctx.n, dtype=bool)
        params = FragmentationParams()
        params.n_epochs = 1
        params.shock_scale = 1.0
        params.size_reference = 1.0  # any non-trivial domain demands full cohesion
        _, history = run_fragmentation_epochs(ctx, [0], land, lambda *_: None, params=params)
        assert len(history) >= 1
        ev = history[0]
        assert ev.epoch == 0
        assert ev.breakaway_capital_face in set(ctx.face_to_settlement)
