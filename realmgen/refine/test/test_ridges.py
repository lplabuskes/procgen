"""Unit tests for realmgen.refine.ridges — drainage divides as ridge polylines."""
from __future__ import annotations

import numpy as np

from realmgen.refine.ridges import _join_at_junctions, _running_mean
from realmgen.refine.river_follow import relief_ramp


def test_relief_ramp_is_linear_between_its_ends():
    r = relief_ramp(np.array([0.0, 0.2, 0.6, 1.0, 3.0]), min_relief=0.2, full_relief=1.0)
    assert np.allclose(r, [0.0, 0.0, 0.5, 1.0, 1.0])


def test_running_mean_keeps_length_and_constants():
    x = np.full(7, 3.0)
    assert np.allclose(_running_mean(x, 5), x)
    assert len(_running_mean(np.arange(3.0), 5)) == 3


def test_junction_pairs_its_two_highest_arms():
    # Junction point 0; arms 1-2-3 (high), 4-5-6 (high), 7-8-9 (low spur).
    relief = np.array([np.nan, 2, 2, 2, 1.5, 1.5, 1.5, 0.1, 0.1, 0.1])
    chains = [[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]]
    lines = _join_at_junctions(chains, frozenset({0}), relief)
    assert sorted(len(line) for line in lines) == [4, 7]
    crest = max(lines, key=len)
    assert set(crest) == {0, 1, 2, 3, 4, 5, 6} and crest.count(0) == 1
    assert crest[3] == 0                                         # through the junction
    assert sorted(min(lines, key=len)) == [0, 7, 8, 9]


def test_closed_loop_stays_one_line():
    lines = _join_at_junctions([[0, 1, 2, 3, 0]], frozenset(), np.ones(4))
    assert lines == [[0, 1, 2, 3, 0]]
