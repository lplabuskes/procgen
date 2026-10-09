"""Unit tests for realmgen.polity.authority.merge_exclaves."""
from __future__ import annotations

import numpy as np

from realmgen.polity.authority import land_edges, merge_exclaves


def _line_adjacency(n: int) -> list:
    return [[j for j in (i - 1, i + 1) if 0 <= j < n] for i in range(n)]


def _setup():
    # Land line 0–6 plus face 7, an island (no land neighbours).
    adj = _line_adjacency(7) + [[]]
    land = np.ones(8, dtype=bool)
    edges = land_edges(adj, land)
    caps = [0, 3]                                      # polity 0 at face 0, polity 1 at face 3
    assign = np.array([0, 0, 1, 1, 1, 0, 1, 0], dtype=np.int32)
    field = np.ones((2, 8))
    return adj, edges, caps, assign, field


def test_merge_exclaves_hands_a_cut_off_piece_to_its_neighbour():
    _, edges, caps, assign, field = _setup()
    out = merge_exclaves(assign, caps, field, edges, max_faces=10)
    assert out[5] == 1                                 # polity 0's piece inside polity 1
    assert out[:5].tolist() == [0, 0, 1, 1, 1]
    assert out[7] == 0                                 # an island has no land rim: kept


def test_merge_exclaves_takes_the_strongest_rim_claimant():
    # 0 1 | 2 | 3 4 5 : polity 2's face 2 is cut off from its seat (5) between 0 and 1.
    adj = _line_adjacency(6)
    edges = land_edges(adj, np.ones(6, dtype=bool))
    assign = np.array([0, 0, 2, 1, 1, 2], dtype=np.int32)
    caps = [0, 4, 5]
    for strong in (0, 1):
        field = np.ones((3, 6))
        field[strong, 2] = 5.0
        assert merge_exclaves(assign, caps, field, edges, max_faces=10)[2] == strong


def test_merge_exclaves_respects_size_cap_and_eligibility():
    _, edges, caps, assign, field = _setup()
    assert np.array_equal(merge_exclaves(assign, caps, field, edges, max_faces=0), assign)
    eligible = np.ones(8, dtype=bool)
    eligible[[4, 6]] = False                           # the rim claimant is outside the territory
    out = merge_exclaves(assign, caps, field, edges, max_faces=10, eligible_mask=eligible)
    assert out[5] == 0
