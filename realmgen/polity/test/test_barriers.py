"""Unit tests for realmgen.polity.barriers — face drainage, divides, ridge costs, exclave merging."""
from __future__ import annotations

import numpy as np
import pytest

from realmgen.polity.authority import land_edges, merge_exclaves
from realmgen.polity.barriers import BarrierParams, add_ridge_costs, divide_relief, face_drainage


def _line_adjacency(n: int) -> list:
    return [[j for j in (i - 1, i + 1) if 0 <= j < n] for i in range(n)]


def _edge_value(adj, per_edge, i, j) -> float:
    return float(per_edge[i][list(adj[i]).index(j)])


# ── drainage / divides ──────────────────────────────────────────────────────


def test_face_drainage_runs_downhill_to_the_water():
    adj = _line_adjacency(5)
    elev = np.array([-1.0, 0.2, 0.4, 0.6, 0.8])
    water = np.array([True, False, False, False, False])
    rec = face_drainage(adj, elev, water)
    assert rec.tolist() == [-1, 0, 1, 2, 3]


def test_face_drainage_through_a_pit_and_a_waterless_landmass():
    adj = _line_adjacency(4) + [[5], [4]]            # faces 4–5: an island with no water
    elev = np.array([-1.0, 0.5, 0.1, 0.6, 0.3, 0.2])
    water = np.array([True, False, False, False, False, False])
    rec = face_drainage(adj, elev, water)
    assert rec[2] == 1                                # the pit spills over its rim
    assert rec[5] == -1 and rec[4] == 5               # the island drains to its lowest face


def _ridge():
    """Sea — slope — peak — slope — sea: one divide, at the peak."""
    adj = _line_adjacency(9)
    elev = np.array([-1.0, 0.2, 0.6, 1.2, 2.0, 1.1, 0.5, 0.1, -1.0])
    water = np.zeros(9, dtype=bool)
    water[[0, 8]] = True
    return adj, elev, water


def test_divide_found_at_the_crest_with_its_relief():
    adj, elev, water = _ridge()
    rec = face_drainage(adj, elev, water)
    relief = divide_relief(adj, elev, water, min_detour=6, floor_hops=2)
    # The peak drains one way; the divide is on its edge to the other side.
    other = 5 if rec[4] == 3 else 3
    r = _edge_value(adj, relief, 4, other)
    floor_peak = elev[rec[rec[4]]]
    floor_other = elev[rec[rec[other]]]
    assert r == pytest.approx(min(elev[4], elev[other]) - max(floor_peak, floor_other))
    assert r > 0.0
    # Symmetric, and nothing on the slopes or at the coast.
    assert _edge_value(adj, relief, other, 4) == pytest.approx(r)
    for i, j in [(1, 2), (2, 3), (5, 6), (6, 7), (0, 1)]:
        assert _edge_value(adj, relief, i, j) == 0.0


def test_parallel_slopes_are_not_a_divide():
    # A 2-wide ramp both rows of which drain to the same sea face.
    #   row a: 1 2 3 ; row b: 4 5 6 ; 0 = sea beside 1 and 4.
    adj = [[1, 4], [0, 2, 4, 5], [1, 3, 5, 6], [2, 6], [0, 1, 5], [4, 6, 1, 2], [5, 3, 2]]
    elev = np.array([-1.0, 0.2, 0.5, 0.9, 0.21, 0.51, 0.91])
    water = np.array([True] + [False] * 6)
    relief = divide_relief(adj, elev, water, min_detour=4, floor_hops=1)
    assert all(float(np.max(r, initial=0.0)) == 0.0 for r in relief)


def test_add_ridge_costs_saturates_at_the_reference_relief():
    costs = [np.array([1.0, 1.0]), np.array([1.0])]
    relief = [np.array([0.25, 3.0]), np.array([0.0])]
    out = add_ridge_costs(costs, relief, BarrierParams(ridge_weight=8.0, ridge_relief_ref=0.5))
    assert out[0].tolist() == pytest.approx([1.0 + 4.0, 1.0 + 8.0])
    assert out[1].tolist() == [1.0]


# ── exclave merging ─────────────────────────────────────────────────────────


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


# ── river crossings ─────────────────────────────────────────────────────────


def test_river_crossing_strength_marks_edges_whose_chord_crosses():
    from core.mesh import Mesh
    from realmgen.polity.barriers import river_crossing_strength
    dual = Mesh.icosphere(2).mesh_dual()
    adj = dual.face_adjacency
    c = np.asarray(dual.face_centroids, dtype=np.float64)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    # A big river along the equator (z = 0), and a stream below the threshold along z = 0.5.
    lon = np.linspace(-np.pi, np.pi, 400)
    big = (np.column_stack([np.cos(lon), np.sin(lon), np.zeros(lon.size)]), np.full(lon.size - 1, 5000.0))
    ring = np.column_stack([np.cos(lon), np.sin(lon), np.full(lon.size, 1.0)])
    small = (ring / np.linalg.norm(ring, axis=1, keepdims=True), np.full(lon.size - 1, 100.0))
    out = river_crossing_strength(adj, c, [big, small], min_flow=300.0, full_flow=4000.0)
    for i, nb in enumerate(adj):
        for k, j in enumerate(nb):
            crosses = c[i, 2] * c[j, 2] < 0.0
            assert (out[i][k] == 1.0) == crosses
            assert out[i][k] == out[j][list(adj[j]).index(i)]


def test_add_barrier_costs_combines_ridge_and_river():
    from realmgen.polity.barriers import add_barrier_costs
    costs = [np.array([1.0, 1.0])]
    out = add_barrier_costs(costs, [np.array([0.5, 0.0])], [np.array([0.0, 0.5])],
                            BarrierParams(ridge_weight=8.0, ridge_relief_ref=0.5, river_weight=6.0))
    assert out[0].tolist() == pytest.approx([9.0, 4.0])
