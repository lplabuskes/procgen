"""Unit tests for realmgen.refine — pure-function / stub-mesh tests.

These avoid a full worldgen build; integration is exercised by the pipeline
smoke test elsewhere.
"""
from __future__ import annotations

import numpy as np

from realmgen.placement.types import Settlement
from realmgen.refine.borders import _weighted_voronoi_offset
from realmgen.refine.settlements import (
    _neighbor_any,
    position_key,
    refine_settlement_positions,
    remap_routes,
)


# ── Stub mesh ──────────────────────────────────────────────────────────────────


class _StubMesh:
    """One 4×4-texel face whose texel (i, j) sits at [j, i, 0]."""

    def __init__(self, centroid):
        self.face_centroids = np.array([centroid], dtype=np.float64)

    def face_texture_positions(self, h, w):
        pos = np.zeros((1, h, w, 3), dtype=np.float64)
        for i in range(h):
            for j in range(w):
                pos[0, i, j] = [j, i, 0.0]
        return pos


def _settlement(face_id=0):
    return Settlement(face_id=face_id, position=np.array([10.0, 10.0, 10.0]),
                      tier=0, score=1.0, population=1000.0)


# ── _neighbor_any ────────────────────────────────────────────────────────────


def test_neighbor_any_orthogonal_only():
    m = np.zeros((3, 3), dtype=bool)
    m[1, 1] = True
    out = _neighbor_any(m)
    assert out[0, 1] and out[2, 1] and out[1, 0] and out[1, 2]
    assert not out[0, 0] and not out[1, 1]  # diagonals / self excluded


# ── settlement snapping ──────────────────────────────────────────────────────


def test_snap_to_ocean_coast():
    # Columns 2-3 are ocean → land cols 0-1; coast = col 1 (adjacent to ocean).
    mesh = _StubMesh(centroid=[1.0, 1.5, 0.0])
    ocean = np.zeros((1, 4, 4, 1), dtype=np.float64)
    ocean[0, :, 2:, 0] = 1.0
    s = _settlement()
    moved = refine_settlement_positions(
        [s], mesh=mesh, ocean_tex=ocean, lake_tex=None,
        fertility=np.array([1.0]), rivers=None,
    )
    assert s.position[0] == 1.0          # snapped onto the coast column
    assert position_key(np.array([10.0, 10.0, 10.0])) in moved


def test_ocean_preferred_over_lake():
    # Col 0 borders a lake (col... actually make col 3 ocean, col 0 lake-side).
    mesh = _StubMesh(centroid=[1.5, 1.5, 0.0])
    ocean = np.zeros((1, 4, 4, 1), dtype=np.float64)
    ocean[0, :, 3, 0] = 1.0              # ocean on col 3 → coast col 2
    lake = np.zeros((1, 4, 4, 1), dtype=np.float64)
    lake[0, :, 0, 0] = 1.0               # lake on col 0 → coast col 1
    s = _settlement()
    refine_settlement_positions(
        [s], mesh=mesh, ocean_tex=ocean, lake_tex=lake,
        fertility=np.array([1.0]), rivers=None,
    )
    assert s.position[0] == 2.0          # ocean coast wins over lake coast


def test_snap_to_river_when_no_coast():
    mesh = _StubMesh(centroid=[0.0, 0.0, 0.0])
    ocean = np.zeros((1, 4, 4, 1), dtype=np.float64)  # all land
    river_pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    rivers = [(river_pts, np.array([1.0]), None)]
    s = _settlement()
    refine_settlement_positions(
        [s], mesh=mesh, ocean_tex=ocean, lake_tex=None,
        fertility=np.array([1.0]), rivers=rivers,
    )
    assert np.allclose(s.position, [0.0, 0.0, 0.0])   # snapped onto the river texel


# ── route remapping ──────────────────────────────────────────────────────────


def test_remap_routes_moves_matching_endpoints():
    old = np.array([1.0, 2.0, 3.0])
    new = np.array([1.1, 2.1, 3.1])
    other = np.array([9.0, 9.0, 9.0])
    moved = {position_key(old): new}
    routes = [(old.copy(), other.copy(), 5.0), (other.copy(), old.copy(), 2.0)]
    out = remap_routes(routes, moved)
    assert np.allclose(out[0][0], new) and np.allclose(out[0][1], other)
    assert np.allclose(out[1][0], other) and np.allclose(out[1][1], new)
    assert out[0][2] == 5.0


# ── weighted-Voronoi border offset ───────────────────────────────────────────


def test_voronoi_offset_equal_control_is_zero():
    a = np.array([-1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    s = np.array([0.0, 0.0, 1.0])           # equidistant from a and b
    d = np.array([1.0, 0.0, 0.0])
    lam = _weighted_voronoi_offset(s, d, a, b, 1.0, 1.0, max_off=10.0)
    assert abs(lam) < 1e-9


def test_voronoi_offset_biases_toward_weaker():
    a = np.array([-1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    s = np.array([0.0, 0.0, 1.0])
    d = np.array([1.0, 0.0, 0.0])
    # a stronger (w_a > w_b): boundary pushed toward b (positive λ along d).
    lam = _weighted_voronoi_offset(s, d, a, b, 2.0, 1.0, max_off=10.0)
    assert lam > 0.0


def test_voronoi_offset_clamped():
    a = np.array([-1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    s = np.array([0.0, 0.0, 1.0])
    d = np.array([1.0, 0.0, 0.0])
    # Unclamped this root is ≈0.785 (see bias test); max_off=0.5 clamps it.
    lam = _weighted_voronoi_offset(s, d, a, b, 2.0, 1.0, max_off=0.5)
    assert lam == 0.5
