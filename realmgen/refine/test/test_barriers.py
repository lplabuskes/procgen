"""Unit tests for realmgen.refine.barriers — the river snap of the polity label field."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from core.mesh import Mesh
from realmgen.refine.barriers import (
    RiverBarriers,
    build_river_barriers,
    flow_ramp,
    pack_river_segments,
)
from realmgen.refine.fill import PolityLabelField
from worldgen.cost.surfaces import EARTH_RADIUS_KM


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _equator(lon0=-0.2, lon1=0.2, n=21):
    lon = np.linspace(lon0, lon1, n)
    return np.column_stack([np.cos(lon), np.sin(lon), np.zeros(n)])


def _at(lon, lat):
    return _unit([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def test_pack_drops_lake_and_low_flow_segments():
    pos = _equator(n=4)
    river = (pos, np.array([500.0, 2000.0, 3000.0]), np.array([False, True, False]))
    seg = pack_river_segments([river], min_flow=1000.0)
    assert seg.shape == (1, 7)
    assert seg[0, 6] == 3000.0
    assert np.allclose(seg[0, :3], pos[2]) and np.allclose(seg[0, 3:6], pos[3])


def test_flow_ramp_is_log_between_min_and_full():
    r = flow_ramp(np.array([500.0, 1000.0, np.sqrt(1000.0 * 8000.0), 8000.0, 50000.0]),
                  min_flow=1000.0, full_flow=8000.0)
    assert r.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])


def _barriers(reach=0.05, g=0.6, pos=None):
    pos = _equator() if pos is None else pos
    m = len(pos) - 1
    return RiverBarriers(pos[:-1], pos[1:], np.full(m, reach), snap_fraction=g)


def test_snap_targets_read_out_on_the_points_own_bank():
    reach, g = 0.05, 0.6
    b = _barriers(reach, g)
    pts = np.array([_at(0.01, 0.02), _at(0.01, -0.03), _at(0.01, 0.08)])
    idx, target = b.snap_targets(pts)
    assert idx.tolist() == [0, 1]                       # the third is beyond reach
    lat = np.arcsin(target[:, 2])
    assert lat[0] == pytest.approx(g * reach + (1 - g) * 0.02, abs=1e-4)
    assert lat[1] == pytest.approx(-(g * reach + (1 - g) * 0.03), abs=1e-4)


def test_snap_targets_fan_round_the_river_end():
    b = _barriers(0.05, 1.0)
    p = _at(0.21, 0.01)                                 # just past the downstream end
    idx, target = b.snap_targets(p[None])
    end = _equator()[-1]

    def tangent(v):                                     # direction in the tangent plane at the end
        return _unit(v - (v @ end) * end)

    assert idx.tolist() == [0]
    assert np.allclose(tangent(target[0] - end), tangent(p - end), atol=1e-3)


def test_nearest_river_whose_band_holds_the_point_decides():
    big = _equator()
    small = np.array([_at(lon, 0.02) for lon in np.linspace(-0.2, 0.2, 21)])   # a parallel stream
    seg = np.concatenate([pack_river_segments([(big, np.full(20, 8000.0))]),
                          pack_river_segments([(small, np.full(20, 1500.0))])])
    b = build_river_barriers(seg, reach_km=0.06 * EARTH_RADIUS_KM, min_flow=1000.0,
                             full_flow=8000.0, snap_fraction=1.0)
    assert b is not None
    stream_reach = 0.06 * np.log(1.5) / np.log(8.0)
    pts = np.array([_at(0.0, 0.021), _at(0.0, 0.045)])   # beside the stream; past its band
    _, target = b.snap_targets(pts)
    lat = np.arcsin(target[:, 2])
    assert lat[0] == pytest.approx(0.02 + stream_reach, abs=2e-4)      # read off the stream
    assert lat[1] == pytest.approx(0.06, abs=2e-4)                     # read off the big river


def test_build_river_barriers_off_and_empty():
    seg = pack_river_segments([(_equator(), np.full(20, 5000.0))])
    assert build_river_barriers(seg, reach_km=0.0, min_flow=1000.0, full_flow=8000.0) is None
    assert build_river_barriers(None, reach_km=50.0, min_flow=1000.0, full_flow=8000.0) is None
    assert build_river_barriers(seg, reach_km=50.0, min_flow=9000.0, full_flow=10000.0) is None


# ── label field ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def dual():
    return Mesh.icosphere(2).mesh_dual()


def test_label_field_border_snaps_onto_a_nearby_river(dual):
    n = len(dual.faces)
    c = np.asarray(dual.face_centroids, dtype=np.float64)
    pof = np.where(c[:, 0] > 0, 0, 1)
    # A river on the small circle x = 0.05 (a little east of the polities' x = 0 split).
    phi = np.linspace(0.1, 1.4, 200)
    river = (_unit(np.column_stack([np.full(phi.size, 0.05), np.cos(phi), np.sin(phi)])),
             np.full(phi.size - 1, 10000.0))
    kw: dict[str, Any] = dict(river_reach_km=0.3 * EARTH_RADIUS_KM, river_min_flow=1000.0,
                              river_full_flow=8000.0, river_snap_fraction=1.0)
    plain = PolityLabelField(dual, pof, np.zeros(n, bool), np.ones(n))
    snapped = PolityLabelField(dual, pof, np.zeros(n, bool), np.ones(n), rivers=[river], **kw)

    xs = np.linspace(-0.25, 0.25, 1001)
    path = _unit(np.column_stack([xs, np.full(xs.size, 0.6), np.full(xs.size, 0.5)]))
    lab = snapped.land_labels(path)
    flips = np.flatnonzero(np.diff(lab) != 0)
    assert len(flips) == 1
    assert lab[0] == 1 and lab[-1] == 0
    river_x = 0.05 / np.sqrt(1.0 + 0.05 ** 2)
    assert path[flips[0], 0] == pytest.approx(river_x, abs=2e-3)       # on the river
    # Without the river the line sits elsewhere; swapping it in via with_rivers matches.
    plain_flip = path[np.flatnonzero(np.diff(plain.land_labels(path)) != 0)[0], 0]
    assert abs(plain_flip - river_x) > 5e-3
    via = PolityLabelField(dual, pof, np.zeros(n, bool), np.ones(n), **kw).with_rivers(
        pack_river_segments([river]))
    assert np.array_equal(via.land_labels(path), lab)


def test_river_cuts_a_cell_off_from_its_own_centroid(dual):
    """A river through one face's cell gives its far bank to the polity beyond, though
    no face-level border runs there."""
    n = len(dual.faces)
    c = np.asarray(dual.face_centroids, dtype=np.float64)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    pof = np.where(c[:, 0] > 0, 0, 1)
    f = int(np.argmax(np.where(pof == 0, -np.abs(c[:, 0] - 0.12), -np.inf)))   # polity 0, near the split
    # A river on the small circle x = x_f - 0.06, between face f's centroid and polity 1.
    x_r = float(c[f, 0]) - 0.06
    phi = np.linspace(-3.0, 3.0, 600)
    r = np.sqrt(1 - x_r ** 2)
    river = (np.column_stack([np.full(phi.size, x_r), r * np.cos(phi), r * np.sin(phi)]),
             np.full(phi.size - 1, 10000.0))
    common: dict[str, Any] = dict(river_reach_km=1.0, river_min_flow=1000.0, river_full_flow=8000.0)
    cut = PolityLabelField(dual, pof, np.zeros(n, bool), np.ones(n), rivers=[river],
                           river_cross_weight=3.0, **common)
    uncut = PolityLabelField(dual, pof, np.zeros(n, bool), np.ones(n), rivers=[river],
                             river_cross_weight=0.0, **common)
    # Points of f's own cell, just across the river from its centroid.
    across = _unit(c[f] - np.array([0.075, 0.0, 0.0]))
    assert int(cut.owner(across[None])[0]) == f
    assert uncut.land_labels(across[None])[0] == 0
    assert cut.land_labels(across[None])[0] == 1


def test_pack_carries_world_river_ids():
    pos = _equator(n=4)
    river = (pos, np.array([2000.0, 2000.0, 2000.0]), np.array([False, True, False]))
    seg = pack_river_segments([river], river_ids=[np.array([-1, 5, 5, 5])])
    assert seg.shape == (2, 8)
    assert seg[:, 7].tolist() == [5.0, 5.0]       # a headwater segment joining river 5 takes its id
