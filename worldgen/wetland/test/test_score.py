"""Wetness score: drainage, flood heights, terms, stamping."""
from __future__ import annotations

import dataclasses

import numpy as np

from worldgen.climate.classification import WETLAND_BIOME
from worldgen.wetland.builder import WetlandParams, stamp_wetland
from worldgen.wetland.score import (
    combine, drainage_targets, flatness, flood_height_above, flood_height_m, flood_term, rain_term,
)


def _chain(n: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR adjacency of a path graph 0 – 1 – … – n−1."""
    nbrs = [[j for j in (i - 1, i + 1) if 0 <= j < n] for i in range(n)]
    indptr = np.cumsum([0] + [len(x) for x in nbrs]).astype(np.int64)
    return indptr, np.concatenate(nbrs).astype(np.int32)


def test_drainage_reaches_first_channel_or_water():
    # sea | 0 m … rising … channel at 3 … ridge at 6 … falls to a pit at 8.
    ground = np.array([-0.1, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.07, 0.06], dtype=np.float64)
    water = np.zeros(9, dtype=bool)
    water[0] = True
    channel = np.zeros(9, dtype=bool)
    channel[3] = True
    indptr, flat = _chain(9)
    t = drainage_targets(ground, water, channel, indptr, flat)
    assert t[0] == 0 and t[1] == 0 and t[2] == 0      # straight to the sea
    assert t[3] == 3                                  # a channel is its own drainage
    assert list(t[4:7]) == [3, 3, 3]                  # down to the channel
    # The pit fills to the ridge and spills over it toward the channel.
    assert t[8] == 3 and t[7] == 3


def test_flood_height_follows_drainage_and_flow():
    p = WetlandParams()
    ground_m = np.array([0.0, 2.0, 50.0, 51.0, 60.0])
    water = np.array([True, False, False, False, False])
    target = np.array([0, 0, 2, 2, 2])
    above = flood_height_above(ground_m, water, np.zeros(5), target, np.array([2]), np.array([8000.0]), p)
    assert np.isclose(above[1], p.shore_flood_height_m - 2.0)
    h = flood_height_m(np.array([8000.0]), p)[0]
    assert np.isclose(above[2], h) and np.isclose(above[3], h - 1.0) and np.isclose(above[4], h - 10.0)
    # Bigger rivers flood higher.
    assert flood_height_m(np.array([50000.0]), p)[0] > h > flood_height_m(np.array([200.0]), p)[0]


def test_terms_are_monotone_and_bounded():
    p = WetlandParams()
    f = flood_term(np.array([-np.inf, -10.0, -p.bank_m, 0.0, p.full_depth_m, 50.0]), p)
    assert f[0] == 0.0 and f[1] == 0.0 and f[2] == 0.0 and f[-1] == 1.0 and f[-2] == 1.0
    assert np.all(np.diff(f) >= 0.0)
    r = rain_term(np.array([100.0, 1000.0, 5000.0]), np.array([1000.0, 200.0, 100.0]), p)
    assert r[0] == 0.0 and r[-1] == 1.0 and 0.0 < r[1] < 1.0
    flat = flatness(np.array([0.0, p.slope_scale_m_per_km / 1000.0, 0.01]), p)
    assert np.isclose(flat[0], 1.0) and np.isclose(flat[1], 0.5) and flat[2] < 0.01
    s = combine(np.array([1.0, 1.0, 0.0]), np.array([0.5, 0.0, 1.0]), np.array([0.5, 1.0, 1.0]))
    assert np.allclose(s, [0.75, 1.0, 0.0])


def test_stamp_marks_unfrozen_land_only():
    biome = np.array([3, 3, 3, 3], dtype=np.uint8)
    score = np.array([0.9, 0.9, 0.9, 0.1])
    land = np.array([True, False, True, True])
    frozen = np.array([False, False, True, False])
    out = stamp_wetland(biome, score, 0.5, land, frozen)
    assert list(out) == [WETLAND_BIOME, 3, 3, 3]


def test_params_threshold_gap_keeps_parent_wetlands():
    from localmap.upsample import LocalUpsampleParams
    from region.upsample import RegionUpsampleParams
    g, r, loc = WetlandParams(), RegionUpsampleParams(), LocalUpsampleParams()
    assert r.wetland_relief_gain <= g.threshold - r.wetland_threshold + 1e-9
    assert loc.wetland_relief_gain <= r.wetland_threshold - loc.wetland_threshold + 1e-9
    assert dataclasses.is_dataclass(g)
