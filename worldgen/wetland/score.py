"""The wetness score: where water stands on the ground, from slope, climate and floods.

    score = flatness × (1 − (1 − rain) · (1 − flood))

- **rain** — local water balance, ``smoothstep(rain_low, rain_high, P / PET)``: rain-fed
  bogs and fens (boreal peatlands, where evaporation is low).
- **flood** — overbank water from rivers, lakes and the sea, by *height above nearest
  drainage*: each texel follows steepest descent (on the depression-filled surface) to
  the first stream-network channel or water texel.  A channel floods
  ``flood_height(flow)`` above its ground, a lake or the sea ``shore_flood_height``
  above its surface; ``flood = ramp(−bank, full, flood level − ground)``.  Flat basins
  draining to a big river sit barely above it and flood wide (Pantanal, Sudd); valley
  sides rise fast and stay dry.  Banks just above the flood get a little — sub-threshold
  here, the riparian marsh the finer tiers' lower thresholds pick up.
- **flatness** — ``1 / (1 + (slope / slope_scale)²)``: water drains off slopes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from worldgen.wetland.builder import WetlandParams

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is an optional accelerator
    def njit(func=None, **_):  # type: ignore[misc]
        return func if func is not None else (lambda f: f)


def smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((np.asarray(x, dtype=np.float64) - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def flood_height_m(flow: np.ndarray, params: "WetlandParams") -> np.ndarray:
    """Water above a channel texel's ground (m): ``flood_height · (flow / flood_ref_flow) ** flood_exponent``."""
    q = np.maximum(np.asarray(flow, dtype=np.float64), 0.0) / max(float(params.flood_ref_flow), 1e-9)
    return float(params.flood_height_m) * q ** float(params.flood_exponent)


@njit(cache=True)
def _steepest_receiver(eff, stop, indptr, flat):  # pragma: no cover - numba
    """Lowest strictly-lower neighbour of each node on the filled surface; -1 at stops and pits."""
    n = eff.shape[0]
    rec = np.full(n, -1, dtype=np.int64)
    for u in range(n):
        if stop[u]:
            continue
        best = eff[u]
        for e in range(indptr[u], indptr[u + 1]):
            v = flat[e]
            if eff[v] < best:
                best = eff[v]
                rec[u] = v
    return rec


def drainage_targets(
    ground_km: np.ndarray, water: np.ndarray, channel: np.ndarray, indptr: np.ndarray, flat: np.ndarray,
) -> np.ndarray:
    """(n,) the first channel or water texel each texel's steepest descent reaches (itself if one).

    Descent runs on the depression-filled surface, so pits drain out over their sills.
    """
    from worldgen.hydrology.routing import priority_flood
    eff, _ = priority_flood(np.asarray(ground_km, dtype=np.float32), np.asarray(water, dtype=bool), indptr, flat)
    stop = np.asarray(water, dtype=bool) | np.asarray(channel, dtype=bool)
    rec = _steepest_receiver(eff.astype(np.float64), stop, np.asarray(indptr, dtype=np.int64),
                             np.asarray(flat, dtype=np.int64))
    target = np.where(rec >= 0, rec, np.arange(len(rec)))
    while True:                                   # pointer doubling to the path's end
        nxt = target[target]
        if np.array_equal(nxt, target):
            return target
        target = nxt


def flood_height_above(
    ground_m: np.ndarray,
    water: np.ndarray,
    surface_m: np.ndarray,
    target: np.ndarray,
    channel_nodes: Optional[np.ndarray],
    channel_flow: Optional[np.ndarray],
    params: "WetlandParams",
) -> np.ndarray:
    """(n,) flood water above each texel's ground (m): its drainage's flood level − ground.

    A channel's flood level is its ground + ``flood_height(flow)``; a lake's or the sea's
    is its surface + ``shore_flood_height``.  Texels draining nowhere get ``-inf``.
    """
    n = len(ground_m)
    level = np.full(n, -np.inf)
    level[water] = surface_m[water] + float(params.shore_flood_height_m)
    if channel_nodes is not None and len(channel_nodes):
        nodes = np.asarray(channel_nodes, dtype=np.int64)
        on_land = ~water[nodes]
        nodes = nodes[on_land]
        level[nodes] = ground_m[nodes] + flood_height_m(np.asarray(channel_flow)[on_land], params)
    return level[target] - ground_m


def rain_term(p_annual_mm: np.ndarray, pet_mm: np.ndarray, params: "WetlandParams") -> np.ndarray:
    ratio = np.asarray(p_annual_mm, dtype=np.float64) / np.maximum(np.asarray(pet_mm, dtype=np.float64), 1.0)
    return smoothstep(float(params.rain_low), float(params.rain_high), ratio)


def flood_term(depth_m: np.ndarray, params: "WetlandParams") -> np.ndarray:
    """``ramp(−bank, full, depth)``: 0 a bank's height above the flood, 1 a full depth under it."""
    depth = np.where(np.isfinite(depth_m), depth_m, -np.inf)
    bank, full = float(params.bank_m), float(params.full_depth_m)
    return np.clip((depth + bank) / max(bank + full, 1e-9), 0.0, 1.0)


def flatness(slope_km_per_km: np.ndarray, params: "WetlandParams") -> np.ndarray:
    s = np.asarray(slope_km_per_km, dtype=np.float64) * 1000.0 / max(float(params.slope_scale_m_per_km), 1e-9)
    return 1.0 / (1.0 + s * s)


def combine(flat: np.ndarray, rain: np.ndarray, flood: np.ndarray) -> np.ndarray:
    return np.clip(flat * (1.0 - (1.0 - rain) * (1.0 - flood)), 0.0, 1.0)
