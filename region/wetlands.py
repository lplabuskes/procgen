"""Wetlands on the region and local maps: the parent's wetness score, refined, at a lower threshold.

The globe marks whole ~14 km texels Wetland (``worldgen/wetland``); each finer tier
reads its parent's ``wetland`` score bilinearly and

- **refines** it by the sub-grid relief: hollows below the parent's surface are
  wetter, rises drier, ``± relief_gain · tanh(Δh / relief_scale)``
  (:func:`refine_wetland`, inside :func:`region.upsample.refine_fields`);
- **adds riparian marsh** along its own routed rivers and streams on flat ground,
  ``riparian_gain · (1 − d / width(flow)) · flatness`` (:func:`add_riparian_wetland`,
  once the tier's rivers exist);

and marks Wetland at its own, lower, threshold.  The relief swing stays below the
gap between the parent's threshold and this one, so the parent's wetlands stay
wetland (bar bilinear softening at their edges); the smaller ones appear around them.
Not conserved: like ``biome``, a classification rather than an amount.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.spatial import KDTree

from worldgen.hydrology.types import REFERENCE_FLOW
from worldgen.wetland.builder import stamp_wetland

if TYPE_CHECKING:
    from region.rivers import RegionRivers
    from region.upsample import RegionFields, RegionUpsampleParams

_EARTH_RADIUS_KM = 6371.0
# Where the score and biome before any riparian marsh are kept, in ``fields.textures``.
_BASE = "_pre_riparian_"


def refine_wetland(
    coarse: np.ndarray,
    elevation_km: np.ndarray,
    elev_coarse_km: np.ndarray,
    params: "RegionUpsampleParams",
) -> np.ndarray:
    """(P,) refined score: the parent's, wetter in sub-grid hollows and drier on rises, in [0, 1]."""
    below_m = (np.asarray(elev_coarse_km, dtype=np.float64) - np.asarray(elevation_km, dtype=np.float64)) * 1000.0
    swing = float(params.wetland_relief_gain) * np.tanh(below_m / max(float(params.wetland_relief_scale_m), 1e-9))
    return np.clip(np.asarray(coarse, dtype=np.float64) + swing, 0.0, 1.0)


def stamp(biome: np.ndarray, score: np.ndarray, params: "RegionUpsampleParams", land: np.ndarray,
          temperature: Optional[np.ndarray]) -> np.ndarray:
    """``biome`` (any shape, matching ``score``/``land``) with Wetland at the tier's threshold."""
    frozen = None
    if temperature is not None:
        frozen = np.asarray(temperature, dtype=np.float64).max(axis=-1).reshape(biome.shape) < 0.0
    return stamp_wetland(biome, score, float(params.wetland_threshold), land, frozen)


def _densify(paths: list, step_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Points every ``step_rad`` (chord) along the paths, and the flow at each."""
    pts, flows = [], []
    for positions, flow, *_ in paths:
        p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        f = np.asarray(flow, dtype=np.float64).reshape(-1)
        if len(p) < 2:
            continue
        seg_f = f[: len(p) - 1] if len(f) >= len(p) - 1 else np.full(len(p) - 1, f.max(initial=0.0))
        a, b = p[:-1], p[1:]
        k = np.maximum(np.ceil(np.linalg.norm(b - a, axis=1) / max(step_rad, 1e-12)).astype(np.int64), 1)
        seg = np.repeat(np.arange(len(a)), k)
        t = ((np.arange(len(seg)) - np.repeat(np.cumsum(k) - k, k)) / k[seg])[:, None]
        pts.append(a[seg] + (b[seg] - a[seg]) * t)
        flows.append(seg_f[seg])
    if not pts:
        return np.zeros((0, 3)), np.zeros(0)
    return np.concatenate(pts), np.concatenate(flows)


def add_riparian_wetland(fields: "RegionFields", rivers: "RegionRivers", params: "RegionUpsampleParams") -> None:
    """Raise ``fields``' wetland score along ``rivers`` (paths and streams) and re-mark Wetland, in place."""
    tex = fields.textures
    if "wetland" not in tex or "biome" not in tex:
        return
    # Idempotent: rivers may be re-routed over the same fields (river tuning changed).
    if _BASE + "wetland" not in tex:
        tex[_BASE + "wetland"], tex[_BASE + "biome"] = tex["wetland"].copy(), tex["biome"].copy()
    tex["wetland"], tex["biome"] = tex[_BASE + "wetland"].copy(), tex[_BASE + "biome"].copy()
    if float(params.wetland_riparian_gain) <= 0.0:
        return
    paths = list(rivers.paths) + list(rivers.streams)
    texel_km = float(np.sqrt(fields.texel_area_km2))
    pts, flow = _densify(paths, 0.5 * texel_km / _EARTH_RADIUS_KM)
    if not len(pts):
        return
    from region.resources import refined_slope

    shape = fields.positions.shape[:3]
    q = np.maximum(flow, 1e-9) / REFERENCE_FLOW
    width_km = np.maximum(float(params.wetland_riparian_width_km) * q ** float(params.wetland_riparian_exponent),
                          float(params.wetland_riparian_min_width_texels) * texel_km)
    reach_km = float(width_km.max())
    pos = fields.positions.reshape(-1, 3)
    dist, k_raw = KDTree(pts).query(pos, distance_upper_bound=reach_km / _EARTH_RADIUS_KM, workers=-1)
    dist, k = np.asarray(dist, dtype=np.float64), np.asarray(k_raw, dtype=np.int64)
    near = np.isfinite(dist)
    if not near.any():
        return
    # Each texel takes the width of its nearest river point.
    closeness = np.zeros(len(pos))
    d_km = dist[near] * _EARTH_RADIUS_KM
    closeness[near] = np.clip(1.0 - d_km / np.maximum(width_km[k[near]], 1e-9), 0.0, 1.0)
    slope = refined_slope(tex["elevation"][..., 0].reshape(shape), texel_km).reshape(-1)
    s = slope * 1000.0 / max(float(params.wetland_riparian_slope_m_per_km), 1e-9)
    boost = float(params.wetland_riparian_gain) * closeness / (1.0 + s * s)

    score = np.clip(tex["wetland"].reshape(-1).astype(np.float64) + boost, 0.0, 1.0)
    water = np.zeros(len(pos), dtype=bool)
    for key in ("ocean_mask", "lake_mask"):
        if key in tex:
            water |= tex[key].reshape(-1) > 0.5
    biome = tex["biome"].reshape(-1)
    biome = stamp(biome, score, params, ~water, tex["temperature"].reshape(len(pos), -1)
                  if "temperature" in tex else None)
    tex["wetland"] = score.reshape(*shape, 1).astype(np.float32)
    tex["biome"] = biome.reshape(*shape, 1).astype(np.uint8)
