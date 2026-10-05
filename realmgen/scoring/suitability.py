"""Per-face settlement suitability in [0, 1].

A weighted sum (so one weak factor doesn't zero a site) of water proximity,
climate habitability, fertility, and resource access, zeroed on water, then
multiplied by a ruggedness penalty.  Ruggedness as a defensive *bonus* belongs
in the situational-value scorer, not here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from worldgen.resources.terrain import face_hops_from_seeds

if TYPE_CHECKING:
    from realmgen.scoring.builder import (
        ClimateHabitabilityParams,
        SuitabilityWeightParams,
        WaterProximityParams,
    )


# ── Water proximity ───────────────────────────────────────────────────────────


def compute_water_proximity(
    mesh,
    freshwater_mask: np.ndarray,
    dist_to_coast_km: np.ndarray,
    *,
    params: "WaterProximityParams",
) -> np.ndarray:
    """Fresh-water and coastal access score, (N,) float32 in [0, 1].

    ``clip(exp(-hops · decay) + coast_weight · exp(-dist_to_coast_km / scale), 0, 1)``,
    hops counted from the ``freshwater_mask`` (rivers + lakes) faces.
    """
    p = params
    n = len(mesh.faces)
    freshwater_mask = np.asarray(freshwater_mask, dtype=bool)
    dist = np.asarray(dist_to_coast_km, dtype=np.float64)

    if freshwater_mask.any():
        hops = face_hops_from_seeds(mesh, freshwater_mask, max_hops=p.freshwater_max_hops)
        fw_score = np.exp(-hops.astype(np.float64) * p.freshwater_decay)
    else:
        fw_score = np.zeros(n, dtype=np.float64)

    coast_score = np.exp(-np.maximum(dist, 0.0) / max(p.coast_scale_km, 1e-9))
    coast_score = np.clip(coast_score, 0.0, 1.0)

    combined = np.clip(fw_score + p.coast_weight * coast_score, 0.0, 1.0)
    return combined.astype(np.float32)


# ── Climate habitability ──────────────────────────────────────────────────────


def compute_climate_habitability(
    t_warmest: np.ndarray,
    t_coldest: np.ndarray,
    p_annual: np.ndarray,
    *,
    params: "ClimateHabitabilityParams",
) -> np.ndarray:
    """Climate habitability, (N,) float32 in [0, 1]: product of three piecewise-linear scores.

    * Growing season (T_warmest °C): 0 below ``t_grow_min_c``, ramps to 1 at ``t_grow_opt_c``.
    * Cold tolerance (T_coldest °C): 0 below ``t_cold_min_c``, ramps to 1 at 0 °C.
    * Precipitation (mm/yr): 0 below ``p_dry_mm``, 1 at ``p_opt_mm``, then down to
      0.5 at and beyond ``p_wet_mm`` (wet but liveable).
    """
    p = params
    tw = np.asarray(t_warmest, dtype=np.float64)
    tc = np.asarray(t_coldest, dtype=np.float64)
    pa = np.asarray(p_annual, dtype=np.float64)

    grow_range = max(p.t_grow_opt_c - p.t_grow_min_c, 1e-9)
    t_grow = np.clip((tw - p.t_grow_min_c) / grow_range, 0.0, 1.0)

    cold_range = max(0.0 - p.t_cold_min_c, 1e-9)
    t_cold = np.clip((tc - p.t_cold_min_c) / cold_range, 0.0, 1.0)

    dry_range = max(p.p_opt_mm - p.p_dry_mm, 1e-9)
    wet_range = max(p.p_wet_mm - p.p_opt_mm, 1e-9)
    p_score = np.where(
        pa < p.p_dry_mm, 0.0,
        np.where(
            pa < p.p_opt_mm, (pa - p.p_dry_mm) / dry_range,
            np.where(
                pa < p.p_wet_mm, 1.0 - 0.5 * (pa - p.p_opt_mm) / wet_range,
                0.5,
            ),
        ),
    )

    return (t_grow * t_cold * p_score).clip(0.0, 1.0).astype(np.float32)


# ── Ruggedness penalty ────────────────────────────────────────────────────────


def compute_ruggedness_penalty(
    face_elev_range_km: np.ndarray,
    *,
    penalty_scale_km: float = 1.0,
) -> np.ndarray:
    """Ruggedness multiplier ``exp(-range / scale)``, (N,) float32 in (0, 1].

    Args:
        face_elev_range_km: (N,) per-face pixel elevation span (km) — the ruggedness
            proxy, since per-pixel slope is noise-dominated.
    """
    r = np.maximum(np.asarray(face_elev_range_km, dtype=np.float64), 0.0)
    return np.exp(-r / max(penalty_scale_km, 1e-9)).astype(np.float32)


# ── Resource access ───────────────────────────────────────────────────────────


def _positions_to_face_ids(positions: np.ndarray, face_centroids: np.ndarray) -> np.ndarray:
    """(M,) nearest-centroid face index for each (M, 3) unit-sphere position.

    Brute-force max dot product, O(M × N).
    """
    pos = np.asarray(positions, dtype=np.float64)
    cen = np.asarray(face_centroids, dtype=np.float64)
    if pos.ndim == 1:
        pos = pos[np.newaxis, :]
    dots = pos @ cen.T  # (M, N)
    return np.argmax(dots, axis=1)


def compute_resource_proximity(
    mesh,
    resource_mask: np.ndarray,
    *,
    resource_decay: float = 0.2,
    resource_max_hops: int = 16,
) -> np.ndarray:
    """Resource-access score, (N,) float32 in [0, 1], via BFS hop-decay from deposit faces.

    ``resource_decay`` is slower than water's since resources travel by trade.
    All zeros when there are no deposits.
    """
    resource_mask = np.asarray(resource_mask, dtype=bool)
    n = len(mesh.faces)
    if not resource_mask.any():
        return np.zeros(n, dtype=np.float32)
    hops = face_hops_from_seeds(mesh, resource_mask, max_hops=resource_max_hops)
    return np.exp(-hops.astype(np.float64) * resource_decay).astype(np.float32)


# ── Combination ───────────────────────────────────────────────────────────────


def combine_suitability(
    is_water: np.ndarray,
    water_proximity: np.ndarray,
    climate: np.ndarray,
    fertility: np.ndarray,
    resource_access: np.ndarray,
    *,
    params: "SuitabilityWeightParams",
) -> np.ndarray:
    """Weight-normalised suitability, (N,) float32 in [0, 1]; water faces 0.

    Factors are (N,) in [0, 1].  Multiply by :func:`compute_ruggedness_penalty` afterwards.
    """
    p = params
    total_w = p.water_weight + p.climate_weight + p.fertility_weight + p.resource_weight
    total_w = max(total_w, 1e-9)

    score = (
        p.water_weight * np.asarray(water_proximity, dtype=np.float64)
        + p.climate_weight * np.asarray(climate, dtype=np.float64)
        + p.fertility_weight * np.asarray(fertility, dtype=np.float64)
        + p.resource_weight * np.asarray(resource_access, dtype=np.float64)
    ) / total_w

    score = np.where(np.asarray(is_water, dtype=bool), 0.0, score)
    return np.clip(score, 0.0, 1.0).astype(np.float32)
