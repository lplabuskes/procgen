"""Per-face settlement suitability scorer.

Four independent factors combine into one [0, 1] suitability scalar per land
face, then a ruggedness penalty is applied multiplicatively.  Each factor is
computed by a pure function that takes pre-extracted arrays (no Globe coupling),
making them easy to test against a lightweight stub mesh.
:func:`combine_suitability` does the final weighted merge and zeros water faces;
:func:`compute_ruggedness_penalty` is multiplied in afterward.

Factor overview (default weights; all tunable via ``SuitabilityParams``):

  * **water proximity**  (w = 0.30) — BFS hop-decay from river + lake seed faces,
    plus an exponential coastal-access term scaled by ocean distance.
  * **climate**          (w = 0.30) — piecewise-linear scores on T_warmest,
    T_coldest, and P_annual, multiplied together (all three must be adequate).
  * **fertility**        (w = 0.25) — direct read of ``fertility_mean`` metadata,
    already in [0, 1] from the resources stage.
  * **resource access**  (w = 0.15) — BFS hop-decay from faces containing mineral
    or rare-bio deposits.

Ruggedness (local elevation span) is a **multiplicative penalty** applied to the
combined score: flat terrain has no penalty; rugged terrain reduces suitability
proportionally, regardless of how good the other factors are.  Defensibility
(ruggedness as a *bonus*) belongs in the static situational-value scorer.

The combination is a weighted sum so that poor resource access does not zero out
an otherwise ideal site — each factor can compensate within its budget.
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
    """Fresh-water and coastal access score in [0, 1].

    Freshwater component uses BFS hop-decay from any river or lake face
    (``freshwater_mask``). Coastal component uses exponential decay over
    ``dist_to_coast_km`` (the climate stage's land → ocean distance in km).

    Combined: ``clip(freshwater_score + coast_weight * coastal_score, 0, 1)``.
    A face adjacent to a river scores ~1.0; a pure coastal face scores
    ``coast_weight`` (default 0.4); a face with both scores are additive up to
    the clip.

    Args:
        mesh: object with ``faces`` and ``face_adjacency``.
        freshwater_mask: (N,) bool — seed faces for BFS (rivers + lakes).
        dist_to_coast_km: (N,) float — land-to-ocean distance in km (0 = coast).
        params: WaterProximityParams — freshwater hop-decay/cap and coastal
            scale/weight.

    Returns:
        (N,) float32 in [0, 1].
    """
    p = params
    n = len(mesh.faces)
    freshwater_mask = np.asarray(freshwater_mask, dtype=bool)
    dist = np.asarray(dist_to_coast_km, dtype=np.float64)

    # Freshwater: BFS hop-decay from river/lake seed faces.
    if freshwater_mask.any():
        hops = face_hops_from_seeds(mesh, freshwater_mask, max_hops=p.freshwater_max_hops)
        fw_score = np.exp(-hops.astype(np.float64) * p.freshwater_decay)
    else:
        fw_score = np.zeros(n, dtype=np.float64)

    # Coastal: exponential decay with land-to-ocean distance.
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
    """Climate habitability score in [0, 1], product of three sub-scores.

    All three sub-scores must be adequate — if any is 0, the result is 0.

    Growing-season score (T_warmest):
      * below ``t_grow_min_c``: 0 (nothing grows)
      * ``t_grow_min_c`` → ``t_grow_opt_c``: linear 0 → 1
      * above ``t_grow_opt_c``: 1 (warm enough is warm enough)

    Cold-tolerance score (T_coldest):
      * below ``t_cold_min_c``: 0 (uninhabitable cold)
      * ``t_cold_min_c`` → 0°C: linear 0 → 1
      * above 0°C: 1

    Precipitation score (P_annual):
      * below ``p_dry_mm``: 0 (desert)
      * ``p_dry_mm`` → ``p_opt_mm``: linear 0 → 1
      * ``p_opt_mm`` → ``p_wet_mm``: linear 1 → 0.5 (wet but liveable)
      * above ``p_wet_mm``: 0.5 (very wet; still habitable)

    Args:
        t_warmest: (N,) float °C — warmest-month temperature per face.
        t_coldest: (N,) float °C — coldest-month temperature per face.
        p_annual:  (N,) float mm/yr — annual precipitation per face.
        params: ClimateHabitabilityParams — the growing/cold/precip thresholds.

    Returns:
        (N,) float32 in [0, 1].
    """
    p = params
    tw = np.asarray(t_warmest, dtype=np.float64)
    tc = np.asarray(t_coldest, dtype=np.float64)
    pa = np.asarray(p_annual, dtype=np.float64)

    # Growing-season score.
    grow_range = max(p.t_grow_opt_c - p.t_grow_min_c, 1e-9)
    t_grow = np.clip((tw - p.t_grow_min_c) / grow_range, 0.0, 1.0)

    # Cold-tolerance score.
    cold_range = max(0.0 - p.t_cold_min_c, 1e-9)
    t_cold = np.clip((tc - p.t_cold_min_c) / cold_range, 0.0, 1.0)

    # Precipitation score — piecewise.
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
    """Ruggedness penalty multiplier in (0, 1] — applied after combine_suitability.

    ``exp(-range / scale)``: flat terrain (range = 0) gives 1.0 (no penalty);
    penalty grows smoothly, reaching exp(-1) ≈ 0.37 at one scale-length. Rugged
    terrain (mountains, gorges) is hard to build on and cultivate, so it reduces
    suitability regardless of how good the climate or fertility are.

    Defensibility (ruggedness as a *bonus*) belongs in the static
    situational-value scorer, not here.

    Args:
        face_elev_range_km: (N,) float — per-face pixel elevation span (km),
            the ruggedness proxy (per-pixel slope is noise-dominated).
        penalty_scale_km: e-folding relief; at this elevation span the penalty
            factor is exp(-1) ≈ 0.37.

    Returns:
        (N,) float32 in (0, 1].
    """
    r = np.maximum(np.asarray(face_elev_range_km, dtype=np.float64), 0.0)
    return np.exp(-r / max(penalty_scale_km, 1e-9)).astype(np.float32)


# ── Resource access ───────────────────────────────────────────────────────────


def _positions_to_face_ids(positions: np.ndarray, face_centroids: np.ndarray) -> np.ndarray:
    """Nearest-centroid face index for each 3D unit-sphere position.

    Uses the dot-product approximation (maximising cosine similarity = nearest
    on the unit sphere). Cost: O(M × N) — fine for a few hundred deposits and
    ~10k faces.

    Args:
        positions:      (M, 3) float — unit-sphere points.
        face_centroids: (N, 3) float — face centroid unit vectors.

    Returns:
        (M,) int — face indices in [0, N).
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
    """Resource-access score in [0, 1] via BFS hop-decay from deposit faces.

    Args:
        mesh: object with ``faces`` and ``face_adjacency``.
        resource_mask: (N,) bool — faces containing a mineral or rare-bio
            deposit (typically sparse).
        resource_decay: BFS hop-decay exponent (slower decay than water since
            resources are exploited via trade routes, not proximity alone).
        resource_max_hops: BFS depth cap.

    Returns:
        (N,) float32 in [0, 1]. Returns all-zeros when no deposits exist
        anywhere (caller may choose to skip this factor by setting weight = 0).
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
    """Weighted-average suitability in [0, 1]; water faces are forced to 0.

    Uses a weighted sum (not product) so that modest resource access does not
    zero out an otherwise ideal site. The weight vector sums to 1 by
    construction (the function normalises it), so changing individual weights
    is safe without also adjusting the others.

    Multiply the result by :func:`compute_ruggedness_penalty` afterward to
    apply the ruggedness penalty.

    Args:
        is_water:        (N,) bool — open water (ocean ∪ lake); these faces
            receive 0 regardless of factor scores.
        water_proximity: (N,) float32 in [0, 1].
        climate:         (N,) float32 in [0, 1].
        fertility:       (N,) float32 in [0, 1].
        resource_access: (N,) float32 in [0, 1].
        params: SuitabilityWeightParams — the four factor weights.

    Returns:
        (N,) float32 in [0, 1].
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
