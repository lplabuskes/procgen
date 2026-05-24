"""Per-face and per-pixel seasonal precipitation."""
from __future__ import annotations

import numpy as np


# Centres of the zonal precipitation bands, in degrees absolute latitude.
_BAND_CENTERS_DEG = np.array([0.0, 30.0, 55.0, 90.0], dtype=np.float64)


def zonal_baseline_mm_per_season(abs_phi_eff_deg: np.ndarray,
                                 *,
                                 itcz_mm_annual: float,
                                 horse_mm_annual: float,
                                 midlat_mm_annual: float,
                                 polar_mm_annual: float,
                                 sigma_deg: float = 15.0) -> np.ndarray:
    """Smooth Gaussian-weighted blend of band values, in mm per season."""
    centres = _BAND_CENTERS_DEG
    values_annual = np.array(
        [itcz_mm_annual, horse_mm_annual, midlat_mm_annual, polar_mm_annual],
        dtype=np.float64,
    )
    dist = abs_phi_eff_deg[..., None] - centres                      # (..., 4)
    weights = np.exp(-0.5 * (dist / max(sigma_deg, 1e-6)) ** 2)      # (..., 4)
    weight_sum = weights.sum(axis=-1)
    weight_sum = np.where(weight_sum < 1e-12, 1.0, weight_sum)
    annual = (weights * values_annual).sum(axis=-1) / weight_sum
    return annual / 4.0


def continentality_factor(distance_to_coast_km: np.ndarray,
                          decay_km: float) -> np.ndarray:
    """Interior dryness factor in [0.2, 1.0]."""
    t = np.tanh(distance_to_coast_km / max(decay_km, 1e-6))
    return 1.0 - 0.8 * t


def orographic_factor(uplift_m_per_s: np.ndarray,
                      k_orographic: float,
                      lo: float = 0.1,
                      hi: float = 3.0) -> np.ndarray:
    """Convert face-relative vertical velocity to a multiplicative precip factor."""
    return np.clip(1.0 + k_orographic * uplift_m_per_s, lo, hi)


def surface_precipitation(latitudes_rad: np.ndarray,
                          winds_m_per_s: np.ndarray,
                          face_elevation_gradient: np.ndarray,
                          distance_to_coast_km: np.ndarray,
                          *,
                          itcz_mm_annual: float,
                          horse_mm_annual: float,
                          midlat_mm_annual: float,
                          polar_mm_annual: float,
                          orographic_k: float,
                          continentality_decay_km: float,
                          itcz_shift_deg: float,
                          season_phases: np.ndarray | None = None,
                          current_wet_delta: np.ndarray | None = None
                          ) -> np.ndarray:
    """Compute (N, K) per-face seasonal precipitation in mm/season.

    Args:
        latitudes_rad: (N,) face latitudes.
        winds_m_per_s: (N, K, 3) per-face per-season wind vectors.
        face_elevation_gradient: (N, 3) elevation gradient (km / km) in 3D,
            tangent to the sphere, computed by the builder from face neighbours.
        distance_to_coast_km: (N,) km.
        current_wet_delta: optional (N, K) hook for current-driven moisture
            adjustment. Applied as `(1 + delta)` multiplier; pass None for MVP.

    Returns:
        (N, K) float32 mm per season.
    """
    from worldgen.climate.winds import SEASON_PHASES
    if season_phases is None:
        season_phases = SEASON_PHASES

    lat_deg = np.rad2deg(latitudes_rad)                                       # (N,)
    itcz_lat = itcz_shift_deg * np.sin(season_phases)                          # (K,)
    phi_eff = lat_deg[:, None] - itcz_lat[None, :]                             # (N, K)
    zonal = zonal_baseline_mm_per_season(
        np.abs(phi_eff),
        itcz_mm_annual=itcz_mm_annual,
        horse_mm_annual=horse_mm_annual,
        midlat_mm_annual=midlat_mm_annual,
        polar_mm_annual=polar_mm_annual,
    )                                                                          # (N, K)

    uplift = np.einsum("nkj,nj->nk", winds_m_per_s, face_elevation_gradient)   # (N, K)
    orog = orographic_factor(uplift, orographic_k)                              # (N, K)

    cont = continentality_factor(distance_to_coast_km, continentality_decay_km)  # (N,)

    p = zonal * orog * cont[:, None]
    if current_wet_delta is not None:
        p = p * (1.0 + current_wet_delta)
    return p.astype(np.float32)


def per_pixel_precipitation(face_precipitation: np.ndarray,
                            face_winds: np.ndarray,
                            face_u_axes: np.ndarray,
                            face_v_axes: np.ndarray,
                            face_u_ranges_km: np.ndarray,
                            face_v_ranges_km: np.ndarray,
                            pixel_elevations_km: np.ndarray,
                            *,
                            orographic_k: float) -> np.ndarray:
    """Broadcast per-face precipitation and overlay a per-pixel orographic correction.

    Args:
        face_precipitation: (N, K) mm/season from `surface_precipitation`.
        face_winds: (N, K, 3) m/s wind vectors at each face.
        face_u_axes, face_v_axes: (N, 3) unit tangent axes per face.
        face_u_ranges_km, face_v_ranges_km: (N,) face UV span in km.
        pixel_elevations_km: (N, H, W) per-pixel elevation, km.
        orographic_k: same K as the face-level calculation; applied to the
            residual (per-pixel uplift minus the per-face mean uplift).

    Returns:
        (N, H, W, K) float32 mm/season.
    """
    n, h, w = pixel_elevations_km.shape
    n_k = face_precipitation.shape[1]

    # Per-pixel slope (∂h/∂u, ∂h/∂v) in km/km. np.gradient returns (axis=row, axis=col) = (∂h/∂v, ∂h/∂u).
    dh_dv, dh_du = np.gradient(pixel_elevations_km, axis=(1, 2))
    pix_du_km = face_u_ranges_km[:, None, None] / max(w - 1, 1)
    pix_dv_km = face_v_ranges_km[:, None, None] / max(h - 1, 1)
    slope_u = dh_du / np.maximum(pix_du_km, 1e-9)                              # (N, H, W)
    slope_v = dh_dv / np.maximum(pix_dv_km, 1e-9)                              # (N, H, W)

    # Wind in face UV: (N, K).
    wind_u = np.einsum("nkj,nj->nk", face_winds, face_u_axes)
    wind_v = np.einsum("nkj,nj->nk", face_winds, face_v_axes)

    # Per-pixel uplift in m/s.
    uplift_pix = (
        wind_u[:, None, None, :] * slope_u[..., None]
        + wind_v[:, None, None, :] * slope_v[..., None]
    )                                                                           # (N, H, W, K)

    # Residual orographic factor, multiplicative, clipped tighter than the face one.
    residual_factor = np.clip(1.0 + orographic_k * uplift_pix, 0.3, 2.0)

    base = face_precipitation[:, None, None, :]                                 # (N, 1, 1, K)
    return (base * residual_factor).astype(np.float32)
