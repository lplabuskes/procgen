"""Per-face and per-pixel surface temperature."""
from __future__ import annotations

import numpy as np


def annual_mean_temperature(latitudes_rad: np.ndarray,
                            equator_c: float,
                            pole_c: float) -> np.ndarray:
    """Latitude-only annual mean temperature in °C.

    Parabolic in |latitude| so equator gets `equator_c` and the poles get
    `pole_c`.
    """
    frac = np.abs(latitudes_rad) / (np.pi / 2.0)
    return equator_c - (equator_c - pole_c) * (frac ** 2)


def thermal_inertia(distance_to_coast_km: np.ndarray,
                    ocean_inertia: float,
                    land_inertia: float,
                    saturation_km: float) -> np.ndarray:
    """Map distance-to-coast to a seasonal-anomaly damping factor in [ocean, land]."""
    t = np.tanh(distance_to_coast_km / max(saturation_km, 1e-6))
    return ocean_inertia + (land_inertia - ocean_inertia) * t


def surface_temperatures(latitudes_rad: np.ndarray,
                         insolation_w_m2: np.ndarray,
                         face_elevations_km: np.ndarray,
                         distance_to_coast_km: np.ndarray,
                         *,
                         equator_annual_c: float,
                         pole_annual_c: float,
                         anomaly_scale: float,
                         lapse_rate_c_per_km: float,
                         ocean_inertia: float,
                         land_inertia: float,
                         inertia_distance_km: float,
                         coastal_current_temp_delta: np.ndarray | None = None
                         ) -> np.ndarray:
    """Compute (N, K) per-face seasonal temperatures in °C.

    Annual mean is a parabolic-in-latitude profile. The seasonal anomaly is
    derived from insolation departures from the annual mean, scaled by
    `anomaly_scale`, then damped by an ocean-vs-land thermal inertia factor.
    Lapse rate cools high faces uniformly across seasons.

    `coastal_current_temp_delta` (shape `(N, K)` in °C) is added at the end as
    a hook for the future currents stage; pass None for the MVP no-op.
    """
    t_annual = annual_mean_temperature(latitudes_rad, equator_annual_c, pole_annual_c)  # (N,)

    q_anomaly = insolation_w_m2 - insolation_w_m2.mean(axis=1, keepdims=True)            # (N, K)
    delta_t_eq = anomaly_scale * q_anomaly                                                # (N, K)

    inertia = thermal_inertia(
        distance_to_coast_km, ocean_inertia, land_inertia, inertia_distance_km
    )                                                                                     # (N,)

    # Lapse rate applies above sea level only — sub-sea-level faces are at the
    # water surface for thermodynamic purposes, not at their bathymetric depth.
    elev_for_lapse = np.maximum(face_elevations_km, 0.0)
    t_face = (
        t_annual[:, None]
        + inertia[:, None] * delta_t_eq
        - lapse_rate_c_per_km * elev_for_lapse[:, None]
    )

    if coastal_current_temp_delta is not None:
        t_face = t_face + coastal_current_temp_delta

    return t_face


def per_pixel_temperatures(face_temperatures_c: np.ndarray,
                           face_elevations_km: np.ndarray,
                           pixel_elevations_km: np.ndarray,
                           lapse_rate_c_per_km: float,
                           pixel_positions: np.ndarray,
                           face_centroids: np.ndarray,
                           face_adjacency: list) -> np.ndarray:
    """IDW-interpolate per-face temperatures to per-pixel, then apply lapse rate.

    Uses the same inverse-distance-weighting scheme as per_pixel_humidity so
    temperatures blend smoothly across face boundaries rather than snapping to
    the nearest face value.  The lapse-rate correction uses the IDW-blended
    reference elevation so it stays consistent with the blended temperature.

    Sub-sea-level pixels are pinned to sea level for lapse purposes.

    Args:
        face_temperatures_c: (N, K) per-face seasonal temperatures, °C.
        face_elevations_km: (N,) per-face elevation, km.
        pixel_elevations_km: (N, H, W) per-pixel elevation, km.
        lapse_rate_c_per_km: lapse rate Γ, °C/km.
        pixel_positions: (N, H, W, 3) unit-sphere pixel positions.
        face_centroids: (N, 3) unit-sphere face centroids.
        face_adjacency: adjacency list from PolygonMesh.

    Returns:
        (N, H, W, K) float32 temperatures in °C.
    """
    N, H, W = pixel_elevations_km.shape
    K = face_temperatures_c.shape[1]
    t = face_temperatures_c.astype(np.float32)
    face_ref_elev = np.maximum(face_elevations_km, 0.0).astype(np.float32)

    idw_temp = np.empty((N, H, W, K), dtype=np.float32)
    idw_ref_elev = np.empty((N, H, W), dtype=np.float32)

    for i in range(N):
        all_idx = [i] + list(face_adjacency[i])
        all_t = t[all_idx]                                   # (M, K)
        all_e = face_ref_elev[all_idx]                       # (M,)
        all_c = face_centroids[all_idx].astype(np.float32)   # (M, 3)

        pix = pixel_positions[i].astype(np.float32)          # (H, W, 3)
        diff = pix[None] - all_c[:, None, None, :]            # (M, H, W, 3)
        dist = np.sqrt((diff * diff).sum(axis=-1))            # (M, H, W)
        w = 1.0 / np.maximum(dist, 1e-9)                     # (M, H, W)
        w_sum = w.sum(axis=0)                                 # (H, W)

        idw_temp[i] = (w[:, :, :, None] * all_t[:, None, None, :]).sum(axis=0) / w_sum[:, :, None]
        idw_ref_elev[i] = (w * all_e[:, None, None]).sum(axis=0) / w_sum

    pixel_above = np.maximum(pixel_elevations_km, 0.0).astype(np.float32)
    correction = -lapse_rate_c_per_km * (pixel_above - idw_ref_elev)[..., None]
    return (idw_temp + correction).astype(np.float32)
