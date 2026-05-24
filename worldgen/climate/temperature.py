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
                           lapse_rate_c_per_km: float) -> np.ndarray:
    """Broadcast per-face seasonal temperatures to per-pixel and re-apply lapse rate.

    Sub-sea-level pixels are pinned to the sea surface for lapse purposes — so
    deep ocean reads the same temperature as the surrounding shelf.

    Args:
        face_temperatures_c: (N, K) per-face seasonal temperatures, °C.
        face_elevations_km: (N,) per-face elevation, km.
        pixel_elevations_km: (N, H, W) per-pixel elevation, km.
        lapse_rate_c_per_km: lapse rate Γ, °C/km.

    Returns:
        (N, H, W, K) float32 temperatures in °C.
    """
    face_above = np.maximum(face_elevations_km, 0.0)[:, None, None]                # (N,1,1)
    pixel_above = np.maximum(pixel_elevations_km, 0.0)                              # (N,H,W)
    base = face_temperatures_c[:, None, None, :]                                    # (N,1,1,K)
    correction = -lapse_rate_c_per_km * (pixel_above - face_above)[..., None]       # (N,H,W,1)
    return (base + correction).astype(np.float32)
