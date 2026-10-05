"""Per-face and per-pixel surface temperature."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from worldgen.climate.builder import TemperatureParams


def annual_mean_temperature(latitudes_rad: np.ndarray,
                            equator_c: float,
                            pole_c: float) -> np.ndarray:
    """Annual mean °C, parabolic in |latitude| from `equator_c` to `pole_c`."""
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
                         params: "TemperatureParams",
                         coastal_current_temp_delta: np.ndarray | None = None
                         ) -> np.ndarray:
    """Compute (N, K) per-face seasonal temperatures in °C.

    Seasonal anomaly = insolation departure × `anomaly_scale`, damped by
    ocean-vs-land thermal inertia; lapse rate applies to all seasons.
    `coastal_current_temp_delta` ((N, K) °C) is added last; None for the
    no-current pass.
    """
    p = params
    t_annual = annual_mean_temperature(latitudes_rad, p.equator_annual_c, p.pole_annual_c)  # (N,)

    q_anomaly = insolation_w_m2 - insolation_w_m2.mean(axis=1, keepdims=True)            # (N, K)
    delta_t_eq = p.anomaly_scale * q_anomaly                                              # (N, K)

    inertia = thermal_inertia(
        distance_to_coast_km, p.ocean_inertia, p.land_inertia, p.inertia_distance_km
    )                                                                                     # (N,)

    # Sub-sea-level faces sit at the water surface, not their bathymetric depth.
    elev_for_lapse = np.maximum(face_elevations_km, 0.0)
    t_face = (
        t_annual[:, None]
        + inertia[:, None] * delta_t_eq
        - p.lapse_rate_c_per_km * elev_for_lapse[:, None]
    )

    if coastal_current_temp_delta is not None:
        t_face = t_face + coastal_current_temp_delta

    return t_face


def per_pixel_temperatures(face_temperatures_c: np.ndarray,
                           face_elevations_km: np.ndarray,
                           pixel_elevations_km: np.ndarray,
                           lapse_rate_c_per_km: float,
                           mesh) -> np.ndarray:
    """Rasterize per-face temperatures to per-pixel via shared-vertex barycentric interpolation.

    Shared vertices make adjacent faces agree at seams. The per-pixel lapse
    correction is relative to the same vertex-averaged reference elevation;
    sub-sea-level pixels count as sea level.

    Args:
        face_temperatures_c: (N, K) per-face seasonal temperatures, °C.
        face_elevations_km: (N,) per-face elevation, km.
        pixel_elevations_km: (N, H, W) per-pixel elevation from elevation texture.
        lapse_rate_c_per_km: lapse rate Γ, °C/km.
        mesh: PolygonMesh.

    Returns:
        (N, H, W, K) float32 temperatures in °C.
    """
    from worldgen.geology.elevation import rasterize_face_values, smooth_texture_seams

    _, H, _ = pixel_elevations_km.shape
    tex_size = H

    t_tex = rasterize_face_values(mesh, face_temperatures_c, tex_size)   # (N, H, W, K)
    ref_tex = rasterize_face_values(mesh, face_elevations_km, tex_size)  # (N, H, W, 1)
    smooth_texture_seams(mesh, t_tex, n_passes=1)

    ref_elev = np.maximum(ref_tex[:, :, :, 0], 0.0).astype(np.float32)
    pix_elev = np.maximum(pixel_elevations_km, 0.0).astype(np.float32)
    t_tex -= (lapse_rate_c_per_km * (pix_elev - ref_elev)).astype(np.float32)[:, :, :, None]
    return t_tex
