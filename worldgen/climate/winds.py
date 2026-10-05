"""Zonal Hadley/Ferrel/Polar surface winds with seasonal ITCZ shift."""
from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

import numpy as np

from worldgen.climate._gradient import _tangent_frames, _tangent_gradient

if TYPE_CHECKING:
    from worldgen.climate.builder import WindParams


SEASON_PHASES = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2], dtype=np.float64)
EARTH_RADIUS_KM = 6371.0


def zonal_winds(face_centroids: np.ndarray,
                latitudes_rad: np.ndarray,
                *,
                params: "WindParams",
                season_phases: np.ndarray | None = None) -> np.ndarray:
    """Compute prevailing surface winds at face centroids per season.

    Args:
        face_centroids: (N, 3) unit-sphere face positions.
        latitudes_rad: (N,) face latitudes in radians.
        params: WindParams — band speeds (trade/westerly/polar) and ITCZ shift.
        season_phases: (K,) seasonal phases in radians. Defaults to four seasons
            anchored at 0, π/2, π, 3π/2.

    Returns:
        (N, K, 3) float32 wind velocity vectors (m/s) tangent to the sphere.
    """
    p = params
    if season_phases is None:
        season_phases = SEASON_PHASES

    east, north = _tangent_frames(face_centroids)                     # (N, 3) each

    lat_deg = np.rad2deg(latitudes_rad)                                # (N,)
    itcz_lat = p.itcz_shift_deg * np.sin(season_phases)               # (K,)
    phi_eff = lat_deg[:, None] - itcz_lat[None, :]                     # (N, K)
    band_dist = np.abs(phi_eff)
    hemi_sign = np.where(phi_eff >= 0.0, 1.0, -1.0)                    # +1 N of thermal equator

    trade = band_dist < 30.0
    westerly = (band_dist >= 30.0) & (band_dist < 60.0)

    # Direction in (east, north) basis. Trade & polar blow toward (W, equator);
    # westerlies blow toward (E, pole). Equatorward = -hemi_sign in `north`.
    u_east = np.where(westerly, 1.0, -1.0)
    v_north = np.where(westerly, hemi_sign, -hemi_sign)

    # Normalise the (u, v) direction so diagonal winds aren't √2× stronger.
    inv_mag = 1.0 / np.sqrt(u_east ** 2 + v_north ** 2)
    u_east *= inv_mag
    v_north *= inv_mag

    speed = np.where(trade, p.trade_speed,
                     np.where(westerly, p.westerly_speed, p.polar_speed))   # (N, K)

    winds = (
        (speed * u_east)[..., None] * east[:, None, :]
        + (speed * v_north)[..., None] * north[:, None, :]
    )                                                                    # (N, K, 3)
    return winds.astype(np.float32)


def pressure_winds(face_centroids: np.ndarray,
                   face_adjacency: Sequence[Sequence[int]],
                   temperature_face: np.ndarray,
                   *,
                   pressure_wind_scale: float,
                   earth_radius_km: float = EARTH_RADIUS_KM) -> np.ndarray:
    """Cookbook pressure-gradient wind contribution from land/sea thermal contrast.

    Temperature anomaly proxies pressure (cold = high). Purely cross-isobar, no
    Coriolis: ``wind = K * grad(T - T_annual)`` — summer onshore, winter offshore
    monsoons, strongest on large-landmass coasts (ocean anomaly is near zero).

    Args:
        face_centroids: (N, 3) unit-sphere face positions.
        face_adjacency: per-face neighbour index lists (from ``PolygonMesh``).
        temperature_face: (N, K) °C, the seasonal surface temperatures.
        pressure_wind_scale: m/s per (°C/km) of anomaly gradient.
        earth_radius_km: sphere radius used to convert unit-sphere distances to km.

    Returns:
        (N, K, 3) float32 wind velocity vectors (m/s) tangent to the sphere.
    """
    n_faces = face_centroids.shape[0]
    n_seasons = temperature_face.shape[1]
    if pressure_wind_scale <= 0.0:
        return np.zeros((n_faces, n_seasons, 3), dtype=np.float32)

    centroids = face_centroids.astype(np.float64)
    east, north = _tangent_frames(centroids)

    anomaly = (
        temperature_face.astype(np.float64)
        - temperature_face.astype(np.float64).mean(axis=1, keepdims=True)
    )                                                                   # (N, K) °C

    grad_a = _tangent_gradient(centroids, face_adjacency, anomaly, earth_radius_km)  # (N, 2, K) °C/km

    # Wind = K · ∇(T_anomaly): blows toward warm anomaly (low pressure).
    wind_u = pressure_wind_scale * grad_a[:, 0, :]                      # (N, K)
    wind_v = pressure_wind_scale * grad_a[:, 1, :]

    winds = (
        wind_u[..., None] * east[:, None, :]
        + wind_v[..., None] * north[:, None, :]
    )                                                                   # (N, K, 3)
    return winds.astype(np.float32)
