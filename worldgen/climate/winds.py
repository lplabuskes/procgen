"""Zonal Hadley/Ferrel/Polar surface winds with seasonal ITCZ shift."""
from __future__ import annotations

import numpy as np


SEASON_PHASES = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2], dtype=np.float64)


def _tangent_frames(face_centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-face east/north unit vectors. At the poles falls back to a fixed frame."""
    z_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    east = np.cross(z_hat, face_centroids)                 # (N, 3)
    east_norm = np.linalg.norm(east, axis=1, keepdims=True)
    polar = east_norm[:, 0] < 1e-6
    fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    east = np.where(polar[:, None], fallback, east / np.maximum(east_norm, 1e-12))
    north = np.cross(face_centroids, east)                 # (N, 3)
    north /= np.maximum(np.linalg.norm(north, axis=1, keepdims=True), 1e-12)
    return east, north


def zonal_winds(face_centroids: np.ndarray,
                latitudes_rad: np.ndarray,
                *,
                trade_speed: float,
                westerly_speed: float,
                polar_speed: float,
                itcz_shift_deg: float,
                season_phases: np.ndarray | None = None) -> np.ndarray:
    """Compute prevailing surface winds at face centroids per season.

    Args:
        face_centroids: (N, 3) unit-sphere face positions.
        latitudes_rad: (N,) face latitudes in radians.
        trade_speed, westerly_speed, polar_speed: band speeds in m/s.
        itcz_shift_deg: amplitude of seasonal ITCZ migration in degrees.
        season_phases: (K,) seasonal phases in radians. Defaults to four seasons
            anchored at 0, π/2, π, 3π/2.

    Returns:
        (N, K, 3) float32 wind velocity vectors (m/s) tangent to the sphere.
    """
    if season_phases is None:
        season_phases = SEASON_PHASES

    east, north = _tangent_frames(face_centroids)                     # (N, 3) each

    lat_deg = np.rad2deg(latitudes_rad)                                # (N,)
    itcz_lat = itcz_shift_deg * np.sin(season_phases)                  # (K,)
    phi_eff = lat_deg[:, None] - itcz_lat[None, :]                     # (N, K)
    band_dist = np.abs(phi_eff)
    hemi_sign = np.where(phi_eff >= 0.0, 1.0, -1.0)                    # +1 N of thermal equator

    trade = band_dist < 30.0
    westerly = (band_dist >= 30.0) & (band_dist < 60.0)
    # polar is the fallthrough

    # Direction in (east, north) basis. Trade & polar blow toward (W, equator);
    # westerlies blow toward (E, pole). Equatorward = -hemi_sign in `north`.
    u_east = np.where(westerly, 1.0, -1.0)
    v_north = np.where(westerly, hemi_sign, -hemi_sign)

    # Normalise the (u, v) direction so diagonal winds aren't √2× stronger.
    inv_mag = 1.0 / np.sqrt(u_east ** 2 + v_north ** 2)
    u_east *= inv_mag
    v_north *= inv_mag

    speed = np.where(trade, trade_speed,
                     np.where(westerly, westerly_speed, polar_speed))   # (N, K)

    # Compose 3D vectors: w = speed * (u_east * east_hat + v_north * north_hat)
    winds = (
        (speed * u_east)[..., None] * east[:, None, :]
        + (speed * v_north)[..., None] * north[:, None, :]
    )                                                                    # (N, K, 3)
    return winds.astype(np.float32)
