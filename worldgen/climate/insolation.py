"""Daily-mean top-of-atmosphere insolation per face per season."""
from __future__ import annotations

import numpy as np


SOLAR_CONSTANT = 1361.0  # W/m²

# Solar declinations at the four canonical seasons.
SEASON_DECLINATIONS_DEG = np.array([0.0, 23.5, 0.0, -23.5], dtype=np.float64)


def seasonal_insolation(latitudes_rad: np.ndarray,
                        declinations_rad: np.ndarray | None = None) -> np.ndarray:
    """Compute daily-mean TOA insolation Q(lat, season) in W/m².

    Args:
        latitudes_rad: (N,) latitudes in radians.
        declinations_rad: (K,) solar declinations in radians. Defaults to the
            four canonical seasons (equinoxes and solstices).

    Returns:
        (N, K) float64 array of daily-mean insolation in W/m².
    """
    if declinations_rad is None:
        declinations_rad = np.deg2rad(SEASON_DECLINATIONS_DEG)

    phi = latitudes_rad[:, None]                   # (N, 1)
    delta = declinations_rad[None, :]              # (1, K)

    tan_product = np.tan(phi) * np.tan(delta)

    # H0 = arccos(-tan φ · tan δ). Undefined (=> polar day/night) when |-tan φ · tan δ| > 1.
    polar_night = tan_product <= -1.0     # -tan·tan ≥ +1 → sun never rises
    polar_day = tan_product >= 1.0        # -tan·tan ≤ -1 → sun never sets

    cos_h0 = np.clip(-tan_product, -1.0, 1.0)
    h0 = np.arccos(cos_h0)                          # (N, K)

    q = (SOLAR_CONSTANT / np.pi) * (
        np.cos(phi) * np.cos(delta) * np.sin(h0) + h0 * np.sin(phi) * np.sin(delta)
    )

    q = np.where(polar_night, 0.0, q)
    q = np.where(polar_day, SOLAR_CONSTANT * np.sin(phi) * np.sin(delta), q)

    return q


def face_latitudes(face_centroids: np.ndarray) -> np.ndarray:
    """Return (N,) latitudes in radians for unit-sphere face centroids."""
    return np.arcsin(np.clip(face_centroids[:, 2], -1.0, 1.0))
