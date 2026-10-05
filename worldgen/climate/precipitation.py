"""Per-pixel seasonal precipitation correction."""
from __future__ import annotations

import numpy as np


def per_pixel_precipitation(pixel_humidity: np.ndarray,
                            face_winds: np.ndarray,
                            face_u_axes: np.ndarray,
                            face_v_axes: np.ndarray,
                            face_u_ranges_km: np.ndarray,
                            face_v_ranges_km: np.ndarray,
                            pixel_elevations_km: np.ndarray,
                            *,
                            orographic_k: float,
                            precipitation_scale: float) -> np.ndarray:
    """Convert per-pixel humidity to precipitation (mm/season) with orographic correction.

    Args:
        pixel_humidity: (N, H, W, K) from per_pixel_humidity.
        face_winds: (N, K, 3) m/s wind vectors at each face.
        face_u_axes, face_v_axes: (N, 3) unit tangent axes per face.
        face_u_ranges_km, face_v_ranges_km: (N,) face UV span in km.
        pixel_elevations_km: (N, H, W) per-pixel elevation, km.
        orographic_k: orographic enhancement coefficient.
        precipitation_scale: annual precipitation at full moisture (mm/yr).

    Returns:
        (N, H, W, K) float32 mm/season.
    """
    n, h, w = pixel_elevations_km.shape

    # Slope in km/km.
    dh_dv, dh_du = np.gradient(pixel_elevations_km, axis=(1, 2))
    pix_du_km = face_u_ranges_km[:, None, None] / max(w - 1, 1)
    pix_dv_km = face_v_ranges_km[:, None, None] / max(h - 1, 1)
    slope_u = dh_du / np.maximum(pix_du_km, 1e-9)                  # (N, H, W)
    slope_v = dh_dv / np.maximum(pix_dv_km, 1e-9)                  # (N, H, W)

    # Wind in face UV frame: (N, K).
    wind_u = np.einsum("nkj,nj->nk", face_winds, face_u_axes)
    wind_v = np.einsum("nkj,nj->nk", face_winds, face_v_axes)

    # Per-pixel uplift (m/s).
    uplift_pix = (
        wind_u[:, None, None, :] * slope_u[..., None]
        + wind_v[:, None, None, :] * slope_v[..., None]
    )                                                               # (N, H, W, K)

    oro_factor = np.clip(1.0 + orographic_k * uplift_pix, 0.3, 2.0)
    mm_per_season = precipitation_scale / 4.0
    return (pixel_humidity * oro_factor * mm_per_season).astype(np.float32)
