"""River flow in physical units: discharge and channel width.

River ``flow`` is a precipitation-weighted texel count — each texel upstream
contributes its annual precipitation over the world mean — so one flow unit is
one global texel's area receiving the mean precipitation.  Multiplying that
water by a runoff ratio gives the discharge; the channel width then follows the
at-a-station hydraulic-geometry fit ``w = a · Q^b`` (Leopold & Maddock; the
global fit of Moody & Troutman gives a ≈ 7 m, b ≈ 0.5: ~3 km for an Amazon, a
few hundred metres for a Rhine).
"""
from __future__ import annotations

import numpy as np

from core.mesh import PolygonMesh
from worldgen.climate.channels import PRECIPITATION

EARTH_RADIUS_KM = 6371.0
SECONDS_PER_YEAR = 3.156e7
# The share of precipitation that leaves as river flow (Earth's land: ~40k of
# ~110k km³/yr).
RUNOFF_RATIO = 0.36
WIDTH_COEFF_M = 7.0
WIDTH_EXPONENT = 0.5


def discharge_per_flow_unit(mesh: PolygonMesh) -> float:
    """m³/s carried by one unit of river flow on ``mesh``; 0 without precipitation."""
    if not PRECIPITATION.present(mesh):
        return 0.0
    precip = PRECIPITATION.read(mesh)
    n, h, w = precip.shape[:3]
    mean_mm_yr = float(precip.astype(np.float32).sum(axis=-1).mean())
    texel_area_m2 = 4.0 * np.pi * EARTH_RADIUS_KM ** 2 / (n * h * w) * 1e6
    return texel_area_m2 * mean_mm_yr * 1e-3 * RUNOFF_RATIO / SECONDS_PER_YEAR


def channel_width_km(discharge_m3s: np.ndarray | float) -> np.ndarray:
    """Bankfull channel width (km) of a river carrying ``discharge_m3s``."""
    q = np.maximum(np.asarray(discharge_m3s, dtype=np.float64), 0.0)
    return WIDTH_COEFF_M * q ** WIDTH_EXPONENT * 1e-3
