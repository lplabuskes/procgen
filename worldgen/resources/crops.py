"""Per-pixel dominant crop classification.

Mirrors the structure of `worldgen/climate/classification.py`: a simple
threshold table (first-match wins) applied to per-pixel growing-season
temperature, annual precipitation, and elevation. Fertility acts as a
gate — pixels below `FERTILITY_GATE` yield CROP_NONE regardless of climate.

Crops are ordered from most-specific to least-specific so the first match
is the tightest ecological fit.
"""
from __future__ import annotations

import numpy as np


CROP_NAMES: list[str] = [
    "None",       # 0 — no viable crop
    "Rice",       # 1
    "Sugarcane",  # 2
    "Maize",      # 3
    "Wheat",      # 4
    "Barley",     # 5
    "Millet",     # 6
    "Cotton",     # 7
    "Coffee",     # 8
]
N_CROPS = len(CROP_NAMES)
CROP_NONE = 0

# Fertility below this threshold → CROP_NONE regardless of climate.
FERTILITY_GATE = 0.08

# ── threshold table ──────────────────────────────────────────────────────────
# Each row: (crop_index, t_grow_min, t_grow_max, p_annual_min, p_annual_max,
#            elev_min_km, elev_max_km)
# Applied in forward order; first match wins.
_CROP_TABLE: list[tuple] = [
    # Sugarcane: hot lowland tropics, very wet.
    (2,  22.0, 50.0, 1800.0, 9999.0, -9.0, 1.0),
    # Coffee: narrow tropical highland band, extremely wet.
    (8,  15.0, 25.0, 2000.0, 9999.0,  0.9, 1.9),
    # Rice: hot and wet (wins over coffee at lowland after sugarcane).
    (1,  20.0, 50.0, 1000.0, 9999.0, -9.0, 2.0),
    # Millet: hot semi-arid — must come before cotton to claim the dry niche.
    (6,  22.0, 50.0,  200.0,  750.0, -9.0, 2.0),
    # Cotton: hot, moderate precipitation.
    (7,  18.0, 50.0,  500.0, 1500.0, -9.0, 2.0),
    # Maize: warm to hot, broad precipitation range.
    (3,  16.0, 50.0,  400.0, 2500.0, -9.0, 3.0),
    # Wheat: cool temperate, moderate precipitation.
    (4,   5.0, 28.0,  250.0, 1500.0, -9.0, 3.5),
    # Barley: cold-tolerant and drought-tolerant — last resort.
    (5,   3.0, 26.0,  150.0, 1200.0, -9.0, 4.5),
]


def default_crop(
    temperature_c: np.ndarray,
    precipitation_mm: np.ndarray,
    elevation_km: float,
    fertility: float,
) -> int:
    """Scalar crop classifier.

    Args:
        temperature_c: (K,) per-season temperatures in °C.
        precipitation_mm: (K,) per-season precipitation in mm.
        elevation_km: pixel elevation.
        fertility: pixel fertility in [0, 1].

    Returns:
        Crop index (0 = None).
    """
    if fertility < FERTILITY_GATE:
        return CROP_NONE

    warm = temperature_c > 5.0
    if warm.any():
        t_grow = float(temperature_c[warm].mean())
    else:
        t_grow = float(temperature_c.mean())

    p_annual = float(precipitation_mm.sum())

    for crop_idx, t_min, t_max, p_min, p_max, e_min, e_max in _CROP_TABLE:
        if (t_min <= t_grow <= t_max
                and p_min <= p_annual <= p_max
                and e_min <= elevation_km <= e_max):
            return int(crop_idx)
    return CROP_NONE


def classify_crops(
    temperature_tex: np.ndarray,
    precipitation_tex: np.ndarray,
    elevation_tex: np.ndarray,
    fertility_tex: np.ndarray,
    ocean_mask_tex: np.ndarray,
) -> np.ndarray:
    """Vectorised crop classifier over (N, H, W, *) textures.

    Args:
        temperature_tex: (N, H, W, K) float32 °C.
        precipitation_tex: (N, H, W, K) float32 mm/season.
        elevation_tex: (N, H, W, 1) float32 km.
        fertility_tex: (N, H, W, 1) float32 in [0, 1].
        ocean_mask_tex: (N, H, W, 1) float32; 1 = ocean.

    Returns:
        (N, H, W, 1) uint8 crop indices.
    """
    t = temperature_tex.astype(np.float32)
    warm = t > 5.0
    warm_count = warm.sum(axis=-1).astype(np.float32)         # (N, H, W)
    warm_sum = np.where(warm, t, 0.0).sum(axis=-1)            # (N, H, W)
    t_grow = np.where(
        warm_count > 0,
        warm_sum / np.maximum(warm_count, 1.0),
        t.mean(axis=-1),
    ).astype(np.float32)                                       # (N, H, W)

    p_annual = precipitation_tex.sum(axis=-1).astype(np.float32)  # (N, H, W)
    elevation = elevation_tex[..., 0].astype(np.float32)           # (N, H, W)
    fertility = fertility_tex[..., 0].astype(np.float32)           # (N, H, W)
    ocean = ocean_mask_tex[..., 0] > 0.5                           # (N, H, W) bool

    # Start with CROP_NONE everywhere; apply table in reverse so earlier
    # (more-specific) rows override later (less-specific) ones.
    crop = np.zeros(t_grow.shape, dtype=np.uint8)
    for crop_idx, t_min, t_max, p_min, p_max, e_min, e_max in reversed(_CROP_TABLE):
        mask = (
            (t_grow >= t_min) & (t_grow <= t_max)
            & (p_annual >= p_min) & (p_annual <= p_max)
            & (elevation >= e_min) & (elevation <= e_max)
        )
        crop = np.where(mask, np.uint8(crop_idx), crop)

    crop = np.where(fertility < FERTILITY_GATE, np.uint8(CROP_NONE), crop)
    crop = np.where(ocean, np.uint8(CROP_NONE), crop)

    return crop[..., None]  # (N, H, W, 1)
