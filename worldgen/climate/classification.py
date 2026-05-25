"""Biome classification — scalar reference + vectorised texture form."""
from __future__ import annotations

import numpy as np


BIOME_NAMES: list[str] = [
    "Icecap",       # 0
    "Tundra",       # 1
    "Taiga",        # 2
    "Temperate",    # 3
    "Steppe",       # 4
    "Savannah",     # 5
    "Tropical",     # 6
    "Semi-Arid",    # 7
    "Desert",       # 8
]
N_BIOMES = len(BIOME_NAMES)


def default_biome(temperature_c: np.ndarray,
                  precipitation_mm: np.ndarray,
                  lat_deg: float) -> int:
    """Scalar classifier: per-season T (°C) and P (mm), latitude (deg) → biome index."""
    t_warmest = float(np.max(temperature_c))
    t_coldest = float(np.min(temperature_c))
    p_annual = float(np.sum(precipitation_mm))
    p_seasonality = (
        (float(np.max(precipitation_mm)) - float(np.min(precipitation_mm)))
        / max(p_annual, 1e-6)
    )

    if t_warmest < 0.0:
        return 0  # Icecap
    if t_warmest < 10.0:
        return 1  # Tundra
    if t_coldest < -10.0:
        return 2  # Taiga
    if p_annual < 300.0 and t_warmest > 18.0:
        return 8  # Desert
    if p_annual < 500.0 and t_warmest > 18.0:
        return 7  # Semi-Arid
    if p_annual < 400.0:
        return 4  # Steppe
    if t_warmest > 22.0 and p_annual > 750.0 and p_seasonality < 0.50:
        return 6  # Tropical
    if t_warmest > 21.0 and p_annual > 350.0:
        return 5  # Savannah
    return 3      # Temperate


def classify_textures(temperature_tex: np.ndarray,
                      precipitation_tex: np.ndarray,
                      latitudes_deg: np.ndarray) -> np.ndarray:
    """Vectorised biome classifier over (N, H, W, K) textures.

    The decision tree must be kept in lock-step with `default_biome`; a unit
    test pins per-pixel agreement.

    Args:
        temperature_tex: (N, H, W, K) float °C.
        precipitation_tex: (N, H, W, K) float mm/season.
        latitudes_deg: (N,) face latitude in degrees (unused at present, but
            kept in the signature so swappable classifiers can use it).

    Returns:
        (N, H, W, 1) uint8 biome indices.
    """
    del latitudes_deg  # not needed for default classifier; kept for API parity

    t_warmest = temperature_tex.max(axis=-1)
    t_coldest = temperature_tex.min(axis=-1)
    p_annual = precipitation_tex.sum(axis=-1)
    p_min = precipitation_tex.min(axis=-1)
    p_max = precipitation_tex.max(axis=-1)
    p_seasonality = (p_max - p_min) / np.maximum(p_annual, 1e-6)

    biome = np.full(t_warmest.shape, 3, dtype=np.uint8)  # default = Temperate

    # Apply rules in REVERSE priority order (last assignment wins), then flip,
    # OR use forward order with a "decided" mask. Use the mask approach so the
    # logic exactly mirrors default_biome's first-match-wins ordering.
    decided = np.zeros(t_warmest.shape, dtype=bool)

    def _assign(mask: np.ndarray, value: int) -> None:
        m = mask & ~decided
        biome[m] = value
        decided[m] = True

    _assign(t_warmest < 0.0, 0)                                          # Icecap
    _assign(t_warmest < 10.0, 1)                                          # Tundra
    _assign(t_coldest < -10.0, 2)                                         # Taiga
    _assign((p_annual < 300.0) & (t_warmest > 18.0), 8)                   # Desert
    _assign((p_annual < 500.0) & (t_warmest > 18.0), 7)                   # Semi-Arid
    _assign(p_annual < 400.0, 4)                                          # Steppe
    _assign((t_warmest > 22.0) & (p_annual > 750.0) & (p_seasonality < 0.50), 6)  # Tropical
    _assign((t_warmest > 21.0) & (p_annual > 350.0), 5)                           # Savannah
    # Fallthrough already initialised to Temperate (3).

    return biome[..., None]
