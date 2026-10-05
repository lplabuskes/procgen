"""Color schemes and display-texture builders for the climate stage."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from core.display import TextureRecipe
from viz.color_stops import interp_stops, interp_stops_scalar, categorical_texture, palette_legend_entries
from worldgen.geology.display import (
    OCEAN_COLOR,
    apply_water_masks,
    ocean_gradient_scheme,
    ocean_categorical_scheme,
)


# Biome palette, indexed by BIOME_NAMES order in classification.py.
BIOME_PALETTE = np.array([
    [0.95, 0.97, 0.99, 1.0],   # 0 Icecap
    [0.78, 0.82, 0.78, 1.0],   # 1 Tundra
    [0.30, 0.45, 0.30, 1.0],   # 2 Taiga
    [0.45, 0.70, 0.30, 1.0],   # 3 Temperate
    [0.78, 0.78, 0.45, 1.0],   # 4 Steppe
    [0.85, 0.78, 0.35, 1.0],   # 5 Savannah
    [0.10, 0.55, 0.20, 1.0],   # 6 Tropical
    [0.85, 0.65, 0.32, 1.0],   # 7 Semi-Arid
    [0.92, 0.85, 0.55, 1.0],   # 8 Desert
    [0.30, 0.55, 0.52, 1.0],   # 9 Wetland
], dtype=np.float32)


# Temperature diverging palette anchored at 0 °C.
# Stops as (°C, RGBA).
_TEMP_STOPS = [
    (-40.0, (0.05, 0.05, 0.35, 1.0)),
    (-20.0, (0.10, 0.30, 0.65, 1.0)),
    (-5.0,  (0.45, 0.65, 0.92, 1.0)),
    (0.0,   (0.95, 0.95, 0.95, 1.0)),
    (10.0,  (0.95, 0.90, 0.55, 1.0)),
    (20.0,  (0.95, 0.65, 0.35, 1.0)),
    (30.0,  (0.85, 0.25, 0.20, 1.0)),
    (40.0,  (0.55, 0.05, 0.10, 1.0)),
]

# Precipitation sequential palette in annual mm (log-spaced anchors).
_PRECIP_STOPS_MM = [
    (50.0,   (0.95, 0.92, 0.55, 1.0)),
    (250.0,  (0.85, 0.85, 0.40, 1.0)),
    (600.0,  (0.55, 0.78, 0.40, 1.0)),
    (1200.0, (0.30, 0.65, 0.55, 1.0)),
    (2400.0, (0.20, 0.45, 0.75, 1.0)),
    (4000.0, (0.10, 0.20, 0.55, 1.0)),
]
_LOG_PRECIP_STOPS = [(np.log10(a), c) for a, c in _PRECIP_STOPS_MM]


# Humidity sequential palette: arid tan → moist blue-green.
# Anchors are in dimensionless q_air units (evaporation_scale=1 gives
# source values ~[0.3, 1.0]; ITCZ convergence can reach ~2.0).
_HUMIDITY_STOPS = [
    (0.05, (0.95, 0.90, 0.65, 1.0)),
    (0.3,  (0.80, 0.75, 0.45, 1.0)),
    (0.7,  (0.50, 0.78, 0.45, 1.0)),
    (1.2,  (0.20, 0.60, 0.65, 1.0)),
    (1.8,  (0.10, 0.30, 0.65, 1.0)),
]


temperature_color = ocean_gradient_scheme("temperature_annual", _TEMP_STOPS)
temperature_warmest_color = ocean_gradient_scheme("temperature_warmest", _TEMP_STOPS)
temperature_coldest_color = ocean_gradient_scheme("temperature_coldest", _TEMP_STOPS)
humidity_color = ocean_gradient_scheme("humidity_annual", _HUMIDITY_STOPS)
biome_color = ocean_categorical_scheme("biome", BIOME_PALETTE, len(BIOME_PALETTE))


def precipitation_color(meta: dict) -> tuple:
    """Per-face annual precipitation color scheme (linear on log mm)."""
    from worldgen.geology.display import is_ocean_face
    if is_ocean_face(meta):
        return OCEAN_COLOR
    p = max(float(meta.get("precipitation_annual", 0.0)), 1.0)
    return interp_stops_scalar(np.log10(p), _LOG_PRECIP_STOPS)


def temperature_legend_entries() -> list[tuple[str, tuple]]:
    """Ordered (label, rgba) entries for a gradient legend, cold→hot."""
    result = []
    for t, c in _TEMP_STOPS:
        label = "0 °C" if t == 0 else f"{int(t):+d} °C"
        result.append((label, c))
    return result


def precipitation_legend_entries() -> list[tuple[str, tuple]]:
    """Ordered (label, rgba) entries for a gradient legend, dry→wet (annual mm)."""
    labels = ["50 mm", "250 mm", "600 mm", "1200 mm", "2400 mm", "4000 mm"]
    return [(lbl, c) for lbl, (_, c) in zip(labels, _PRECIP_STOPS_MM)]


def biome_legend_entries() -> list[tuple[str, tuple]]:
    """(name, rgba) pairs for all biomes in index order."""
    from worldgen.climate.classification import BIOME_NAMES
    return palette_legend_entries(BIOME_NAMES, BIOME_PALETTE)


def _render_temperature(season: int | None):
    def render(tex: Mapping[str, np.ndarray]) -> np.ndarray:
        """(N, H, W, K) °C → (N, H, W, 4) RGBA, one season or the annual mean; ocean masked."""
        t = tex["temperature"]
        values = t.mean(axis=-1) if season is None else t[..., season]
        return apply_water_masks(
            interp_stops(values, _TEMP_STOPS),
            elevation_tex=tex.get("elevation"), ocean_mask_tex=tex.get("ocean_mask"),
        )
    return render


def _render_precipitation(tex: Mapping[str, np.ndarray]) -> np.ndarray:
    """(N, H, W, K) mm/season → (N, H, W, 4) RGBA on an annual log scale; ocean masked."""
    annual = np.maximum(tex["precipitation"].sum(axis=-1), 1.0)
    return apply_water_masks(
        interp_stops(np.log10(annual), _LOG_PRECIP_STOPS),
        elevation_tex=tex.get("elevation"), ocean_mask_tex=tex.get("ocean_mask"),
    )


def _render_biome(tex: Mapping[str, np.ndarray]) -> np.ndarray:
    """Biome indices through BIOME_PALETTE; ocean and lakes masked."""
    rgba = categorical_texture(tex["biome"], BIOME_PALETTE, len(BIOME_PALETTE))
    return apply_water_masks(
        rgba,
        elevation_tex=tex.get("elevation"),
        ocean_mask_tex=tex.get("ocean_mask"),
        lake_mask_tex=tex.get("lake_mask"),
    )


# The climate layers, for the world map and (from refined data) regions alike.
# Seasons follow SEASON_LABELS: 1 is summer, 3 is winter.
BIOME_RECIPE = TextureRecipe(_render_biome, sources=("biome",), relief=True)
ANNUAL_TEMPERATURE_RECIPE = TextureRecipe(_render_temperature(None), sources=("temperature",))
SUMMER_TEMPERATURE_RECIPE = TextureRecipe(_render_temperature(1), sources=("temperature",))
WINTER_TEMPERATURE_RECIPE = TextureRecipe(_render_temperature(3), sources=("temperature",))
PRECIPITATION_RECIPE = TextureRecipe(_render_precipitation, sources=("precipitation",))
