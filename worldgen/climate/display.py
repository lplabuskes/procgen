"""Color schemes and display-texture builders for the climate stage."""
from __future__ import annotations

import numpy as np

from viz.color_stops import interp_stops
from worldgen.geology.display import OCEAN_COLOR

_OCEAN_RGBA = np.array(OCEAN_COLOR, dtype=np.float32)


# 9 biome palette, indexed by BIOME_NAMES order in classification.py.
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


def _is_ocean_face(meta: dict) -> bool:
    if "is_ocean" in meta:
        return bool(meta["is_ocean"])
    return float(meta.get("elevation", 0.0)) < 0.0


def temperature_color(meta: dict, key: str = "temperature_annual") -> tuple:
    """Per-face temperature color scheme, anchored at 0 °C."""
    if _is_ocean_face(meta):
        return OCEAN_COLOR
    t = float(meta.get(key, 0.0))
    return tuple(interp_stops(np.array(t, dtype=np.float32), _TEMP_STOPS).tolist())


def precipitation_color(meta: dict) -> tuple:
    """Per-face annual precipitation color scheme (linear on log mm)."""
    if _is_ocean_face(meta):
        return OCEAN_COLOR
    p = max(float(meta.get("precipitation_annual", 0.0)), 1.0)
    log_stops = [(np.log10(a), c) for a, c in _PRECIP_STOPS_MM]
    return tuple(interp_stops(np.array(np.log10(p), dtype=np.float32), log_stops).tolist())


def humidity_color(meta: dict) -> tuple:
    """Per-face annual mean humidity color scheme."""
    if _is_ocean_face(meta):
        return OCEAN_COLOR
    h = float(meta.get("humidity_annual", 0.0))
    return tuple(interp_stops(np.array(h, dtype=np.float32), _HUMIDITY_STOPS).tolist())


def biome_color(meta: dict) -> tuple:
    """Per-face biome color (categorical)."""
    if _is_ocean_face(meta):
        return OCEAN_COLOR
    idx = int(meta.get("biome", 0))
    if 0 <= idx < len(BIOME_PALETTE):
        return tuple(BIOME_PALETTE[idx].tolist())
    return (0.5, 0.5, 0.5, 1.0)


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
    return [(name, tuple(BIOME_PALETTE[i].tolist())) for i, name in enumerate(BIOME_NAMES)]


def _apply_ocean_mask(rgba: np.ndarray,
                      elevation_tex: np.ndarray | None,
                      ocean_mask_tex: np.ndarray | None = None) -> np.ndarray:
    if ocean_mask_tex is not None:
        ocean = ocean_mask_tex[..., 0] > 0
    elif elevation_tex is not None:
        ocean = elevation_tex[..., 0] < 0.0
    else:
        return rgba
    rgba[ocean] = _OCEAN_RGBA
    return rgba


def build_temperature_display_texture(temperature_tex: np.ndarray,
                                      season: int | None = None,
                                      elevation_tex: np.ndarray | None = None,
                                      ocean_mask_tex: np.ndarray | None = None) -> np.ndarray:
    """Convert (N, H, W, K) °C texture to (N, H, W, 4) RGBA.

    season=None returns annual mean across all K channels; otherwise picks one.
    Ocean pixels (per ocean_mask_tex, or elevation < 0 as fallback) are overwritten
    with the ocean color.
    """
    if season is None:
        values = temperature_tex.mean(axis=-1)
    else:
        values = temperature_tex[..., season]
    rgba = interp_stops(values, _TEMP_STOPS)
    return _apply_ocean_mask(rgba, elevation_tex, ocean_mask_tex)


def build_precipitation_display_texture(precipitation_tex: np.ndarray,
                                        elevation_tex: np.ndarray | None = None,
                                        ocean_mask_tex: np.ndarray | None = None) -> np.ndarray:
    """Convert (N, H, W, K) mm/season texture to (N, H, W, 4) RGBA (annual log scale)."""
    annual = np.maximum(precipitation_tex.sum(axis=-1), 1.0)
    log_stops = [(np.log10(a), c) for a, c in _PRECIP_STOPS_MM]
    rgba = interp_stops(np.log10(annual), log_stops)
    return _apply_ocean_mask(rgba, elevation_tex, ocean_mask_tex)


def build_biome_display_texture(mesh) -> np.ndarray:
    """Look up ``"biome"`` indices in BIOME_PALETTE → (N, H, W, 4) RGBA.

    Ocean pixels are overpainted with OCEAN_COLOR (using the ocean_mask texture
    if present, otherwise falling back to elevation < 0). Inland lake pixels are
    overpainted with LAKE_COLOR.
    """
    from worldgen.hydrology.builder import LAKE_COLOR

    biome_tex = mesh.face_textures["biome"]              # (N, H, W, 1) uint8
    idx = biome_tex[..., 0].astype(np.int32)
    idx = np.clip(idx, 0, len(BIOME_PALETTE) - 1)
    rgba = BIOME_PALETTE[idx].copy()
    if "ocean_mask" in mesh.face_textures:
        ocean = mesh.face_textures["ocean_mask"][..., 0] > 0
        rgba[ocean] = np.asarray(OCEAN_COLOR, dtype=rgba.dtype)
    elif "elevation" in mesh.face_textures:
        elev_tex = mesh.face_textures["elevation"]       # (N, H, W, 1)
        ocean = elev_tex[..., 0] < 0.0
        rgba[ocean] = np.asarray(OCEAN_COLOR, dtype=rgba.dtype)
    if "lake_mask" in mesh.face_textures:
        lake_tex = mesh.face_textures["lake_mask"]       # (N, H, W, 1)
        rgba[lake_tex[:, :, :, 0] > 0] = LAKE_COLOR
    return rgba


