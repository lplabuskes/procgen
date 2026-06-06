"""Color schemes and display-texture builders for the resources stage."""
from __future__ import annotations

import numpy as np

from viz.color_stops import interp_stops
from worldgen.geology.display import OCEAN_COLOR
from worldgen.hydrology.builder import LAKE_COLOR
from worldgen.resources.crops import CROP_NAMES, N_CROPS
from worldgen.resources.minerals import MINERAL_NAMES, MINERAL_PALETTE
from worldgen.resources.rare_bio import RARE_BIO_NAMES, RARE_BIO_PALETTE


_OCEAN_RGBA = np.array(OCEAN_COLOR, dtype=np.float32)
_LAKE_RGBA = np.array(LAKE_COLOR, dtype=np.float32)


# Fertility palette: barren gray → tan → green → deep green.
_FERTILITY_STOPS = [
    (0.00, (0.55, 0.55, 0.52, 1.0)),
    (0.15, (0.78, 0.72, 0.50, 1.0)),
    (0.40, (0.70, 0.78, 0.40, 1.0)),
    (0.65, (0.40, 0.70, 0.32, 1.0)),
    (0.85, (0.18, 0.55, 0.22, 1.0)),
    (1.00, (0.05, 0.35, 0.12, 1.0)),
]

# Forest density palette: open ground tan → light green → dark forest.
_FOREST_STOPS = [
    (0.00, (0.82, 0.78, 0.58, 1.0)),
    (0.20, (0.70, 0.78, 0.48, 1.0)),
    (0.45, (0.40, 0.70, 0.38, 1.0)),
    (0.70, (0.18, 0.52, 0.22, 1.0)),
    (1.00, (0.05, 0.30, 0.10, 1.0)),
]


def _is_ocean_face(meta: dict) -> bool:
    if "is_ocean" in meta:
        return bool(meta["is_ocean"])
    return float(meta.get("elevation", 0.0)) < 0.0


def fertility_color(meta: dict) -> tuple:
    """Per-face fertility color (mean of pixel values)."""
    if _is_ocean_face(meta):
        return OCEAN_COLOR
    v = float(meta.get("fertility_mean", 0.0))
    return tuple(interp_stops(np.array(v, dtype=np.float32), _FERTILITY_STOPS).tolist())


def forest_density_color(meta: dict) -> tuple:
    """Per-face forest-density color (mean of pixel values)."""
    if _is_ocean_face(meta):
        return OCEAN_COLOR
    v = float(meta.get("forest_density_mean", 0.0))
    return tuple(interp_stops(np.array(v, dtype=np.float32), _FOREST_STOPS).tolist())


def fertility_legend_entries() -> list[tuple[str, tuple]]:
    labels = ["barren", "0.15", "0.40", "0.65", "0.85", "lush"]
    return [(lbl, c) for lbl, (_, c) in zip(labels, _FERTILITY_STOPS)]


def forest_density_legend_entries() -> list[tuple[str, tuple]]:
    labels = ["open", "0.20", "0.45", "0.70", "dense"]
    return [(lbl, c) for lbl, (_, c) in zip(labels, _FOREST_STOPS)]


def _apply_water_masks(
    rgba: np.ndarray,
    elevation_tex: np.ndarray | None,
    ocean_mask_tex: np.ndarray | None,
    lake_mask_tex: np.ndarray | None,
) -> np.ndarray:
    if ocean_mask_tex is not None:
        ocean = ocean_mask_tex[..., 0] > 0
        rgba[ocean] = _OCEAN_RGBA
    elif elevation_tex is not None:
        ocean = elevation_tex[..., 0] < 0.0
        rgba[ocean] = _OCEAN_RGBA
    if lake_mask_tex is not None:
        lake = lake_mask_tex[..., 0] > 0
        rgba[lake] = _LAKE_RGBA
    return rgba


def build_fertility_display_texture(
    fertility_tex: np.ndarray,
    elevation_tex: np.ndarray | None = None,
    ocean_mask_tex: np.ndarray | None = None,
    lake_mask_tex: np.ndarray | None = None,
) -> np.ndarray:
    """Convert (N, H, W, 1) fertility texture to (N, H, W, 4) RGBA."""
    values = fertility_tex[..., 0]
    rgba = interp_stops(values, _FERTILITY_STOPS)
    return _apply_water_masks(rgba, elevation_tex, ocean_mask_tex, lake_mask_tex)


# Crop palette — one RGBA per CROP_NAMES entry.
# None: muted gray; crops: visually distinct natural hues.
_CROP_PALETTE = np.array([
    [0.70, 0.70, 0.65, 1.0],  # 0  None      — muted gray
    [0.20, 0.65, 0.50, 1.0],  # 1  Rice       — teal-green (flooded paddies)
    [0.92, 0.78, 0.10, 1.0],  # 2  Sugarcane  — amber/gold
    [0.88, 0.92, 0.20, 1.0],  # 3  Maize      — yellow-green
    [0.92, 0.82, 0.48, 1.0],  # 4  Wheat      — golden straw
    [0.72, 0.88, 0.58, 1.0],  # 5  Barley     — pale green
    [0.82, 0.60, 0.20, 1.0],  # 6  Millet     — sandy brown
    [0.95, 0.92, 0.82, 1.0],  # 7  Cotton     — cream/ivory
    [0.38, 0.20, 0.08, 1.0],  # 8  Coffee     — dark roast brown
], dtype=np.float32)


def crop_color(meta: dict) -> tuple:
    """Per-face dominant-crop color."""
    if _is_ocean_face(meta):
        return OCEAN_COLOR
    idx = int(meta.get("dominant_crop", 0))
    idx = max(0, min(idx, N_CROPS - 1))
    return tuple(_CROP_PALETTE[idx].tolist())


def crop_legend_entries() -> list[tuple[str, tuple]]:
    return [(name, tuple(_CROP_PALETTE[i].tolist())) for i, name in enumerate(CROP_NAMES)]


def build_dominant_crop_display_texture(
    crop_tex: np.ndarray,
    ocean_mask_tex: np.ndarray | None = None,
    lake_mask_tex: np.ndarray | None = None,
) -> np.ndarray:
    """Convert (N, H, W, 1) uint8 crop texture to (N, H, W, 4) RGBA."""
    idx = np.clip(crop_tex[..., 0].astype(np.int32), 0, N_CROPS - 1)
    rgba = _CROP_PALETTE[idx].copy()                          # (N, H, W, 4)
    return _apply_water_masks(rgba, None, ocean_mask_tex, lake_mask_tex)


def mineral_legend_entries() -> list[tuple[str, tuple]]:
    return [(name, tuple(MINERAL_PALETTE[i].tolist())) for i, name in enumerate(MINERAL_NAMES)]


def rare_bio_legend_entries() -> list[tuple[str, tuple]]:
    return [(name, tuple(RARE_BIO_PALETTE[i].tolist())) for i, name in enumerate(RARE_BIO_NAMES)]


def build_forest_density_display_texture(
    forest_tex: np.ndarray,
    elevation_tex: np.ndarray | None = None,
    ocean_mask_tex: np.ndarray | None = None,
    lake_mask_tex: np.ndarray | None = None,
) -> np.ndarray:
    """Convert (N, H, W, 1) forest density texture to (N, H, W, 4) RGBA."""
    values = forest_tex[..., 0]
    rgba = interp_stops(values, _FOREST_STOPS)
    return _apply_water_masks(rgba, elevation_tex, ocean_mask_tex, lake_mask_tex)
