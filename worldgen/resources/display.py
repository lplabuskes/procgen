"""Color schemes and display-texture builders for the resources stage."""
from __future__ import annotations

import numpy as np

from core.display import TextureRecipe
from viz.color_stops import gradient_texture, categorical_texture, interp_stops_scalar, palette_legend_entries
from worldgen.geology.display import apply_water_masks, is_ocean_face, ocean_gradient_scheme, ocean_categorical_scheme
from worldgen.resources.crops import CROP_NAMES, N_CROPS
from worldgen.resources.minerals import MINERAL_NAMES, MINERAL_PALETTE
from worldgen.resources.rare_bio import RARE_BIO_NAMES, RARE_BIO_PALETTE


# Fertility palette: barren gray → tan → green → deep green.
_FERTILITY_STOPS = [
    (0.00, (0.55, 0.55, 0.52, 1.0)),
    (0.15, (0.78, 0.72, 0.50, 1.0)),
    (0.40, (0.70, 0.78, 0.40, 1.0)),
    (0.65, (0.40, 0.70, 0.32, 1.0)),
    (0.85, (0.18, 0.55, 0.22, 1.0)),
    (1.00, (0.05, 0.35, 0.12, 1.0)),
]

# Aquatic fertility palette: deep dark blue → teal → bright aqua.
_WATER_FERTILITY_STOPS = [
    (0.00, (0.04, 0.18, 0.30, 1.0)),
    (0.20, (0.08, 0.38, 0.55, 1.0)),
    (0.45, (0.12, 0.58, 0.72, 1.0)),
    (0.70, (0.18, 0.78, 0.82, 1.0)),
    (1.00, (0.25, 0.92, 0.88, 1.0)),
]

# Forest density palette: open ground tan → light green → dark forest.
_FOREST_STOPS = [
    (0.00, (0.82, 0.78, 0.58, 1.0)),
    (0.20, (0.70, 0.78, 0.48, 1.0)),
    (0.45, (0.40, 0.70, 0.38, 1.0)),
    (0.70, (0.18, 0.52, 0.22, 1.0)),
    (1.00, (0.05, 0.30, 0.10, 1.0)),
]


def _fertility_color(meta: dict) -> tuple:
    val = meta.get("fertility_mean", 0.0)
    if is_ocean_face(meta) or meta.get("is_water", False):
        return interp_stops_scalar(val, _WATER_FERTILITY_STOPS)
    return interp_stops_scalar(val, _FERTILITY_STOPS)


fertility_color = _fertility_color
forest_density_color = ocean_gradient_scheme("forest_density_mean", _FOREST_STOPS)


def fertility_legend_entries() -> list[tuple[str, tuple]]:
    labels = ["barren", "0.15", "0.40", "0.65", "0.85", "lush"]
    return [(lbl, c) for lbl, (_, c) in zip(labels, _FERTILITY_STOPS)]


def forest_density_legend_entries() -> list[tuple[str, tuple]]:
    labels = ["open", "0.20", "0.45", "0.70", "dense"]
    return [(lbl, c) for lbl, (_, c) in zip(labels, _FOREST_STOPS)]


# Crop palette — one RGBA per CROP_NAMES entry.
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


crop_color = ocean_categorical_scheme("dominant_crop", _CROP_PALETTE, N_CROPS)


def crop_legend_entries() -> list[tuple[str, tuple]]:
    return palette_legend_entries(CROP_NAMES, _CROP_PALETTE)


def mineral_legend_entries() -> list[tuple[str, tuple]]:
    return palette_legend_entries(MINERAL_NAMES, MINERAL_PALETTE)


def rare_bio_legend_entries() -> list[tuple[str, tuple]]:
    return palette_legend_entries(RARE_BIO_NAMES, RARE_BIO_PALETTE)


def build_fertility_display_texture(
    fertility_tex: np.ndarray,
    elevation_tex: np.ndarray | None = None,
    ocean_mask_tex: np.ndarray | None = None,
    lake_mask_tex: np.ndarray | None = None,
) -> np.ndarray:
    """Convert (N, H, W, 1) fertility texture to (N, H, W, 4) RGBA."""
    rgba = gradient_texture(fertility_tex, _FERTILITY_STOPS)
    water_mask = np.zeros(fertility_tex.shape[:3], dtype=bool)
    if ocean_mask_tex is not None:
        water_mask |= ocean_mask_tex[..., 0] > 0
    elif elevation_tex is not None:
        water_mask |= elevation_tex[..., 0] < 0.0
    if lake_mask_tex is not None:
        water_mask |= lake_mask_tex[..., 0] > 0
    if water_mask.any():
        water_rgba = gradient_texture(fertility_tex, _WATER_FERTILITY_STOPS)
        rgba[water_mask] = water_rgba[water_mask]
    return rgba


def build_forest_density_display_texture(
    forest_tex: np.ndarray,
    elevation_tex: np.ndarray | None = None,
    ocean_mask_tex: np.ndarray | None = None,
    lake_mask_tex: np.ndarray | None = None,
) -> np.ndarray:
    """Convert (N, H, W, 1) forest density texture to (N, H, W, 4) RGBA."""
    rgba = gradient_texture(forest_tex, _FOREST_STOPS)
    return apply_water_masks(
        rgba, elevation_tex=elevation_tex, ocean_mask_tex=ocean_mask_tex, lake_mask_tex=lake_mask_tex
    )


def build_dominant_crop_display_texture(
    crop_tex: np.ndarray,
    ocean_mask_tex: np.ndarray | None = None,
    lake_mask_tex: np.ndarray | None = None,
) -> np.ndarray:
    """Convert (N, H, W, 1) uint8 crop texture to (N, H, W, 4) RGBA."""
    rgba = categorical_texture(crop_tex, _CROP_PALETTE, N_CROPS)
    return apply_water_masks(rgba, ocean_mask_tex=ocean_mask_tex, lake_mask_tex=lake_mask_tex)


# Resource layers as recipes, for the world map and refined regions alike.
def _water(tex) -> dict:
    return dict(
        elevation_tex=tex.get("elevation"), ocean_mask_tex=tex.get("ocean_mask"),
        lake_mask_tex=tex.get("lake_mask"),
    )


FERTILITY_RECIPE = TextureRecipe(
    lambda tex: build_fertility_display_texture(tex["fertility"], **_water(tex)), sources=("fertility",), relief=True,
)
FOREST_DENSITY_RECIPE = TextureRecipe(
    lambda tex: build_forest_density_display_texture(tex["forest_density"], **_water(tex)),
    sources=("forest_density",), relief=True,
)
DOMINANT_CROP_RECIPE = TextureRecipe(
    lambda tex: build_dominant_crop_display_texture(
        tex["dominant_crop"], ocean_mask_tex=tex.get("ocean_mask"), lake_mask_tex=tex.get("lake_mask"),
    ),
    sources=("dominant_crop",), relief=True,
)
