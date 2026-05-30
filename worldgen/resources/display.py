"""Color schemes and display-texture builders for the resources stage."""
from __future__ import annotations

import numpy as np

from worldgen.climate.display import _interp_stops
from worldgen.geology.display import OCEAN_COLOR
from worldgen.hydrology.builder import LAKE_COLOR


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
    return tuple(_interp_stops(np.array(v, dtype=np.float32), _FERTILITY_STOPS).tolist())


def forest_density_color(meta: dict) -> tuple:
    """Per-face forest-density color (mean of pixel values)."""
    if _is_ocean_face(meta):
        return OCEAN_COLOR
    v = float(meta.get("forest_density_mean", 0.0))
    return tuple(_interp_stops(np.array(v, dtype=np.float32), _FOREST_STOPS).tolist())


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
        ocean = ocean_mask_tex[..., 0] > 0.5
        rgba[ocean] = _OCEAN_RGBA
    elif elevation_tex is not None:
        ocean = elevation_tex[..., 0] < 0.0
        rgba[ocean] = _OCEAN_RGBA
    if lake_mask_tex is not None:
        lake = lake_mask_tex[..., 0] > 0.5
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
    rgba = _interp_stops(values, _FERTILITY_STOPS)
    return _apply_water_masks(rgba, elevation_tex, ocean_mask_tex, lake_mask_tex)


def build_forest_density_display_texture(
    forest_tex: np.ndarray,
    elevation_tex: np.ndarray | None = None,
    ocean_mask_tex: np.ndarray | None = None,
    lake_mask_tex: np.ndarray | None = None,
) -> np.ndarray:
    """Convert (N, H, W, 1) forest density texture to (N, H, W, 4) RGBA."""
    values = forest_tex[..., 0]
    rgba = _interp_stops(values, _FOREST_STOPS)
    return _apply_water_masks(rgba, elevation_tex, ocean_mask_tex, lake_mask_tex)
