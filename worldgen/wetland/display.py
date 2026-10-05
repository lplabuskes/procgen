"""Colour scheme for the wetland-score debug layer."""
from __future__ import annotations

import numpy as np

from core.display import TextureRecipe
from viz.color_stops import gradient_texture
from worldgen.geology.display import apply_water_masks

# Dry (stone) → damp (olive) → wetland (teal, the Wetland biome colour).
_WETLAND_STOPS = [
    (0.00, (0.62, 0.58, 0.52, 1.0)),
    (0.25, (0.62, 0.64, 0.45, 1.0)),
    (0.50, (0.30, 0.55, 0.52, 1.0)),
    (1.00, (0.08, 0.30, 0.35, 1.0)),
]


def wetland_color(meta: dict) -> tuple:
    """Per-face fallback: the face's wetland share."""
    from worldgen.geology.display import OCEAN_COLOR, is_ocean_face
    from viz.color_stops import interp_stops_scalar
    if is_ocean_face(meta):
        return OCEAN_COLOR
    return interp_stops_scalar(float(meta.get("wetland_fraction", 0.0)), _WETLAND_STOPS)


def wetland_legend_entries() -> list[tuple[str, tuple]]:
    labels = ["dry", "damp", "wetland", "saturated"]
    return [(lbl, c) for lbl, (_, c) in zip(labels, _WETLAND_STOPS)]


def _render_wetland(tex) -> np.ndarray:
    score = np.clip(np.asarray(tex["wetland"], dtype=np.float32), 0.0, 1.0)
    return apply_water_masks(
        gradient_texture(score, _WETLAND_STOPS),
        ocean_mask_tex=tex.get("ocean_mask"), lake_mask_tex=tex.get("lake_mask"),
    )


WETLAND_RECIPE = TextureRecipe(_render_wetland, sources=("wetland",), relief=True)
