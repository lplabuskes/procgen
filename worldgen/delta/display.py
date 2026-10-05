"""Colour scheme for the delta debug layer."""
from __future__ import annotations

import numpy as np

from core.display import TextureRecipe
from viz.color_stops import gradient_texture
from worldgen.geology.display import apply_water_masks, ocean_gradient_scheme

# Off the delta (stone) → delta plain (green).
_DELTA_STOPS = [
    (0.00, (0.55, 0.52, 0.48, 1.0)),
    (1.00, (0.35, 0.65, 0.25, 1.0)),
]

delta_color = ocean_gradient_scheme("delta_fraction", _DELTA_STOPS, default=0.0)


def delta_legend_entries() -> list[tuple[str, tuple]]:
    labels = ["off delta", "delta plain"]
    return [(lbl, c) for lbl, (_, c) in zip(labels, _DELTA_STOPS)]


def _render_delta(tex) -> np.ndarray:
    value = np.clip(np.asarray(tex["delta"], dtype=np.float32)[..., :1], 0.0, 1.0)
    return apply_water_masks(
        gradient_texture(value, _DELTA_STOPS),
        ocean_mask_tex=tex.get("ocean_mask"), lake_mask_tex=tex.get("lake_mask"),
    )


DELTA_RECIPE = TextureRecipe(_render_delta, sources=("delta",), relief=True)
