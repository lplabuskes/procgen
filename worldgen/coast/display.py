"""Colour scheme for the coast-smoothness debug layer."""
from __future__ import annotations

import numpy as np

from core.display import TextureRecipe
from viz.color_stops import gradient_texture
from worldgen.geology.display import apply_water_masks, ocean_gradient_scheme

# Rugged (dark slate) → smooth (sand).
_COAST_STOPS = [
    (0.00, (0.20, 0.22, 0.30, 1.0)),
    (0.35, (0.45, 0.42, 0.45, 1.0)),
    (0.65, (0.72, 0.64, 0.48, 1.0)),
    (1.00, (0.95, 0.86, 0.60, 1.0)),
]

coast_color = ocean_gradient_scheme("coast_smoothness", _COAST_STOPS, default=0.5)


def coast_legend_entries() -> list[tuple[str, tuple]]:
    labels = ["rugged", "", "", "smooth"]
    return [(lbl, c) for lbl, (_, c) in zip(labels, _COAST_STOPS)]


def _render_coast(tex) -> np.ndarray:
    smooth = np.clip(np.asarray(tex["coast_smoothness"], dtype=np.float32), 0.0, 1.0)
    return apply_water_masks(
        gradient_texture(smooth, _COAST_STOPS),
        ocean_mask_tex=tex.get("ocean_mask"), lake_mask_tex=tex.get("lake_mask"),
    )


# Drawn on land with water flat, so the colour reads along the shore it styles.
COAST_RECIPE = TextureRecipe(_render_coast, sources=("coast_smoothness",), relief=True)
