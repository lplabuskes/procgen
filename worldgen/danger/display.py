"""Color scheme for the danger stage."""
from __future__ import annotations

import numpy as np

from core.display import TextureRecipe
from viz.color_stops import gradient_texture
from worldgen.geology.display import apply_water_masks, ocean_gradient_scheme

# Neutral land → amber → red → deep red
_DANGER_STOPS = [
    (0.00, (0.72, 0.52, 0.25, 1.0)),
    (0.25, (0.88, 0.55, 0.10, 1.0)),
    (0.50, (0.85, 0.18, 0.05, 1.0)),
    (0.75, (0.65, 0.05, 0.05, 1.0)),
    (1.00, (0.30, 0.00, 0.05, 1.0)),
]

danger_color = ocean_gradient_scheme("danger", _DANGER_STOPS, default=0.0)


def danger_legend_entries() -> list[tuple[str, tuple]]:
    labels = ["safe", "0.25", "0.50", "0.75", "lethal"]
    return [(lbl, c) for lbl, (_, c) in zip(labels, _DANGER_STOPS)]


def _render_danger(tex) -> np.ndarray:
    """Danger over land as the scheme colours it; water flat, like the other
    refined layers. Values past 1 (the peaks of a smoothed field) read as lethal."""
    danger = np.clip(np.asarray(tex["danger"], dtype=np.float32), 0.0, 1.0)
    return apply_water_masks(
        gradient_texture(danger, _DANGER_STOPS),
        ocean_mask_tex=tex.get("ocean_mask"), lake_mask_tex=tex.get("lake_mask"),
    )


# Danger is per face on the globe, so the world map draws the colour function;
# a region, which refines it into a texture, draws this.
DANGER_RECIPE = TextureRecipe(_render_danger, sources=("danger",))
