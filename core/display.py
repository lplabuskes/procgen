"""DisplayRegistry — the Qt-free surface stages contribute visualization to.

Each stage's ``register_display`` adds named color schemes (callables over a
face's metadata dict), feature overlays, and hover contributors here. A viz
adapter (in the ``viz`` layer) later walks the registry and wires it into a
concrete ``MapWindow`` — keeping all Qt/OpenGL out of ``core``.

Hover is decomposed: every stage registers a contributor returning a list of
status fragments for a hovered face; the registry concatenates them, so the old
monolithic hover callback dissolves into per-stage pieces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.world import World

# A color scheme maps a face's metadata dict to an RGBA tuple.
SchemeFn = Callable[[dict], Sequence[float]]
# A hover contributor maps (face_idx, uv_or_None, world) to status fragments.
HoverFn = Callable[[int, "Optional[np.ndarray]", "World"], List[str]]
# Refined values for a set of texture keys, or None when not all are refined.
LocalHoverFn = Callable[[Tuple[str, ...]], Optional[List[str]]]
# A legend is (kind, entries) where kind is e.g. "gradient" | "categorical".
Legend = Tuple[str, list]


# Reserved recipe source: ``(N, H, W, 3)`` unit-sphere texel positions, for layers
# that are a function of position rather than of stored data. A region view
# supplies its own texel positions under this key.
POSITIONS_KEY = "positions"


@dataclass(frozen=True)
class TextureRecipe:
    """How a scheme's RGBA texture is made from data textures.

    ``fn`` maps a ``{key: (N, H, W, C)}`` texture dict to ``(N, H, W, 4)`` RGBA.
    It is written once, by the stage that owns the layer, and applied to any
    texture set on the same keys: the globe's face textures for the world map,
    or a region's refined textures for the region view — so both views colour
    a layer identically by construction.
    """

    fn: Callable[[Mapping[str, np.ndarray]], np.ndarray]
    # Keys ``fn`` needs; optional extras (water masks, …) it reads with ``.get``.
    sources: Tuple[str, ...]
    # The layer reads better with hillshading where the view has sub-texel relief.
    relief: bool = False

    def available(self, textures: Mapping[str, np.ndarray]) -> bool:
        return all(k in textures for k in self.sources)

    def __call__(self, textures: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.fn(textures)


@dataclass
class SchemeSpec:
    fn: SchemeFn
    texture: Optional[str] = None      # associated per-face RGBA texture, if any
    legend: Optional[Legend] = None
    recipe: Optional[TextureRecipe] = None   # how ``texture`` is built from data


@dataclass
class HoverSpec:
    fn: HoverFn
    # Face-texture keys this contributor reports. A regional view that has
    # refined all of them shows its own values in place of this contributor.
    refined: Tuple[str, ...] = ()


@dataclass
class OverlaySpec:
    feature: str
    label: str
    checked: bool = True
    legend: Optional[Legend] = None


class DisplayRegistry:
    """Accumulates display contributions from stages; consumed by the viz layer."""

    def __init__(self) -> None:
        self.schemes: dict[str, SchemeSpec] = {}
        self.overlays: List[OverlaySpec] = []
        self.hover_fns: List[HoverSpec] = []

    def add_scheme(
        self,
        name: str,
        fn: SchemeFn,
        *,
        texture: Optional[str] = None,
        legend: Optional[Legend] = None,
        recipe: Optional[TextureRecipe] = None,
    ) -> None:
        """Register a colour scheme.

        ``texture`` names a per-face RGBA texture drawn in place of ``fn``;
        ``recipe`` is how that texture is made, which lets a regional view
        rebuild the layer from its own refined data.
        """
        self.schemes[name] = SchemeSpec(fn, texture, legend, recipe)

    def add_overlay(
        self,
        feature: str,
        label: str,
        *,
        checked: bool = True,
        legend: Optional[Legend] = None,
    ) -> None:
        self.overlays.append(OverlaySpec(feature, label, checked, legend))

    def add_hover(self, fn: HoverFn, *, refined: Tuple[str, ...] = ()) -> None:
        """Register a hover contributor.

        ``refined`` names the face textures it reports on (e.g. ``("elevation",)``);
        see ``hover``'s ``local``.
        """
        self.hover_fns.append(HoverSpec(fn, tuple(refined)))

    def hover(
        self,
        face_idx: int,
        uv: "Optional[np.ndarray]",
        world: "World",
        local: Optional[LocalHoverFn] = None,
    ) -> List[str]:
        """Concatenate every registered contributor's fragments for a face.

        ``local`` lets a regional view substitute its refined values: for each
        contributor with ``refined`` keys, ``local(keys)`` is tried first and, if
        it returns fragments, they replace the contributor's global ones in place.
        """
        parts: List[str] = []
        for spec in self.hover_fns:
            replaced = local(spec.refined) if local is not None and spec.refined else None
            parts.extend(replaced if replaced is not None else spec.fn(face_idx, uv, world))
        return parts
