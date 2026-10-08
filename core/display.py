"""DisplayRegistry — the Qt-free surface stages contribute visualization to.

Each stage's ``register_display`` adds color schemes, feature overlays and hover
contributors (whose status fragments are concatenated); ``viz`` wires the
registry into a ``MapWindow``, keeping Qt/OpenGL out of ``core``.
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
# that are a function of position; a region view supplies its own.
POSITIONS_KEY = "positions"
# Reserved optional recipe input: ``(M, 7)`` ``[a, b, flow]`` river segments a view
# draws (``realmgen.refine.barriers.pack_river_segments``), for layers that follow them.
RIVER_SEGMENTS_KEY = "river_segments"


@dataclass(frozen=True)
class TextureRecipe:
    """How a scheme's RGBA texture is made from data textures.

    ``fn`` maps ``{key: (N, H, W, C)}`` to ``(N, H, W, 4)`` RGBA, applied to the
    globe's textures or a region's refined ones, so both views colour alike.
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
    # Face-texture keys reported; a regional view with all of them refined shows its own values.
    refined: Tuple[str, ...] = ()
    # Shown on the world map only; a regional view describes this itself.
    world_only: bool = False


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
        """Register a colour scheme.  ``texture`` (a per-face RGBA texture) is drawn
        in place of ``fn``; ``recipe`` lets a regional view rebuild it from refined data."""
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

    def add_hover(self, fn: HoverFn, *, refined: Tuple[str, ...] = (), world_only: bool = False) -> None:
        """Register a hover contributor.  ``refined``: face textures it reports
        (see ``hover``'s ``local``); ``world_only`` drops it from regional views."""
        self.hover_fns.append(HoverSpec(fn, tuple(refined), world_only))

    def hover(
        self,
        face_idx: int,
        uv: "Optional[np.ndarray]",
        world: "World",
        local: Optional[LocalHoverFn] = None,
    ) -> List[str]:
        """Concatenate every contributor's fragments for a face.

        With ``local`` (a regional view), ``local(spec.refined)`` replaces a
        contributor's fragments when not ``None``; ``world_only`` ones are skipped.
        """
        parts: List[str] = []
        for spec in self.hover_fns:
            if local is not None and spec.world_only:
                continue
            replaced = local(spec.refined) if local is not None and spec.refined else None
            parts.extend(replaced if replaced is not None else spec.fn(face_idx, uv, world))
        return parts
