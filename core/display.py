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
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.world import World

# A color scheme maps a face's metadata dict to an RGBA tuple.
SchemeFn = Callable[[dict], tuple]
# A hover contributor maps (face_idx, uv_or_None, world) to status fragments.
HoverFn = Callable[[int, "Optional[np.ndarray]", "World"], List[str]]
# Refined values for a set of texture keys, or None when not all are refined.
LocalHoverFn = Callable[[Tuple[str, ...]], Optional[List[str]]]
# A legend is (kind, entries) where kind is e.g. "gradient" | "categorical".
Legend = Tuple[str, list]


@dataclass
class SchemeSpec:
    fn: SchemeFn
    texture: Optional[str] = None      # associated per-face RGBA texture, if any
    legend: Optional[Legend] = None


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
    ) -> None:
        self.schemes[name] = SchemeSpec(fn, texture, legend)

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
