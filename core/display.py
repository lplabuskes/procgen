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

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.world import World

# A color scheme maps a face's metadata dict to an RGBA tuple.
SchemeFn = Callable[[dict], tuple]
# A hover contributor maps (face_idx, uv_or_None, world) to status fragments.
HoverFn = Callable[[int, "Optional[np.ndarray]", "World"], List[str]]
# A legend is (kind, entries) where kind is e.g. "gradient" | "categorical".
Legend = Tuple[str, list]


@dataclass
class SchemeSpec:
    fn: SchemeFn
    texture: Optional[str] = None      # associated per-face RGBA texture, if any
    legend: Optional[Legend] = None


@dataclass
class OverlaySpec:
    feature: str
    label: str
    checked: bool = True


class DisplayRegistry:
    """Accumulates display contributions from stages; consumed by the viz layer."""

    def __init__(self) -> None:
        self.schemes: dict[str, SchemeSpec] = {}
        self.overlays: List[OverlaySpec] = []
        self.hover_fns: List[HoverFn] = []

    def add_scheme(
        self,
        name: str,
        fn: SchemeFn,
        *,
        texture: Optional[str] = None,
        legend: Optional[Legend] = None,
    ) -> None:
        self.schemes[name] = SchemeSpec(fn, texture, legend)

    def add_overlay(self, feature: str, label: str, *, checked: bool = True) -> None:
        self.overlays.append(OverlaySpec(feature, label, checked))

    def add_hover(self, fn: HoverFn) -> None:
        self.hover_fns.append(fn)

    def hover(self, face_idx: int, uv: "Optional[np.ndarray]", world: "World") -> List[str]:
        """Concatenate every registered contributor's fragments for a face."""
        parts: List[str] = []
        for fn in self.hover_fns:
            parts.extend(fn(face_idx, uv, world))
        return parts
