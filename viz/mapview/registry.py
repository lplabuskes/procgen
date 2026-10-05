"""Adapter: apply a (Qt-free) ``DisplayRegistry`` to a ``MapWindow``.

Keeps ``core`` and the stages free of any Qt dependency.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from core.display import DisplayRegistry, LocalHoverFn
from core.world import World


def apply_registry(reg: DisplayRegistry, window, world: World) -> None:
    """Wire every scheme, overlay, legend, and hover contributor into ``window``."""
    # Color schemes (combo-box order == registration order).
    window.set_color_schemes({name: spec.fn for name, spec in reg.schemes.items()})
    for name, spec in reg.schemes.items():
        if spec.texture is not None:
            window.associate_scheme_texture(name, spec.texture)
        if spec.recipe is not None:
            window.associate_scheme_recipe(name, spec.recipe)
        if spec.legend is not None:
            kind, entries = spec.legend
            window.register_legend(name, kind, entries)

    # Feature overlays (+ their optional legends, keyed by feature name).
    for ov in reg.overlays:
        window.add_overlay(ov.feature, label=ov.label, checked=ov.checked)
        if ov.legend is not None:
            kind, entries = ov.legend
            window.register_legend(ov.feature, kind, entries)

    # Every stage's hover fragments for a face; the region view passes ``local``
    # to swap in its refined values.
    def _hover_parts(
        face_src: int, uv: Optional[np.ndarray] = None, local: Optional[LocalHoverFn] = None
    ) -> list[str]:
        return reg.hover(face_src, uv, world, local)

    window.set_hover_parts(_hover_parts)
