"""Adapter: apply a (Qt-free) ``DisplayRegistry`` to a ``MapWindow``.

This is the one place that knows both the registry shape and the Qt window API.
Stages contribute schemes/overlays/hover to a ``DisplayRegistry`` in ``core``;
this function performs the actual window wiring, so ``core`` and the stages stay
free of any Qt dependency.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from core.display import DisplayRegistry
from core.world import World


def apply_registry(reg: DisplayRegistry, window, world: World) -> None:
    """Wire every scheme, overlay, legend, and hover contributor into ``window``."""
    # Color schemes (combo-box order == registration order).
    window.set_color_schemes({name: spec.fn for name, spec in reg.schemes.items()})
    for name, spec in reg.schemes.items():
        if spec.texture is not None:
            window.associate_scheme_texture(name, spec.texture)
        if spec.legend is not None:
            kind, entries = spec.legend
            window.register_legend(name, kind, entries)

    # Feature overlays (+ their optional legends, keyed by feature name).
    for ov in reg.overlays:
        window.add_overlay(ov.feature, label=ov.label, checked=ov.checked)
        if ov.legend is not None:
            kind, entries = ov.legend
            window.register_legend(ov.feature, kind, entries)

    # Decomposed hover: concatenate every stage's fragments into the status bar.
    def _hover(face_src: Optional[int], uv: Optional[np.ndarray] = None) -> None:
        if face_src is None:
            window.set_status("")
            return
        window.set_status("  |  ".join(reg.hover(face_src, uv, world)))

    window.set_hover_callback(_hover)
