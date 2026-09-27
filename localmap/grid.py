"""The local map's hex grid: :class:`RegionHexGrid` over one parent hexagon."""
from __future__ import annotations

import math

import numpy as np

from localmap.geometry import PARENT_CENTER, PARENT_RADIUS
from region.grid import RegionHexGrid


def in_flat_hexagon(offsets: np.ndarray, radius: float) -> np.ndarray:
    """``(..., 2)`` offsets from a flat-top hexagon's centre → inside it (bool)."""
    x, y = np.abs(offsets[..., 0]), np.abs(offsets[..., 1])
    sqrt3 = math.sqrt(3.0)
    return (
        (y <= sqrt3 * 0.5 * radius + 1e-12)
        & (sqrt3 * x + y <= sqrt3 * radius + 1e-12)
        & (sqrt3 * x - y <= sqrt3 * radius + 1e-12)
    )


class LocalHexGrid(RegionHexGrid):
    """Flat-top hexagons over a local map, kept only where they overlap the
    parent hexagon (~2/3 of the square) — the corners belong to neighbouring
    local maps."""

    def _layout(self) -> np.ndarray:
        centers = super()._layout()
        # A hex overlapping the parent has its centre within R of it, so inside
        # the parent grown by R in inradius (circumradius + 2R/√3) — a superset.
        reach = PARENT_RADIUS + self.radius * 2.0 / math.sqrt(3.0)
        return centers[in_flat_hexagon(centers - PARENT_CENTER, reach)]

    def texture_in_map(self, tex_size: int) -> np.ndarray:
        """(M, T, T) bool: texels inside the parent hexagon — the map's domain."""
        return in_flat_hexagon(self.texture_local(tex_size) - PARENT_CENTER, PARENT_RADIUS)
