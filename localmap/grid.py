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
    parent hexagon grown by ``halo`` (local units) — the band past the parent the
    map conserves over so it agrees with its neighbours (``LocalUpsampleParams.
    conservation_halo_parent_texels``); the square's corners beyond belong to
    neighbouring local maps."""

    def __init__(self, geometry, n_cols: int = 16, halo: float = 0.0) -> None:  # type: ignore[no-untyped-def]
        self.halo = max(float(halo), 0.0)
        super().__init__(geometry, n_cols)

    def _layout(self) -> np.ndarray:
        centers = super()._layout()
        # A hex overlapping the grown parent has its centre within R of it, so inside
        # it grown by R in inradius (circumradius + 2R/√3) — a superset.
        reach = PARENT_RADIUS + self.halo * 2.0 / math.sqrt(3.0) + self.radius * 2.0 / math.sqrt(3.0)
        return centers[in_flat_hexagon(centers - PARENT_CENTER, reach)]

    def texture_in_halo(self, tex_size: int) -> np.ndarray:
        """(M, T, T) bool: texels inside the parent hexagon grown by ``halo`` (inradius)."""
        grown = PARENT_RADIUS + self.halo * 2.0 / math.sqrt(3.0)
        return in_flat_hexagon(self.texture_local(tex_size) - PARENT_CENTER, grown)

    def texture_in_map(self, tex_size: int) -> np.ndarray:
        """(M, T, T) bool: texels inside the parent hexagon — the map's domain."""
        return in_flat_hexagon(self.texture_local(tex_size) - PARENT_CENTER, PARENT_RADIUS)
