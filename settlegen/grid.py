"""The settlement map's grid: one cell whose texture square is the whole frame.

A :class:`RegionHexGrid` so every region / local consumer (corridor rasters,
``snap_to_land``, channel cover, footprints) reads a settlement map unchanged.
With ``n_cols = 1`` the radius is 0.5, so the one texture square, centred at
``(0.5, 0.5)``, is exactly ``[0, 1]²``; every texel counts as inside its cell.
"""
from __future__ import annotations

import numpy as np

from region.geometry import RegionGeometry
from region.grid import RegionHexGrid


class SettlementGrid(RegionHexGrid):
    """One cell covering the settlement frame ``[0, 1]²``."""

    def __init__(self, geometry: RegionGeometry) -> None:
        super().__init__(geometry, n_cols=1)

    def _layout(self) -> np.ndarray:
        return np.array([[0.5, 0.5]], dtype=np.float64)

    def texture_in_hexagon(self, tex_size: int) -> np.ndarray:
        """(1, T, T) all True: the frame is a square, not a hexagon."""
        if tex_size < 1:
            raise ValueError(f"tex_size must be >= 1, got {tex_size}")
        return np.ones((1, tex_size, tex_size), dtype=bool)
