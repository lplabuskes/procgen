"""A local map's frame: one regional hexagon, blown up to its own ``[0, 1]²``.

The local square is the parent hex's texture square — side ``2·R`` in the
region's ``(s, t)`` frame, centred on the hex — mapped affinely onto ``[0, 1]²``.
Everything geometric is borrowed from the region: its frame extends smoothly
past the cell (so a hex hanging over the cell edge is fine), it is right-handed,
and ``up_direction`` / ``local_to_lonlat`` / ``center_ecef`` all work through
:meth:`local_to_ecef` unchanged.  The parent hexagon is flat-top with
circumradius 0.5 around ``(0.5, 0.5)`` in local units.
"""
from __future__ import annotations

import numpy as np

from region.geometry import RegionGeometry
from region.grid import RegionHexGrid

#: The parent hexagon in local units: flat-top, centred, circumradius 0.5.
PARENT_CENTER = 0.5
PARENT_RADIUS = 0.5


class LocalGeometry(RegionGeometry):
    """Geometry of the local map inside regional hex ``hex_index`` of ``grid``.

    ``region_id`` (and the rest of :class:`RegionGeometry`'s state) is the
    parent region's; ``hex_index`` names the hex.
    """

    def __init__(self, parent: RegionGeometry, grid: RegionHexGrid, hex_index: int) -> None:
        if not 0 <= hex_index < len(grid):
            raise ValueError(f"hex_index {hex_index} out of range [0, {len(grid)})")
        super().__init__(parent.region_id, parent.antimeridian)
        self.parent = parent
        self.hex_index = int(hex_index)
        self.scale = 2.0 * grid.radius                              # local unit, in region units
        self.origin = grid.centers[hex_index] - grid.radius         # region (s, t) of local (0, 0)

    def to_region(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Local ``(a, b)`` → the parent region's ``(s, t)``."""
        return (
            self.origin[0] + self.scale * np.asarray(a, dtype=np.float64),
            self.origin[1] + self.scale * np.asarray(b, dtype=np.float64),
        )

    def local_to_ecef(self, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        return self.parent.local_to_ecef(*self.to_region(s, t))

    def ecef_to_local(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s, t, valid = self.parent.ecef_to_local(positions)
        return (s - self.origin[0]) / self.scale, (t - self.origin[1]) / self.scale, valid
