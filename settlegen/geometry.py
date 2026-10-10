"""A settlement map's frame: a square around one town or city, blown up to its own ``[0, 1]²``.

The square (side ``scale``, centred on ``center``, both in the owning local map's
``(s, t)``) maps affinely onto ``[0, 1]²``.  Everything else goes through the
local map's frame, which goes through the region's: it extends smoothly past
the local map's edge and is right-handed.
"""
from __future__ import annotations

import numpy as np

from region.geometry import RegionGeometry


class SettlementGeometry(RegionGeometry):
    """Geometry of a settlement map inside the local map with geometry ``parent``.

    ``region_id`` (and the rest of :class:`RegionGeometry`'s state) is the
    owning local map's region's.
    """

    def __init__(self, parent: RegionGeometry, center: np.ndarray, side: float) -> None:
        if not side > 0.0:
            raise ValueError(f"side must be positive, got {side}")
        super().__init__(parent.region_id, parent.antimeridian)
        self.parent = parent
        self.center = np.asarray(center, dtype=np.float64).reshape(2)
        self.scale = float(side)                                 # settlement unit, in local units
        self.origin = self.center - 0.5 * self.scale             # local (s, t) of settlement (0, 0)

    def to_parent(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Settlement ``(a, b)`` → the local map's ``(s, t)``."""
        return (
            self.origin[0] + self.scale * np.asarray(a, dtype=np.float64),
            self.origin[1] + self.scale * np.asarray(b, dtype=np.float64),
        )

    def local_to_ecef(self, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        return self.parent.local_to_ecef(*self.to_parent(s, t))

    def ecef_to_local(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s, t, valid = self.parent.ecef_to_local(positions)
        return (s - self.origin[0]) / self.scale, (t - self.origin[1]) / self.scale, valid
