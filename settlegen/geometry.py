"""A settlement map's frame: a square around one town or city, blown up to its own ``[0, 1]²``.

The square (side ``scale``, centred on ``center``, both in the owning local map's
``(s, t)``) is squared to north: ``a`` runs due east and ``b`` due north on the
ground at its centre, equally long, so the frame is a true square there.  It maps
affinely onto ``[0, 1]²``.  Everything else goes through the local map's frame,
which goes through the region's: it extends smoothly past the local map's edge
and is right-handed.
"""
from __future__ import annotations

import numpy as np

from region.geometry import RegionGeometry, north_tangent


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
        self.scale = float(side)                                 # settlement unit, in local units (√area)
        # Ground east and north at the centre, in the local map's (s, t); scaled to
        # the frame's area (side² local units), which keeps them a ground square.
        jac = parent.jacobian(float(self.center[0]), float(self.center[1]))
        up = parent.local_to_ecef(self.center[:1], self.center[1:])[0]
        north = north_tangent(up, parent.antimeridian)
        axes, *_ = np.linalg.lstsq(jac, np.column_stack([np.cross(north, up), north]), rcond=None)
        self.axes = axes * (self.scale / np.sqrt(abs(np.linalg.det(axes))))   # columns: local (s, t) of a, b
        self._inverse = np.linalg.inv(self.axes)

    def to_parent(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Settlement ``(a, b)`` → the local map's ``(s, t)``."""
        da = np.asarray(a, dtype=np.float64) - 0.5
        db = np.asarray(b, dtype=np.float64) - 0.5
        return (
            self.center[0] + self.axes[0, 0] * da + self.axes[0, 1] * db,
            self.center[1] + self.axes[1, 0] * da + self.axes[1, 1] * db,
        )

    def local_to_ecef(self, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        return self.parent.local_to_ecef(*self.to_parent(s, t))

    def ecef_to_local(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s, t, valid = self.parent.ecef_to_local(positions)
        ds, dt = s - self.center[0], t - self.center[1]
        inv = self._inverse
        return inv[0, 0] * ds + inv[0, 1] * dt + 0.5, inv[1, 0] * ds + inv[1, 1] * dt + 0.5, valid
