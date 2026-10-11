"""The settlement map's planar frame: metres from its centre, along its axes.

Everything a site or layout holds is geometry in these coordinates (plan
Decision 8): ``x`` along the frame's ``a`` axis, ``y`` along ``b`` (screen-up
unrotated; :meth:`SettlementGeometry.up_direction` gives north).  A frame spans
≤10 km, where the gnomonic patch is flat to well under 0.1%.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon, box

from settlegen.geometry import SettlementGeometry


@dataclass(frozen=True)
class Frame:
    """``[0, 1]²`` of ``geometry`` as a ``side_m`` square of metres centred on 0."""

    geometry: SettlementGeometry
    side_m: float

    @property
    def half(self) -> float:
        return 0.5 * self.side_m

    def to_xy(self, ecef: np.ndarray) -> np.ndarray:
        """``(N, 3)`` unit vectors → ``(N, 2)`` metres."""
        a, b, _ = self.geometry.ecef_to_local(np.asarray(ecef, dtype=np.float64).reshape(-1, 3))
        return np.column_stack([(a - 0.5) * self.side_m, (b - 0.5) * self.side_m])

    def to_ecef(self, xy: np.ndarray) -> np.ndarray:
        """``(N, 2)`` metres → ``(N, 3)`` unit vectors."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        return self.geometry.local_to_ecef(xy[:, 0] / self.side_m + 0.5, xy[:, 1] / self.side_m + 0.5)

    def texel_xy(self, tex_size: int) -> np.ndarray:
        """``(T, T, 2)`` texel centres in metres; row 0 is the top (largest ``y``), as textures are."""
        c = ((np.arange(tex_size) + 0.5) / tex_size - 0.5) * self.side_m
        xx, yy = np.meshgrid(c, c[::-1])
        return np.stack([xx, yy], axis=-1)

    def box(self) -> Polygon:
        return box(-self.half, -self.half, self.half, self.half)
