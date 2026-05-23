"""River data structures and sphere-surface smoothing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class RiverNode:
    position: np.ndarray          # 3D point on globe surface, shape (3,)
    elevation: float
    face_id: int
    uv: Tuple[float, float]       # UV coordinate within the face in [0,1]^2


@dataclass
class RiverSegment:
    start: RiverNode
    end: RiverNode
    flow_rate: float              # accumulated upstream count at segment start
    tributary_count: int = 0


@dataclass
class River:
    segments: List[RiverSegment] = field(default_factory=list)
    source: Optional[RiverNode] = None
    mouth: Optional[RiverNode] = None

    def as_polyline(self) -> List[np.ndarray]:
        if not self.segments:
            return []
        pts = [self.segments[0].start.position]
        pts += [s.end.position for s in self.segments]
        return pts

    def flow_rates(self) -> List[float]:
        """Returns one flow value per segment (parallel to as_polyline edges)."""
        return [s.flow_rate for s in self.segments]


def laplacian_smooth_sphere(
    polyline: List[np.ndarray],
    iterations: int = 8,
    alpha: float = 0.5,
) -> List[np.ndarray]:
    """Approximating Laplacian smoothing on the unit sphere.

    Each interior point is pulled toward the midpoint of its two neighbours by
    `alpha`, then re-projected to the sphere. Endpoints (source and mouth) are
    pinned. Unlike an interpolating spline (Catmull-Rom), this displaces the
    control points themselves, so grid-aligned staircases and sharp corners
    are flattened out.
    """
    if len(polyline) < 3:
        return [p.astype(np.float32, copy=True) for p in polyline]

    pts = np.stack(polyline).astype(np.float64)
    for _ in range(iterations):
        mids = 0.5 * (pts[:-2] + pts[2:])
        pts[1:-1] += alpha * (mids - pts[1:-1])
        norms = np.linalg.norm(pts[1:-1], axis=1, keepdims=True)
        np.divide(pts[1:-1], norms, out=pts[1:-1], where=norms > 0)
    return [pts[i].astype(np.float32) for i in range(len(pts))]
