"""Smoothing polylines in any dimension."""
from __future__ import annotations

import numpy as np


def chaikin(points, iterations: int, closed: bool = False) -> np.ndarray:
    """Chaikin corner-cutting of ``(P, D)`` ``points``: each segment's ¼ and ¾ points.

    Open lines keep their endpoints (so chains meeting at a junction stay joined);
    a ``closed`` ring is cut all round, including its closing segment.
    """
    pts = np.asarray(points)
    for _ in range(max(iterations, 0)):
        if len(pts) < 3:
            break
        a = pts
        b = np.roll(pts, -1, axis=0) if closed else pts[1:]
        if not closed:
            a = pts[:-1]
        cut = np.stack([0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b], axis=1).reshape(-1, pts.shape[1])
        pts = cut if closed else np.concatenate([pts[:1], cut, pts[-1:]])
    return pts
