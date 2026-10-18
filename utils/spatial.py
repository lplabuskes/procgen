"""Nearest-point queries on point sets (KD-trees) and hashable position keys."""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import KDTree


def position_key(pos: np.ndarray) -> tuple[float, float, float]:
    """Hashable key for a unit-sphere position, stable for centroid-derived points."""
    p = np.asarray(pos, dtype=np.float64).ravel()
    return (round(float(p[0]), 9), round(float(p[1]), 9), round(float(p[2]), 9))


def kdtree(points: Optional[np.ndarray]) -> Optional[KDTree]:
    """A KD-tree over ``points``, or None when there are none."""
    return KDTree(points) if points is not None and len(points) else None


def near_any(tree: Optional[KDTree], points: np.ndarray, radius: float) -> np.ndarray:
    """``(P,)`` True where ``points`` lie within ``radius`` of a ``tree`` point (none when ``tree`` is None)."""
    if tree is None or points.shape[0] == 0:
        return np.zeros(points.shape[0], dtype=bool)
    # Bounded: points farther than ``radius`` from every tree point come back as inf, sooner.
    return tree.query(points, workers=-1, distance_upper_bound=radius)[0] < radius


# Chord distance for a road end to match a settlement's old position (float32 round-off).
_MOVE_TOLERANCE = 1e-6


def move_points(points: np.ndarray, moved: Optional[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """``points`` with any at an ``old`` position of ``moved = (old, new)`` replaced by its ``new``."""
    if moved is None or len(moved[0]) == 0 or len(points) == 0:
        return points
    old, new = (np.asarray(a, dtype=np.float64).reshape(-1, 3) for a in moved)
    dist, near = KDTree(old).query(points, workers=-1)
    hit = dist < _MOVE_TOLERANCE
    out = points.copy()
    out[hit] = new[np.asarray(near)[hit]]
    return out
