"""Flat-top hexagons in 2-D."""
from __future__ import annotations

import math

import numpy as np


def in_flat_hexagon(offsets: np.ndarray, radius: float) -> np.ndarray:
    """``(..., 2)`` offsets from a flat-top hexagon's centre → inside it (bool)."""
    x, y = np.abs(offsets[..., 0]), np.abs(offsets[..., 1])
    # The intersection of three slabs.
    sqrt3 = math.sqrt(3.0)
    return (
        (y <= sqrt3 * 0.5 * radius + 1e-12)
        & (sqrt3 * x + y <= sqrt3 * radius + 1e-12)
        & (sqrt3 * x - y <= sqrt3 * radius + 1e-12)
    )
