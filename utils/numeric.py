"""Small numeric helpers."""
from __future__ import annotations

import numpy as np


def smoothstep(lo: float, hi: float, x):
    """Hermite ``0 → 1`` ramp of ``x`` between ``lo`` and ``hi`` (a step at ``hi`` when they are equal)."""
    x = np.asarray(x, dtype=np.float64)
    t = (x >= hi).astype(np.float64) if hi == lo else np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
