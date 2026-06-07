"""Polity data types and display constants."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

# Categorical palette for territory fill.  Distinct, slightly muted hues so the
# capital settlement dots (drawn by the settlements overlay) stay legible on top.
# Colors are assigned by ``polity_id % len(POLITY_COLORS)`` at render time.
POLITY_COLORS: List[tuple] = [
    (0.84, 0.37, 0.34, 1.0),  # red
    (0.36, 0.55, 0.80, 1.0),  # blue
    (0.45, 0.68, 0.40, 1.0),  # green
    (0.80, 0.62, 0.30, 1.0),  # amber
    (0.62, 0.46, 0.74, 1.0),  # purple
    (0.36, 0.70, 0.68, 1.0),  # teal
    (0.83, 0.55, 0.40, 1.0),  # orange
    (0.74, 0.50, 0.62, 1.0),  # mauve
    (0.55, 0.62, 0.34, 1.0),  # olive
    (0.45, 0.58, 0.66, 1.0),  # slate
    (0.78, 0.45, 0.52, 1.0),  # rose
    (0.50, 0.66, 0.52, 1.0),  # sage
]

# Land that no polity reached (remote wilderness / out of range).
UNCLAIMED_COLOR = (0.32, 0.30, 0.27, 1.0)


@dataclass
class Polity:
    """A territorial polity grown from a capital settlement.

    ``polity_id`` indexes the polity within the build (0-based, aligned with the
    per-face ``polity_id`` metadata).  ``power`` is the total population of every
    settlement inside the territory — the strength measure used for border
    tension's power-balance term.
    """

    polity_id: int
    capital_face: int
    capital_position: np.ndarray            # (3,) unit-sphere centroid of the capital face
    power: float                            # total member-settlement population
    n_faces: int                            # territory size (face count)
    member_settlement_faces: List[int] = field(default_factory=list)
