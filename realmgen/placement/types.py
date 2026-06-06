"""Settlement data types and display constants."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TIER_NAMES = ["Capital", "City", "Town", "Village"]

# RGBA display colors per tier.
TIER_COLORS: list[tuple[float, float, float, float]] = [
    (1.00, 0.85, 0.10, 1.0),  # gold   — capital
    (0.35, 0.60, 0.90, 1.0),  # blue   — city
    (0.40, 0.75, 0.40, 1.0),  # green  — town
    (0.80, 0.68, 0.50, 1.0),  # tan    — village
]

# Base display radii in pixels at zoom = 1.
TIER_RADII: list[float] = [8.0, 5.5, 3.5, 2.2]


@dataclass
class Settlement:
    """A provisionally placed settlement on the land face graph.

    Population is not assigned here — that happens in the Population &
    Calibration stage.  ``tier`` indexes into ``TIER_NAMES`` /
    ``TIER_COLORS`` / ``TIER_RADII``.
    """

    face_id: int
    position: np.ndarray   # (3,) unit-sphere centroid of the face
    tier: int              # 0=capital, 1=city, 2=town, 3=village
    score: float           # combined placement score at seeding time
