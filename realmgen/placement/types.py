"""Settlement data types and display constants."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TIER_NAMES = ["Capital", "City", "Town", "Village"]

# RGBA display colors per tier — used at visualisation time, keyed by display tier
# derived from population (see ``population_display_tier``), not the seeding tier.
TIER_COLORS: list[tuple[float, float, float, float]] = [
    (1.00, 0.85, 0.10, 1.0),  # gold   — capital
    (0.35, 0.60, 0.90, 1.0),  # blue   — city
    (0.40, 0.75, 0.40, 1.0),  # green  — town
    (0.80, 0.68, 0.50, 1.0),  # tan    — village
]

# Base display radii in pixels at zoom = 1.
TIER_RADII: list[float] = [8.0, 5.5, 3.5, 2.2]


def population_display_tier(population: float, max_pop: float) -> int:
    """Map a settlement's population to a 0-3 display tier.

    Tier is derived purely from ``population / max_pop`` so that visualisation
    reflects actual importance rather than the seeding label.
    """
    if max_pop <= 0 or population <= 0:
        return 3
    frac = population / max_pop
    if frac >= 0.5:
        return 0
    if frac >= 0.15:
        return 1
    if frac >= 0.04:
        return 2
    return 3


@dataclass
class Settlement:
    """A placed settlement on the land face graph.

    ``tier`` is assigned by the seeder and serves only as a seeding label
    (it governs exclusion radius and seeds ``influence`` / ``demand``).
    No downstream computation reads ``tier`` — all generation operates on
    ``influence``, ``demand``, and ``population``.  Display tiers are derived
    at render time via :func:`population_display_tier`.
    """

    face_id: int
    position: np.ndarray   # (3,) unit-sphere centroid of the face
    tier: int              # seeding label: 0=capital, 1=city, 2=town, 3=village
    score: float           # combined placement score at seeding time
    influence: float = 0.0   # hinterland priority offset; lower = wider territory
    demand: float = 1.0      # trade demand weight (seeded from tier)
    population: float = 0.0  # set by Population & Calibration stage
