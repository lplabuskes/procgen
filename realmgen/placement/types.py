"""Settlement data types and display constants."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Every modeled settlement is city-class at this mesh resolution (a face is
# province-scale); the tiers are rank *within* the set of major cities.
TIER_NAMES = ["Great City", "Major City", "City", "Minor City"]

# Bit flags for what a city's refine-stage site offered, so finer tiers re-site it
# without losing it (a harbour city stays on the coast, a river city on its river).
SITE_KIND_OCEAN_COAST, SITE_KIND_LAKE_COAST, SITE_KIND_RIVER, SITE_KIND_CONFLUENCE = 1, 2, 4, 8


def format_population(population: float) -> str:
    """Humanize an urban headcount for display: 9400 → '9.4k', 132000 → '132k'."""
    p = float(population)
    if p >= 1e6:
        return f"{p / 1e6:.2g}M"
    if p >= 1e3:
        return f"{p / 1e3:.3g}k"
    return f"{p:.0f}"


# RGBA per display tier (from ``population_display_tier``, not the seeding tier).
TIER_COLORS: list[tuple[float, float, float, float]] = [
    (1.00, 0.85, 0.10, 1.0),  # gold   — great city
    (0.35, 0.60, 0.90, 1.0),  # blue   — major city
    (0.40, 0.75, 0.40, 1.0),  # green  — city
    (0.80, 0.68, 0.50, 1.0),  # tan    — minor city
]

# Base display radii in pixels at zoom = 1.
TIER_RADII: list[float] = [8.0, 5.5, 3.5, 2.2]


def population_display_tier(population: float, max_pop: float) -> int:
    """Map ``population / max_pop`` to a 0-3 display tier."""
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

    ``tier`` is only a seeding label (exclusion radius, initial population); all
    generation reads ``population``.

    ``position`` is at the finest tier that has refined it: ``site_position`` is the
    refine stage's site, ``region_position`` the region's refinement (``refined``),
    then the local map's (``local_refined``); see ``region/cities.py``.
    """

    face_id: int
    position: np.ndarray   # (3,) unit-sphere, current (finest refinement)
    tier: int              # seeding label: 0=capital, 1=city, 2=town, 3=village
    score: float           # combined placement score at seeding time
    population: float = 0.0    # seeded from tier initial_population; updated by PopulationStage
    defensibility: float = 1.0  # composite multiplier used by PolityStage; set by PopulationStage
    has_walls: bool = False     # whether the settlement has defensive walls
    wall_strength: float = 0.0  # wall contribution to defensibility (0 if no walls)
    site_kind: int = 0          # SITE_KIND_* flags of the refine stage's site
    site_position: np.ndarray = field(default_factory=lambda: np.zeros(0))    # (3,)
    region_position: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (3,)
    refined: bool = False           # re-sited by its region
    local_refined: bool = False     # re-sited by its local map
    # Per land↔water link its trade uses, strongest first: the tags of the water
    # that serves it (``realmgen.trade.water_links``).  Every siting tier keeps it there.
    water_links: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        if self.site_position.shape != (3,):
            self.site_position = self.position.copy()
        if self.region_position.shape != (3,):
            self.region_position = self.position.copy()

    def set_site(self, position: np.ndarray, kind: int = 0) -> None:
        """Put the city on a new stage site, dropping any finer refinement."""
        self.site_position = np.asarray(position, dtype=np.float64).copy()
        self.site_kind = int(kind)
        self.reset_refinement()

    def reset_refinement(self) -> None:
        """Back to the stage site (the region refinement is stale)."""
        self.position = self.site_position.copy()
        self.region_position = self.site_position.copy()
        self.refined = self.local_refined = False

    def reset_local_refinement(self) -> None:
        """Back to the region's refinement (the local refinement is stale)."""
        self.position = self.region_position.copy()
        self.local_refined = False
