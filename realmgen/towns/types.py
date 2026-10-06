"""The ``towns`` world feature: the settlements below the city floor."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

import numpy as np

# A town's kind is what its chosen site offered (harbour first, then mine, river, route).
TOWN_MARKET, TOWN_PORT, TOWN_RIVER, TOWN_MINING, TOWN_ROUTE = range(5)
TOWN_KIND_NAMES = ("Market town", "Port town", "River town", "Mining town", "Waypoint town")
TOWN_KIND_COLORS: list[tuple[float, float, float, float]] = [
    (0.93, 0.90, 0.80, 1.0),  # cream      — market
    (0.55, 0.78, 0.95, 1.0),  # sky blue   — port
    (0.45, 0.85, 0.78, 1.0),  # teal       — river
    (0.78, 0.55, 0.45, 1.0),  # rust       — mining
    (0.92, 0.78, 0.45, 1.0),  # amber      — waypoint
]


@dataclass
class Towns:
    """Every town on the globe, as parallel arrays (``K`` towns).

    ``parent`` indexes the ``settlements`` city whose hinterland holds the town
    (``-1`` if none). ``site_positions`` is the stage's global-texel site;
    ``positions`` is the finest refinement so far: region (``refined``,
    ``region_positions``, within ~one global texel; ``region/towns.py``), then
    local (``local_refined``; ``localmap/settlements.py``). Walls (``has_walls``,
    ``wall_strength``) are rolled by the town stage, more rarely than a city's;
    data only. Villages never have walls.
    """

    positions: np.ndarray    # (K, 3) unit-sphere, current (refined where ``refined``)
    population: np.ndarray   # (K,) people
    face_id: np.ndarray      # (K,) owning global face
    parent: np.ndarray       # (K,) settlement index or -1
    kind: np.ndarray         # (K,) TOWN_* code
    site_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))   # (K, 3)
    refined: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))   # (K,)
    region_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # (K, 3)
    local_refined: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))  # (K,)
    has_walls: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))  # (K,)
    wall_strength: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (K,) 0 without walls

    def __post_init__(self) -> None:
        n = self.positions.shape[0]
        if self.site_positions.shape[0] != n:
            self.site_positions = self.positions.copy()
        if self.region_positions.shape[0] != n:
            self.region_positions = self.positions.copy()
        if self.refined.shape[0] != n:
            self.refined = np.zeros(n, dtype=bool)
        if self.local_refined.shape[0] != n:
            self.local_refined = np.zeros(n, dtype=bool)
        if self.has_walls.shape[0] != n:
            self.has_walls = np.zeros(n, dtype=bool)
        if self.wall_strength.shape[0] != n:
            self.wall_strength = np.zeros(n, dtype=np.float64)

    def __len__(self) -> int:
        return int(self.population.shape[0])

    def reset_refinement(self) -> None:
        """Put every town back on its stage site (region refinement is stale)."""
        self.positions[:] = self.site_positions
        self.region_positions[:] = self.site_positions
        self.refined[:] = False
        self.local_refined[:] = False

    def reset_local_refinement(self) -> None:
        """Put every town back on its region's refinement (local refinement is stale)."""
        self.positions[:] = self.region_positions
        self.local_refined[:] = False

    @classmethod
    def empty(cls) -> "Towns":
        return cls(
            positions=np.zeros((0, 3), dtype=np.float64),
            population=np.zeros(0, dtype=np.float64),
            face_id=np.zeros(0, dtype=np.int32),
            parent=np.zeros(0, dtype=np.int32),
            kind=np.zeros(0, dtype=np.int8),
        )

    def pack(self) -> Dict[str, np.ndarray]:
        return {
            "town_position": self.positions.astype(np.float32),
            "town_site_position": self.site_positions.astype(np.float32),
            "town_refined": self.refined.astype(np.bool_),
            "town_region_position": self.region_positions.astype(np.float32),
            "town_local_refined": self.local_refined.astype(np.bool_),
            "town_population": self.population.astype(np.float64),
            "town_face_id": self.face_id.astype(np.int32),
            "town_parent": self.parent.astype(np.int32),
            "town_kind": self.kind.astype(np.int8),
            "town_has_walls": self.has_walls.astype(np.bool_),
            "town_wall_strength": self.wall_strength.astype(np.float64),
        }

    @classmethod
    def unpack(cls, arrays: Mapping[str, np.ndarray]) -> "Towns":
        return cls(
            positions=np.asarray(arrays["town_position"], dtype=np.float64).reshape(-1, 3),
            population=np.asarray(arrays["town_population"], dtype=np.float64),
            face_id=np.asarray(arrays["town_face_id"], dtype=np.int32),
            parent=np.asarray(arrays["town_parent"], dtype=np.int32),
            kind=np.asarray(arrays["town_kind"], dtype=np.int8),
            site_positions=np.asarray(arrays["town_site_position"], dtype=np.float64).reshape(-1, 3),
            refined=np.asarray(arrays["town_refined"], dtype=bool),
            region_positions=np.asarray(arrays["town_region_position"], dtype=np.float64).reshape(-1, 3),
            local_refined=np.asarray(arrays["town_local_refined"], dtype=bool),
            has_walls=np.asarray(arrays["town_has_walls"], dtype=bool),
            wall_strength=np.asarray(arrays["town_wall_strength"], dtype=np.float64),
        )
