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

    ``parent`` is the index into the ``settlements`` feature of the city whose
    hinterland holds the town's face (``-1`` outside every hinterland).

    ``site_positions`` is where the town stage put each town, on a global texel.
    ``positions`` is where the town actually is: the same, until a region view
    refines it onto the region's finer terrain and streams (``refined``), which
    only ever moves it within about one global texel of its site
    (``region/towns.py``).
    """

    positions: np.ndarray    # (K, 3) unit-sphere, current (refined where ``refined``)
    population: np.ndarray   # (K,) people
    face_id: np.ndarray      # (K,) owning global face
    parent: np.ndarray       # (K,) settlement index or -1
    kind: np.ndarray         # (K,) TOWN_* code
    site_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))   # (K, 3)
    refined: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))   # (K,)

    def __post_init__(self) -> None:
        if self.site_positions.shape[0] != self.positions.shape[0]:
            self.site_positions = self.positions.copy()
        if self.refined.shape[0] != self.positions.shape[0]:
            self.refined = np.zeros(self.positions.shape[0], dtype=bool)

    def __len__(self) -> int:
        return int(self.population.shape[0])

    def reset_refinement(self) -> None:
        """Put every town back on its stage site (region refinement is stale)."""
        self.positions[:] = self.site_positions
        self.refined[:] = False

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
            "town_population": self.population.astype(np.float64),
            "town_face_id": self.face_id.astype(np.int32),
            "town_parent": self.parent.astype(np.int32),
            "town_kind": self.kind.astype(np.int8),
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
        )
