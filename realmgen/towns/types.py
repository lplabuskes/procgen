"""The ``towns`` world feature: the settlements below the city floor."""
from __future__ import annotations

from dataclasses import dataclass
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
    """

    positions: np.ndarray    # (K, 3) unit-sphere
    population: np.ndarray   # (K,) people
    face_id: np.ndarray      # (K,) owning global face
    parent: np.ndarray       # (K,) settlement index or -1
    kind: np.ndarray         # (K,) TOWN_* code

    def __len__(self) -> int:
        return int(self.population.shape[0])

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
        )
