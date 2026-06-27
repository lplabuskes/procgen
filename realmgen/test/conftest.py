"""Shared test helpers for all realmgen tests."""
from __future__ import annotations

import numpy as np

from realmgen.placement.types import Settlement


def _settlement(
    face_id: int = 0,
    tier: int = 0,
    pop: float = 0.0,
    score: float = 1.0,
    position: np.ndarray | None = None,
) -> Settlement:
    if position is None:
        position = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return Settlement(face_id=face_id, position=position, tier=tier, score=score, population=pop)
