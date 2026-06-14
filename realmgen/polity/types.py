"""Polity data types and display constants."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, NamedTuple, Optional

import numpy as np


class ControlClass(IntEnum):
    """How a settlement relates to the polity claiming it (projected authority
    vs local power): ``DIRECT`` = directly-ruled province, ``VASSAL`` = founds a
    child polity under its claimant, ``INDEPENDENT`` = founds a new sovereign."""

    DIRECT = 0
    VASSAL = 1
    INDEPENDENT = 2


class Scalars(NamedTuple):
    """The four per-polity authority scalars — one (C,) array each, capital-aligned."""

    cohesion: np.ndarray
    infrastructure: np.ndarray
    legitimacy: np.ndarray
    maritime_reach: np.ndarray

    def take(self, idx) -> "Scalars":
        """Sub-select/reorder polities; always returns 1-D arrays."""
        return Scalars(*(np.atleast_1d(np.asarray(a)[idx]) for a in self))


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
    """A node in the polity forest grown from a capital settlement.

    Sovereigns are roots (``liege is None``); vassals point at a parent via
    ``liege``. ``polity_id`` aligns with the per-face ``polity_id`` metadata (the
    *directly governing* polity — sovereign or vassal). Scalars: ``cohesion``
    (stability), ``infrastructure`` (projection range), ``legitimacy`` (field
    magnitude), ``maritime_reach`` (sea projection). ``power`` is own-territory
    member population; ``domain_power`` adds descendant vassals.
    """

    polity_id: int
    capital_face: int
    capital_position: np.ndarray            # (3,) unit-sphere centroid of the capital face
    # Per-polity scalars (see class docstring).
    cohesion: float = 1.0
    infrastructure: float = 1.0
    legitimacy: float = 1.0
    maritime_reach: float = 0.0
    # Hierarchy.
    liege: Optional[int] = None             # parent polity id; None ⇒ sovereign root
    autonomy: float = 0.0                   # 0 for sovereigns; >0 for vassals
    depth: int = 0                          # 0 = sovereign root
    # Strength + territory.
    power: float = 0.0                      # own-territory member-settlement population
    domain_power: float = 0.0               # own + all descendant-vassal members
    n_faces: int = 0                        # own-territory size (face count)
    territory_faces: List[int] = field(default_factory=list)            # faces directly governed
    member_settlement_faces: List[int] = field(default_factory=list)    # settlement faces directly governed
    vassals: List[int] = field(default_factory=list)                    # child polity ids
