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


# Sovereign palette — assigned to independent nations by ``polity_id % len(POLITY_COLORS)``.
# Rich, saturated, and dark enough (L ≈ 0.38–0.42) that vassal tints (lightened in
# make_polity_tree_color) land visibly lighter while staying clearly in the same family.
# Yellow-green band (~75°–115°) is intentionally skipped to avoid muddy colours.
POLITY_COLORS: List[tuple] = [
    (0.69, 0.19, 0.11, 1.0),  # crimson      [  8°]
    (0.72, 0.44, 0.12, 1.0),  # orange       [ 32°]
    (0.71, 0.67, 0.13, 1.0),  # gold         [ 55°]
    (0.14, 0.62, 0.21, 1.0),  # forest green [128°]
    (0.10, 0.70, 0.38, 1.0),  # sea green    [148°]
    (0.13, 0.63, 0.54, 1.0),  # teal         [170°]
    (0.10, 0.58, 0.70, 1.0),  # cerulean     [192°]
    (0.14, 0.37, 0.70, 1.0),  # sky blue     [215°]
    (0.10, 0.15, 0.70, 1.0),  # indigo       [235°]
    (0.26, 0.13, 0.67, 1.0),  # cobalt       [255°]
    (0.42, 0.10, 0.70, 1.0),  # purple       [272°]
    (0.57, 0.15, 0.65, 1.0),  # violet       [290°]
    (0.70, 0.10, 0.65, 1.0),  # fuchsia      [305°]
    (0.66, 0.14, 0.49, 1.0),  # magenta      [320°]
    (0.70, 0.10, 0.30, 1.0),  # rose         [340°]
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
