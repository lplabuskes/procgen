"""What a layout holds: its walls, blocks and the whole plan, in frame metres."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon


@dataclass
class Walls:
    """The core's walls."""

    kind: str                    # "palisade" / "stone"
    line: MultiLineString        # open along the water
    towers: np.ndarray           # (n, 2)
    gates: np.ndarray            # (n, 2): where arterials pass
    width_m: float = 5.0


@dataclass
class Block:
    """One block between streets."""

    polygon: Polygon
    district: str
    core: bool                   # inside the old core
    lots: np.ndarray             # (k,) shapely polygons


@dataclass
class SettlementLayout:
    """See the module docstring.  All in frame metres."""

    anchors: dict[str, np.ndarray]           # kind → (2,)
    core: MultiPolygon
    walls: Optional[Walls]
    citadel: MultiPolygon
    plaza: MultiPolygon
    precinct: MultiPolygon
    arterials: list[LineString]              # main streets, then the wall street and secondary arterials
    bridges: np.ndarray                      # (n, 2)
    streets: list[LineString]                # minor streets' centre lines
    blocks: list[Block]
    points: dict[str, np.ndarray] = field(default_factory=dict)   # wells, shrines: (n, 2)
    arterial_m: float = 10.0
    quays: MultiPolygon = field(default_factory=MultiPolygon)
    piers: list[Polygon] = field(default_factory=list)
    green: MultiPolygon = field(default_factory=MultiPolygon)        # a village green
    fill_footprint: bool = True              # the footprint is streets between the blocks (not a village)
    # The built-up ground when the layout grew the site's footprint (walls taking in gardens).
    footprint: Optional[MultiPolygon] = None
    # Per arterial: "main", "wall" (inside the walls) or "secondary"; empty — all main.
    arterial_kinds: list[str] = field(default_factory=list)
    secondary_m: float = 7.0
    # Lanes into deep blocks: centre lines of the dead ends and the through ways.
    closes: list[LineString] = field(default_factory=list)
    alleys: list[LineString] = field(default_factory=list)
    lane_m: float = 3.0

    def arterial_width(self, i: int) -> float:
        kind = self.arterial_kinds[i] if i < len(self.arterial_kinds) else "main"
        return self.arterial_m if kind == "main" else self.secondary_m

    @staticmethod
    def empty() -> "SettlementLayout":
        e = MultiPolygon()
        return SettlementLayout({}, e, None, e, e, e, [], np.zeros((0, 2)), [], [])

    def district_areas_ha(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for b in self.blocks:
            out[b.district] = out.get(b.district, 0.0) + b.polygon.area / 1e4
        return out

    @property
    def n_lots(self) -> int:
        return int(sum(len(b.lots) for b in self.blocks))
