"""What a layout holds: its walls, blocks and the whole plan, in frame metres."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

from settlegen.layout.lots import derive_lots, dividers_of, seeds_of


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
    """One block between streets (a village: one street side's run of plots): the ground
    its lots share and the lines dividing it; the lots are derived from them
    (:mod:`settlegen.layout.lots`) — they tile their ground, sharing edges."""

    polygon: Polygon
    district: str
    core: bool                   # inside the old core
    dividers: MultiLineString = field(default_factory=MultiLineString)   # between lots and along their backs
    seeds: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # (k, 2): a point in each lot
    # (k,) str per lot: a trade or a household's use (settlegen.layout.households); "" or
    # None — the district's.
    uses: Optional[np.ndarray] = None
    _lots: Optional[np.ndarray] = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def of_lots(cls, polygon: Polygon, district: str, core: bool, lots: Sequence[Polygon] | np.ndarray,
                uses: Optional[np.ndarray] = None, dividers: Optional[MultiLineString] = None) -> "Block":
        """The block whose lots (tiling part of ``polygon``) are ``lots`` — which are what its
        dividers derive, so they are kept rather than derived again.  ``dividers``: the lines
        between them where known (else traced from their outlines)."""
        block = cls(polygon=polygon, district=district, core=core,
                    dividers=dividers if dividers is not None else dividers_of(polygon, lots),
                    seeds=seeds_of(lots), uses=uses)
        block._lots = np.asarray(lots, dtype=object)
        return block

    @property
    def lots(self) -> np.ndarray:
        """``(k,)`` lot polygons, one per seed (derived once)."""
        if self._lots is None:
            self._lots = derive_lots(self.polygon, self.dividers, self.seeds)
        return self._lots

    def use(self, i: int) -> str:
        return "" if self.uses is None else str(self.uses[i])

    def without(self, i: int) -> "Block":
        """The block with lot ``i`` open ground (its dividers kept)."""
        keep = np.arange(len(self.seeds)) != i
        out = Block(polygon=self.polygon, district=self.district, core=self.core, dividers=self.dividers,
                    seeds=self.seeds[keep], uses=None if self.uses is None else self.uses[keep])
        out._lots = None if self._lots is None else self._lots[keep]
        return out

    def lot_district(self, i: int) -> str:
        """The district a lot reads as (a village's manor: the elite's)."""
        return "elite" if self.use(i) == "manor" else self.district


@dataclass
class Buildings:
    """Every building of a layout (settlegen.layout.buildings), as parallel arrays."""

    polygons: np.ndarray         # (n,) shapely polygons
    kinds: np.ndarray            # (n,) str: "house", "row", "barn", "workshop", "hall", "church", …
    lots: np.ndarray             # (n, 2) int: (block, lot) the building stands on; (-1, -1) on open ground

    @staticmethod
    def empty() -> "Buildings":
        return Buildings(np.zeros(0, dtype=object), np.zeros(0, dtype=object), np.zeros((0, 2), dtype=np.int64))

    def __len__(self) -> int:
        return len(self.polygons)


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
    buildings: Buildings = field(default_factory=Buildings.empty)

    def arterial_width(self, i: int) -> float:
        kind = self.arterial_kinds[i] if i < len(self.arterial_kinds) else "main"
        return self.arterial_m if kind == "main" else self.secondary_m

    @staticmethod
    def empty() -> "SettlementLayout":
        e = MultiPolygon()
        return SettlementLayout({}, e, None, e, e, e, [], np.zeros((0, 2)), [], [])

    def district_areas_ha(self) -> dict[str, float]:
        """Hectares per district: blocks, or (a village, not filling its footprint) plots."""
        out: dict[str, float] = {}
        for b in self.blocks:
            if self.fill_footprint:
                out[b.district] = out.get(b.district, 0.0) + b.polygon.area / 1e4
                continue
            for i, lot in enumerate(b.lots):
                out[b.lot_district(i)] = out.get(b.lot_district(i), 0.0) + lot.area / 1e4
        return out

    @property
    def n_lots(self) -> int:
        return int(sum(len(b.lots) for b in self.blocks))
