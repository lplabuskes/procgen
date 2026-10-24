"""What is where on a layout, in words: the hover text and the one-line summary."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import shapely
from shapely.geometry import Point

from settlegen.layout.districts import OPEN_DISTRICTS
from settlegen.layout.geometry import _axes
from settlegen.layout.households import HOUSEHOLD_LABELS, TRADES, use_label
from settlegen.layout.types import Block, SettlementLayout
from settlegen.profile.builder import SettlementProfile

if TYPE_CHECKING:
    from settlegen.site.builder import SettlementSite


_DISTRICT_LABELS = {
    "residential": "Houses", "suburb": "Suburb", "gardens": "Gardens", "elite": "Elite quarter",
    "market": "Market", "quay": "Quayside", "graveyard": "Graveyard", "square": "Square",
    "sugar mills": "Sugar mills",
}


def district_label(district: str, tier: int) -> str:
    """How a district reads (a village's houses are farmsteads, its elite plot the manor)."""
    if tier == 5:
        return {"residential": "Farmstead", "elite": "Manor"}.get(district, district.capitalize())
    if tier == 4 and district == "graveyard":
        return "Churchyard"
    if district in _DISTRICT_LABELS:
        return _DISTRICT_LABELS[district]
    return f"{district.capitalize()}' quarter"


def block_at(layout: SettlementLayout, xy: np.ndarray) -> Optional[Block]:
    """The block holding ``xy`` (an STR tree, built once per layout)."""
    tree = getattr(layout, "_block_tree", None)
    if tree is None:
        tree = shapely.STRtree([b.polygon for b in layout.blocks])
        setattr(layout, "_block_tree", tree)
    hits = tree.query(Point(float(xy[0]), float(xy[1])), predicate="within")
    return layout.blocks[int(hits[0])] if len(hits) else None


def describe_at(layout: SettlementLayout, site: "SettlementSite", xy: np.ndarray, tier: int) -> Optional[str]:
    """What of the settlement is at ``xy`` (frame metres), or None."""
    pt = Point(float(xy[0]), float(xy[1]))
    village = tier == 5
    if any(p.contains(pt) for p in layout.piers):
        return "Jetty" if village else "Pier"
    walls = layout.walls
    if walls is not None and not walls.line.is_empty and walls.line.distance(pt) <= 0.5 * walls.width_m + 2.0:
        near_tower = len(walls.towers) and float(np.min(np.linalg.norm(walls.towers - xy, axis=1))) <= walls.width_m
        return "Tower" if near_tower else ("Stone wall" if walls.kind == "stone" else "Palisade")
    for geom, label in ((layout.citadel, "Citadel"), (layout.plaza, "Market street" if tier == 4 else "Market place"),
                        (layout.precinct, "Church and churchyard" if tier >= 4 else "Temple precinct"),
                        (layout.green, "Village green"), (layout.quays, "Quay")):
        if not geom.is_empty and geom.contains(pt):
            return label
    for i, line in enumerate(layout.arterials):
        if line.distance(pt) <= 0.5 * layout.arterial_width(i):
            kind = layout.arterial_kinds[i] if i < len(layout.arterial_kinds) else "main"
            return "Village street" if village else {"wall": "Wall street", "secondary": "Through street"}.get(
                kind, "Main street")
    for lanes, label in ((layout.closes, "Close"), (layout.alleys, "Alley")):
        if any(ln.distance(pt) <= 0.5 * layout.lane_m + 0.5 for ln in lanes):
            return label
    block = block_at(layout, xy)
    if block is not None:
        label = district_label(block.district, tier)
        for i, lot in enumerate(block.lots):
            if lot.contains(pt):
                _, _, _, length, depth = _axes(lot)
                what = use_label(block.use(i)) or label
                return f"{what} · plot ~{min(length, depth):.0f} × {max(length, depth):.0f} m"
        return label if block.district in OPEN_DISTRICTS or not len(block.lots) else f"{label} · yard"
    if not site.water.channels.is_empty and site.water.channels.contains(pt):
        return "River"                   # the sea and lakes the ground's own readout names
    built = layout.footprint if layout.footprint is not None else site.footprint
    if layout.fill_footprint and built.contains(pt):
        return "Street"
    return None


def summary(layout: SettlementLayout, profile: SettlementProfile) -> str:
    """One line on the settlement and its layout."""
    pol = profile.polity
    who = f"{profile.tier_name} ({profile.kind}) · {profile.population:,.0f} people"
    rule = "" if pol.control == "none" else f" · {pol.control}" + (" capital" if pol.capital else "")
    if pol.march >= 0.3:
        rule += " march"
    walls = f"{layout.walls.kind} walls" if layout.walls is not None else "unwalled"
    uses = [b.use(i) for b in layout.blocks for i in range(len(b.lots))]
    trades = sum(u in TRADES for u in uses)
    if profile.tier == 5:
        counts = {k: uses.count(k) for k in HOUSEHOLD_LABELS}
        plan = ", ".join([f"{counts['farm']} farmsteads", f"{trades} craftsmen"]
                         + [f"{counts[k]} {k}s' cottages" for k in ("fisher", "miner") if counts[k]])
    else:
        plan = f"{len(layout.blocks):,} blocks, {layout.n_lots:,} plots ({trades:,} shops)" + \
            (" (planned)" if profile.planned else "")
    n = len(layout.piers)
    docks = f" · {n} pier{'s' if n != 1 else ''}" if n else ""
    return f"{who}{rule} · {walls} · {plan}{docks}"
