"""The local maps' roads and river and lake legs on the site: each map's on the ground it owns, chained
across the seams between maps (either end may join: roads have no direction), and their sea legs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from localmap.builder import LocalMap
from region.roads import RegionRoads, RouteSegment
from settlegen.terrain.source import LocalMapsSource


# A road cut at one map's edge continues on a neighbour's within this many local
# road-raster nodes (its vertices are that far apart before smoothing).
_SEAM_GAP_NODES = 2.5


@dataclass(eq=False)
class _Piece:
    """A run of one local map's road on the ground that map owns."""

    pos: np.ndarray           # (K, 3)
    flows: np.ndarray         # (K - 1,)
    cut_start: bool           # its first vertex is where the map's ground ends
    cut_end: bool
    map_index: int
    pin_start: bool = False   # a water leg's end pinned to the land at a port
    pin_end: bool = False

    def reversed(self) -> "_Piece":
        return _Piece(self.pos[::-1].copy(), self.flows[::-1].copy(), self.cut_end, self.cut_start, self.map_index,
                      self.pin_end, self.pin_start)


def _pieces(maps: Sequence[LocalMap], source: LocalMapsSource, water: bool = False) -> list[_Piece]:
    """Each map's roads (``water``: river and lake legs) on the ground that map owns."""
    out = []
    for i, m in enumerate(maps):
        r = m.roads
        paths = zip(r.water_polylines, r.water_flows, r.water_pins) if water else \
            zip(r.polylines, r.polyline_flows, [(False, False)] * len(r.polylines))
        for verts, flows, pins in paths:
            pos = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
            if len(pos) < 2:
                continue
            idx = np.flatnonzero(source.map_of(pos) == i)
            if idx.size == 0:
                continue
            breaks = np.flatnonzero(np.diff(idx) > 1)
            for a, b in zip(np.concatenate([[idx[0]], idx[breaks + 1]]), np.concatenate([idx[breaks], [idx[-1]]])):
                if b <= a:
                    continue
                out.append(_Piece(
                    pos=pos[a:b + 1], flows=np.asarray(flows, dtype=np.float64)[a:b],
                    cut_start=a > 0, cut_end=b < len(pos) - 1, map_index=i,
                    pin_start=a == 0 and bool(pins[0]), pin_end=b == len(pos) - 1 and bool(pins[1]),
                ))
    return out


def _chain(pieces: list[_Piece], gap: float) -> list[_Piece]:
    """Join each road cut at its map's edge to the nearest cut end of a road on
    another map within ``gap`` (chord).  Roads have no direction: either end may join."""
    pieces = list(pieces)
    joined = True
    while joined:
        joined = False
        best: Optional[tuple[float, _Piece, bool, _Piece, bool]] = None
        for p in pieces:
            for p_end in (False, True):
                if not (p.cut_end if p_end else p.cut_start):
                    continue
                at = p.pos[-1] if p_end else p.pos[0]
                for q in pieces:
                    if q is p or q.map_index == p.map_index:
                        continue
                    for q_end in (False, True):
                        if not (q.cut_end if q_end else q.cut_start):
                            continue
                        d = float(np.linalg.norm((q.pos[-1] if q_end else q.pos[0]) - at))
                        if d < gap and (best is None or d < best[0]):
                            best = (d, p, p_end, q, q_end)
        if best is None:
            break
        _, p, p_end, q, q_end = best
        head = p if p_end else p.reversed()           # cut at its end
        tail = q.reversed() if q_end else q           # cut at its start
        pieces.remove(p)
        pieces.remove(q)
        pieces.append(_Piece(
            pos=np.concatenate([head.pos, tail.pos]),
            flows=np.concatenate([head.flows, [max(float(head.flows[-1]), float(tail.flows[0]))], tail.flows]),
            cut_start=head.cut_start, cut_end=tail.cut_end, map_index=tail.map_index,
            pin_start=head.pin_start, pin_end=tail.pin_end,
        ))
        joined = True
    return pieces


def _sea_legs(maps: Sequence[LocalMap]) -> list[RouteSegment]:
    """Every map's sea legs, once each (neighbours carry the same region legs)."""
    seen: set = set()
    out = []
    for m in maps:
        for a, b, f in m.roads.sea:
            key = (tuple(np.round(np.asarray(a, dtype=np.float64), 7)),
                   tuple(np.round(np.asarray(b, dtype=np.float64), 7)))
            if key in seen:
                continue
            seen.add(key)
            out.append((np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), float(f)))
    return out


def parent_roads(maps: Sequence[LocalMap], antimeridian: float, road_raster_texels: float) -> RegionRoads:
    """The local maps' roads and river and lake legs over the ground each owns,
    chained across seams, and their sea legs."""
    source = LocalMapsSource([m.fields for m in maps], antimeridian)
    f0 = maps[0].fields
    gap = _SEAM_GAP_NODES * float(road_raster_texels) * f0.grid.texel_spacing_rad(f0.tex_size)
    pieces = _chain(_pieces(maps, source), gap)
    waters = _chain(_pieces(maps, source, water=True), gap)
    sea = _sea_legs(maps)
    return RegionRoads(
        region_id=f0.region_id, segments=list(sea), sea=sea,
        polylines=[p.pos for p in pieces], polyline_flows=[p.flows for p in pieces],
        water_polylines=[p.pos for p in waters], water_flows=[p.flows for p in waters],
        water_pins=[(p.pin_start, p.pin_end) for p in waters],
    )
