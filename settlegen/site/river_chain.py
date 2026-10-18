"""The local maps' rivers and streams on the site: each map's on the ground it owns, chained across the
seams between maps, then as lines in frame metres with their ends tidied at the frame's edge."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from localmap.builder import LocalMap
from region.rivers import RegionRivers, RiverPath
from settlegen.frame import Frame
from settlegen.site.types import RiverLine
from settlegen.terrain.source import LocalMapsSource
from worldgen.hydrology.discharge import channel_width_km


# A path cut at one map's edge continues on a neighbour's within this many local texels.
_SEAM_GAP_TEXELS = 3.0


@dataclass(eq=False)
class _Piece:
    """A run of one local map's path on the ground that map owns."""

    pos: np.ndarray
    flows: np.ndarray
    in_water: np.ndarray
    ids: np.ndarray
    starts: bool              # holds the path's first vertex
    ends: bool                # holds its last vertex
    map_index: int


def _pieces(
    maps: Sequence[LocalMap], source: LocalMapsSource, streams: bool,
) -> list[_Piece]:
    out = []
    for i, m in enumerate(maps):
        paths = m.rivers.streams if streams else m.rivers.paths
        ids = [np.full(len(p[0]), -1, dtype=np.int64) for p in paths] if streams else m.rivers.river_ids()
        for (positions, flows, in_water), river_ids in zip(paths, ids):
            pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
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
                    in_water=np.asarray(in_water, dtype=bool)[a:b], ids=np.asarray(river_ids)[a:b + 1],
                    starts=a == 0, ends=b == len(pos) - 1, map_index=i,
                ))
    return out


def _chain(pieces: list[_Piece], gap: float) -> list[_Piece]:
    """Join each piece cut at its map's edge to the piece continuing it on a neighbour:
    its first vertex within ``gap`` (chord) of the cut, the same world river if known."""
    pieces = list(pieces)
    joined = True
    while joined:
        joined = False
        for p in pieces:
            if p.ends:
                continue
            best, best_d = None, gap
            for q in pieces:
                if q is p or q.starts or q.map_index == p.map_index:
                    continue
                if p.ids[-1] >= 0 and q.ids[0] >= 0 and p.ids[-1] != q.ids[0]:
                    continue
                d = float(np.linalg.norm(q.pos[0] - p.pos[-1]))
                if d < best_d:
                    best, best_d = q, d
            if best is None:
                continue
            q = best
            pieces.remove(q)
            p.pos = np.concatenate([p.pos, q.pos])
            p.flows = np.concatenate([p.flows, [max(float(p.flows[-1]), float(q.flows[0]))], q.flows])
            p.in_water = np.concatenate([p.in_water, [False], q.in_water])
            p.ids = np.concatenate([p.ids, q.ids])
            p.ends, p.map_index = q.ends, q.map_index
            joined = True
            break
    return pieces


def parent_rivers(maps: Sequence[LocalMap], antimeridian: float) -> tuple[RegionRivers, list[RiverPath]]:
    """``(rivers, streams)`` of the local maps over the ground each owns, chained across seams."""
    source = LocalMapsSource([m.fields for m in maps], antimeridian)
    gap = _SEAM_GAP_TEXELS * maps[0].fields.grid.texel_spacing_rad(maps[0].fields.tex_size)
    rivers = _chain(_pieces(maps, source, streams=False), gap)
    streams = _chain(_pieces(maps, source, streams=True), gap)
    paths: list[RiverPath] = [(p.pos, p.flows, p.in_water) for p in rivers]
    return (
        RegionRivers(region_id=maps[0].fields.region_id, paths=paths, path_rivers=[p.ids for p in rivers]),
        [(p.pos, p.flows, p.in_water) for p in streams],
    )


# How far along a line from its end (m) a hook at the frame's edge, or an end short of it, is tidied.
_END_REACH_M = 600.0


def _tidy_ends(line: RiverLine, half: float, reach_m: float = _END_REACH_M) -> RiverLine:
    """``line`` with each end that hooks back in at the frame's edge (outside within
    ``reach_m`` of it, the end itself inside) cut at its last vertex outside, and each
    end left inside within ``reach_m`` of the edge, heading out, carried straight past it."""
    for _ in range(2):              # the start, then (reversed) the end
        xy = line.xy
        out = np.max(np.abs(xy), axis=1) > half
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        hooks = np.flatnonzero(out & (s <= reach_m))
        if not out[0] and hooks.size and hooks[-1] < len(xy) - 1:
            k = int(hooks[-1])
            line = RiverLine(xy=xy[k:], flows=line.flows[k:], width_m=line.width_m[k:],
                             in_water=line.in_water[k:], minor=line.minor)
        elif not out[0]:
            heading = xy[0] - xy[1]
            heading = heading / max(float(np.linalg.norm(heading)), 1e-12)
            gap = half - float(np.max(np.abs(xy[0])))
            axis = int(np.argmax(np.abs(xy[0])))
            outward = float(np.sign(xy[0][axis]) * heading[axis])
            if gap < reach_m and outward > 0.3:
                start = xy[0] + heading * (gap + 50.0) / outward
                line = RiverLine(xy=np.vstack([start, xy]), flows=np.r_[line.flows[:1], line.flows],
                                 width_m=np.r_[line.width_m[:1], line.width_m],
                                 in_water=np.r_[line.in_water[:1], line.in_water], minor=line.minor)
        line = RiverLine(xy=line.xy[::-1].copy(), flows=line.flows[::-1].copy(), width_m=line.width_m[::-1].copy(),
                         in_water=line.in_water[::-1].copy(), minor=line.minor)
    return line


def river_lines(rivers: RegionRivers, frame: Frame, discharge_scale: float) -> list[RiverLine]:
    """The re-traced rivers and streams in frame metres, their ends tidied at the frame's edge."""
    out = []
    for paths, minor in ((rivers.paths, False), (rivers.streams, True)):
        for positions, flows, in_water in paths:
            pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
            if len(pos) < 2:
                continue
            f = np.asarray(flows, dtype=np.float64)[:len(pos) - 1]
            out.append(_tidy_ends(RiverLine(
                xy=frame.to_xy(pos), flows=f, width_m=channel_width_km(f * discharge_scale) * 1000.0,
                in_water=np.asarray(in_water, dtype=bool)[:len(pos) - 1], minor=minor,
            ), frame.half))
    return out
