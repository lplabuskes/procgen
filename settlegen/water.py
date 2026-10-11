"""A settlement site's water, as vectors: the sea, lakes, shorelines, rivers and their channels.

- **Sea and lakes** are traced from the refined terrain (:mod:`settlegen.contour`):
  the zero crossing of height above the water surface — sea level, or the nearest
  lake's surface — with its sign forced to the refined masks, so the polygons
  are the masks' water with smooth shores rather than a texel staircase.
- **Rivers and streams** are the local maps' (each clipped to the ground its map
  owns, and chained across a seam between maps) re-traced over the frame's
  relief within about one local texel (:func:`localmap.rivers.build_local_rivers`
  one tier down), then widened to their bankfull width
  (:func:`localmap.banks.river_channels`) into channel polygons.

Qt-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import shapely
from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

from core.params import param_field as _f, paramclass
from localmap.builder import LocalMap
from localmap.rivers import LocalRiverParams, build_local_rivers
from region.rivers import RegionRivers, RiverPath
from region.upsample import SEA_LEVEL_KM, RegionFields
from settlegen.contour import regions_below
from settlegen.frame import Frame
from settlegen.source import LocalMapsSource
from worldgen.hydrology.discharge import channel_width_km, discharge_per_flow_unit

if TYPE_CHECKING:
    from core.world import World

# Water / dry decided by the masks, the shore placed by the heights: never exactly on the level.
_EPS_M = 0.01
# A path cut at one map's edge continues on a neighbour's within this many local texels.
_SEAM_GAP_TEXELS = 3.0
# Narrowest channel polygon (m): streams are drawn, however small.
_MIN_CHANNEL_M = 1.0


@paramclass
class SettlementRiverParams(LocalRiverParams):
    """:class:`LocalRiverParams` at settlement scale (raster in settlement texels, corridor in local texels)."""

    # ~45 m: the 15 m raster keeps the router's 45° turns; this and 12 passes smooth them out.
    raster_texels: float = _f(3.0, label="River raster spacing (texels)", min=0.5, max=8.0, step=0.5)
    # Past the frame, in frame units (the frame is 1).
    margin: float = _f(0.02, label="River raster margin", min=0.0, max=0.2, step=0.01)
    corridor_parent_texels: float = _f(
        1.0, label="Corridor half-width (local texels)", min=0.25, max=3.0, step=0.25)
    smoothing_iterations: int = _f(12, label="Path smoothing iterations", min=0, max=40)


@dataclass
class RiverLine:
    """One re-traced river or stream on the site, in frame metres."""

    xy: np.ndarray            # (K, 2)
    flows: np.ndarray         # (K - 1,) world flow per edge
    width_m: np.ndarray       # (K - 1,) bankfull width per edge
    in_water: np.ndarray      # (K - 1,) bool: under a lake or the sea (not drawn)
    minor: bool               # a stream, not a world river


@dataclass
class SiteWater:
    """The site's water.  ``rivers`` are the re-traced paths on the sphere (for routing roads)."""

    sea: MultiPolygon
    lakes: MultiPolygon
    shorelines: MultiLineString
    lines: list[RiverLine]
    channels: MultiPolygon
    rivers: RegionRivers = field(repr=False)

    @property
    def bodies(self) -> MultiPolygon:
        """Sea, lakes and river channels: all the site's water."""
        merged = shapely.union_all([self.sea, self.lakes, self.channels])
        return _as_multipolygon(merged)


def _as_multipolygon(geom) -> MultiPolygon:  # type: ignore[no-untyped-def]
    if isinstance(geom, MultiPolygon):
        return geom
    if isinstance(geom, Polygon):
        return MultiPolygon([geom]) if not geom.is_empty else MultiPolygon()
    return MultiPolygon([g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon) and not g.is_empty])


# ── sea and lakes ────────────────────────────────────────────────────────────


def water_regions(fields: RegionFields, frame: Frame) -> tuple[MultiPolygon, MultiPolygon]:
    """``(sea, lakes)`` polygons of the refined masks, shores at the water surface."""
    t = fields.tex_size
    xy = frame.texel_xy(t)
    elev_m = fields.textures["elevation"][0, ..., 0].astype(np.float64) * 1000.0
    out = []
    for key, level in (("ocean_mask", None), ("lake_mask", "lake")):
        mask = fields.textures.get(key)
        if mask is None or not (mask[0, ..., 0] > 0.5).any():
            out.append(MultiPolygon())
            continue
        wet = mask[0, ..., 0] > 0.5
        if level is None:
            surface = np.full_like(elev_m, SEA_LEVEL_KM * 1000.0)
        else:
            # Each texel reads the surface of its nearest lake texel.
            lake_surface = fields.textures["filled_height"][0, ..., 0].astype(np.float64) * 1000.0
            nearest = np.asarray(distance_transform_edt(~wet, return_distances=False, return_indices=True))
            surface = lake_surface[nearest[0], nearest[1]]
        height = elev_m - surface
        signed = np.where(wet, np.minimum(height, -_EPS_M), np.maximum(height, _EPS_M))
        out.append(regions_below(signed, xy, frame.box()))
    return out[0], out[1]


def shorelines(water: MultiPolygon, frame: Frame) -> MultiLineString:
    """The water's edge inside the frame (its boundary, less the frame's own edge)."""
    if water.is_empty:
        return MultiLineString()
    edge = frame.box().exterior.buffer(1e-3)
    lines = shapely.line_merge(shapely.difference(water.boundary, edge))
    if lines.is_empty:
        return MultiLineString()
    parts = [lines] if isinstance(lines, LineString) else list(getattr(lines, "geoms", []))
    return MultiLineString([p for p in parts if isinstance(p, LineString) and not p.is_empty])


# ── rivers ───────────────────────────────────────────────────────────────────


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


def river_lines(rivers: RegionRivers, frame: Frame, discharge_scale: float) -> list[RiverLine]:
    out = []
    for paths, minor in ((rivers.paths, False), (rivers.streams, True)):
        for positions, flows, in_water in paths:
            pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
            if len(pos) < 2:
                continue
            f = np.asarray(flows, dtype=np.float64)[:len(pos) - 1]
            out.append(RiverLine(
                xy=frame.to_xy(pos), flows=f, width_m=channel_width_km(f * discharge_scale) * 1000.0,
                in_water=np.asarray(in_water, dtype=bool)[:len(pos) - 1], minor=minor,
            ))
    return out


def channel_polygons(lines: Sequence[RiverLine], clip: Optional[Polygon] = None) -> MultiPolygon:
    """The ground the channels cover: each dry segment widened to its bankfull width
    (clipped to ``clip``, if given)."""
    a, b, half = [], [], []
    for line in lines:
        dry = ~line.in_water
        a.append(line.xy[:-1][dry])
        b.append(line.xy[1:][dry])
        half.append(np.maximum(0.5 * line.width_m[dry], 0.5 * _MIN_CHANNEL_M))
    if not a or not sum(len(x) for x in a):
        return MultiPolygon()
    a_xy, b_xy, h = np.concatenate(a), np.concatenate(b), np.concatenate(half)
    # Flat ends, so a channel stops square at a shore or the frame instead of in a disc
    # its full width; each joint between two segments rounded at the narrower one's width.
    parts = list(shapely.buffer(shapely.linestrings(np.stack([a_xy, b_xy], axis=1)), h, cap_style="flat"))
    joint = np.linalg.norm(a_xy[1:] - b_xy[:-1], axis=1) < 1e-6
    if joint.any():
        k = np.flatnonzero(joint)
        parts.extend(shapely.buffer(shapely.points(a_xy[k + 1]), np.minimum(h[k], h[k + 1]), quad_segs=4))
    merged = shapely.union_all(parts)
    return _as_multipolygon(merged.intersection(clip) if clip is not None else merged)


def build_site_water(
    globe: "World", fields: RegionFields, frame: Frame, maps: Sequence[LocalMap],
    params: Optional[SettlementRiverParams] = None,
) -> SiteWater:
    """The site's water.  See the module docstring."""
    params = params or SettlementRiverParams()
    sea, lakes = water_regions(fields, frame)
    parent, streams = parent_rivers(maps, globe.antimeridian)
    rivers = (build_local_rivers(fields, maps[0].fields, parent, globe, params, streams)
              if parent.paths or streams else RegionRivers(region_id=fields.region_id, paths=[]))
    lines = river_lines(rivers, frame, discharge_per_flow_unit(globe.mesh))
    still = _as_multipolygon(shapely.union_all([sea, lakes]))
    return SiteWater(
        sea=sea, lakes=lakes, shorelines=shorelines(still, frame),
        lines=lines, channels=channel_polygons(lines, frame.box()), rivers=rivers,
    )
