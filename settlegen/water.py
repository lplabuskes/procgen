"""A settlement site's water, as vectors: the sea, lakes, shorelines, rivers and their channels.

- **Sea and lakes** are traced from the refined terrain (:mod:`settlegen.contour`):
  the zero crossing of height above the water surface — sea level, or the nearest
  lake's surface — with its sign forced to the refined masks, so the polygons
  are the masks' water with smooth shores rather than a texel staircase.  Where
  the shore is within ``_FLAT_SHORE_M`` of the water, the heights can't place it
  (a forced sign alone traces the staircase): the mask, smoothed, does.
- **Rivers and streams** are the local maps' (each clipped to the ground its map
  owns, and chained across a seam between maps) re-traced over the frame's
  relief within about one local texel (:func:`localmap.rivers.build_local_rivers`
  one tier down), then widened into channels (:func:`natural_channels`): their
  bankfull width, wandering by ``width_wander`` and each bank by
  ``bank_wander`` over a few widths, with islands (eyots) in rivers at least
  ``island_min_width_m`` wide, the channel widened around each so both arms
  keep the river's width.  A channel runs on into the sea or lake it meets, so
  the two meet without a shore line across the mouth.  A line's end that hooks
  back in at the frame's edge (the router, pinned at its raster's edge) is cut
  where it leaves, and an end left just inside is carried straight out, so
  channels reach the frame's edge whole (:func:`_tidy_ends`).

Qt-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence
import zlib

import numpy as np
import shapely
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon

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
# Below this height above (or depth under) the water, the smoothed mask places the shore (σ in texels).
_FLAT_SHORE_M = 1.0
_FLAT_SHORE_SIGMA = 1.5
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
    # The channel's shape (natural_channels).
    width_wander: float = _f(0.15, label="Channel width wander (lognormal σ)", min=0.0, max=0.6, step=0.01)
    bank_wander: float = _f(0.1, label="Channel bank wander (× width)", min=0.0, max=0.5, step=0.01)
    wander_widths: float = _f(4.0, label="Channel wander length (× width)", min=0.5, max=20.0, step=0.5)
    island_min_width_m: float = _f(60.0, label="Narrowest river with islands (m)", min=5.0, max=1000.0, step=5.0)
    islands_per_km: float = _f(0.6, label="Islands per km of river", min=0.0, max=5.0, step=0.05)
    # Mean island breadth (× the river's width) and length (× its breadth).
    island_width: float = _f(0.3, label="Island breadth (× river width)", min=0.05, max=2.0, step=0.05)
    island_length: float = _f(7.0, label="Island length (× breadth)", min=1.5, max=20.0, step=0.5)
    # The channel widens round an island by this share of its breadth (1: both arms keep the river's width).
    island_widening: float = _f(0.5, label="Channel widening at an island (× breadth)", min=0.0, max=1.0, step=0.05)
    # Land between a channel and the water it runs into narrower than this goes under water.
    min_land_m: float = _f(75.0, label="Narrowest land beside a channel (m)", min=0.0, max=500.0, step=5.0)
    # The channel's course is the line smoothed over this many widths.
    course_smoothing: float = _f(1.0, label="Channel course smoothing (× width)", min=0.0, max=5.0, step=0.1)


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
    """The site's water.  ``rivers`` are the re-traced paths on the sphere (for routing roads);
    ``channels`` reach as far past the frame as they do (:func:`natural_channels`)."""

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
        smooth = np.abs(gaussian_filter(np.where(wet, -1.0, 1.0), _FLAT_SHORE_SIGMA, mode="nearest"))
        least = np.maximum(_FLAT_SHORE_M * smooth, _EPS_M)
        signed = np.where(wet, np.minimum(height, -least), np.maximum(height, least))
        out.append(regions_below(signed, xy, frame.box()))
    return out[0], out[1]


def shorelines(water: MultiPolygon, frame: Frame, channels: Optional[MultiPolygon] = None) -> MultiLineString:
    """The water's edge inside the frame (its boundary, less the frame's own edge and
    the mouths of ``channels``)."""
    if water.is_empty:
        return MultiLineString()
    edge = frame.box().exterior.buffer(1e-3)
    if channels is not None and not channels.is_empty:
        edge = edge.union(channels.buffer(0.5))
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


def channel_polygons(lines: Sequence[RiverLine], clip: Optional[Polygon] = None) -> MultiPolygon:
    """Each dry segment widened to its bankfull width (clipped to ``clip``, if given):
    the channels' plain outline, without :func:`natural_channels`' wander and islands."""
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


_DENSIFY_M = 10.0
# How far (× its width) a channel runs on past an end that meets water, the frame's edge or another channel.
_END_RUN_WIDTHS = 2.0


def _line_rng(line: RiverLine) -> np.random.Generator:
    """A stream of its own per line (its ends, to the metre), so a channel is the same however often it is built."""
    key = np.round(np.concatenate([line.xy[0], line.xy[-1]])).astype(np.int64)
    return np.random.default_rng([zlib.crc32(key.tobytes()), len(line.xy)])


def _smooth_noise(n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """``n`` samples of unit-variance noise correlated over ``sigma`` samples."""
    sigma = max(float(sigma), 0.5)
    pad = int(np.ceil(3.0 * sigma))
    raw = gaussian_filter1d(rng.standard_normal(n + 2 * pad), sigma, mode="wrap")[pad:pad + n]
    # Filtered white noise has std 1 / sqrt(2 sqrt(pi) sigma): scaled by theory, not by a short sample.
    return raw * np.sqrt(2.0 * np.sqrt(np.pi) * sigma)


def _lens(u: np.ndarray, power: float) -> np.ndarray:
    """1 at ``u`` = 0.5, falling to 0 at 0 and 1 (and outside)."""
    return np.clip(1.0 - (2.0 * u - 1.0) ** 2, 0.0, 1.0) ** power


def _end_heading(xy: np.ndarray, reach: float) -> np.ndarray:
    """Unit direction out of ``xy``'s last vertex: the chord from ``reach`` back along it
    (a chord, not the last segment, so a kink at the very end doesn't steer it)."""
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy[::-1], axis=0), axis=1))]
    back = xy[::-1][int(min(np.searchsorted(s, reach), len(xy) - 1))]
    d = xy[-1] - back
    if float(np.linalg.norm(d)) < 1e-9:
        d = xy[-1] - xy[-2]
    return d / max(float(np.linalg.norm(d)), 1e-12)


def _line_channel(
    line: RiverLine, params: SettlementRiverParams,
    extend: tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None),
    avoid: Sequence[tuple[LineString, float]] = (),
) -> tuple[list[Polygon], list[Polygon]]:
    """``(bank quads, islands)`` of one line's channel.  ``avoid``: the other lines and
    their widths, which islands keep clear of (a confluence partway along this line).
    See :func:`natural_channels`."""
    xy = np.asarray(line.xy, dtype=np.float64)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    keep = np.r_[True, seg > 1e-9]
    xy = xy[keep]
    width = np.r_[line.width_m, line.width_m[-1:]][keep]
    wet = np.r_[line.in_water, line.in_water[-1:]][keep]
    if len(xy) < 2:
        return [], []
    # Ends that run on (into water, past the frame, into another channel) carried
    # straight on, so the band overruns what cuts it rather than stopping square short.
    length0 = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    pre = post = 0.0

    def heading(given: np.ndarray, out_of: np.ndarray, w: float) -> np.ndarray:
        own = _end_heading(out_of, min(w, length0 / 3.0))
        if float(np.linalg.norm(given)) < 0.5:
            return own
        # Into still water: halfway between the line's own heading and the water's middle.
        mid = own + given
        norm = float(np.linalg.norm(mid))
        return mid / norm if norm > 1e-6 else given

    if extend[0] is not None:
        pre = _END_RUN_WIDTHS * max(float(width[0]), _MIN_CHANNEL_M)
        xy = np.vstack([xy[0] + heading(extend[0], xy[::-1], float(width[0])) * pre, xy])
        width, wet = np.r_[width[:1], width], np.r_[wet[:1], wet]
    if extend[1] is not None:
        post = _END_RUN_WIDTHS * max(float(width[-1]), _MIN_CHANNEL_M)
        xy = np.vstack([xy, xy[-1] + heading(extend[1], xy, float(width[-1])) * post])
        width, wet = np.r_[width, width[-1:]], np.r_[wet, wet[-1:]]
    # Densified, so the wander and the islands have vertices to bend.
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    n = np.maximum(np.ceil(seg / _DENSIFY_M).astype(int), 1)
    k = np.repeat(np.arange(len(seg)), n)
    u = np.concatenate([np.arange(m) / m for m in n])
    p = np.concatenate([xy[k] + u[:, None] * (xy[k + 1] - xy[k]), xy[-1:]])
    w = np.maximum(np.r_[width[k], width[-2:-1]], _MIN_CHANNEL_M)
    dry = ~np.r_[wet[k], wet[-2:-1]]
    # The course smoothed over about a width: a channel's bends are wide for its width
    # (offset from a tight bend, a wide river's inner bank balloons into a lobe).
    sigma_c = float(params.course_smoothing) * float(np.median(w)) / _DENSIFY_M
    if sigma_c > 0.5 and len(p) > 3:
        p = np.column_stack([gaussian_filter1d(p[:, j], sigma_c, mode="nearest") for j in range(2)])
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    # The line's own stretch (islands keep to it), less the runs on.
    s_lo = float(s[-1]) * pre / max(pre + length0 + post, 1e-9)
    s_hi = float(s[-1]) * (pre + length0) / max(pre + length0 + post, 1e-9)
    tangent = np.gradient(p, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])

    rng = _line_rng(line)
    sigma = float(params.wander_widths) * float(np.median(w)) / _DENSIFY_M
    half = 0.5 * w * np.exp(float(params.width_wander) * _smooth_noise(len(p), sigma, rng))
    shift = float(params.bank_wander) * w * _smooth_noise(len(p), sigma, rng)
    left = np.maximum(half + shift, 0.25 * w)
    right = np.maximum(half - shift, 0.25 * w)

    islands: list[Polygon] = []
    total = s_hi - s_lo
    count = rng.poisson(float(params.islands_per_km) * total / 1000.0) if total > 0 else 0
    taken: list[tuple[float, float]] = []
    for at in np.sort(rng.uniform(s_lo, s_hi, count)):
        i = int(np.searchsorted(s, at))
        i = min(i, len(p) - 1)
        if w[i] < float(params.island_min_width_m):
            continue
        breadth = float(params.island_width) * w[i] * rng.uniform(0.7, 1.3)
        length = float(params.island_length) * breadth * rng.uniform(0.7, 1.3)
        a, b = at - 0.5 * length, at + 0.5 * length
        # Clear of the line's ends (a mouth, a confluence, the frame) and of other islands, by half a width.
        lo, hi = a - 0.5 * w[i], b + 0.5 * w[i]
        if lo < s_lo or hi > s_hi or any(lo < t1 and hi > t0 for t0, t1 in taken):
            continue
        span = (s >= lo) & (s <= hi)
        if not dry[span].all() or (w[span] < float(params.island_min_width_m)).any():
            continue
        ends = np.flatnonzero(span)[[0, -1]]
        if float(np.dot(tangent[ends[0]], tangent[ends[1]])) < np.cos(np.radians(60.0)):
            continue              # not on a sharp bend
        # Clear of a channel joining (or leaving) partway along: half a width past both banks.
        stretch = LineString(p[span]) if span.sum() >= 2 else Point(p[i])
        if any(other.distance(stretch) < w[i] + 0.5 * ow for other, ow in avoid):
            continue
        taken.append((lo, hi))
        # The channel widens by a share of the island's breadth, tapered over twice its
        # length, so the banks swell gently rather than bulge.
        ta, tb = a - 0.5 * length, b + 0.5 * length
        widen = 0.5 * float(params.island_widening) * breadth * _lens((s - ta) / (tb - ta), 1.0)
        left, right = left + widen, right + widen
        on = (s >= a) & (s <= b)
        if on.sum() < 3:
            continue
        # Blunt at its head, tapering to a long tail downstream (lines run downstream),
        # on one line along the channel's middle there.
        u = (s[on] - a) / (b - a)
        shape = u ** 0.5 * (1.0 - u) ** 1.5
        thick = 0.5 * breadth * shape / max(float(shape.max()), 1e-12)
        centre = float(np.mean(0.5 * (left[on] - right[on])))
        mid = p[on] + normal[on] * centre
        ring = np.concatenate([mid + normal[on] * thick[:, None], (mid - normal[on] * thick[:, None])[::-1]])
        isle = shapely.make_valid(Polygon(ring))
        parts = [g for g in getattr(isle, "geoms", [isle]) if isinstance(g, Polygon) and not g.is_empty]
        if parts:
            islands.append(max(parts, key=lambda g: g.area))

    lb, rb = p + normal * left[:, None], p - normal * right[:, None]
    quads = shapely.make_valid(shapely.polygons(np.stack([lb[:-1], lb[1:], rb[1:], rb[:-1], lb[:-1]], axis=1)))
    return list(np.asarray(quads)), islands


_RunOn = Optional[np.ndarray]


def _runs_on(line: RiverLine, k: int, lines: Sequence[RiverLine], others: list[LineString],
             still: MultiPolygon, half: Optional[float]) -> tuple[_RunOn, _RunOn]:
    """How each end of ``line`` runs on: ``None`` (a source: it doesn't), a unit heading
    (into the still water within about a width: towards its middle there, which
    :func:`_line_channel` blends with the line's own heading), or a zero vector (on along the line: past the
    frame's edge, ``half``, or into another line's channel at a confluence)."""
    out: list[_RunOn] = []
    for end, w in ((line.xy[0], float(line.width_m[0])), (line.xy[-1], float(line.width_m[-1]))):
        reach = 1.5 * max(w, _MIN_CHANNEL_M) + 20.0
        pt = Point(end)
        if not still.is_empty and still.distance(pt) < reach:
            water = still.intersection(pt.buffer(2.0 * reach))
            to = np.array([water.centroid.x, water.centroid.y]) - end if not water.is_empty else np.zeros(2)
            norm = float(np.linalg.norm(to))
            out.append(to / norm if norm > 1e-6 else np.zeros(2))
        elif (half is not None and half - float(np.max(np.abs(end))) < reach) \
                or any(j != k and others[j].distance(pt) < reach + 0.5 * float(np.max(lines[j].width_m))
                       for j in range(len(lines))):
            out.append(np.zeros(2))
        else:
            out.append(None)
    return out[0], out[1]


def natural_channels(
    lines: Sequence[RiverLine], still: MultiPolygon, params: Optional[SettlementRiverParams] = None,
    half: Optional[float] = None,
) -> MultiPolygon:
    """The ground the channels cover, less the ``still`` water (sea and lakes) they run
    into and their islands.  Each line's channel is its bankfull width, wandering
    (``width_wander``, lognormal), its centre off the line (``bank_wander`` × width),
    both correlated over ``wander_widths`` widths; rivers at least
    ``island_min_width_m`` wide get ``islands_per_km`` islands (Poisson): long and
    narrow, ``island_width`` × the width broad and ``island_length`` × that long
    (±30%), blunt upstream and tapering downstream, along the channel's middle,
    clear of the line's ends, sharp bends and other lines (a width plus half theirs); the channel widens round each by
    ``island_widening`` × its breadth, tapered over twice its length.  Unclipped: as far as the lines reach.
    In-water stretches are widened too, so a channel meets its sea or lake whole, and
    land between the two narrower than ``min_land_m`` goes under the channel.  The
    course is the line smoothed over ``course_smoothing`` widths, and an end that
    meets water, another line or the frame's edge (``half``, the frame's half-side)
    runs on straight two widths (into still water, turned towards its middle), so the band
    is cut there, not stopped square."""
    params = params or SettlementRiverParams()
    quads, islands = [], []
    others = [LineString(line.xy) for line in lines]
    for k, line in enumerate(lines):
        avoid = [(others[j], float(np.max(lines[j].width_m))) for j in range(len(lines)) if j != k]
        q, i = _line_channel(line, params, _runs_on(line, k, lines, others, still, half), avoid)
        quads.extend(q)
        islands.extend(i)
    if not quads:
        return MultiPolygon()
    merged = shapely.union_all(quads)
    d = 0.5 * float(params.min_land_m)
    if d > 0 and not still.is_empty:
        # Closed against the still water, near the channels only.
        near = merged.buffer(4.0 * d)
        closed = shapely.union_all([merged, still.intersection(near)]).buffer(d).buffer(-d)
        merged = shapely.union_all([merged, closed.intersection(merged.buffer(2.0 * d))])
    # The quads' union can leave slivers unfilled on tight bends: only islands are holes.
    merged = _as_multipolygon(merged)
    merged = MultiPolygon([Polygon(g.exterior) for g in merged.geoms])
    cut = shapely.union_all([*islands, still]) if islands or not still.is_empty else None
    if cut is not None and not cut.is_empty:
        merged = merged.difference(cut)
    return _as_multipolygon(merged)


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
    channels = natural_channels(lines, still, params, frame.half)
    return SiteWater(
        sea=sea, lakes=lakes, shorelines=shorelines(still, frame, channels),
        lines=lines, channels=channels, rivers=rivers,
    )
