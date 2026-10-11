"""A settlement site's roads, as vectors: re-traced roads, gates, landings and crossings.

- **Roads** are the local maps' (each clipped to the ground its map owns, and
  chained across a seam between maps) re-traced over the frame's relief within
  about one local texel (:func:`localmap.roads.build_local_roads` one tier down).
  A river here is many raster nodes wide, so it is a channel, not a line: entering
  it pays the world's ford cost, each step in it costs as much as climbing a
  ``channel_climb`` slope (roads cross the short way), and the world's
  along-river discount goes to a ``bank_m`` band beside it (roads keep to the banks).
- **River and lake legs** (the world's water trade off the open sea) are the
  local maps' too, chained the same way, re-routed over the site's still water
  and river channels (:func:`region.roads._water_cost`), each pinned to the land
  at its port as the tiers above pin it.
- **Landings**: where a leg pinned at a port first reaches water (a quay on its
  sea, lake or river).  A sea leg is straight from a coast end on the region's
  ~9 km raster, so the road that reached it lands where it passes nearest the
  traced shore in the frame, and is cut there; a sea leg with no road (its port
  is on the water) lands at the shore nearest its start, if near the frame.
- **Crossings**: every point a road crosses a dry river or stream channel's
  centreline, a bridge where the channel is at least ``bridge_width_m`` wide or
  the settlement is a city, else a ford.
- **Gates**: where a road leaves the footprint for good, facing out along the
  road (:func:`gates`).

Two routes re-traced onto one way (the local maps draw them overlapping too)
cross a river, or leave the footprint, once: their flows are summed.

Qt-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional, Sequence

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, box
from shapely.ops import nearest_points

from core.params import param_field as _f, paramclass
from localmap.builder import LocalMap
from localmap.roads import LocalRoadParams
from region.corridor import Raster, corridor_reach, node_spacing_rad, raster_frame, sample_raster
from region.display import snap_to_land
from region.roads import (
    RegionRoads, RouteSegment, _road_cost, _segments, _stretches, _water_cost, route_polyline,
)
from region.upsample import EARTH_RADIUS_KM, RegionFields
from settlegen.frame import Frame
from settlegen.source import LocalMapsSource
from settlegen.water import SiteWater, channel_polygons

if TYPE_CHECKING:
    from core.world import World
    from worldgen.cost.builder import LandCostParams

# A road cut at one map's edge continues on a neighbour's within this many local
# road-raster nodes (its vertices are that far apart before smoothing).
_SEAM_GAP_NODES = 2.5
# Points closer than this (m) along one road are one gate / crossing.
_SAME_POINT_M = 2.0


@paramclass
class SettlementRoadParams(LocalRoadParams):
    """:class:`LocalRoadParams` at settlement scale (raster in settlement texels, corridor in
    local texels), plus where sea legs land and which crossings are bridged."""

    # ~45 m, as the rivers'.
    raster_texels: float = _f(3.0, label="Road raster spacing (texels)", min=1.0, max=8.0, step=0.5)
    margin: float = _f(0.02, label="Road raster margin", min=0.0, max=0.2, step=0.01)
    corridor_parent_texels: float = _f(
        1.0, label="Corridor half-width (local texels)", min=0.25, max=3.0, step=0.25)
    smoothing_iterations: int = _f(8, label="Road smoothing iterations", min=0, max=20)
    # The region puts a sea leg's coast end on its ~9 km raster: the shore can be a way off.
    landing_reach_km: float = _f(2.0, label="Landing reach (km)", min=0.1, max=10.0, step=0.1)
    bridge_width_m: float = _f(20.0, label="Bridge from channel width (m)", min=0.0, max=500.0, step=1.0)
    # A river here is many raster nodes wide: roads cross it the short way and keep to its banks.
    # Each step along a channel costs as much as climbing this slope (km/km): the
    # ruggedness of the street-scale relief would otherwise send roads down the flat river bed.
    channel_climb: float = _f(1.0, label="Channel step cost (as slope)", min=0.0, max=5.0, step=0.1)
    bank_m: float = _f(60.0, label="River bank band (m)", min=0.0, max=500.0, step=5.0)
    water_corridor_parent_texels: float = _f(
        1.5, label="Water corridor half-width (local texels)", min=0.25, max=6.0, step=0.25)


@dataclass
class RoadLine:
    """One re-traced road on the site, in frame metres."""

    xy: np.ndarray            # (K, 2)
    flows: np.ndarray         # (K - 1,) trade flow per edge


@dataclass
class Gate:
    """Where a road crosses the footprint's edge."""

    xy: np.ndarray            # (2,)
    direction: np.ndarray     # (2,) unit, along the road, outwards
    flow: float
    road: int                 # index into ``SiteRoads.lines``


@dataclass
class Landing:
    """Where a water leg leaves the land: a quay."""

    xy: np.ndarray            # (2,)
    direction: np.ndarray     # (2,) unit, onto the water
    flow: float
    water: Literal["sea", "lake", "river"]


@dataclass
class Crossing:
    """Where a road crosses a river or stream."""

    xy: np.ndarray            # (2,)
    road: int                 # index into ``SiteRoads.lines``
    river: int                # index into ``SiteWater.lines``
    width_m: float            # the channel's bankfull width there
    flow: float               # the road's
    bridge: bool              # else a ford


@dataclass
class SiteRoads:
    """The site's roads.  ``roads`` are the re-traced roads and river and lake legs
    on the sphere, with the sea legs from their landings (for drawing in the world's form)."""

    lines: list[RoadLine]
    water: list[RoadLine]                               # river and lake legs
    sea: list[tuple[np.ndarray, np.ndarray, float]]     # (a xy, b xy, flow): a at the landing if it has one
    landings: list[Landing]
    crossings: list[Crossing]
    gates: list[Gate]
    roads: RegionRoads = field(repr=False)


# ── the local maps' roads, chained ───────────────────────────────────────────


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


# ── landings ─────────────────────────────────────────────────────────────────


def _split_at_coast(parent: RegionRoads, reach_m: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """``parent``'s polylines and flows, split at any inner vertex that is a sea
    leg's coast end (two port hops from one city share their coast point: the
    region chains them city → coast → city)."""
    polylines = [np.asarray(p, dtype=np.float64) for p in parent.polylines]
    flows = [np.asarray(f, dtype=np.float64) for f in parent.polyline_flows]
    for a, _, _ in parent.sea:
        a = np.asarray(a, dtype=np.float64)
        best, best_d = None, reach_m
        for i, p in enumerate(polylines):
            d = np.linalg.norm(p - a, axis=1) * EARTH_RADIUS_KM * 1000.0
            v = int(np.argmin(d))
            if d[v] < best_d:
                best, best_d = (i, v), float(d[v])
        if best is None or best[1] in (0, len(polylines[best[0]]) - 1):
            continue
        i, v = best
        p, f = polylines.pop(i), flows.pop(i)
        polylines += [p[:v + 1], p[v:]]
        flows += [f[:v], f[v:]]
    # The two halves of such a loop are one road.
    keep: list[int] = []
    for i, p in enumerate(polylines):
        twin = next((j for j in keep if len(polylines[j]) == len(p) and (
            np.allclose(polylines[j], p) or np.allclose(polylines[j], p[::-1]))), None)
        if twin is None:
            keep.append(i)
        else:
            flows[twin] = flows[twin] + (flows[i] if np.allclose(polylines[twin], p) else flows[i][::-1])
    return [polylines[i] for i in keep], [flows[i] for i in keep]


def _landings(
    parent: RegionRoads, water: SiteWater, frame: Frame, reach_m: float, merge_m: float,
) -> tuple[list[Landing], list[tuple[np.ndarray, np.ndarray, float]], list[np.ndarray]]:
    """``(landings, sea legs in frame metres, parent polylines)``.

    A sea leg's coast end is the region's (its raster is ~9 km): the road that
    reached it (the local map snapped it onto its land; the road end nearest it,
    within ``reach_m``) lands where it passes nearest the shore in the frame, and
    is cut there.  A leg with no road lands at the shore nearest its coast end.
    """
    polylines, flows = _split_at_coast(parent, reach_m)
    parent.polyline_flows = flows
    # Matched before any road is cut at a landing: several legs can share a road's coast end.
    ends = np.array([[p[0], p[-1]] for p in polylines]).reshape(-1, 2, 3)
    legs: list[tuple[np.ndarray, np.ndarray, float]] = []
    landings: list[Landing] = []
    shore = water.shorelines
    frame_box = frame.box()
    landed: dict[int, np.ndarray] = {}            # polyline → its landing, once cut
    for a, b, f in parent.sea:
        a = np.asarray(a, dtype=np.float64)
        a_xy, b_xy = frame.to_xy(np.stack([a, np.asarray(b, dtype=np.float64)]))
        if shore.is_empty:
            legs.append((a_xy, b_xy, f))
            continue
        road, at_end = None, True
        if len(ends):
            d = np.linalg.norm(ends - a, axis=2) * EARTH_RADIUS_KM * 1000.0
            k = int(np.argmin(d))
            if d.flat[k] <= reach_m:
                road, at_end = divmod(k, 2)[0], bool(divmod(k, 2)[1])
        if road is not None and road in landed:
            land_xy = landed[road]
        else:
            if road is not None:
                xy = frame.to_xy(polylines[road])
                inside = LineString(xy).intersection(frame_box)
                if inside.is_empty:
                    legs.append((a_xy, b_xy, f))
                    continue
                on, near = nearest_points(shore, inside)
            else:
                if not frame_box.buffer(reach_m).contains(Point(a_xy)):
                    legs.append((a_xy, b_xy, f))
                    continue
                on, near = nearest_points(shore, Point(a_xy))
            land_xy = np.array([on.x, on.y])
            if on.distance(near) > reach_m:
                legs.append((a_xy, b_xy, f))
                continue
            if road is not None:
                # Cut the road at its vertex nearest the landing, then on to it.
                pos = polylines[road] if at_end else polylines[road][::-1]
                k = int(np.argmin(np.linalg.norm(frame.to_xy(pos) - land_xy, axis=1)))
                pos = np.concatenate([pos[:max(k, 1)], frame.to_ecef(land_xy)])
                polylines[road] = pos if at_end else pos[::-1].copy()
                flows = np.asarray(parent.polyline_flows[road], dtype=np.float64)
                n = len(pos) - 1
                parent.polyline_flows[road] = flows[:n] if at_end else flows[::-1][:n][::-1].copy()
                landed[road] = land_xy
        legs.append((land_xy, b_xy, f))
        for landing in landings:
            if np.linalg.norm(landing.xy - land_xy) < merge_m:
                landing.flow += float(f)
                break
        else:
            heading = b_xy - land_xy
            norm = float(np.linalg.norm(heading))
            on_pt = Point(land_xy)
            sea_d = water.sea.distance(on_pt) if not water.sea.is_empty else np.inf
            lake_d = water.lakes.distance(on_pt) if not water.lakes.is_empty else np.inf
            kind: Literal["sea", "lake", "river"] = "sea" if sea_d <= lake_d else "lake"
            landings.append(Landing(
                xy=land_xy, direction=heading / norm if norm > 0 else np.zeros(2), flow=float(f), water=kind))
    return landings, legs, polylines


def _quays(
    legs: list[RoadLine], pins: list[tuple[bool, bool]], water: SiteWater, channels: MultiPolygon, frame: Frame,
) -> list[Landing]:
    """Where each river or lake leg pinned at a port in the frame first reaches
    water (the nearest bank, for a port on the water)."""
    still = water.sea.union(water.lakes)
    wet = still.union(channels)
    frame_box = frame.box()
    out = []
    for leg, pin in zip(legs, pins):
        for pinned, xy, flows in ((pin[0], leg.xy, leg.flows), (pin[1], leg.xy[::-1], leg.flows[::-1])):
            if not pinned or not frame_box.contains(Point(xy[0])):
                continue
            if wet.contains(Point(xy[0])):
                # The port is drawn on water here (the finer channel covers the
                # local map's bank): its quay is the nearest bank.
                bank = nearest_points(wet.boundary, Point(xy[0]))[0]
                at, heading = np.array([bank.x, bank.y]), xy[0] - np.array([bank.x, bank.y])
                if float(np.linalg.norm(heading)) < 1e-9:
                    heading = xy[1] - xy[0]
                hit = 1
            else:
                hit = next((k for k in range(1, len(xy)) if wet.contains(Point(xy[k]))), None)
                if hit is None:
                    continue
                prev, first_wet = xy[hit - 1], xy[hit]
                entry = _points(LineString([prev, first_wet]).intersection(wet.boundary))
                at = min(entry, key=lambda p: float(np.linalg.norm(p - prev))) if entry else first_wet
                heading = first_wet - prev
            heading = heading / max(float(np.linalg.norm(heading)), 1e-12)
            on = Point(at + heading)
            kind: Literal["sea", "lake", "river"] = (
                "sea" if water.sea.contains(on) else "lake" if water.lakes.contains(on) else "river")
            out.append(Landing(xy=np.asarray(at, dtype=np.float64), direction=heading, flow=float(flows[hit - 1]),
                               water=kind))
    return out


def _merge_landings(landings: list[Landing], merge_m: float) -> list[Landing]:
    out: list[Landing] = []
    for landing in landings:
        same = next((q for q in out if np.linalg.norm(q.xy - landing.xy) < merge_m), None)
        if same is None:
            out.append(landing)
        else:
            same.flow += landing.flow
    return out


# ── crossings and gates ──────────────────────────────────────────────────────


def _nearest_edge(xy: np.ndarray, p: np.ndarray) -> int:
    """Index of the edge of polyline ``xy`` nearest point ``p``."""
    a, b = xy[:-1], xy[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18), 0.0, 1.0)
    return int(np.argmin(np.linalg.norm(a + t[:, None] * ab - p, axis=1)))


def _points(geom) -> list[np.ndarray]:  # type: ignore[no-untyped-def]
    """The points of an intersection (a touching stretch counts by its ends)."""
    if geom.is_empty:
        return []
    if isinstance(geom, Point):
        return [np.array([geom.x, geom.y])]
    if isinstance(geom, LineString):
        c = np.asarray(geom.coords)
        return [c[0], c[-1]]
    return [q for g in getattr(geom, "geoms", []) for q in _points(g)]


def _dedupe(points: list[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for p in points:
        if all(np.linalg.norm(p - q) >= _SAME_POINT_M for q in out):
            out.append(p)
    return out


def crossings(
    lines: list[RoadLine], water: SiteWater, bridge_width_m: float, always_bridge: bool, merge_m: float,
) -> list[Crossing]:
    """Every point a road crosses a dry river or stream centreline; roads that
    share a way (two routes re-traced onto one path) cross once, flows summed."""
    out: list[Crossing] = []
    for j, river in enumerate(water.lines):
        dry = ~river.in_water
        if not dry.any():
            continue
        segs = shapely.linestrings(np.stack([river.xy[:-1][dry], river.xy[1:][dry]], axis=1))
        rline = shapely.union_all(segs)
        for i, road in enumerate(lines):
            for p in _dedupe(_points(LineString(road.xy).intersection(rline))):
                flow = float(road.flows[_nearest_edge(road.xy, p[None])])
                same = [c for c in out if c.river == j and np.linalg.norm(c.xy - p) < merge_m]
                if same:
                    same[0].flow += flow
                    continue
                width = float(river.width_m[_nearest_edge(river.xy, p[None])])
                out.append(Crossing(
                    xy=p, road=i, river=j, width_m=width, flow=flow,
                    bridge=bool(always_bridge or width >= bridge_width_m),
                ))
    return out


def gates(lines: list[RoadLine], footprint: MultiPolygon, merge_m: float) -> list[Gate]:
    """Where each road leaves ``footprint`` for good, facing out along the road: the
    inner end of each stretch outside it that runs on to the road's end (a stretch
    between two parts of the footprint — a bridge, a bay — is no gate).  Roads that
    share a way leave by one gate, flows summed."""
    if footprint.is_empty:
        return []
    edge = footprint.boundary
    out: list[Gate] = []
    for i, road in enumerate(lines):
        ends = (road.xy[0], road.xy[-1])
        outside = LineString(road.xy).difference(footprint)
        for part in getattr(outside, "geoms", [outside]):
            if part.is_empty or not isinstance(part, LineString):
                continue
            c = np.asarray(part.coords)
            first_on, last_on = (edge.distance(Point(c[k])) < 1e-6 for k in (0, -1))
            leads_out = [any(np.linalg.norm(c[k] - e) < 1e-6 for e in ends) for k in (0, -1)]
            if first_on and leads_out[1] and not last_on:
                p, ahead = c[0], c[1]
            elif last_on and leads_out[0] and not first_on:
                p, ahead = c[-1], c[-2]
            else:
                continue
            k = _nearest_edge(road.xy, p[None])
            same = [g for g in out if np.linalg.norm(g.xy - p) < merge_m]
            if same:
                same[0].flow += float(road.flows[k])
                continue
            along = (ahead - p) / max(float(np.linalg.norm(ahead - p)), 1e-12)
            out.append(Gate(xy=p, direction=along, flow=float(road.flows[k]), road=i))
    return out


def _land_at(lines: list[RoadLine], landings: list[Landing], tol_m: float) -> None:
    """End each road that stops within ``tol_m`` of a landing exactly on it
    (re-tracing snaps a shore point a texel onto land)."""
    for landing in landings:
        for road in lines:
            for end in (0, -1):
                if float(np.linalg.norm(road.xy[end] - landing.xy)) < tol_m:
                    road.xy[end] = landing.xy


def _channel_nodes(
    raster: Raster, frame: Frame, water: SiteWater, bank_m: float, node_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``(channel flow, bank flow)`` per raster node: the flow of the nearest dry
    channel segment on nodes in its channel (at least ~a node wide, so no road
    slips through between nodes) or in the ``bank_m`` band beside it; 0 elsewhere."""
    size = raster.n ** 2
    channel, bank = np.zeros(size), np.zeros(size)
    segs, flows, half = [], [], []
    for line in water.lines:
        dry = ~line.in_water
        segs.append(np.stack([line.xy[:-1][dry], line.xy[1:][dry]], axis=1))
        flows.append(line.flows[dry])
        half.append(0.5 * line.width_m[dry])
    if not segs or not sum(len(x) for x in flows):
        return channel, bank
    seg = np.concatenate(segs)
    flow, half_w = np.concatenate(flows), np.maximum(np.concatenate(half), 0.75 * node_m)
    nodes = np.flatnonzero(raster.sampled)
    xy = frame.to_xy(raster.positions[nodes])
    tree = shapely.STRtree(np.asarray(shapely.linestrings(seg)))
    near_node, near_seg = tree.query(shapely.points(xy), predicate="dwithin", distance=float(half_w.max() + bank_m))
    if near_node.size == 0:
        return channel, bank
    a, b = seg[near_seg, 0], seg[near_seg, 1]
    ab = b - a
    p = xy[near_node]
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18), 0.0, 1.0)
    edge = np.linalg.norm(a + t[:, None] * ab - p, axis=1) - half_w[near_seg]   # < 0 inside the channel
    # Each node's nearest channel edge decides.
    order = np.lexsort((edge, near_node))
    first = np.ones(order.size, dtype=bool)
    first[1:] = near_node[order][1:] != near_node[order][:-1]
    pick = order[first]
    node, d, f = nodes[near_node[pick]], edge[pick], flow[near_seg[pick]]
    inside = d <= 0.0
    beside = ~inside & (d <= bank_m)
    channel[node[inside]] = np.maximum(f[inside], 1e-9)
    bank[node[beside]] = np.maximum(f[beside], 1e-9)
    return channel, bank


def _route(
    fields: RegionFields, parent: RegionFields, roads: RegionRoads, globe: "World", frame: Frame, water: SiteWater,
    params: SettlementRoadParams, land: Optional["LandCostParams"], danger_scale: float, land_reach: float,
) -> RegionRoads:
    """:func:`localmap.roads.build_local_roads` one tier down, with the rivers as
    channels (:func:`_channel_nodes`) rather than lines: roads keep out of them,
    river and lake legs sail them."""
    from worldgen.cost.builder import LandCostParams
    land = land or LandCostParams()
    lo, step, n = raster_frame(fields, params.raster_texels, params.margin)
    hi = lo + step * (n - 1)
    node_rad = node_spacing_rad(fields, step)
    node_m = node_rad * EARTH_RADIUS_KM * 1000.0
    # The local roads follow a river's centre line: the corridor reaches past its widest channel's bank.
    widest = max((float(line.width_m.max()) for line in water.lines if len(line.width_m)), default=0.0)
    parent_rad = parent.grid.texel_spacing_rad(parent.tex_size)
    clear = (0.5 * widest + float(params.bank_m)) / node_m + 1.5
    radius_nodes = max(float(params.corridor_parent_texels) * parent_rad / node_rad, clear, 1.5)
    water_radius = max(float(params.water_corridor_parent_texels) * parent_rad / node_rad, clear, 1.5)

    def nearby(polylines, flow_lists, extra):  # type: ignore[no-untyped-def]
        lines, kept = [], []
        for verts, flows, more in zip(polylines, flow_lists, extra):
            verts = np.asarray(verts, dtype=np.float64)
            s, t, valid = fields.geometry.ecef_to_local(verts)
            if len(verts) >= 2 and (valid & (s >= lo) & (s <= hi) & (t >= lo) & (t <= hi)).any():
                lines.append((s, t))
                kept.append((verts, np.asarray(flows, dtype=np.float64), more))
        return lines, kept

    lines, polys = nearby(roads.polylines, roads.polyline_flows, [None] * len(roads.polylines))
    water_lines, waters = nearby(roads.water_polylines, roads.water_flows, roads.water_pins)
    out = RegionRoads(region_id=fields.region_id, segments=[])
    if not polys and not waters:
        return out
    reach = corridor_reach(lines, lo, step, n, radius_nodes + 2.0)
    reach |= corridor_reach(water_lines, lo, step, n, water_radius + 2.0)
    raster = sample_raster(fields, globe, lo, step, n, reach, keys=("biome", "forest_density", "danger"))
    channel, bank = _channel_nodes(raster, frame, water, float(params.bank_m), node_m)
    face_km = float(np.sqrt(np.mean(globe.mesh.face_areas))) * EARTH_RADIUS_KM
    edge_cost = _road_cost(raster, channel, land, danger_scale, node_rad * EARTH_RADIUS_KM, face_km,
                           bank_flow=bank, channel_climb=float(params.channel_climb))
    # Ends inside a channel go to its nearest bank in the raster: a road that started
    # and ended in one would stay in it rather than pay to ford it again.
    inner = box(*((np.array([lo, lo, hi, hi]) - 0.5) * frame.side_m)).buffer(-node_m)
    channels = channel_polygons(water.lines)          # past the frame, as far as the raster
    banks = channels.boundary.intersection(inner) if not channels.is_empty else None
    smoothing = int(params.smoothing_iterations)
    for verts, flows, _ in polys:
        verts = verts.copy()
        verts[[0, -1]] = snap_to_land(fields, verts[[0, -1]], land_reach)
        s, t, _ = fields.geometry.ecef_to_local(verts)
        for st in _stretches(s, t, verts, flows, lo, hi):
            # Cut ends at the raster edge become real ends, so they can be moved.
            pts = np.where(st.real[:, None], st.ecef, fields.geometry.local_to_ecef(st.s, st.t))
            if banks is not None and not banks.is_empty:
                xy = frame.to_xy(pts[[0, -1]])
                for k, end in ((0, 0), (1, -1)):
                    if channels.contains(Point(xy[k])):
                        on = nearest_points(banks, Point(xy[k]))[0]
                        shift = np.array([on.x, on.y]) - xy[k]
                        shift *= 1.0 + 0.5 * node_m / max(float(np.linalg.norm(shift)), 1e-9)
                        pts[end] = frame.to_ecef(xy[k] + shift)[0]
            for path, edge_flows, _ in route_polyline(
                fields, raster, pts, st.flows, radius_nodes, edge_cost, smoothing,
            ):
                out.polylines.append(path)
                out.polyline_flows.append(np.asarray(edge_flows, dtype=np.float64))
    if waters:
        wet = raster.sampled & (raster.water | (channel > 0.0))
        water_cost = _water_cost(wet, float(params.water_land_cost))
        for verts, flows, pin in waters:
            # Pinned ends are ports on land: keep them off the site's water.
            verts = verts.copy()
            for k, end in enumerate((0, -1)):
                if pin[k]:
                    verts[end] = snap_to_land(fields, verts[[end]], land_reach)[0]
            for path, edge_flows, pins in route_polyline(
                fields, raster, verts, flows, water_radius, water_cost, smoothing, wet=wet, pin=pin,
            ):
                out.water_polylines.append(path)
                out.water_flows.append(np.asarray(edge_flows, dtype=np.float64))
                out.water_pins.append(pins)
    return out


def build_site_roads(
    globe: "World", fields: RegionFields, frame: Frame, maps: Sequence[LocalMap], water: SiteWater,
    params: Optional[SettlementRoadParams] = None, *, always_bridge: bool = False,
    local_road_raster_texels: float = 2.0, land: Optional["LandCostParams"] = None, danger_scale: float = 2.0,
) -> SiteRoads:
    """The site's roads.  See the module docstring.  ``local_road_raster_texels`` is the
    local maps' road raster spacing (how far apart their vertices are, for joining at seams).
    No gates yet: the footprint follows the roads (:func:`set_gates`)."""
    params = params or SettlementRoadParams()
    parent = parent_roads(maps, globe.antimeridian, local_road_raster_texels)
    texel_m = frame.side_m / fields.tex_size
    merge_m = merge_distance_m(params, frame, fields)
    landings, legs, polylines = _landings(parent, water, frame, float(params.landing_reach_km) * 1000.0, merge_m)
    parent.polylines = polylines
    routed = _route(fields, maps[0].fields, parent, globe, frame, water, params, land, danger_scale,
                    4.0 * fields.grid.texel_spacing_rad(fields.tex_size))

    def as_lines(polylines: list[np.ndarray], flow_lists: list[np.ndarray]) -> list[RoadLine]:
        out = []
        for p, f in zip(polylines, flow_lists):
            xy = frame.to_xy(p)
            # A stub from a city to itself (a route's two ends snapped together).
            if len(xy) >= 2 and float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) > texel_m:
                out.append(RoadLine(xy=xy, flows=np.asarray(f, dtype=np.float64)))
        return out

    lines = as_lines(routed.polylines, routed.polyline_flows)
    _land_at(lines, landings, merge_m)
    kept = [k for k, p in enumerate(routed.water_polylines) if len(as_lines([p], [routed.water_flows[k]]))]
    water_legs = [as_lines([routed.water_polylines[k]], [routed.water_flows[k]])[0] for k in kept]
    quays = _quays(water_legs, [routed.water_pins[k] for k in kept], water, channel_polygons(water.lines), frame)
    landings = _merge_landings(landings + quays, merge_m)

    sea_ecef: list[RouteSegment] = [
        (frame.to_ecef(a)[0].astype(np.float32), frame.to_ecef(b)[0].astype(np.float32), f) for a, b, f in legs]
    roads = RegionRoads(region_id=fields.region_id, segments=list(sea_ecef), sea=sea_ecef)
    for line in lines:
        pos = frame.to_ecef(line.xy)
        roads.polylines.append(pos)
        roads.polyline_flows.append(line.flows)
        roads.segments.extend(_segments(pos, line.flows))
    for k, line in zip(kept, water_legs):
        pos = frame.to_ecef(line.xy)
        roads.water_polylines.append(pos)
        roads.water_flows.append(line.flows)
        roads.water_pins.append(routed.water_pins[k])
        roads.segments.extend(_segments(pos, line.flows))
    return SiteRoads(
        lines=lines, water=water_legs, sea=legs, landings=landings,
        crossings=crossings(lines, water, float(params.bridge_width_m), always_bridge, merge_m),
        gates=[], roads=roads,
    )


def merge_distance_m(params: SettlementRoadParams, frame: Frame, fields: RegionFields) -> float:
    """Two things nearer than a road-raster node are one."""
    return float(params.raster_texels) * frame.side_m / fields.tex_size


def set_gates(
    roads: SiteRoads, footprint: MultiPolygon, params: SettlementRoadParams, frame: Frame, fields: RegionFields,
) -> None:
    """``roads.gates`` on ``footprint`` (:func:`gates`)."""
    roads.gates = gates(roads.lines, footprint, merge_distance_m(params, frame, fields))
