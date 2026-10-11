"""A settlement site's roads, as vectors: re-traced roads, gates, landings and crossings.

- **Roads** are the local maps' (each clipped to the ground its map owns, and
  chained across a seam between maps) re-traced over the frame's relief within
  about one local texel (:func:`localmap.roads.build_local_roads` one tier down).
  A river here is many raster nodes wide, so it is a channel, not a line: entering
  it pays the world's ford cost, each step in it costs as much as climbing a
  ``channel_climb`` slope (roads cross the short way), and the world's
  along-river discount goes to a ``bank_m`` band beside it (roads keep to the banks).
- **Landings**: a sea leg's coast end is the region's (on its ~9 km raster), so
  the road that reached it lands where it passes nearest the traced shore in
  the frame (sea or lake), and is cut there; a leg with no road lands at the
  shore nearest its coast end, if that is near the frame.
- **Crossings**: every point a road crosses a dry river or stream channel's
  centreline, a bridge where the channel is at least ``bridge_width_m`` wide or
  the settlement is a city, else a ford.
- **Gates**: every point a road crosses the footprint's edge, facing out along
  the road (:func:`gates`).

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
from region.roads import RegionRoads, RouteSegment, _road_cost, _stretches, route_polyline
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
# How far (m) past a gate its outward side is tested.
_STEP_M = 0.5


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
    """Where a sea leg meets the shore."""

    xy: np.ndarray            # (2,)
    direction: np.ndarray     # (2,) unit, towards the leg's far end
    flow: float
    water: Literal["sea", "lake"]


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
    """The site's roads.  ``roads`` are the re-traced roads on the sphere, with the
    sea legs from their landings (for drawing in the world's form)."""

    lines: list[RoadLine]
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

    def reversed(self) -> "_Piece":
        return _Piece(self.pos[::-1].copy(), self.flows[::-1].copy(), self.cut_end, self.cut_start, self.map_index)


def _pieces(maps: Sequence[LocalMap], source: LocalMapsSource) -> list[_Piece]:
    out = []
    for i, m in enumerate(maps):
        for verts, flows in zip(m.roads.polylines, m.roads.polyline_flows):
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
    """The local maps' roads over the ground each owns, chained across seams, and their sea legs."""
    source = LocalMapsSource([m.fields for m in maps], antimeridian)
    f0 = maps[0].fields
    gap = _SEAM_GAP_NODES * float(road_raster_texels) * f0.grid.texel_spacing_rad(f0.tex_size)
    pieces = _chain(_pieces(maps, source), gap)
    sea = _sea_legs(maps)
    return RegionRoads(
        region_id=f0.region_id, segments=list(sea), sea=sea,
        polylines=[p.pos for p in pieces], polyline_flows=[p.flows for p in pieces],
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
            kind: Literal["sea", "lake"] = "sea" if sea_d <= lake_d else "lake"
            landings.append(Landing(
                xy=land_xy, direction=heading / norm if norm > 0 else np.zeros(2), flow=float(f), water=kind))
    return landings, legs, polylines


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
    """Every point a road crosses ``footprint``'s edge, facing out along the road;
    roads that share a way leave by one gate, flows summed."""
    if footprint.is_empty:
        return []
    edge = footprint.boundary
    out: list[Gate] = []
    for i, road in enumerate(lines):
        for p in _dedupe(_points(LineString(road.xy).intersection(edge))):
            k = _nearest_edge(road.xy, p[None])
            same = [g for g in out if np.linalg.norm(g.xy - p) < merge_m]
            if same:
                same[0].flow += float(road.flows[k])
                continue
            along = road.xy[k + 1] - road.xy[k]
            along = along / max(float(np.linalg.norm(along)), 1e-12)
            # Outwards: the way along the road that leaves the footprint.
            if footprint.contains(Point(p + _STEP_M * along)):
                along = -along
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
    channels (:func:`_channel_nodes`) rather than lines."""
    from worldgen.cost.builder import LandCostParams
    land = land or LandCostParams()
    lo, step, n = raster_frame(fields, params.raster_texels, params.margin)
    hi = lo + step * (n - 1)
    node_rad = node_spacing_rad(fields, step)
    node_m = node_rad * EARTH_RADIUS_KM * 1000.0
    # The local roads follow a river's centre line: the corridor reaches past its widest channel's bank.
    widest = max((float(line.width_m.max()) for line in water.lines if len(line.width_m)), default=0.0)
    radius_nodes = max(
        float(params.corridor_parent_texels) * parent.grid.texel_spacing_rad(parent.tex_size) / node_rad,
        (0.5 * widest + float(params.bank_m)) / node_m + 1.5, 1.5)
    lines, polys = [], []
    for verts, flows in zip(roads.polylines, roads.polyline_flows):
        verts = np.asarray(verts, dtype=np.float64)
        s, t, valid = fields.geometry.ecef_to_local(verts)
        if len(verts) >= 2 and (valid & (s >= lo) & (s <= hi) & (t >= lo) & (t <= hi)).any():
            lines.append((s, t))
            polys.append((verts, np.asarray(flows, dtype=np.float64)))
    if not polys:
        return RegionRoads(region_id=fields.region_id, segments=[])
    reach = corridor_reach(lines, lo, step, n, radius_nodes + 2.0)
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
    polylines, polyline_flows = [], []
    for verts, flows in polys:
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
                        out = np.array([on.x, on.y]) - xy[k]
                        out *= 1.0 + 0.5 * node_m / max(float(np.linalg.norm(out)), 1e-9)
                        pts[end] = frame.to_ecef(xy[k] + out)[0]
            for path, edge_flows in route_polyline(
                fields, raster, pts, st.flows, radius_nodes, edge_cost, int(params.smoothing_iterations),
            ):
                polylines.append(path)
                polyline_flows.append(np.asarray(edge_flows, dtype=np.float64))
    return RegionRoads(region_id=fields.region_id, segments=[], polylines=polylines, polyline_flows=polyline_flows)


def build_site_roads(
    globe: "World", fields: RegionFields, frame: Frame, maps: Sequence[LocalMap], water: SiteWater,
    footprint: MultiPolygon, params: Optional[SettlementRoadParams] = None, *, always_bridge: bool = False,
    local_road_raster_texels: float = 2.0, land: Optional["LandCostParams"] = None, danger_scale: float = 2.0,
) -> SiteRoads:
    """The site's roads.  See the module docstring.  ``local_road_raster_texels`` is the
    local maps' road raster spacing (how far apart their vertices are, for joining at seams)."""
    params = params or SettlementRoadParams()
    parent = parent_roads(maps, globe.antimeridian, local_road_raster_texels)
    texel_m = frame.side_m / fields.tex_size
    # Two things nearer than a road-raster node are one.
    merge_m = float(params.raster_texels) * texel_m
    landings, legs, polylines = _landings(parent, water, frame, float(params.landing_reach_km) * 1000.0, merge_m)
    parent.polylines = polylines
    routed = _route(fields, maps[0].fields, parent, globe, frame, water, params, land, danger_scale,
                    4.0 * fields.grid.texel_spacing_rad(fields.tex_size))
    lines = []
    for p, f in zip(routed.polylines, routed.polyline_flows):
        xy = frame.to_xy(p)
        # A stub from a city to itself (a route's two ends snapped together).
        if len(xy) >= 2 and float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) > texel_m:
            lines.append(RoadLine(xy=xy, flows=np.asarray(f, dtype=np.float64)))
    _land_at(lines, landings, merge_m)

    sea_ecef: list[RouteSegment] = [
        (frame.to_ecef(a)[0].astype(np.float32), frame.to_ecef(b)[0].astype(np.float32), f) for a, b, f in legs]
    segments: list[RouteSegment] = list(sea_ecef)
    on_sphere = []
    for line in lines:
        pos = frame.to_ecef(line.xy)
        on_sphere.append(pos)
        segments.extend((pos[k].astype(np.float32), pos[k + 1].astype(np.float32), float(line.flows[k]))
                        for k in range(len(pos) - 1))
    roads = RegionRoads(
        region_id=fields.region_id, segments=segments, polylines=on_sphere,
        polyline_flows=[line.flows for line in lines], sea=sea_ecef,
    )
    return SiteRoads(
        lines=lines, sea=legs, landings=landings,
        crossings=crossings(lines, water, float(params.bridge_width_m), always_bridge, merge_m),
        gates=gates(lines, footprint, merge_m), roads=roads,
    )
