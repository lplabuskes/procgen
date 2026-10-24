"""A settlement site's roads, as vectors: re-traced roads, gates, landings and crossings.

- **Roads** are the local maps' (:mod:`settlegen.site.road_chain`: each clipped to the ground its map owns, and
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
- **Landings** (:mod:`settlegen.site.landings`): where a leg pinned at a port first reaches water (a quay on its
  sea, lake or river).  A sea leg is straight from a coast end on the region's
  ~9 km raster, so the road that reached it lands where it passes nearest the
  traced shore in the frame, and is cut there; a sea leg with no road (its port
  is on the water) lands at the shore nearest its start, if near the frame.
- **Crossings** (:mod:`settlegen.site.crossings`): every point a road crosses a dry river or stream channel's
  centreline, a bridge where the channel is at least ``bridge_width_m`` wide or
  the settlement is a city, else a ford.
- **Gates**: where a road leaves the footprint for good, facing out along the
  road (:func:`settlegen.site.crossings.gates`).

Two routes re-traced onto one way (the local maps draw them overlapping too)
cross a river, or leave the footprint, once: their flows are summed.

Qt-free.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points

from localmap.builder import LocalMap
from region.corridor import Raster, corridor_reach, node_spacing_rad, raster_frame, sample_raster
from region.display import snap_to_land
from region.roads import RegionRoads, RouteSegment, _road_cost, _segments, _stretches, _water_cost, route_polyline
from region.upsample import RegionFields
from settlegen.frame import Frame
from settlegen.site.crossings import crossings, gates
from settlegen.site.landings import _land_at, _landings, _merge_landings, _quays
from settlegen.site.params import SettlementRoadParams
from settlegen.site.road_chain import parent_roads
from settlegen.site.types import RoadLine, SiteRoads, SiteWater
from utils.sphere import EARTH_RADIUS_KM

if TYPE_CHECKING:
    from core.world import World
    from worldgen.cost.builder import LandCostParams


def _channel_nodes(
    raster: Raster, frame: Frame, water: SiteWater, bank_m: float, node_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``(channel flow, bank flow)`` per raster node: the flow of the nearest dry
    channel segment on nodes in its channel (the drawn channel, and at least ~a
    node wide about the line, so no road slips through between nodes) or in the
    ``bank_m`` band beside it; 0 elsewhere."""
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
    # The drawn channel wanders and widens round its islands, up to about this past the line's width.
    slack = float(half_w.max())
    near_node, near_seg = tree.query(shapely.points(xy), predicate="dwithin",
                                     distance=float(2.0 * half_w.max() + bank_m + slack))
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
    if not water.channels.is_empty:
        # The channel as drawn: in it is in it, its bank is beside it.
        pts = shapely.points(xy[near_node[pick]])
        drawn = np.asarray(shapely.distance(water.channels, pts))
        d = np.minimum(d, np.where(shapely.contains(water.channels, pts), -1.0, drawn))
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
    channels = water.channels                  # past the frame, as far as the lines
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


def _route_direction(globe: "World", frame: Frame) -> Optional[tuple[np.ndarray, float]]:
    """``(unit xy, flow)``: the way of the world's land trade-route hop nearest the
    settlement, and the least flow a road is drawn at (the median hop's); None without routes."""
    from realmgen.trade.network import ROUTE_LAND
    routes = globe.get_feature("trade_routes") or []
    if not routes:
        return None
    a = np.array([np.asarray(r[0], dtype=np.float64) for r in routes])
    b = np.array([np.asarray(r[1], dtype=np.float64) for r in routes])
    flow = float(np.median([float(r[2]) for r in routes]))
    layers = globe.get_feature("trade_route_layers")
    if layers is not None and len(layers) == len(routes) and (np.asarray(layers) == ROUTE_LAND).any():
        land = np.asarray(layers) == ROUTE_LAND
        a, b = a[land], b[land]
    c = frame.to_ecef(np.asarray(frame.settlement_xy, dtype=np.float64)[None])[0]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", c - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18), 0.0, 1.0)
    k = int(np.argmin(np.linalg.norm(a + t[:, None] * ab - c, axis=1)))
    d = ab[k] - float(ab[k] @ c) * c                 # the hop's way in the tangent plane here
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return None
    eps = 0.1 * frame.side_m / (EARTH_RADIUS_KM * 1000.0)
    u = frame.to_xy((c + eps * d / n)[None])[0] - frame.to_xy(c[None])[0]
    return u / max(float(np.linalg.norm(u)), 1e-12), flow


def through_road(
    globe: "World", frame: Frame, water: SiteWater, step_m: float,
) -> tuple[Optional[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """``(way, halves)``: a straight road through the settlement the way (unit xy) of
    the nearest land trade route (:func:`_route_direction`), from frame edge to frame
    edge, cut where it meets the sea or a lake, as ``(ecef (K, 3), flows (K - 1,))`` per
    half.  Each half runs out from the settlement, so the re-traced road keeps to the
    settlement rather than wandering its corridor.  ``(None, [])`` if there is none."""
    way = _route_direction(globe, frame)
    if way is None:
        return None, []
    u, flow = way
    centre = Point(*frame.settlement_xy)
    line: BaseGeometry = LineString([frame.settlement_xy - frame.side_m * u, frame.settlement_xy + frame.side_m * u])
    line = line.intersection(frame.box())
    still = shapely.union_all([water.sea, water.lakes])
    if not still.is_empty:
        line = line.difference(still)
    parts = [g for g in shapely.get_parts(line) if isinstance(g, LineString) and g.length > step_m]
    if not parts:
        return None, []
    piece = min(parts, key=lambda g: g.distance(centre))
    at = piece.project(centre)
    out = []
    for end in (0.0, piece.length):
        if abs(end - at) <= step_m:
            continue
        d = np.linspace(at, end, max(int(np.ceil(abs(end - at) / step_m)), 1) + 1)
        xy = np.asarray([(p.x, p.y) for p in shapely.line_interpolate_point(piece, d)])
        out.append((frame.to_ecef(xy), np.full(len(xy) - 1, flow)))
    return (u if out else None), out


def build_site_roads(
    globe: "World", fields: RegionFields, frame: Frame, maps: Sequence[LocalMap], water: SiteWater,
    params: Optional[SettlementRoadParams] = None, *, always_bridge: bool = False,
    local_road_raster_texels: float = 2.0, land: Optional["LandCostParams"] = None, danger_scale: float = 2.0,
    through_within_m: Optional[float] = None,
) -> SiteRoads:
    """The site's roads.  See the module docstring.  ``local_road_raster_texels`` is the
    local maps' road raster spacing (how far apart their vertices are, for joining at seams).
    ``through_within_m``: if no road comes this near the settlement, add :func:`through_road`.
    No gates yet: the footprint follows the roads (:func:`set_gates`)."""
    params = params or SettlementRoadParams()
    parent = parent_roads(maps, globe.antimeridian, local_road_raster_texels)
    texel_m = frame.side_m / fields.tex_size
    merge_m = merge_distance_m(params, frame, fields)
    landings, legs, polylines = _landings(parent, water, frame, float(params.landing_reach_km) * 1000.0, merge_m)
    parent.polylines = polylines
    through: Optional[np.ndarray] = None
    if through_within_m is not None and not any(
            len(p) >= 2 and LineString(frame.to_xy(p)).distance(Point(*frame.settlement_xy)) <= through_within_m
            for p in parent.polylines):
        through, halves = through_road(globe, frame, water, 2.0 * merge_m)
        for pos, flows in halves:
            parent.polylines.append(pos)
            parent.polyline_flows.append(flows)
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
    quays = _quays(water_legs, [routed.water_pins[k] for k in kept], water, water.channels, frame)
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
        gates=[], roads=roads, through=through,
    )


def merge_distance_m(params: SettlementRoadParams, frame: Frame, fields: RegionFields) -> float:
    """Two things nearer than a road-raster node are one."""
    return float(params.raster_texels) * frame.side_m / fields.tex_size


def set_gates(
    roads: SiteRoads, footprint: MultiPolygon, params: SettlementRoadParams, frame: Frame, fields: RegionFields,
) -> None:
    """``roads.gates`` on ``footprint`` (:func:`settlegen.site.crossings.gates`)."""
    roads.gates = gates(roads.lines, footprint, merge_distance_m(params, frame, fields))
