"""Local roads: :mod:`region.roads` one tier down.

Each regional road polyline is re-traced within ``corridor_parent_texels``
regional texels of it (:func:`region.roads.route_polyline`), with the world's
land-cost model per local raster node plus the local rivers' fords and
along-river discount.  River and lake legs are re-traced the same way over the
local water and rivers (:func:`region.roads._water_cost`), keeping their pinned
port ends.  Sea legs and undrawn routes pass through straight.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from core.params import param_field as _f, paramclass
from region.corridor import corridor_reach, node_spacing_rad, raster_frame, sample_raster
from region.display import snap_to_land
from region.rivers import RegionRivers
from region.roads import (
    RegionRoadParams,
    RegionRoads,
    RouteSegment,
    _river_nodes,
    _road_cost,
    _segments,
    _water_cost,
    _wet_nodes,
    move_points,
    route_polyline,
)
from region.upsample import EARTH_RADIUS_KM, RegionFields

if TYPE_CHECKING:
    from core.world import World
    from worldgen.cost.builder import LandCostParams


@paramclass
class LocalRoadParams:
    """Tuning for local road refinement."""

    # Raster spacing as a multiple of the local texel spacing.
    raster_texels: float = _f(2.0, label="Road raster spacing (texels)", min=1.0, max=6.0, step=0.5)
    margin: float = _f(0.03, label="Road raster margin", min=0.0, max=0.2, step=0.01)
    # Corridor half-width, in regional texels (the region's road raster is 2 texels).
    corridor_parent_texels: float = _f(
        2.0, label="Corridor half-width (regional texels)", min=0.25, max=6.0, step=0.25)
    smoothing_iterations: int = _f(6, label="Road smoothing iterations", min=0, max=20)
    # River and lake legs: corridor half-width in regional texels, and land-step cost (as the region's).
    water_corridor_parent_texels: float = _f(
        3.0, label="Water corridor half-width (regional texels)", min=0.25, max=8.0, step=0.25)
    water_land_cost: float = _f(
        RegionRoadParams().water_land_cost, label="Water route land-step cost", min=1.0, max=200.0, step=1.0)


def _near(
    segments: list[RouteSegment], local: RegionFields, lo: float, hi: float, samples: int = 9,
) -> list[RouteSegment]:
    """The straight ``segments`` that pass through the raster box ``[lo, hi]²``."""
    if not segments:
        return []
    frac = np.linspace(0.0, 1.0, samples)[None, :, None]
    a = np.array([np.asarray(sg[0], dtype=np.float64) for sg in segments])[:, None, :]
    b = np.array([np.asarray(sg[1], dtype=np.float64) for sg in segments])[:, None, :]
    pts = a + frac * (b - a)
    pts = pts / np.linalg.norm(pts, axis=2, keepdims=True)
    s, t, valid = local.geometry.ecef_to_local(pts.reshape(-1, 3))
    inside = (valid & (s >= lo) & (s <= hi) & (t >= lo) & (t <= hi)).reshape(len(segments), samples)
    hits = np.asarray(inside.any(axis=1), dtype=bool).tolist()
    return [sg for sg, hit in zip(segments, hits) if hit]


def build_local_roads(
    local: RegionFields,
    region: RegionFields,
    region_roads: RegionRoads,
    globe: "World",
    rivers: Optional[RegionRivers] = None,
    params: Optional[LocalRoadParams] = None,
    land: Optional["LandCostParams"] = None,
    danger_scale: float = 2.0,
    land_reach: Optional[float] = None,
    moved: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> RegionRoads:
    """Re-route ``region_roads``' land roads over the local map ``local`` of ``region``.

    ``land``, ``danger_scale`` and ``land_reach`` are as for
    :func:`region.roads.build_region_roads` (``land_reach`` defaults to six local
    texels); ``moved = (old, new)`` moves road ends at re-sited cities.
    """
    from worldgen.cost.builder import LandCostParams
    params = params or LocalRoadParams()
    land = land or LandCostParams()
    lo, step, n = raster_frame(local, params.raster_texels, params.margin)
    hi = lo + step * (n - 1)
    undrawn = _near(region_roads.undrawn, local, lo, hi)
    sea = _near(region_roads.sea, local, lo, hi)
    spacing = local.grid.texel_spacing_rad(local.tex_size)
    land_reach = 6.0 * spacing if land_reach is None else float(land_reach)

    node_rad = node_spacing_rad(local, step)
    radius_nodes = max(
        float(params.corridor_parent_texels) * region.grid.texel_spacing_rad(region.tex_size) / node_rad, 1.5
    )
    parent_rad = region.grid.texel_spacing_rad(region.tex_size)
    water_radius = max(float(params.water_corridor_parent_texels) * parent_rad / node_rad, 1.5)

    def nearby(polylines, flow_lists, extra):
        lines, kept = [], []
        for verts, flows, more in zip(polylines, flow_lists, extra):
            verts = np.asarray(verts, dtype=np.float64)
            s, t, valid = local.geometry.ecef_to_local(verts)
            near = valid & (s >= lo) & (s <= hi) & (t >= lo) & (t <= hi)
            if len(verts) >= 2 and near.any():
                lines.append((s, t))
                kept.append((verts, np.asarray(flows, dtype=np.float64), more))
        return lines, kept

    lines, polys = nearby(region_roads.polylines, region_roads.polyline_flows, [None] * len(region_roads.polylines))
    water_lines, waters = nearby(region_roads.water_polylines, region_roads.water_flows, region_roads.water_pins)
    if not polys and not waters:
        return RegionRoads(region_id=local.region_id, segments=list(sea), undrawn=undrawn, sea=sea)

    reach = corridor_reach(lines, lo, step, n, radius_nodes + 2.0)
    reach |= corridor_reach(water_lines, lo, step, n, water_radius + 2.0)
    raster = sample_raster(local, globe, lo, step, n, reach, keys=("biome", "forest_density", "danger"))
    river_flow, _ = _river_nodes(rivers, local, raster)
    face_km = float(np.sqrt(np.mean(globe.mesh.face_areas))) * EARTH_RADIUS_KM
    edge_cost = _road_cost(raster, river_flow, land, danger_scale, node_rad * EARTH_RADIUS_KM, face_km)

    segments: list[RouteSegment] = list(sea)
    out = RegionRoads(region_id=local.region_id, segments=segments, undrawn=undrawn, sea=sea)
    smoothing = int(params.smoothing_iterations)
    for verts, flows, _ in polys:
        # Ends are settlements or coast points: move them off the local water.
        verts = verts.copy()
        verts[[0, -1]] = snap_to_land(local, move_points(verts[[0, -1]], moved), land_reach)
        for path, edge_flows, _ in route_polyline(
            local, raster, verts, flows, radius_nodes, edge_cost, smoothing,
        ):
            out.polylines.append(path)
            out.polyline_flows.append(np.asarray(edge_flows, dtype=np.float64))
            segments.extend(_segments(path, edge_flows))
    if waters:
        wet = _wet_nodes(raster, river_flow)
        water_cost = _water_cost(wet, float(params.water_land_cost))
        for verts, flows, pin in waters:
            # Pinned ends are ports on land: follow a moved city, and keep off the local water.
            verts = verts.copy()
            for k, end in enumerate((0, -1)):
                if pin[k]:
                    verts[end] = snap_to_land(local, move_points(verts[[end]], moved), land_reach)[0]
            for path, edge_flows, pins in route_polyline(
                local, raster, verts, flows, water_radius, water_cost, smoothing, wet=wet, pin=pin,
            ):
                out.water_polylines.append(path)
                out.water_flows.append(np.asarray(edge_flows, dtype=np.float64))
                out.water_pins.append(pins)
                segments.extend(_segments(path, edge_flows))
    return out
