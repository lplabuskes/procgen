"""Local roads: the region's re-routed roads re-routed again over the local terrain.

The region turned the world's face-graph trade routes into roads over its
refined terrain (:mod:`region.roads`); a local map takes those road polylines
and re-traces each within ``corridor_parent_texels`` regional texels of it, with
the world's own land-cost model evaluated per local raster node
(:func:`region.roads.route_polyline`): biome, forest and danger per face-width,
climbing per km, and the local rivers' fords and along-river discount.  Ends at
a settlement land exactly on it.

Sea legs and the routes the world map does not draw are passed through, straight,
as the region has them (those that cross the local raster).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from core.params import param_field as _f, paramclass
from region.corridor import corridor_reach, node_spacing_rad, raster_frame, sample_raster
from region.display import snap_to_land
from region.rivers import RegionRivers
from region.roads import RegionRoads, RouteSegment, _river_nodes, _road_cost, route_polyline
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
    # Half-width of the corridor a regional road may move within, in regional
    # texels: the region routed it on a raster two of its texels apart.
    corridor_parent_texels: float = _f(
        2.0, label="Corridor half-width (regional texels)", min=0.25, max=6.0, step=0.25)
    smoothing_iterations: int = _f(6, label="Road smoothing iterations", min=0, max=20)


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
) -> RegionRoads:
    """Re-route ``region_roads``' land roads over the local map ``local`` of ``region``.

    ``land``, ``danger_scale`` and ``land_reach`` are as for
    :func:`region.roads.build_region_roads` (``land_reach`` defaulting to six
    local texels).
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

    # The region's polylines that come near the raster, with their local coords.
    node_rad = node_spacing_rad(local, step)
    radius_nodes = max(
        float(params.corridor_parent_texels) * region.grid.texel_spacing_rad(region.tex_size) / node_rad, 1.5
    )
    lines, polys = [], []
    for verts, flows in zip(region_roads.polylines, region_roads.polyline_flows):
        verts = np.asarray(verts, dtype=np.float64)
        s, t, valid = local.geometry.ecef_to_local(verts)
        near = valid & (s >= lo) & (s <= hi) & (t >= lo) & (t <= hi)
        if len(verts) >= 2 and near.any():
            lines.append((s, t))
            polys.append((verts, np.asarray(flows, dtype=np.float64)))
    if not polys:
        return RegionRoads(region_id=local.region_id, segments=list(sea), undrawn=undrawn, sea=sea)

    reach = corridor_reach(lines, lo, step, n, radius_nodes + 2.0)
    raster = sample_raster(local, globe, lo, step, n, reach, keys=("biome", "forest_density", "danger"))
    river_flow, _ = _river_nodes(rivers, local, raster)
    face_km = float(np.sqrt(np.mean(globe.mesh.face_areas))) * EARTH_RADIUS_KM
    edge_cost = _road_cost(raster, river_flow, land, danger_scale, node_rad * EARTH_RADIUS_KM, face_km)

    segments: list[RouteSegment] = list(sea)
    polylines: list[np.ndarray] = []
    polyline_flows: list[np.ndarray] = []
    for verts, flows in polys:
        # A road's ends are settlements (or the region's coast points): moved off
        # the local water, like the region moves them off its own.
        verts = verts.copy()
        verts[[0, -1]] = snap_to_land(local, verts[[0, -1]], land_reach)
        for path, edge_flows in route_polyline(
            local, raster, verts, flows, radius_nodes, edge_cost, int(params.smoothing_iterations),
        ):
            polylines.append(path)
            polyline_flows.append(np.asarray(edge_flows, dtype=np.float64))
            for k in range(len(path) - 1):
                segments.append((path[k].astype(np.float32), path[k + 1].astype(np.float32), float(edge_flows[k])))
    return RegionRoads(
        region_id=local.region_id, segments=segments, polylines=polylines, polyline_flows=polyline_flows,
        undrawn=undrawn, sea=sea,
    )
