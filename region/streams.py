"""Regional streams — the region's own drainage network below the world's.

The world's network stops at ``stream_flow_threshold`` (~20,000 km² catchment);
local maps want the brooks below it.  Once per region, on the rivers' raster:

- Priority-flood from the outlets (refined sea and the globe's lakes, the
  re-routed rivers, the raster edge); each node drains to the neighbour it was
  flooded from, so streams run through small lakes (``in_lake`` there).
- Each node contributes catchment × precipitation / world mean, in world flow
  units (one global texel at mean rain).  Flow is capped at the world's stream
  threshold: anything larger the world routed elsewhere.
- Nodes above ``min_catchment_km2`` are channels, cut into main stems with
  :func:`worldgen.hydrology.rivers.main_stem_paths`.

Streams end on the region's rivers, so never cross one, and every local map of
a region re-routes the same network (:mod:`localmap.rivers`).  The region view
doesn't draw them; nothing is conserved (the world has no channels this small).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from core.params import param_field as _f, paramclass
from region.corridor import centreline, node_spacing_rad, raster_frame, sample_raster
from region.drainage import accumulate, priority_flood
from region.rivers import RegionRivers, RiverPath
from region.upsample import EARTH_RADIUS_KM, RegionFields
from worldgen.hydrology.rivers import main_stem_paths

if TYPE_CHECKING:
    from core.world import World


@paramclass
class RegionStreamParams:
    """Tuning for the regional stream network the local maps draw."""

    # Smallest channel catchment, km² (raster nodes are ~4.5 km apart).
    min_catchment_km2: float = _f(300.0, label="Min catchment (km²)", min=20.0, max=5000.0, step=10.0)
    # Stems shorter than this many raster nodes are dropped as stubs.
    min_nodes: int = _f(3, label="Min stream length (texels)", min=2, max=20)
    # Raster spacing in regional texels, and margin past the cell; as the rivers'.
    raster_texels: float = _f(1.0, label="Stream raster spacing (texels)", min=0.5, max=4.0, step=0.25)
    margin: float = _f(0.08, label="Stream raster margin", min=0.0, max=0.3, step=0.01)


def _river_nodes(fields: RegionFields, rivers: RegionRivers, lo: float, step: float, n: int) -> np.ndarray:
    """``(n²,)`` mask of the raster nodes the region's rivers run through."""
    hit = np.zeros((n, n), dtype=bool)
    hi = lo + step * (n - 1)
    for positions, _, _ in rivers.paths:
        s, t, valid = fields.geometry.ecef_to_local(np.asarray(positions, dtype=np.float64))
        inside = valid & (s >= lo) & (s <= hi) & (t >= lo) & (t <= hi)
        # One run per visit, so a path leaving and re-entering is not bridged.
        idx = np.flatnonzero(inside)
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if run.size:
                rows, cols = centreline(lo, step, n, s[run], t[run])
                hit[rows, cols] = True
    return hit.ravel()


def _flow_unit_km2(globe: "World") -> float:
    """Catchment of one world flow unit at mean rain, km²: one global texel."""
    mesh = globe.mesh
    gw = mesh.face_textures["elevation"].shape[2]
    spacing = float(np.sqrt(np.mean(mesh.face_areas))) / max(gw - 1, 1)
    return (spacing * EARTH_RADIUS_KM) ** 2


def build_region_streams(
    fields: RegionFields,
    globe: "World",
    rivers: RegionRivers,
    params: Optional[RegionStreamParams] = None,
) -> list[RiverPath]:
    """The region's streams below the world's network, ending on ``rivers``.

    ``(positions, flows, in_lake)`` polylines, headwater → outlet, largest flow
    first.  Empty without a ``"streams"`` feature or precipitation.
    """
    params = params or RegionStreamParams()
    streams = globe.get_feature("streams")
    mesh = globe.mesh
    if streams is None or len(streams) == 0 or "precipitation" not in mesh.face_textures:
        return []
    cap = float(streams.flow[streams.channel].min()) if streams.channel.any() else float("inf")

    lo, step, n = raster_frame(fields, params.raster_texels, params.margin)
    raster = sample_raster(fields, globe, lo, step, n, np.ones(n * n, dtype=bool), keys=("precipitation",))
    edge = np.zeros((n, n), dtype=bool)
    edge[[0, -1], :] = edge[:, [0, -1]] = True
    edge = edge.ravel()
    # Small lakes aren't outlets: streams run through them (hidden there).
    outlet = raster.sinks | edge | _river_nodes(fields, rivers, lo, step, n)

    parent, order, _ = priority_flood(raster.values["elevation"].astype(np.float64), outlet, n)
    unit = _flow_unit_km2(globe)
    node_km2 = (node_spacing_rad(fields, step) * EARTH_RADIUS_KM) ** 2
    mean_rain = max(float(mesh.face_textures["precipitation"].sum(axis=-1).mean()), 1e-6)
    weight = np.where(outlet, 0.0, raster.values["precipitation"] / mean_rain * node_km2)
    catchment = accumulate(parent, order, weight)        # km² at mean rain

    channel = (catchment >= float(params.min_catchment_km2)) & ~outlet
    flow = np.minimum(catchment / unit, cap)
    out: list[RiverPath] = []
    for path in main_stem_paths(parent, catchment, channel, outlet, int(params.min_nodes)):
        nodes = np.asarray(path, dtype=np.int64)
        if edge[nodes[-1]] and not raster.sinks[nodes[-1]]:
            # Drains off the raster: cut there, it ends beyond.
            nodes = nodes[:-1]
            if len(nodes) < 2:
                continue
        flows = np.maximum.accumulate(flow[nodes[:-1]]).astype(np.float32)
        in_lake = raster.small_lake[nodes[:-1]] & raster.small_lake[nodes[1:]]
        out.append((raster.positions[nodes].astype(np.float32), flows, in_lake))
    out.sort(key=lambda p: -float(p[1].max()))
    return out
