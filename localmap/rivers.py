"""Local rivers: :mod:`region.rivers` one tier down.

Each regional river path is re-traced within about one regional texel of its
line over the local relief (:func:`region.rivers.route_stream_paths`), largest
flow first; one ending inside the map ends on what it drains into (stem, lake or
coast).  Flow is the region's, which is the world's.

The region's own streams (:mod:`region.streams`, below the world's
``stream_flow_threshold``) are re-traced after the rivers and go to
:attr:`RegionRivers.streams`: villages site on them, but towns, cities and
deltas keep to the world's channels.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.spatial import KDTree

from core.params import param_field as _f, paramclass
from region.corridor import node_spacing_rad, raster_frame
from region.rivers import RegionRiverParams, RegionRivers, RiverPath, StreamPath, route_stream_paths
from region.upsample import RegionFields

if TYPE_CHECKING:
    from core.world import World

# Chord under which two path vertices are one point (~0.6 m; region vertices are float32).
_SAME_POINT = 1e-7


@paramclass
class LocalRiverParams:
    """Tuning for local river refinement."""

    # Raster spacing as a multiple of the local texel spacing.
    raster_texels: float = _f(1.0, label="River raster spacing (texels)", min=0.5, max=4.0, step=0.25)
    # How far past the local square the raster reaches, in local units (the map is 1).
    margin: float = _f(0.03, label="River raster margin", min=0.0, max=0.2, step=0.01)
    # Half-width of the corridor a regional river may move within, in regional texels.
    corridor_parent_texels: float = _f(
        1.0, label="Corridor half-width (regional texels)", min=0.25, max=3.0, step=0.25)
    valley_seeking: float = _f(2.0, label="Valley seeking", min=0.0, max=6.0, step=0.25)
    smoothing_iterations: int = _f(8, label="Path smoothing iterations", min=0, max=20)


def _runs(inside: np.ndarray) -> list[tuple[int, int]]:
    """``(first, last)`` index of every run of True in ``inside``."""
    idx = np.flatnonzero(inside)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def _starts_on_path(rivers: list[RiverPath]) -> np.ndarray:
    """Per path, whether its first vertex is a vertex of another path (a delta branch leaving its parent)."""
    out = np.zeros(len(rivers), dtype=bool)
    owner = np.concatenate([np.full(len(p[0]), k) for k, p in enumerate(rivers)] or [np.zeros(0, np.int64)])
    if owner.size == 0:
        return out
    tree = KDTree(np.concatenate([np.asarray(p[0], dtype=np.float64).reshape(-1, 3) for p in rivers]))
    for k, (positions, _, _) in enumerate(rivers):
        if len(positions):
            near = tree.query_ball_point(np.asarray(positions[0], dtype=np.float64), _SAME_POINT)
            out[k] = bool(np.any(owner[near] != k))
    return out


def _region_paths(
    local: RegionFields, rivers: list[RiverPath], lo: float, hi: float, minor: bool = False,
    river_ids: Optional[list[np.ndarray]] = None,
) -> list[StreamPath]:
    """The region's river paths inside the local raster, largest flow first.

    One run per visit to the raster; a run ``ends_here`` (on a sink or stem, not
    the raster edge) only when it holds the region path's last vertex.  Each
    keeps to land where the region's path is drawn, and a run holding the first
    vertex of a path that starts on another (a delta distributary) starts on its
    re-route.
    """
    out: list[StreamPath] = []
    starts = _starts_on_path(rivers)
    for k, (positions, flows, in_water) in enumerate(rivers):
        pts = np.asarray(positions, dtype=np.float64)
        f = np.asarray(flows, dtype=np.float64)
        if len(pts) < 2:
            continue
        s, t, valid = local.geometry.ecef_to_local(pts)
        inside = valid & (s >= lo) & (s <= hi) & (t >= lo) & (t <= hi)
        for i, j in _runs(inside):
            if j <= i:
                continue
            out.append(StreamPath(
                s=s[i:j + 1], t=t[i:j + 1], edge_flows=np.maximum.accumulate(f[i:j]),
                ends_here=j == len(pts) - 1, minor=minor,
                hidden_edges=np.asarray(in_water, dtype=bool)[i:j],
                starts_on_path=i == 0 and bool(starts[k]),
                river_ids=None if river_ids is None else np.asarray(river_ids[k])[i:j + 1],
            ))
    out.sort(key=lambda p: -float(p.edge_flows.max()))
    return out


def build_local_rivers(
    local: RegionFields,
    region: RegionFields,
    region_rivers: RegionRivers,
    globe: "World",
    params: Optional[LocalRiverParams] = None,
    region_streams: Optional[list[RiverPath]] = None,
) -> RegionRivers:
    """Re-route ``region_rivers``, then ``region_streams`` (:func:`region.streams.build_region_streams`;
    none when omitted), over the local map ``local`` of ``region``."""
    params = params or LocalRiverParams()
    lo, step, n = raster_frame(local, params.raster_texels, params.margin)
    hi = lo + step * (n - 1)
    parent_spacing = region.grid.texel_spacing_rad(region.tex_size)
    radius_nodes = max(
        float(params.corridor_parent_texels) * parent_spacing / node_spacing_rad(local, step), 1.5
    )
    paths = (_region_paths(local, region_rivers.paths, lo, hi, river_ids=region_rivers.river_ids())
             + _region_paths(local, region_streams or [], lo, hi, minor=True))
    if not paths:
        return RegionRivers(region_id=local.region_id, paths=[])
    routing = RegionRiverParams(
        valley_seeking=float(params.valley_seeking), smoothing_iterations=int(params.smoothing_iterations),
    )
    return route_stream_paths(local, globe, paths, lo, step, n, radius_nodes, routing)
