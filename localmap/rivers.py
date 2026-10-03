"""Local rivers: the region's re-routed rivers re-routed again over the local relief.

A region re-traces the world's stream network within about one global texel of
its world centreline (:mod:`region.rivers`); a local map does the same one tier
down: each regional river path is re-traced within about one *regional* texel of
the region's line, over the local relief, with the same valley-seeking cost and
the same routing loop (:func:`region.rivers.route_stream_paths`).  Paths are
routed largest flow first, and one that ends inside the map ends on whatever it
drains into near its regional end — the stem it joins, a lake, or the local
coast.  Flow is the region's, which is the world's.

**Then the region's streams.**  The world's network stops at its
``stream_flow_threshold``, coarse for villages; the region adds its own streams
below that (:mod:`region.streams`), and a local map re-traces them the same way,
after the rivers, so each ends on the river or stream it drains into.  They go
to :attr:`RegionRivers.streams`: drawn and sited on by villages, but towns,
cities and deltas keep to the world's channels.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from core.params import param_field as _f, paramclass
from region.corridor import node_spacing_rad, raster_frame
from region.rivers import RegionRiverParams, RegionRivers, RiverPath, StreamPath, route_stream_paths
from region.upsample import RegionFields

if TYPE_CHECKING:
    from core.world import World


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


def _region_paths(
    local: RegionFields, rivers: list[RiverPath], lo: float, hi: float, minor: bool = False,
) -> list[StreamPath]:
    """The region's river paths inside the local raster, largest flow first.

    A path leaving and re-entering the raster becomes one run per visit; a run
    ends *here* (on a sink or a stem, not at the raster edge) only when it holds
    the region path's last vertex.
    """
    out: list[StreamPath] = []
    for positions, flows, _ in rivers:
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
    paths = (_region_paths(local, region_rivers.paths, lo, hi)
             + _region_paths(local, region_streams or [], lo, hi, minor=True))
    if not paths:
        return RegionRivers(region_id=local.region_id, paths=[])
    routing = RegionRiverParams(
        valley_seeking=float(params.valley_seeking), smoothing_iterations=int(params.smoothing_iterations),
    )
    return route_stream_paths(local, globe, paths, lo, step, n, radius_nodes, routing)
