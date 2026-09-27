"""Local rivers: the region's re-routed rivers re-routed again over the local relief.

A region re-traces the world's stream network within about one global texel of
its world centreline (:mod:`region.rivers`); a local map does the same one tier
down: each regional river path is re-traced within about one *regional* texel of
the region's line, over the local relief, with the same valley-seeking cost and
the same routing loop (:func:`region.rivers.route_stream_paths`).  Paths are
routed largest flow first, and one that ends inside the map ends on whatever it
drains into near its regional end — the stem it joins, a lake, or the local
coast.  Flow is the region's, which is the world's.

**Only the world's channels exist here.**  The region never invents a channel,
and neither does a local map: every stream is one of the hydrology stage's
(down to its ``stream_flow_threshold``), which is coarse for villages to sit on.
:func:`_region_paths` is where a finer local network would join them — see the
note there.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from core.params import param_field as _f, paramclass
from region.corridor import node_spacing_rad, raster_frame
from region.rivers import RegionRiverParams, RegionRivers, StreamPath, route_stream_paths
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
    local: RegionFields, rivers: RegionRivers, lo: float, hi: float,
) -> list[StreamPath]:
    """The region's river paths inside the local raster, largest flow first.

    A path leaving and re-entering the raster becomes one run per visit; a run
    ends *here* (on a sink or a stem, not at the raster edge) only when it holds
    the region path's last vertex.

    **A finer local stream network would join here.**  Flow accumulation over the
    local relief (D8 on the local elevation, seeded with the regional rivers'
    flow where they enter) would add channels below the world's
    ``stream_flow_threshold`` as extra :class:`StreamPath`\\ s — the tributaries
    villages sit on.  They would need no corridor (they are local facts), so
    their paths would be the accumulation's own, routed after these.
    """
    out: list[StreamPath] = []
    for positions, flows, _ in rivers.paths:
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
                ends_here=j == len(pts) - 1,
            ))
    out.sort(key=lambda p: -float(p.edge_flows.max()))
    return out


def build_local_rivers(
    local: RegionFields,
    region: RegionFields,
    region_rivers: RegionRivers,
    globe: "World",
    params: Optional[LocalRiverParams] = None,
) -> RegionRivers:
    """Re-route ``region_rivers`` over the local map ``local`` of ``region``."""
    params = params or LocalRiverParams()
    lo, step, n = raster_frame(local, params.raster_texels, params.margin)
    hi = lo + step * (n - 1)
    parent_spacing = region.grid.texel_spacing_rad(region.tex_size)
    radius_nodes = max(
        float(params.corridor_parent_texels) * parent_spacing / node_spacing_rad(local, step), 1.5
    )
    paths = _region_paths(local, region_rivers, lo, hi)
    if not paths:
        return RegionRivers(region_id=local.region_id, paths=[])
    routing = RegionRiverParams(
        valley_seeking=float(params.valley_seeking), smoothing_iterations=int(params.smoothing_iterations),
    )
    return route_stream_paths(local, globe, paths, lo, step, n, radius_nodes, routing)
