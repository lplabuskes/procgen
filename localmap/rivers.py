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


def _clipped_runs(
    s: np.ndarray, t: np.ndarray, valid: np.ndarray, lo: float, hi: float,
) -> list[tuple[int, int, float, float]]:
    """Each stretch of the polyline ``(s, t)`` inside the box ``[lo, hi]²``, clipped at its edge.

    ``(i, j, u0, u1)``: segments ``i … j - 1``, the first entered at fraction ``u0``
    of its length and the last left at ``u1`` (1 and 0 when a vertex is inside),
    so a stretch reaches the edge however far apart the vertices are, and one
    crossing a corner with no vertex inside is kept.  Segments with an end not
    ``valid`` are skipped.
    """
    out: list[tuple[int, int, float, float]] = []
    run: Optional[list] = None
    for k in range(len(s) - 1):
        u0, u1 = 0.0, 1.0
        if valid[k] and valid[k + 1]:
            # Liang–Barsky against the box.
            for p0, d in ((s[k], s[k + 1] - s[k]), (t[k], t[k + 1] - t[k])):
                for q, r in ((p0 - lo, -d), (hi - p0, d)):
                    if r == 0.0:
                        if q < 0.0:
                            u0, u1 = 1.0, 0.0
                    elif r < 0.0:
                        u0 = max(u0, q / r)
                    else:
                        u1 = min(u1, q / r)
        else:
            u0, u1 = 1.0, 0.0
        if u1 <= u0:
            if run is not None:
                out.append((run[0], k, run[1], run[2]))
                run = None
            continue
        if run is None or u0 > 0.0:
            if run is not None:
                out.append((run[0], k, run[1], run[2]))
            run = [k, u0, u1]
        else:
            run[2] = u1
        if u1 < 1.0:
            out.append((run[0], k + 1, run[1], run[2]))
            run = None
    if run is not None:
        out.append((run[0], len(s) - 1, run[1], run[2]))
    return out


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


def _tributary_ends(rivers: list[RiverPath]) -> list[np.ndarray]:
    """Per path, the indices of its vertices another path ends on (where a tributary joins)."""
    ends = [np.asarray(p[0][-1], dtype=np.float64) for p in rivers if len(p[0])]
    out: list[np.ndarray] = []
    for positions, _, _ in rivers:
        pts = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        if not ends or len(pts) < 2:
            out.append(np.zeros(0, dtype=np.int64))
            continue
        near = KDTree(pts[:-1]).query_ball_point(np.stack(ends), _SAME_POINT)
        out.append(np.unique(np.concatenate([np.asarray(k, dtype=np.int64) for k in near])))
    return out


def _region_paths(
    local: RegionFields, rivers: list[RiverPath], lo: float, hi: float, minor: bool = False,
    river_ids: Optional[list[np.ndarray]] = None,
) -> list[StreamPath]:
    """The region's river paths inside the local raster, largest flow first.

    One run per visit to the raster, clipped at its edge (:func:`_clipped_runs`); a run
    ``ends_here`` (on a sink or stem, not the raster edge) only when it holds the
    region path's last vertex.  Each
    keeps to land where the region's path is drawn, and a run holding the first
    vertex of a path that starts on another (a delta distributary) starts on its
    re-route.  A ``minor`` (stream) run is routed through only the vertices its
    tributaries end on: its flow is catchment, rising at every vertex, so the
    flow-jump test would pin it to the region's every vertex (cusps between).
    """
    out: list[StreamPath] = []
    starts = _starts_on_path(rivers)
    joins = _tributary_ends(rivers) if minor else None
    for k, (positions, flows, in_water) in enumerate(rivers):
        pts = np.asarray(positions, dtype=np.float64)
        f = np.asarray(flows, dtype=np.float64)
        if len(pts) < 2:
            continue
        s, t, valid = local.geometry.ecef_to_local(pts)
        for i, j, u0, u1 in _clipped_runs(s, t, valid, lo, hi):
            rs, rt = s[i:j + 1].copy(), t[i:j + 1].copy()
            rs[0], rt[0] = s[i] + u0 * (s[i + 1] - s[i]), t[i] + u0 * (t[i + 1] - t[i])
            rs[-1], rt[-1] = s[j - 1] + u1 * (s[j] - s[j - 1]), t[j - 1] + u1 * (t[j] - t[j - 1])
            out.append(StreamPath(
                s=rs, t=rt, edge_flows=np.maximum.accumulate(f[i:j]),
                ends_here=j == len(pts) - 1 and u1 == 1.0, minor=minor,
                hidden_edges=np.asarray(in_water, dtype=bool)[i:j],
                starts_on_path=i == 0 and u0 == 0.0 and bool(starts[k]),
                river_ids=None if river_ids is None else np.asarray(river_ids[k])[i:j + 1],
                through=None if joins is None else [int(v) - i for v in joins[k] if i < v < j],
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
