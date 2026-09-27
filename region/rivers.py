"""Regional rivers — the world's stream network re-routed over the refined relief.

The hydrology stage saves its drainage tree down to a *stream* threshold far
below the rivers' (:class:`worldgen.hydrology.types.StreamNetwork`): which texels
carry a channel, where each drains, and how much water.  Those are facts at the
global resolution, and the town stage already sites towns on them.  A region keeps
all of it and refines only what its finer grid can see — *where exactly* each
channel runs:

- The network inside the region is cut into main stems with the global stage's
  own :func:`worldgen.hydrology.rivers.main_stem_paths`, so a region's rivers are
  the world's rivers extended by their sub-threshold tributaries, with the same
  confluences.
- Each path is re-traced through a *corridor* about one global texel either side
  of its world centreline — the uncertainty the global grid leaves — as the
  least-cost route through the refined relief, where cost rises with the sub-grid
  elevation anomaly.  It follows the refined valley floor but can never change
  valley or basin.  Stems are routed before their tributaries, and a path whose
  world mouth is here ends on whatever it drains into near that mouth: its
  re-routed stem, a lake, or the *refined* coast.  Flow is the network's own.

Everything happens on one uniform raster over the region's local ``(s, t)``
frame, ``margin`` past the cell, at the regional texel spacing — but only the
nodes some corridor can reach are ever sampled (refined fields from the nearest
hex texel, global fields beyond the textures).  Paths leaving the raster are cut
at its edge; the view clips to the cell long before that.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.spatial import KDTree

from core.params import param_field as _f, paramclass
from region.corridor import (
    EdgeCost,
    Raster,
    corridor_reach,
    node_spacing_rad,
    raster_frame,
    route_corridor,
    sample_raster,
)
from region.upsample import EARTH_RADIUS_KM, RegionFields
from worldgen.hydrology.rivers import main_stem_paths
from worldgen.hydrology.types import StreamNetwork, laplacian_smooth_sphere

if TYPE_CHECKING:
    from core.world import World

# The (positions, flows, in_lake) polyline form of the world's "rivers" feature.
RiverPath = tuple[np.ndarray, np.ndarray, np.ndarray]

# A river crossing a lake passes over sub-grid islands and spits in the refined
# shoreline; land runs this short between water along a path count as water, so
# the crossing is hidden whole rather than drawn as dashes.
_SPECKLE_NODES = 2


@paramclass
class RegionRiverParams:
    """Tuning for regional river refinement."""

    # Raster spacing as a multiple of the regional texel spacing.  1 matches the
    # refined relief; coarser is faster but blurs the valleys it follows.
    raster_texels: float = _f(1.0, label="River raster spacing (texels)", min=0.5, max=4.0, step=0.25)
    # How far past the cell edge the raster reaches, in local units (the cell is 1).
    margin: float = _f(0.08, label="River raster margin", min=0.0, max=0.3, step=0.01)
    # Half-width of the corridor a global river may move within, in global texels.
    corridor_global_texels: float = _f(1.0, label="Corridor half-width (global texels)", min=0.25, max=3.0, step=0.25)
    # How strongly a re-routed river prefers low ground: the path cost of a
    # texel is exp(valley_seeking · anomaly / σ), σ the anomaly spread in the corridor.
    valley_seeking: float = _f(2.0, label="Valley seeking", min=0.0, max=6.0, step=0.25)
    # Stream-network stems shorter than this many global texels are dropped as
    # stubs.  (How far down the network goes is the hydrology stage's stream
    # threshold — a world param, since towns are sited on the same streams.)
    min_global_nodes: int = _f(3, label="Min stream length (global texels)", min=2, max=20)
    smoothing_iterations: int = _f(8, label="Path smoothing iterations", min=0, max=20)


@dataclass
class RegionRivers:
    """Regional river polylines, in the world ``"rivers"`` feature's form."""

    region_id: int
    # The stream network's stems re-routed over the refined relief: the world's
    # rivers (extended upstream by their largest streams) and the streams below
    # the river threshold alike, told apart only by their flow.
    paths: list[RiverPath]
    # (K, 3) where a path ends on another one — the confluences town siting rewards.
    confluences: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # (J, 3) where a path ends on the refined sea, and the flow it arrives with.
    mouths: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    mouth_flows: np.ndarray = field(default_factory=lambda: np.zeros(0))


# ── re-routing ───────────────────────────────────────────────────────────────


def _valley_cost(raster: Raster, valley_seeking: float) -> EdgeCost:
    """Edge cost that seeks the refined valley floor.

    Node cost is exponential in the sub-grid anomaly, normalised by its spread in
    the corridor, so a river seeks the valley on a plain as much as in the
    mountains.
    """
    cache: dict = {}

    def edge_cost(sub: np.ndarray, a: np.ndarray, b: np.ndarray, length: np.ndarray) -> np.ndarray:
        if cache.get("sub") is not sub:
            anomaly = raster.anomaly[sub]
            spread = max(float(anomaly.std()), 1e-6)
            cache["sub"] = sub
            cache["cost"] = np.exp(np.clip(valley_seeking * (anomaly - anomaly.mean()) / spread, -20.0, 20.0))
        cost = cache["cost"]
        return length * 0.5 * (cost[a] + cost[b])

    return edge_cost


def _route_run(
    raster: Raster,
    s: np.ndarray,
    t: np.ndarray,
    radius_nodes: float,
    sinks: np.ndarray,
    ends_here: bool,
    params: RegionRiverParams,
) -> Optional[np.ndarray]:
    """Least-cost node path along one in-raster stretch of a stream path.

    The path is confined to nodes within ``radius_nodes`` of the world polyline.
    It starts at the stretch's first vertex and ends at its last — or, when the
    stream's world mouth is in this stretch (``ends_here``), at the cheapest
    ``sinks`` node near that mouth (its re-routed stem, a lake, the refined sea).
    The stem this river joins was re-routed within its own corridor, so it can
    sit up to a corridor width off the world confluence: the search widens to
    twice the corridor around the mouth.
    """
    return route_corridor(
        raster, s, t, radius_nodes, _valley_cost(raster, float(params.valley_seeking)),
        sinks=sinks if ends_here else None, mouth_radius=2.0 * radius_nodes,
    )


def _run_flows(
    raster: Raster, nodes: np.ndarray, s: np.ndarray, t: np.ndarray, flows: np.ndarray
) -> np.ndarray:
    """Per-edge flow for a re-routed path: the world flow at the nearest world vertex."""
    x = (s - raster.lo) / raster.step
    y = (t - raster.lo) / raster.step
    rows, cols = np.divmod(nodes, raster.n)
    mids_r = 0.5 * (rows[:-1] + rows[1:])
    mids_c = 0.5 * (cols[:-1] + cols[1:])
    nearest = np.argmin((mids_c[:, None] - x[None, :]) ** 2 + (mids_r[:, None] - y[None, :]) ** 2, axis=1)
    edge_flow = flows[np.minimum(nearest, len(flows) - 1)]
    return np.maximum.accumulate(edge_flow).astype(np.float32)


@dataclass
class _StreamPath:
    """One main stem of the network inside the raster, in local coordinates."""

    s: np.ndarray
    t: np.ndarray
    edge_flows: np.ndarray       # (len - 1,) world flow, monotonic
    ends_here: bool              # ends on a sink or a stem, not at the raster edge


def _stream_paths(
    streams: StreamNetwork, fields: RegionFields, lo: float, step: float, n: int, min_nodes: int,
) -> list[_StreamPath]:
    """The network's main stems inside the raster, largest flow first."""
    s, t, valid = fields.geometry.ecef_to_local(np.asarray(streams.positions, dtype=np.float64))
    hi = lo + step * (n - 1)
    inside = valid & (s >= lo) & (s <= hi) & (t >= lo) & (t <= hi)
    idx = np.flatnonzero(inside)
    local = np.full(len(streams), -1, dtype=np.int64)
    local[idx] = np.arange(idx.size)
    downstream = streams.receiver[idx]
    receiver = np.where(downstream >= 0, local[np.maximum(downstream, 0)], -1)
    receiver[downstream < 0] = -1
    # Where the tree carries on beyond the raster, a path is cut, not ended.
    leaves_raster = (downstream >= 0) & (receiver < 0)

    flow = streams.flow[idx].astype(np.float64)
    out = []
    for path in main_stem_paths(receiver, flow, streams.channel[idx], streams.stop[idx], min_nodes):
        nodes = np.asarray(path, dtype=np.int64)
        running = np.maximum.accumulate(flow[nodes])
        out.append(_StreamPath(
            s=s[idx][nodes], t=t[idx][nodes], edge_flows=running[:-1],
            ends_here=not (receiver[nodes[-1]] < 0 and leaves_raster[nodes[-1]]),
        ))
    out.sort(key=lambda p: -float(p.edge_flows.max()))
    return out


# ── entry point ──────────────────────────────────────────────────────────────


def _fill_short_gaps(flags: np.ndarray, max_gap: int) -> np.ndarray:
    """``flags`` with runs of up to ``max_gap`` False entries between two True ones set."""
    out = flags.copy()
    true_at = np.flatnonzero(flags)
    for a, b in zip(true_at[:-1], true_at[1:]):
        if 1 < b - a <= max_gap + 1:
            out[a + 1:b] = True
    return out


def _to_path(raster: Raster, nodes: np.ndarray, flows: np.ndarray, params: RegionRiverParams) -> RiverPath:
    pts = [raster.positions[k] for k in nodes]
    smooth = laplacian_smooth_sphere(pts, iterations=int(params.smoothing_iterations), alpha=0.5)
    water = _fill_short_gaps(raster.ocean[nodes] | raster.lake[nodes], _SPECKLE_NODES)
    # Hidden where both ends are under water, so a river still reaches its coast.
    in_water = water[:-1] & water[1:]
    return np.stack(smooth).astype(np.float32), np.asarray(flows, dtype=np.float32), in_water


def build_region_rivers(
    fields: RegionFields,
    globe: "World",
    params: Optional[RegionRiverParams] = None,
) -> RegionRivers:
    """Re-route the world's stream network through ``fields``' refined relief.

    Needs the hydrology stage's ``"streams"`` feature; a world without one gets
    no rivers.  Deterministic, like the refinement itself.
    """
    params = params or RegionRiverParams()
    streams = globe.get_feature("streams")
    if streams is None or len(streams) == 0:
        return RegionRivers(region_id=fields.region_id, paths=[])
    lo, step, n = raster_frame(fields, params.raster_texels, params.margin)

    # Corridor half-width in raster nodes: global texel spacing over node spacing.
    mesh = globe.mesh
    gw = mesh.face_textures["elevation"].shape[2]
    global_spacing = float(np.sqrt(np.mean(mesh.face_areas))) / max(gw - 1, 1)
    radius_nodes = max(float(params.corridor_global_texels) * global_spacing / node_spacing_rad(fields, step), 1.5)

    paths = _stream_paths(streams, fields, lo, step, n, int(params.min_global_nodes))
    reach = corridor_reach([(p.s, p.t) for p in paths], lo, step, n, 2.0 * radius_nodes + 2.0)
    raster = sample_raster(fields, globe, lo, step, n, reach)

    occupied = np.zeros(n * n, dtype=bool)
    on_path = np.zeros(n * n, dtype=np.int64)
    routed: list[tuple[np.ndarray, np.ndarray]] = []
    for p in paths:
        sinks = occupied | raster.ocean | raster.lake
        nodes = _route_run(raster, p.s, p.t, radius_nodes, sinks, p.ends_here, params)
        if nodes is None or len(nodes) < 2:
            continue
        routed.append((nodes, _run_flows(raster, nodes, p.s, p.t, p.edge_flows)))
        occupied[nodes] = True
        on_path[nodes] += 1

    # A path ending on another one (not on water) is a confluence.
    water = raster.ocean | raster.lake
    ends = [int(nodes[-1]) for nodes, _ in routed if on_path[nodes[-1]] > 1 and not water[nodes[-1]]]
    drawn = [_to_path(raster, nodes, flows, params) for nodes, flows in routed]
    at_sea = [i for i, (nodes, _) in enumerate(routed) if raster.ocean[nodes[-1]]]
    return RegionRivers(
        region_id=fields.region_id,
        paths=drawn,
        confluences=raster.positions[ends].reshape(-1, 3),
        mouths=np.array([drawn[i][0][-1] for i in at_sea], dtype=np.float64).reshape(-1, 3),
        mouth_flows=np.array([float(drawn[i][1][-1]) for i in at_sea], dtype=np.float64),
    )


def snap_deltas(deltas: np.ndarray, rivers: RegionRivers, reach: float) -> np.ndarray:
    """Move each world delta onto the mouth of its re-routed river.

    A delta sits on a world river's mouth, which the region re-routed onto the
    refined coast.  Each delta moves to the largest-flow regional mouth within
    ``reach`` (chord on the unit sphere) — the river itself rather than a stream
    that happens to reach the sea closer by; one with no mouth that near stays.
    """
    out = np.array(deltas, dtype=np.float64).reshape(-1, 3)
    if len(rivers.mouths) == 0 or len(out) == 0:
        return out
    tree = KDTree(rivers.mouths)
    for i, near in enumerate(tree.query_ball_point(out, reach)):
        if near:
            out[i] = rivers.mouths[max(near, key=lambda j: rivers.mouth_flows[j])]
    return out


def river_length_km(path: RiverPath) -> float:
    """Great-circle length of a river polyline, km."""
    pts = np.asarray(path[0], dtype=np.float64)
    dots = np.clip(np.einsum("ij,ij->i", pts[:-1], pts[1:]), -1.0, 1.0)
    return float(np.arccos(dots).sum()) * EARTH_RADIUS_KM
