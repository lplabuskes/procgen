"""Regional rivers — the world's stream network re-routed over the refined relief.

The hydrology stage's :class:`worldgen.hydrology.types.StreamNetwork` (down to a
stream threshold far below the rivers') fixes which texels carry a channel, where
each drains and how much flows.  A region refines only where each channel runs:

- The network is cut into main stems with
  :func:`worldgen.hydrology.rivers.main_stem_paths`, so regional rivers are the
  world's extended by sub-threshold tributaries, with the same confluences.
- Each path is re-traced within ~one global texel of its world centreline as the
  least-cost route through the refined relief (cost rises with the sub-grid
  anomaly), so it follows the valley floor but never changes basin.  Stems go
  before tributaries; a path whose mouth is here ends on its stem, a lake, or the
  refined coast.

All on one uniform raster over the local ``(s, t)`` frame, ``margin`` past the
cell.  Paths are cut at its edge.  A path that leaves a lake (or the sea) and
re-enters the same one is hidden for the whole stretch between.
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
from region.distributaries import WATER_COST, DistributaryParams, add_distributaries
from region.upsample import EARTH_RADIUS_KM, RegionFields
from worldgen.hydrology.rivers import main_stem_paths
from worldgen.hydrology.types import StreamNetwork, laplacian_smooth_sphere

if TYPE_CHECKING:
    from core.world import World

# The (positions, flows, in_lake) polyline form of the world's "rivers" feature.
RiverPath = tuple[np.ndarray, np.ndarray, np.ndarray]

# Land runs this short between water along a path count as water, so a lake
# crossing over sub-grid islands is hidden whole rather than drawn as dashes.
_SPECKLE_NODES = 2


@paramclass
class RegionRiverParams:
    """Tuning for regional river refinement."""

    # In regional texels; coarser is faster but blurs the valleys.
    raster_texels: float = _f(1.0, label="River raster spacing (texels)", min=0.5, max=4.0, step=0.25)
    # How far past the cell edge the raster reaches, in local units (the cell is 1).
    margin: float = _f(0.08, label="River raster margin", min=0.0, max=0.3, step=0.01)
    # Half-width of the corridor a global river may move within, in global texels.
    corridor_global_texels: float = _f(1.0, label="Corridor half-width (global texels)", min=0.25, max=3.0, step=0.25)
    # Node cost is exp(valley_seeking · anomaly / σ), σ the corridor's anomaly spread.
    valley_seeking: float = _f(2.0, label="Valley seeking", min=0.0, max=6.0, step=0.25)
    # Shorter stems are dropped (network depth is the hydrology stage's stream threshold).
    min_global_nodes: int = _f(3, label="Min stream length (global texels)", min=2, max=20)
    smoothing_iterations: int = _f(8, label="Path smoothing iterations", min=0, max=20)
    distributaries: DistributaryParams = field(
        default_factory=DistributaryParams, metadata={"label": "Delta distributaries"}
    )


@dataclass
class RegionRivers:
    """Regional river polylines, in the world ``"rivers"`` feature's form."""

    region_id: int
    # Re-routed stems: world rivers and sub-threshold streams alike, told apart by flow.
    paths: list[RiverPath]
    # (K, 3) where a path ends on another one — the confluences town siting rewards.
    confluences: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # (J, 3) where a path ends on the refined sea, and the flow it arrives with.
    mouths: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    mouth_flows: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Local maps only: the region's own streams (region.streams), re-routed; used
    # by villages but not among confluences or mouths.
    streams: list[RiverPath] = field(default_factory=list)


# ── re-routing ───────────────────────────────────────────────────────────────


def _valley_cost(raster: Raster, valley_seeking: float, node_factor: Optional[np.ndarray] = None) -> EdgeCost:
    """Edge cost exponential in the sub-grid anomaly, normalised by its corridor spread.

    ``node_factor`` (``(n²,)``) multiplies each node's cost.
    """
    cache: dict = {}

    def edge_cost(sub: np.ndarray, a: np.ndarray, b: np.ndarray, length: np.ndarray) -> np.ndarray:
        if cache.get("sub") is not sub:
            anomaly = raster.anomaly[sub]
            spread = max(float(anomaly.std()), 1e-6)
            cache["sub"] = sub
            cost = np.exp(np.clip(valley_seeking * (anomaly - anomaly.mean()) / spread, -20.0, 20.0))
            cache["cost"] = cost if node_factor is None else cost * node_factor[sub]
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
    node_factor: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Least-cost node path along one in-raster stretch, within ``radius_nodes`` of the world line.

    With ``ends_here`` it ends on the cheapest ``sinks`` node within twice the
    corridor of the mouth (the joined stem may sit a corridor width off).
    """
    return route_corridor(
        raster, s, t, radius_nodes, _valley_cost(raster, float(params.valley_seeking), node_factor),
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
class StreamPath:
    """One path to re-route, in the map's local coordinates."""

    s: np.ndarray
    t: np.ndarray
    edge_flows: np.ndarray       # (len - 1,) world flow, monotonic
    ends_here: bool              # ends on a sink or a stem, not at the raster edge
    minor: bool = False          # a regional stream (goes to RegionRivers.streams)
    # (len - 1,) the parent's hidden (under-water) segments; off them water is dear. None: water free.
    hidden_edges: Optional[np.ndarray] = None
    # Starts on another path (a delta distributary): starts on that path's re-route.
    starts_on_path: bool = False


def _stream_paths(
    streams: StreamNetwork, fields: RegionFields, lo: float, step: float, n: int, min_nodes: int,
) -> list[StreamPath]:
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
    # Where the tree continues beyond the raster, a path is cut, not ended.
    leaves_raster = (downstream >= 0) & (receiver < 0)

    flow = streams.flow[idx].astype(np.float64)
    out = []
    for path in main_stem_paths(receiver, flow, streams.channel[idx], streams.stop[idx], min_nodes):
        nodes = np.asarray(path, dtype=np.int64)
        running = np.maximum.accumulate(flow[nodes])
        out.append(StreamPath(
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


def _hide_reentries(water: np.ndarray, bodies: np.ndarray) -> np.ndarray:
    """``water`` set from each body's first node along the path to its last.

    A path that leaves a lake (or the sea) and comes back to the same one never
    really left: the stretch between is the world's in-lake course crossing a
    refined shore, not a river.
    """
    out = water.copy()
    for body in np.unique(bodies[bodies > 0]):
        at = np.flatnonzero(bodies == body)
        out[at[0]:at[-1] + 1] = True
    return out


def _hidden_nodes(raster: Raster, nodes: np.ndarray) -> np.ndarray:
    """Per node of a routed path, whether it counts as under water (not drawn)."""
    water = _hide_reentries(raster.water[nodes], raster.water_bodies[nodes])
    return _fill_short_gaps(water, _SPECKLE_NODES)


def _water_factor(
    raster: Raster, s: np.ndarray, t: np.ndarray, hidden_edges: np.ndarray, radius_nodes: float,
) -> np.ndarray:
    """``(n²,)`` node cost factor: water dear except within the corridor of the parent's hidden stretches.

    A re-route keeps to land wherever its parent ran on land, so it neither
    hugs a refined shore nor gets hidden as a re-entry (:func:`_hide_reentries`).
    """
    free = np.zeros(raster.n * raster.n, dtype=bool)
    wet = np.flatnonzero(np.asarray(hidden_edges, dtype=bool))
    if wet.size:
        runs = np.split(wet, np.flatnonzero(np.diff(wet) > 1) + 1)
        lines = [(s[r[0]:r[-1] + 2], t[r[0]:r[-1] + 2]) for r in runs]
        free = corridor_reach(lines, raster.lo, raster.step, raster.n, radius_nodes)
    return np.where(raster.water & ~free, WATER_COST, 1.0)


def _start_on_drawn(
    raster: Raster, on_path: np.ndarray, s: np.ndarray, t: np.ndarray, reach_nodes: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``(s, t)`` with its first vertex moved to the nearest drawn path node within ``reach_nodes``."""
    cand = np.flatnonzero(on_path > 0)
    if cand.size == 0:
        return s, t
    rows, cols = np.divmod(cand, raster.n)
    gap = np.hypot(cols - (s[0] - raster.lo) / raster.step, rows - (t[0] - raster.lo) / raster.step)
    k = int(np.argmin(gap))
    if gap[k] > reach_nodes:
        return s, t
    s, t = s.copy(), t.copy()
    s[0] = raster.lo + raster.step * cols[k]
    t[0] = raster.lo + raster.step * rows[k]
    return s, t


def _to_path(
    raster: Raster, nodes: np.ndarray, flows: np.ndarray, hidden: np.ndarray, params: RegionRiverParams,
) -> RiverPath:
    pts = [raster.positions[k] for k in nodes]
    smooth = laplacian_smooth_sphere(pts, iterations=int(params.smoothing_iterations), alpha=0.5)
    # Hidden where both ends are under water, so a river still reaches its coast.
    in_water = hidden[:-1] & hidden[1:]
    return np.stack(smooth).astype(np.float32), np.asarray(flows, dtype=np.float32), in_water


def build_region_rivers(
    fields: RegionFields,
    globe: "World",
    params: Optional[RegionRiverParams] = None,
) -> RegionRivers:
    """Re-route the world's ``"streams"`` network through ``fields``' refined relief."""
    params = params or RegionRiverParams()
    streams = globe.get_feature("streams")
    if streams is None or len(streams) == 0:
        return RegionRivers(region_id=fields.region_id, paths=[])
    lo, step, n = raster_frame(fields, params.raster_texels, params.margin)

    mesh = globe.mesh
    gw = mesh.face_textures["elevation"].shape[2]
    global_spacing = float(np.sqrt(np.mean(mesh.face_areas))) / max(gw - 1, 1)
    radius_nodes = max(float(params.corridor_global_texels) * global_spacing / node_spacing_rad(fields, step), 1.5)

    paths = _stream_paths(streams, fields, lo, step, n, int(params.min_global_nodes))
    rivers = route_stream_paths(fields, globe, paths, lo, step, n, radius_nodes, params)
    return add_distributaries(fields, globe, rivers, params)


def route_stream_paths(
    fields: RegionFields,
    globe: "World",
    paths: list[StreamPath],
    lo: float,
    step: float,
    n: int,
    radius_nodes: float,
    params: RegionRiverParams,
) -> RegionRivers:
    """Re-route ``paths`` (largest flow first) over ``fields``' raster ``(lo, step, n)``.

    Tier-agnostic: a local map feeds the region's rivers, then its streams (``minor``).
    """
    # Sampled whole, not just the corridors, so a lake is one water body (_hide_reentries).
    raster = sample_raster(fields, globe, lo, step, n, np.ones(n * n, dtype=bool))

    # Only drawn stretches take tributaries or count towards confluences: a
    # hidden one is under water, or a lake crossing (_hide_reentries) over land.
    occupied = np.zeros(n * n, dtype=bool)
    on_path = np.zeros(n * n, dtype=np.int64)
    routed: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    minor: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    starts_on: list[bool] = []
    for p in paths:
        sinks = occupied | raster.ocean | raster.lake
        s, t = p.s, p.t
        if p.starts_on_path:
            s, t = _start_on_drawn(raster, on_path, s, t, 2.0 * radius_nodes)
        factor = None if p.hidden_edges is None else _water_factor(raster, s, t, p.hidden_edges, radius_nodes)
        nodes = _route_run(raster, s, t, radius_nodes, sinks, p.ends_here, params, factor)
        if nodes is None or len(nodes) < 2:
            continue
        hidden = _hidden_nodes(raster, nodes)
        flows = _run_flows(raster, nodes, s, t, p.edge_flows)
        (minor if p.minor else routed).append((nodes, flows, hidden))
        occupied[nodes[~hidden]] = True
        if not p.minor:
            on_path[nodes[~hidden]] += 1
            starts_on.append(p.starts_on_path)

    # A path ending on another one's drawn stretch is a confluence.
    ends = [int(nodes[-1]) for nodes, _, hidden in routed
            if on_path[nodes[-1]] - (0 if hidden[-1] else 1) > 0]
    drawn = [_to_path(raster, nodes, flows, hidden, params) for nodes, flows, hidden in routed]
    # A branch's first point exactly on the line it leaves, as the region draws it.
    for i in np.flatnonzero(starts_on):
        others = [drawn[j][0] for j in range(len(drawn)) if j != i]
        if others:
            line = np.concatenate(others)
            drawn[i][0][0] = line[int(np.argmin(np.linalg.norm(line - drawn[i][0][0][None, :], axis=1)))]
    at_sea = [i for i, (nodes, _, _) in enumerate(routed) if raster.ocean[nodes[-1]]]
    return RegionRivers(
        region_id=fields.region_id,
        paths=drawn,
        confluences=raster.positions[ends].reshape(-1, 3),
        mouths=np.array([drawn[i][0][-1] for i in at_sea], dtype=np.float64).reshape(-1, 3),
        mouth_flows=np.array([float(drawn[i][1][-1]) for i in at_sea], dtype=np.float64),
        streams=[_to_path(raster, nodes, flows, hidden, params) for nodes, flows, hidden in minor],
    )


def snap_deltas(deltas: np.ndarray, rivers: RegionRivers, reach: float) -> np.ndarray:
    """Move each world delta to the largest-flow regional mouth within ``reach`` (chord), if any."""
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
    # Chord form: arccos of a float32 dot product loses sub-km steps entirely.
    chords = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return float((2.0 * np.arcsin(np.clip(0.5 * chords, 0.0, 1.0))).sum()) * EARTH_RADIUS_KM
