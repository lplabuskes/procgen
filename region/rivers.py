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
from scipy.ndimage import distance_transform_edt
from scipy.spatial import KDTree

from core.params import param_field as _f, paramclass
from region.corridor import (
    CROSS_COST,
    EdgeCost,
    Raster,
    centreline,
    corridor_reach,
    diagonal_corners,
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

# A branch (``starts_on_path``) leaves its parent without the separation cost within this many nodes.
_LEAVING_NODES = 3.0
# Beyond this many nodes the separation cost (e^(−d/2)) is negligible.
_SEPARATION_REACH = 8
# A junction is cut at a crossing with the line it meets within this many segments of its end.
_JUNCTION_SEGMENTS = 6


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
    # Cost × (1 + separation · e^(−d/2)) d nodes from an earlier path, so two
    # channels sharing a valley run side by side rather than braid once smoothed.
    separation: float = _f(3.0, label="Channel separation", min=0.0, max=10.0, step=0.5)
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
    # Per path, per vertex: the index of the world river (``"rivers"``) it is part of, -1 none
    # (a sub-threshold headwater, a distributary). Empty when unknown.
    path_rivers: list[np.ndarray] = field(default_factory=list)

    def river_ids(self) -> list[np.ndarray]:
        """``path_rivers``, or all -1 when it isn't aligned with ``paths``."""
        if len(self.path_rivers) == len(self.paths):
            return self.path_rivers
        return [np.full(len(p[0]), -1, dtype=np.int64) for p in self.paths]


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


def _run_ids(raster: Raster, nodes: np.ndarray, s: np.ndarray, t: np.ndarray,
             ids: Optional[np.ndarray]) -> np.ndarray:
    """Per node of a re-routed path, the world river id of the nearest source vertex (-1 without ids)."""
    if ids is None:
        return np.full(len(nodes), -1, dtype=np.int64)
    x = (s - raster.lo) / raster.step
    y = (t - raster.lo) / raster.step
    rows, cols = np.divmod(nodes, raster.n)
    nearest = np.argmin((cols[:, None] - x[None, :]) ** 2 + (rows[:, None] - y[None, :]) ** 2, axis=1)
    return np.asarray(ids, dtype=np.int64)[nearest]


def _meet_ids(ids: np.ndarray, new_len: int, at_end: bool) -> np.ndarray:
    """``ids`` re-aligned to a path ``meet_line`` cut to ``new_len`` vertices (the cut point
    takes the id of the segment it lies on)."""
    old_len = len(ids)
    if new_len == old_len:
        return ids
    if at_end:
        return np.concatenate([ids[:new_len - 1], ids[new_len - 2:new_len - 1]])
    seg = old_len - new_len
    return np.concatenate([ids[seg + 1:seg + 2], ids[seg + 1:]])


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
    # (len,) world river index per vertex (RegionRivers.path_rivers); None: none.
    river_ids: Optional[np.ndarray] = None


def stream_river_ids(streams: StreamNetwork, river_texels: Optional[list]) -> np.ndarray:
    """``(K,)`` the world river (index into ``"rivers"``) each stream node is part of, -1 none.

    ``river_texels`` is the ``"river_texels"`` feature. A tributary's last texel is
    its confluence on the stem, so a texel inside one river wins over a river's end.
    """
    out = np.full(len(streams), -1, dtype=np.int64)
    if not river_texels:
        return out
    order = np.argsort(streams.nodes, kind="stable")
    sorted_nodes = streams.nodes[order]
    for interior in (False, True):
        for r, tex in enumerate(river_texels):
            tex = np.asarray(tex, dtype=np.int64)
            tex = tex[:-1] if interior else tex[-1:]
            if tex.size == 0:
                continue
            at = np.clip(np.searchsorted(sorted_nodes, tex), 0, len(sorted_nodes) - 1)
            hit = sorted_nodes[at] == tex
            out[order[at[hit]]] = r
    return out


def _stream_paths(
    streams: StreamNetwork, fields: RegionFields, lo: float, step: float, n: int, min_nodes: int,
    river_ids: Optional[np.ndarray] = None,
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
        ids = None
        if river_ids is not None:
            ids = river_ids[idx][nodes].copy()
            # The last node is where it joins its stem (that river's texel): it is still this path.
            ids[-1] = ids[-2]
        out.append(StreamPath(
            s=s[idx][nodes], t=t[idx][nodes], edge_flows=running[:-1],
            ends_here=not (receiver[nodes[-1]] < 0 and leaves_raster[nodes[-1]]),
            river_ids=ids,
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


def _avoid_factor(
    raster: Raster, taken: np.ndarray, blocked: np.ndarray, s: np.ndarray, t: np.ndarray,
    p: StreamPath, radius_nodes: float, separation: float,
) -> Optional[np.ndarray]:
    """``(n²,)`` node cost factor keeping a path off the ones routed before it.

    ``taken``: their drawn nodes; ``blocked``: those plus their diagonal corners
    (:func:`~region.corridor.diagonal_corners`).  Blocked nodes cost
    ``CROSS_COST``, others near a taken one ``1 + separation · e^(−d/2)`` (so a
    tributary meets its stem head-on rather than alongside, which smoothing would
    turn into a crossing).  Near its mouth (:func:`_route_run`'s disc) a taken
    node is free: the first one touched ends the path, so it joins rather than
    crosses.  A branch starts on its parent and may leave it without the
    separation cost.
    """
    if not blocked.any():
        return None
    n = raster.n
    # Only the corridor (and mouth disc) is routed over: work in a window around
    # it, wide enough that the separation term has died away at its edge.
    rows, cols = centreline(raster.lo, raster.step, n, s, t)
    pad = int(np.ceil(2.0 * radius_nodes)) + 1 + _SEPARATION_REACH
    r0, r1 = max(int(rows.min()) - pad, 0), min(int(rows.max()) + pad + 1, n)
    c0, c1 = max(int(cols.min()) - pad, 0), min(int(cols.max()) + pad + 1, n)
    win = (np.arange(r0, r1)[:, None] * n + np.arange(c0, c1)[None, :]).ravel()
    d = np.asarray(distance_transform_edt(~taken[win].reshape(r1 - r0, c1 - c0))).ravel()
    near = 1.0 + separation * np.exp(-d / 2.0)
    wr, wc = np.divmod(win, n)

    def disc(k: int, radius: float) -> np.ndarray:
        return np.hypot(wc - (s[k] - raster.lo) / raster.step, wr - (t[k] - raster.lo) / raster.step) <= radius

    if p.starts_on_path:
        near[disc(0, _LEAVING_NODES)] = 1.0
    local = np.where(blocked[win], CROSS_COST, near)
    if p.ends_here:
        local[disc(-1, 2.0 * radius_nodes) & taken[win]] = 1.0
    factor = np.ones(n * n)
    factor[win] = local
    if p.starts_on_path:
        start, inside = raster.index(s[:1], t[:1])
        if inside[0]:
            factor[start[0]] = 1.0
    return factor


def _trim_crossed_head(nodes: np.ndarray, blocked: np.ndarray, reach_nodes: float) -> np.ndarray:
    """``nodes`` starting past the last earlier path it crosses within ``reach_nodes`` steps of its source.

    A source is placed no better than the world's grid; one that lands beside a
    stem re-routed onto its side would otherwise begin by crossing it.
    """
    head = np.flatnonzero(blocked[nodes[:int(reach_nodes) + 1]])
    return nodes if head.size == 0 else nodes[int(head[-1]) + 1:]


def _join_at_first_touch(n: int, nodes: np.ndarray, owner: np.ndarray) -> np.ndarray:
    """A path ending on another one, cut where it first comes beside (8-neighbour) it, ending on it.

    The stem's re-route can move across the tributary's course upstream of the
    world confluence; the tributary joins it there rather than crossing it.
    """
    stem = int(owner[nodes[-1]])
    if stem < 0:
        return nodes
    on_stem = (owner == stem).reshape(n, n)
    rows, cols = np.divmod(nodes, n)
    for k in range(len(nodes) - 1):
        r, c = int(rows[k]), int(cols[k])
        if on_stem[r, c]:
            return nodes[:k + 1] if k > 0 else nodes
        # 4-neighbours first, so a join never steps diagonally past a stem node.
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and on_stem[rr, cc]:
                return np.append(nodes[:k + 1], rr * n + cc)
    return nodes


def _segment_crossings(a: np.ndarray, b: np.ndarray) -> list[tuple[int, float]]:
    """``(segment, t)`` along polyline ``a`` (``(M, 3)``, near unit sphere) wherever it crosses polyline ``b``.

    Planar, on the tangent plane at ``a[0]`` — exact enough over a junction, and
    well conditioned for the sub-km segments here (unlike the globe's
    :func:`~worldgen.hydrology.crossings.crossing_pairs`).  Touches at an end of
    either segment don't count.
    """
    ref = a[0] / np.linalg.norm(a[0])
    e1 = np.cross(ref, [0.0, 0.0, 1.0] if abs(ref[2]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ref, e1)
    pa, pb = np.stack([a @ e1, a @ e2], axis=1), np.stack([b @ e1, b @ e2], axis=1)
    r, q = np.diff(pa, axis=0), np.diff(pb, axis=0)            # (M-1, 2), (K-1, 2)
    d = r[:, None, 0] * q[None, :, 1] - r[:, None, 1] * q[None, :, 0]
    w = pb[None, :-1, :] - pa[:-1, None, :]                    # b start − a start
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[..., 0] * q[None, :, 1] - w[..., 1] * q[None, :, 0]) / d
        u = (w[..., 0] * r[:, None, 1] - w[..., 1] * r[:, None, 0]) / d
    eps = 1e-4
    hit = (np.abs(d) > 0) & (t > eps) & (t < 1 - eps) & (u > eps) & (u < 1 - eps)
    return [(int(i), float(t[i, j])) for i, j in zip(*np.nonzero(hit))]


def meet_line(
    path: RiverPath, line: np.ndarray, at_end: bool, cut: Optional[list[np.ndarray]] = None,
) -> RiverPath:
    """``path`` with its end (or start) on the polyline ``line``.

    That point moves to ``line``'s nearest vertex; then, if ``path`` crosses
    ``line`` (or any of ``cut``, default ``[line]``) within ``_JUNCTION_SEGMENTS``
    of it, it is cut at the crossing nearest that end instead.
    """
    pts, flows, in_water = path
    out = np.array(pts, dtype=np.float64)
    k = -1 if at_end else 0
    out[k] = line[int(np.argmin(np.linalg.norm(line - out[k][None, :], axis=1)))]
    m = min(_JUNCTION_SEGMENTS, len(out) - 1)
    lo = len(out) - 1 - m if at_end else 0
    window = out[lo:lo + m + 1]
    hits = [h for other in (cut if cut is not None else [line])
            for h in _segment_crossings(window, np.asarray(other, dtype=np.float64))]
    if hits:
        si, t = (min if at_end else max)(hits, key=lambda h: h[0] + h[1])
        seg = lo + si
        cp = out[seg] + t * (out[seg + 1] - out[seg])
        cp *= np.linalg.norm(out[seg]) / np.linalg.norm(cp)
        if at_end:
            out, flows, in_water = np.vstack([out[:seg + 1], cp]), flows[:seg + 1], in_water[:seg + 1]
        else:
            out, flows, in_water = np.vstack([cp, out[seg + 1:]]), flows[seg:], in_water[seg:]
    return out.astype(np.asarray(pts).dtype), flows, in_water


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

    ids = stream_river_ids(streams, globe.get_feature("river_texels"))
    paths = _stream_paths(streams, fields, lo, step, n, int(params.min_global_nodes), ids)
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
    # Occupied plus the corners of its diagonal steps (_avoid_factor).
    blocked = np.zeros(n * n, dtype=bool)
    on_path = np.zeros(n * n, dtype=np.int64)
    # The (last) path drawn through each node; -1 none.
    owner = np.full(n * n, -1, dtype=np.int64)
    # Per routed path: its index in ``paths``, nodes, flows, hidden, and the paths
    # it starts on / ends on (-1: none).
    done: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, int, int]] = []
    node_ids: dict[int, np.ndarray] = {}
    for i, p in enumerate(paths):
        sinks = occupied | raster.sinks
        s, t = p.s, p.t
        if p.starts_on_path:
            s, t = _start_on_drawn(raster, on_path, s, t, 2.0 * radius_nodes)
        factor = _avoid_factor(raster, occupied, blocked, s, t, p, radius_nodes, float(params.separation))
        if p.hidden_edges is not None:
            water = _water_factor(raster, s, t, p.hidden_edges, radius_nodes)
            factor = water if factor is None else factor * water
        nodes = _route_run(raster, s, t, radius_nodes, sinks, p.ends_here, params, factor)
        if nodes is None or len(nodes) < 2:
            continue
        if not p.starts_on_path:
            nodes = _trim_crossed_head(nodes, blocked, 2.0 * radius_nodes)
            if len(nodes) < 2:
                continue
            if p.ends_here:
                nodes = _join_at_first_touch(n, nodes, owner)
        hidden = _hidden_nodes(raster, nodes)
        flows = _run_flows(raster, nodes, s, t, p.edge_flows)
        node_ids[i] = _run_ids(raster, nodes, s, t, p.river_ids)
        parent = int(owner[nodes[0]]) if p.starts_on_path else -1
        stem = int(owner[nodes[-1]]) if p.ends_here and not hidden[-1] else -1
        done.append((i, nodes, flows, hidden, parent, stem))
        owner[nodes[~hidden]] = i
        occupied[nodes[~hidden]] = True
        blocked[nodes[~hidden]] = True
        blocked[diagonal_corners(n, nodes, ~(hidden[:-1] & hidden[1:]))] = True
        if not p.minor:
            on_path[nodes[~hidden]] += 1

    # Smoothed independently, a junction may stop short of the line it meets or
    # poke through it: put it on that line as drawn (stems before their tributaries).
    lines: dict[int, np.ndarray] = {}
    drawn: dict[int, RiverPath] = {}
    drawn_ids: dict[int, np.ndarray] = {}
    for i, nodes, flows, hidden, parent, stem in done:
        path = _to_path(raster, nodes, flows, hidden, params)
        ids = node_ids[i]
        if parent in lines:
            path = meet_line(path, lines[parent], at_end=False)
            ids = _meet_ids(ids, len(path[0]), at_end=False)
        if stem in lines:
            path = meet_line(path, lines[stem], at_end=True)
            ids = _meet_ids(ids, len(path[0]), at_end=True)
        drawn[i] = path
        drawn_ids[i] = ids
        lines[i] = np.asarray(path[0], dtype=np.float64)

    major = [(i, nodes) for i, nodes, *_ in done if not paths[i].minor]
    # A path ending on another one's drawn stretch is a confluence.
    ends = [int(nodes[-1]) for i, nodes, _, hidden, _, _ in done
            if not paths[i].minor and on_path[nodes[-1]] - (0 if hidden[-1] else 1) > 0]
    at_sea = [i for i, nodes in major if raster.ocean[nodes[-1]]]
    return RegionRivers(
        region_id=fields.region_id,
        paths=[drawn[i] for i, _ in major],
        confluences=raster.positions[ends].reshape(-1, 3),
        mouths=np.array([drawn[i][0][-1] for i in at_sea], dtype=np.float64).reshape(-1, 3),
        mouth_flows=np.array([float(drawn[i][1][-1]) for i in at_sea], dtype=np.float64),
        streams=[drawn[i] for i, *_ in done if paths[i].minor],
        path_rivers=[drawn_ids[i] for i, _ in major],
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
