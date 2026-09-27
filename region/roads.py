"""Regional roads — the world's trade routes re-routed over the refined terrain.

The trade stage routes goods over the face graph, so a world route is a chain of
straight hops between face centroids, ~225 km apart.  Which places trade, over
which faces, and how much, are world facts.  A region refines only *where the
road runs* between them:

- The world's route edges are chained into polylines at their junctions, and the
  land chains are re-traced (:func:`region.corridor.route_corridor`) through a
  corridor ``corridor_face_fraction`` of a face spacing either side of the world
  line — the uncertainty a face-graph route leaves.
- The cost is the world's land-cost model evaluated per raster node instead of
  per face, in the same units: a node's base cost (biome difficulty, forest,
  and danger × the realm's movement scale) is charged per face-width travelled,
  climbing is charged ``ruggedness_cost_per_km`` per km of refined relief, and a
  regional river costs the world's ford penalty to cross and gives its corridor
  discount to travel along.  Refined water is impassable.  So a road takes the
  pass and the valley floor, fords rivers where they are small, and keeps out of
  dangerous forest when there is a way round.
- Sea legs stay straight.  A port hop (land face → water face) is split at the
  refined coast: its land part is routed with the chain it continues, the rest
  is drawn straight out to sea.
- Only routes the world map draws are re-routed (flow at or above the world's
  median, the renderer's threshold); the rest are kept as they are, for town
  siting.  Ends at a settlement land exactly on its marker (moved onto the
  refined land when the refined shoreline flooded it).

Everything is deterministic.  Routes leaving the raster are cut at its edge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from core.params import param_field as _f, paramclass
from region.corridor import (
    Raster,
    corridor_reach,
    node_spacing_rad,
    raster_frame,
    route_corridor,
    sample_raster,
)
from region.display import snap_to_land
from region.rivers import RegionRivers
from region.sampling import GlobalSampler
from region.upsample import EARTH_RADIUS_KM, RegionFields
from worldgen.cost.surfaces import BIOME_TRAVERSAL_COST
from worldgen.hydrology.types import laplacian_smooth_sphere

if TYPE_CHECKING:
    from core.world import World
    from worldgen.cost.builder import LandCostParams

# One trade-route hop, in the world "trade_routes" feature's form.
RouteSegment = tuple[np.ndarray, np.ndarray, float]


@paramclass
class RegionRoadParams:
    """Tuning for regional road refinement."""

    # Raster spacing as a multiple of the regional texel spacing.  Roads cross
    # whole faces, so a coarser raster than the rivers' is plenty.
    raster_texels: float = _f(2.0, label="Road raster spacing (texels)", min=1.0, max=6.0, step=0.5)
    margin: float = _f(0.08, label="Road raster margin", min=0.0, max=0.3, step=0.01)
    # Half-width of the corridor a world route may move within, as a fraction of
    # the face spacing (~225 km at the default mesh).
    corridor_face_fraction: float = _f(
        0.35, label="Corridor half-width (face spacings)", min=0.05, max=1.0, step=0.05)
    smoothing_iterations: int = _f(6, label="Road smoothing iterations", min=0, max=20)


@dataclass
class RegionRoads:
    """Regional trade routes, as world-form ``(start, end, flow)`` segments."""

    region_id: int
    # The re-routed land roads and the straight sea legs, drawn in place of the
    # world's routes.
    segments: list[RouteSegment]
    # World routes below the drawing threshold, kept straight: towns still read them.
    undrawn: list[RouteSegment] = field(default_factory=list)
    # (K, 3) re-routed road polylines, for town siting.
    polylines: list[np.ndarray] = field(default_factory=list)
    # The straight sea legs among ``segments``.
    sea: list[RouteSegment] = field(default_factory=list)


def _key(p: np.ndarray) -> tuple[float, float, float]:
    from realmgen.refine.settlements import position_key
    return position_key(p)


# ── the cost ─────────────────────────────────────────────────────────────────


def _river_nodes(rivers: Optional[RegionRivers], fields: RegionFields, raster: Raster) -> tuple[np.ndarray, np.ndarray]:
    """``(n²,)`` flow of the regional river on each raster node (0 off-river).

    Rivers are rasterised 4-connected — a diagonal step marks both corners — so
    an 8-connected road cannot slip between two diagonal river nodes unfording.
    """
    n = raster.n
    flow = np.zeros(n * n)
    if rivers is None:
        return flow, flow > 0.0
    for positions, flows, in_water in rivers.paths:
        pts = np.asarray(positions, dtype=np.float64)
        f = np.asarray(flows, dtype=np.float64)
        dry = ~np.asarray(in_water, dtype=bool)
        if len(pts) < 2 or not dry.any():
            continue
        s, t, valid = fields.geometry.ecef_to_local(pts)
        x = (s - raster.lo) / raster.step
        y = (t - raster.lo) / raster.step
        for k in np.flatnonzero(dry & valid[:-1] & valid[1:]):
            m = max(int(np.ceil(2.0 * np.hypot(x[k + 1] - x[k], y[k + 1] - y[k]))), 1)
            frac = np.arange(m + 1) / m
            cols = np.rint(x[k] + frac * (x[k + 1] - x[k])).astype(np.int64)
            rows = np.rint(y[k] + frac * (y[k + 1] - y[k])).astype(np.int64)
            # Close diagonal gaps.
            cols = np.concatenate([cols, cols[1:]])
            rows = np.concatenate([rows, rows[:-1]])
            ok = (cols >= 0) & (cols < n) & (rows >= 0) & (rows < n)
            idx = rows[ok] * n + cols[ok]
            flow[idx] = np.maximum(flow[idx], f[k])
    return flow, flow > 0.0


def _road_cost(
    raster: Raster, river_flow: np.ndarray, land: "LandCostParams", danger_scale: float, node_km: float,
    face_km: float,
):
    """Per-edge road cost in the world's land-cost units (one face-width step of
    base cost 1 costs 1)."""
    biome = np.clip(np.rint(raster.values.get("biome", np.full(raster.n ** 2, 3.0))), 0,
                    len(BIOME_TRAVERSAL_COST) - 1).astype(np.int64)
    base = float(land.biome_cost_scale) * BIOME_TRAVERSAL_COST[biome].astype(np.float64)
    if "forest_density" in raster.values:
        base = base + float(land.forest_cost_strength) * np.clip(raster.values["forest_density"], 0.0, 1.0)
    if "danger" in raster.values:
        base = base + float(danger_scale) * np.maximum(raster.values["danger"], 0.0)
    elevation = raster.values["elevation"]
    water = raster.water
    ref = max(float(land.river_flow_reference), 1e-9)
    weight = np.minimum(river_flow / ref, 1.0)
    on_river = river_flow > 0.0
    ford = np.minimum(float(land.river_ford_strength) * weight, float(land.river_max_mult) - 1.0)
    along = np.maximum(1.0 - float(land.river_corridor_strength) * weight, float(land.river_min_mult))
    per_step = node_km / face_km
    climb_per_km = float(land.ruggedness_cost_per_km)

    def edge_cost(sub: np.ndarray, a: np.ndarray, b: np.ndarray, length: np.ndarray) -> np.ndarray:
        na, nb = sub[a], sub[b]
        cost = 0.5 * (base[na] + base[nb]) * length * per_step
        both = on_river[na] & on_river[nb]
        cost = np.where(both, cost * np.minimum(along[na], along[nb]), cost)
        entering = on_river[nb] & ~on_river[na]
        cost = cost + np.where(entering, ford[nb] * base[nb], 0.0)
        cost = cost + climb_per_km * np.abs(elevation[nb] - elevation[na])
        return np.where(water[na] | water[nb], np.inf, cost)

    return edge_cost


# ── chaining the world's hops ────────────────────────────────────────────────


def _chains(edges: list[tuple[int, int, float]], n_nodes: int) -> list[tuple[list[int], list[float]]]:
    """Cut an undirected edge list into maximal chains through degree-2 nodes.

    Returns ``(nodes, edge flows)`` per chain; a closed loop is cut at one node.
    """
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n_nodes)]
    for e, (a, b, _) in enumerate(edges):
        adj[a].append((b, e))
        adj[b].append((a, e))
    used = np.zeros(len(edges), dtype=bool)
    chains = []

    def walk(start: int, e: int) -> tuple[list[int], list[float]]:
        nodes, flows = [start], []
        cur = start
        while True:
            used[e] = True
            a, b, f = edges[e]
            nxt = b if a == cur else a
            nodes.append(nxt)
            flows.append(f)
            cur = nxt
            if len(adj[cur]) != 2:
                return nodes, flows
            e2 = [x for _, x in adj[cur] if not used[x]]
            if not e2:
                return nodes, flows
            e = e2[0]

    for v in range(n_nodes):
        if len(adj[v]) != 2:
            for _, e in adj[v]:
                if not used[e]:
                    chains.append(walk(v, e))
    for e in range(len(edges)):              # loops: every node has degree 2
        if not used[e]:
            chains.append(walk(edges[e][0], e))
    return chains


def _coast_point(
    raster: Raster, fields: RegionFields, land_end: np.ndarray, sea_end: np.ndarray,
) -> Optional[np.ndarray]:
    """Where the straight hop from ``land_end`` to ``sea_end`` first meets refined
    water, as the last dry raster node before it; None if it never does."""
    pts = np.stack([land_end, sea_end])
    s, t, valid = fields.geometry.ecef_to_local(pts)
    if not valid.all():
        return None
    x = (s - raster.lo) / raster.step
    y = (t - raster.lo) / raster.step
    m = max(int(np.ceil(2.0 * np.hypot(x[1] - x[0], y[1] - y[0]))), 1)
    frac = np.arange(m + 1) / m
    cols = np.rint(x[0] + frac * (x[1] - x[0])).astype(np.int64)
    rows = np.rint(y[0] + frac * (y[1] - y[0])).astype(np.int64)
    n = raster.n
    inside = (cols >= 0) & (cols < n) & (rows >= 0) & (rows < n)
    if not inside.all():
        return None
    idx = rows * n + cols
    wet = raster.water[idx] & raster.sampled[idx]
    hit = np.flatnonzero(wet)
    if hit.size == 0 or hit[0] == 0:
        return None
    return raster.positions[idx[hit[0] - 1]].copy()


def _dry_node(raster: Raster, node: int, radius: float) -> int:
    """``node``, or the nearest dry sampled node within ``radius`` nodes of it."""
    if not raster.water[node] and raster.sampled[node]:
        return node
    n = raster.n
    r0, c0 = divmod(node, n)
    rad = int(np.ceil(radius))
    rr, cc = np.mgrid[max(r0 - rad, 0):min(r0 + rad + 1, n), max(c0 - rad, 0):min(c0 + rad + 1, n)]
    cand = (rr * n + cc).ravel()
    ok = raster.sampled[cand] & ~raster.water[cand]
    if not ok.any():
        return node
    d = np.hypot(rr.ravel() - r0, cc.ravel() - c0)
    d = np.where(ok, d, np.inf)
    return int(cand[int(np.argmin(d))])


@dataclass
class _Stretch:
    """One in-raster stretch of a chain: local vertices, which are real world
    vertices (the others are cuts at the raster edge), and the hop flows."""

    s: np.ndarray
    t: np.ndarray
    ecef: np.ndarray          # (K, 3); rows of cut ends are placeholders
    real: np.ndarray          # (K,) bool
    flows: np.ndarray         # (K - 1,)


def _clip(sa: float, ta: float, sb: float, tb: float, lo: float, hi: float) -> tuple[float, float]:
    """The point where the segment from inside ``a`` to outside ``b`` leaves the box."""
    lam = 1.0
    for a, b in ((sa, sb), (ta, tb)):
        if b < lo:
            lam = min(lam, (lo - a) / (b - a))
        elif b > hi:
            lam = min(lam, (hi - a) / (b - a))
    return sa + lam * (sb - sa), ta + lam * (tb - ta)


def _stretches(
    s: np.ndarray, t: np.ndarray, ecef: np.ndarray, flows: np.ndarray, lo: float, hi: float,
) -> list[_Stretch]:
    """The chain cut at the raster edge into its inside stretches."""
    inside = (s >= lo) & (s <= hi) & (t >= lo) & (t <= hi)
    out = []
    i, k = 0, len(s)
    while i < k:
        if not inside[i]:
            i += 1
            continue
        j = i
        while j + 1 < k and inside[j + 1]:
            j += 1
        ss, tt = list(s[i:j + 1]), list(t[i:j + 1])
        pts, real = list(ecef[i:j + 1]), [True] * (j + 1 - i)
        fl = list(flows[i:j])
        if i > 0:
            cs, ct = _clip(s[i], t[i], s[i - 1], t[i - 1], lo, hi)
            ss.insert(0, cs)
            tt.insert(0, ct)
            pts.insert(0, ecef[i])
            real.insert(0, False)
            fl.insert(0, flows[i - 1])
        if j < k - 1:
            cs, ct = _clip(s[j], t[j], s[j + 1], t[j + 1], lo, hi)
            ss.append(cs)
            tt.append(ct)
            pts.append(ecef[j])
            real.append(False)
            fl.append(flows[j])
        if len(ss) >= 2:
            out.append(_Stretch(np.array(ss), np.array(tt), np.array(pts), np.array(real), np.array(fl)))
        i = j + 1
    return out


def _assign_flows(path: np.ndarray, verts: np.ndarray, flows: np.ndarray) -> np.ndarray:
    """Per-edge flow along a routed ``path`` (P, 3): each world hop's flow on the
    stretch between the path points nearest its two ends."""
    near = np.argmin(np.linalg.norm(path[None, :, :] - verts[:, None, :], axis=2), axis=1)
    near = np.maximum.accumulate(near)
    out = np.full(len(path) - 1, flows[0] if len(flows) else 0.0)
    for k in range(len(flows)):
        out[near[k]:max(near[k + 1], near[k] + 1)] = flows[k]
    return out


# ── entry point ──────────────────────────────────────────────────────────────


def build_region_roads(
    fields: RegionFields,
    globe: "World",
    rivers: Optional[RegionRivers] = None,
    params: Optional[RegionRoadParams] = None,
    land: Optional["LandCostParams"] = None,
    danger_scale: float = 2.0,
    land_reach: Optional[float] = None,
) -> RegionRoads:
    """Re-route the world's trade routes over ``fields``' refined terrain.

    ``land`` and ``danger_scale`` are the world's land-cost tuning and the realm's
    danger movement scale; ``land_reach`` is how far (chord) a settlement on the
    refined water is moved onto land (default: six regional texels).
    """
    from worldgen.cost.builder import LandCostParams
    params = params or RegionRoadParams()
    land = land or LandCostParams()
    routes = globe.get_feature("trade_routes") or []
    empty = RegionRoads(region_id=fields.region_id, segments=[])
    if not routes:
        return empty
    mesh = globe.mesh
    lo, step, n = raster_frame(fields, params.raster_texels, params.margin)
    hi = lo + step * (n - 1)
    spacing = fields.grid.texel_spacing_rad(fields.tex_size)
    land_reach = 6.0 * spacing if land_reach is None else float(land_reach)

    # Only what the map draws is re-routed; the rest is kept for the towns.
    flows = np.array([float(r[2]) for r in routes])
    threshold = float(np.median(flows))
    starts = np.array([np.asarray(r[0], dtype=np.float64) for r in routes])
    ends = np.array([np.asarray(r[1], dtype=np.float64) for r in routes])
    s0, t0, v0 = fields.geometry.ecef_to_local(starts)
    s1, t1, v1 = fields.geometry.ecef_to_local(ends)
    face_rad = float(np.sqrt(np.mean(mesh.face_areas)))
    node_rad = node_spacing_rad(fields, step)
    slack = 2.0 * face_rad * step / node_rad          # two hops, in local units

    def near_raster(s: np.ndarray, t: np.ndarray) -> np.ndarray:
        return (s >= lo - slack) & (s <= hi + slack) & (t >= lo - slack) & (t <= hi + slack)

    here = v0 & v1 & (near_raster(s0, t0) | near_raster(s1, t1))
    drawn = here & (flows >= threshold)
    undrawn = [routes[i] for i in np.flatnonzero(here & ~drawn)]
    if not drawn.any():
        return RegionRoads(region_id=fields.region_id, segments=[], undrawn=undrawn)

    # Land or sea at each hop end, by the world's land-passability of its face.
    cs = globe.get_feature("cost_surfaces")
    sampler = GlobalSampler(mesh)
    idx = np.flatnonzero(drawn)
    f0, _, _ = sampler.locate(starts[idx])
    f1, _, _ = sampler.locate(ends[idx])
    if cs is not None:
        land0, land1 = cs.land_passable[f0], cs.land_passable[f1]
    else:
        land0 = land1 = np.ones(len(idx), dtype=bool)

    # Sample the raster around every land and port hop.
    radius_nodes = max(float(params.corridor_face_fraction) * face_rad / node_rad, 1.5)
    any_land = land0 | land1
    lines = [(np.array([s0[i], s1[i]]), np.array([t0[i], t1[i]])) for i in idx[any_land]]
    reach = corridor_reach(lines, lo, step, n, radius_nodes + 2.0)
    raster = sample_raster(fields, globe, lo, step, n, reach, keys=("biome", "forest_density", "danger"))

    # Nodes: every hop end, plus a coast point per port hop.
    keys: dict = {}
    points: list[np.ndarray] = []

    def node(p: np.ndarray) -> int:
        k = _key(p)
        if k not in keys:
            keys[k] = len(points)
            points.append(np.asarray(p, dtype=np.float64))
        return keys[k]

    land_edges: list[tuple[int, int, float]] = []
    sea: list[RouteSegment] = []
    for j, i in enumerate(idx):
        a, b, f = starts[i], ends[i], float(flows[i])
        if land0[j] and land1[j]:
            land_edges.append((node(a), node(b), f))
        elif land0[j] or land1[j]:
            land_end, sea_end = (a, b) if land0[j] else (b, a)
            coast = _coast_point(raster, fields, land_end, sea_end)
            if coast is None:
                land_edges.append((node(a), node(b), f))
                continue
            land_edges.append((node(land_end), node(coast), f))
            sea.append((coast.astype(np.float32), np.asarray(sea_end, dtype=np.float32), f))
        else:
            sea.append((np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32), f))

    river_flow, _ = _river_nodes(rivers, fields, raster)
    node_km = node_rad * EARTH_RADIUS_KM
    edge_cost = _road_cost(raster, river_flow, land, danger_scale, node_km, face_rad * EARTH_RADIUS_KM)

    pts = np.array(points)
    ends_on_land = snap_to_land(fields, pts, land_reach)
    segments: list[RouteSegment] = list(sea)
    polylines: list[np.ndarray] = []
    for chain, chain_flows in _chains(land_edges, len(points)):
        verts = ends_on_land[chain]
        s, t, _ = fields.geometry.ecef_to_local(verts)
        for st in _stretches(s, t, verts, np.asarray(chain_flows), lo, hi):
            anchors = np.where(st.real[:, None], st.ecef, fields.geometry.local_to_ecef(st.s, st.t))
            routed = _route_stretch(raster, st.s, st.t, radius_nodes, edge_cost)
            if routed is None:
                # No dry way through (an island hop the refined coast cut): the world line.
                path, edge_flows = anchors, st.flows
            else:
                path = raster.positions[routed].copy()
                # A world end lands exactly on its point (a settlement's marker); a
                # cut at the raster edge keeps its node.
                if st.real[0]:
                    path[0] = st.ecef[0]
                if st.real[-1]:
                    path[-1] = st.ecef[-1]
                path = np.stack(laplacian_smooth_sphere(list(path), iterations=int(params.smoothing_iterations)))
                path = path.astype(np.float64)
                edge_flows = _assign_flows(path, anchors, st.flows)
            polylines.append(path)
            for k in range(len(path) - 1):
                segments.append((path[k].astype(np.float32), path[k + 1].astype(np.float32), float(edge_flows[k])))
    return RegionRoads(region_id=fields.region_id, segments=segments, undrawn=undrawn, polylines=polylines, sea=sea)


def _route_stretch(
    raster: Raster, s: np.ndarray, t: np.ndarray, radius_nodes: float, edge_cost,
) -> Optional[np.ndarray]:
    """Raster nodes of the least-cost road along one in-raster stretch, its ends
    moved onto the nearest dry node."""
    ends, _ = raster.index(np.array([s[0], s[-1]]), np.array([t[0], t[-1]]))
    a = _dry_node(raster, int(ends[0]), radius_nodes)
    b = _dry_node(raster, int(ends[1]), radius_nodes)
    ra, ca = divmod(a, raster.n)
    rb, cb = divmod(b, raster.n)
    s = s.copy()
    t = t.copy()
    s[0], t[0] = raster.lo + raster.step * ca, raster.lo + raster.step * ra
    s[-1], t[-1] = raster.lo + raster.step * cb, raster.lo + raster.step * rb
    nodes = route_corridor(raster, s, t, radius_nodes, edge_cost)
    if nodes is None or len(nodes) < 2:
        return None
    return nodes
