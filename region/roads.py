"""Regional roads — the world's trade routes re-routed over the refined terrain.

World routes are straight hops between face centroids (~225 km); which places
trade and how much are world facts.  A region refines only where the road runs:

- Route edges are chained into polylines at junctions; land chains are re-traced
  (:func:`region.corridor.route_corridor`) within ``corridor_face_fraction`` of a
  face spacing of the world line.
- Cost is the world's land-cost model per raster node, in the same units: base
  cost (biome, forest, danger × movement scale) per face-width, plus
  ``ruggedness_cost_per_km`` per km climbed; regional rivers charge the ford
  penalty to cross and discount travel along them.  Refined water is impassable.
- Water hops keep to the water (the world's ``trade_route_layers``): river and
  lake legs are chained and re-traced over refined water and the regional rivers,
  land costing ``water_land_cost`` times as much; open-sea legs stay straight.  A
  port hop is split at the refined coast, the road ending where the boat leaves.
- Only routes the world map draws (flow ≥ median) are re-routed; the rest are
  kept for town siting.  Settlement ends land exactly on their markers; ``moved``
  carries cities re-sited by ``region/cities.py``.

Routes leaving the raster are cut at its edge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.spatial import KDTree

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
from realmgen.trade.network import ROUTE_LAND, ROUTE_TRANSFER
from worldgen.cost.surfaces import BIOME_TRAVERSAL_COST
from worldgen.geology.channels import IS_OCEAN
from worldgen.hydrology.types import laplacian_smooth_sphere

if TYPE_CHECKING:
    from core.world import World
    from worldgen.cost.builder import LandCostParams

# One trade-route hop, in the world "trade_routes" feature's form.
RouteSegment = tuple[np.ndarray, np.ndarray, float]


@paramclass
class RegionRoadParams:
    """Tuning for regional road refinement."""

    # In regional texels; coarser than the rivers' raster, as roads cross whole faces.
    raster_texels: float = _f(2.0, label="Road raster spacing (texels)", min=1.0, max=6.0, step=0.5)
    margin: float = _f(0.08, label="Road raster margin", min=0.0, max=0.3, step=0.01)
    # Corridor half-width as a fraction of the face spacing (~225 km).
    corridor_face_fraction: float = _f(
        0.35, label="Corridor half-width (face spacings)", min=0.05, max=1.0, step=0.05)
    smoothing_iterations: int = _f(6, label="Road smoothing iterations", min=0, max=20)
    # River and lake legs: corridor half-width (the river can be half a face from the
    # centroid line) and the cost of a step over land relative to one over water.
    water_corridor_face_fraction: float = _f(
        0.5, label="Water corridor half-width (face spacings)", min=0.05, max=1.0, step=0.05)
    water_land_cost: float = _f(20.0, label="Water route land-step cost", min=1.0, max=200.0, step=1.0)


@dataclass
class RegionRoads:
    """Regional trade routes, as world-form ``(start, end, flow)`` segments."""

    region_id: int
    # Re-routed land roads and straight sea legs, drawn in place of the world's.
    segments: list[RouteSegment]
    # World routes below the drawing threshold, kept straight: towns still read them.
    undrawn: list[RouteSegment] = field(default_factory=list)
    # (K, 3) re-routed road polylines, for town siting.
    polylines: list[np.ndarray] = field(default_factory=list)
    # The straight sea legs among ``segments``.
    sea: list[RouteSegment] = field(default_factory=list)
    # (K - 1,) per-edge flow of each of ``polylines``, for the tier below.
    polyline_flows: list[np.ndarray] = field(default_factory=list)
    # Re-routed river and lake legs (also in ``segments``), their per-edge flows, and
    # whether each end is pinned to a place on land (a port or the coast point of a port hop).
    water_polylines: list[np.ndarray] = field(default_factory=list)
    water_flows: list[np.ndarray] = field(default_factory=list)
    water_pins: list[tuple[bool, bool]] = field(default_factory=list)


# Chord distance for a road end to match a settlement's old position (float32 round-off).
_MOVE_TOLERANCE = 1e-6


def move_points(points: np.ndarray, moved: Optional[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """``points`` with any at an ``old`` position of ``moved = (old, new)`` replaced by its ``new``."""
    if moved is None or len(moved[0]) == 0 or len(points) == 0:
        return points
    old, new = (np.asarray(a, dtype=np.float64).reshape(-1, 3) for a in moved)
    dist, near = KDTree(old).query(points, workers=-1)
    hit = dist < _MOVE_TOLERANCE
    out = points.copy()
    out[hit] = new[np.asarray(near)[hit]]
    return out


def _key(p: np.ndarray) -> tuple[float, float, float]:
    from realmgen.refine.settlements import position_key
    return position_key(p)


# ── the cost ─────────────────────────────────────────────────────────────────


def _river_nodes(rivers: Optional[RegionRivers], fields: RegionFields, raster: Raster) -> tuple[np.ndarray, np.ndarray]:
    """``(n²,)`` flow of the regional river on each raster node (0 off-river).

    Rasterised 4-connected so an 8-connected road cannot slip diagonally across unfording.
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
    face_km: float, bank_flow: Optional[np.ndarray] = None, channel_climb: float = 0.0,
):
    """Per-edge road cost in world land-cost units (one face-width step of base cost 1 costs 1).

    Entering a ``river_flow`` node pays the ford; a step between two of them gets
    the along-river discount — or, when the rivers are wider than a node (a
    settlement map: ``river_flow`` on the channel, ``bank_flow`` on a band beside
    it), costs as much more as climbing at a slope of ``channel_climb`` (km/km),
    and the discount goes to steps between two ``bank_flow`` nodes instead.
    """
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
    along_flow = river_flow if bank_flow is None else bank_flow
    on_bank = along_flow > 0.0
    along = np.maximum(
        1.0 - float(land.river_corridor_strength) * np.minimum(along_flow / ref, 1.0), float(land.river_min_mult))
    per_step = node_km / face_km
    climb_per_km = float(land.ruggedness_cost_per_km)

    def edge_cost(sub: np.ndarray, a: np.ndarray, b: np.ndarray, length: np.ndarray) -> np.ndarray:
        na, nb = sub[a], sub[b]
        cost = 0.5 * (base[na] + base[nb]) * length * per_step
        if bank_flow is not None:
            in_channel = climb_per_km * float(channel_climb) * length * node_km
            cost = np.where(on_river[na] & on_river[nb], cost + in_channel, cost)
        both = on_bank[na] & on_bank[nb]
        cost = np.where(both, cost * np.minimum(along[na], along[nb]), cost)
        entering = on_river[nb] & ~on_river[na]
        cost = cost + np.where(entering, ford[nb] * base[nb], 0.0)
        cost = cost + climb_per_km * np.abs(elevation[nb] - elevation[na])
        return np.where(water[na] | water[nb], np.inf, cost)

    return edge_cost


def _wet_nodes(raster: Raster, river_flow: np.ndarray) -> np.ndarray:
    """``(n²,)`` nodes a boat can use: refined water or a regional river."""
    return raster.sampled & (raster.water | (river_flow > 0.0))


def _water_cost(wet: np.ndarray, land_cost: float):
    """Per-edge cost of a river or lake leg: one per step on water, ``land_cost`` per step on land."""
    node = np.where(wet, 1.0, float(land_cost))

    def edge_cost(sub: np.ndarray, a: np.ndarray, b: np.ndarray, length: np.ndarray) -> np.ndarray:
        return 0.5 * (node[sub[a]] + node[sub[b]]) * length

    return edge_cost


# ── chaining the world's hops ────────────────────────────────────────────────


def _chains(
    edges: list[tuple[int, int, float]], n_nodes: int, stops: Optional[np.ndarray] = None,
) -> list[tuple[list[int], list[float]]]:
    """Cut an undirected edge list into maximal chains through degree-2 nodes.

    ``stops`` (``(n_nodes,)`` bool) also end chains: a routed chain lands exactly
    on its ends only.  Returns ``(nodes, edge flows)`` per chain; a closed loop is
    cut at one node.
    """
    stop = np.zeros(n_nodes, dtype=bool) if stops is None else np.asarray(stops, dtype=bool)
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
            if len(adj[cur]) != 2 or stop[cur]:
                return nodes, flows
            e2 = [x for _, x in adj[cur] if not used[x]]
            if not e2:
                return nodes, flows
            e = e2[0]

    for v in range(n_nodes):
        if len(adj[v]) != 2 or stop[v]:
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
    """Where the hop ``land_end`` → ``sea_end`` meets refined water: the last dry raster node before it,
    ``land_end`` itself when the water starts there, or None when the hop meets none on the raster."""
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
    # Up to where the hop leaves the raster.
    k = int(np.argmin(inside)) if not inside.all() else len(inside)
    idx = rows[:k] * n + cols[:k]
    hit = np.flatnonzero(raster.water[idx] & raster.sampled[idx])
    if hit.size == 0:
        return None
    if hit[0] == 0:
        return land_end
    return raster.positions[idx[hit[0] - 1]].copy()


def _nearest_node(raster: Raster, node: int, radius: float, allowed: np.ndarray) -> int:
    """``node``, or the nearest ``allowed`` node within ``radius`` nodes of it."""
    if allowed[node]:
        return node
    n = raster.n
    r0, c0 = divmod(node, n)
    rad = int(np.ceil(radius))
    rr, cc = np.mgrid[max(r0 - rad, 0):min(r0 + rad + 1, n), max(c0 - rad, 0):min(c0 + rad + 1, n)]
    cand = (rr * n + cc).ravel()
    ok = allowed[cand]
    if not ok.any():
        return node
    d = np.hypot(rr.ravel() - r0, cc.ravel() - c0)
    d = np.where(ok, d, np.inf)
    return int(cand[int(np.argmin(d))])


@dataclass
class _Stretch:
    """One in-raster stretch of a chain; ``real`` marks world vertices (others are edge cuts)."""

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
    """Per-edge flow along a routed ``path`` (P, 3), each hop's flow between the points nearest its ends."""
    near = np.argmin(np.linalg.norm(path[None, :, :] - verts[:, None, :], axis=2), axis=1)
    near = np.maximum.accumulate(near)
    out = np.full(len(path) - 1, flows[0] if len(flows) else 0.0)
    for k in range(len(flows)):
        out[near[k]:max(near[k + 1], near[k] + 1)] = flows[k]
    return out


# ── entry point ──────────────────────────────────────────────────────────────

# What a drawn world hop becomes: a road, a port hop (road to the coast, then
# water), a river/lake leg, a straight open-sea leg, or nothing (a river switch).
_LAND, _PORT, _WATER, _SEA, _SKIP = range(5)


def _hop_kinds(
    globe: "World", f0: np.ndarray, f1: np.ndarray, layers: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(kind, land0, ocean)`` per drawn hop from its ends' faces ``f0``/``f1`` and world ``layers``.

    ``land0``: the start face is land-passable; ``ocean``: per face (all faces).
    Without ``layers`` the kind is guessed from the faces, river faces counting as land.
    """
    mesh = globe.mesh
    cs = globe.get_feature("cost_surfaces")
    lp = (np.asarray(cs.land_passable, dtype=bool) if cs is not None
          else np.ones(len(mesh.faces), dtype=bool))
    ocean = np.asarray(IS_OCEAN.read(mesh), dtype=bool) if IS_OCEAN.present(mesh) else ~lp
    land0, land1 = lp[f0], lp[f1]
    if layers is None:
        kind = np.select([land0 & land1, land0 | land1], [_LAND, _PORT], _SEA)
    else:
        kind = np.select(
            [layers == ROUTE_LAND, (layers == ROUTE_TRANSFER) & (f0 == f1), layers == ROUTE_TRANSFER,
             ocean[f0] & ocean[f1]],
            [_LAND, _SKIP, _PORT, _SEA], _WATER)
    return kind, land0, ocean


def build_region_roads(
    fields: RegionFields,
    globe: "World",
    rivers: Optional[RegionRivers] = None,
    params: Optional[RegionRoadParams] = None,
    land: Optional["LandCostParams"] = None,
    danger_scale: float = 2.0,
    land_reach: Optional[float] = None,
    moved: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> RegionRoads:
    """Re-route the world's trade routes over ``fields``' refined terrain.

    ``land_reach`` is how far (chord) a settlement on refined water is moved onto
    land (default six texels).  ``moved = (old, new)`` maps settlements' world
    sites to where they now stand.
    """
    from worldgen.cost.builder import LandCostParams
    params = params or RegionRoadParams()
    land = land or LandCostParams()
    routes = globe.get_feature("trade_routes") or []
    empty = RegionRoads(region_id=fields.region_id, segments=[])
    if not routes:
        return empty
    layers = globe.get_feature("trade_route_layers")
    if layers is not None and len(layers) != len(routes):
        layers = None
    mesh = globe.mesh
    lo, step, n = raster_frame(fields, params.raster_texels, params.margin)
    hi = lo + step * (n - 1)
    spacing = fields.grid.texel_spacing_rad(fields.tex_size)
    land_reach = 6.0 * spacing if land_reach is None else float(land_reach)

    flows = np.array([float(r[2]) for r in routes])
    threshold = float(np.median(flows))
    starts = move_points(np.array([np.asarray(r[0], dtype=np.float64) for r in routes]), moved)
    ends = move_points(np.array([np.asarray(r[1], dtype=np.float64) for r in routes]), moved)
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

    sampler = GlobalSampler(mesh)
    idx = np.flatnonzero(drawn)
    f0, _, _ = sampler.locate(starts[idx])
    f1, _, _ = sampler.locate(ends[idx])
    kinds, land0, ocean = _hop_kinds(globe, f0, f1, None if layers is None else np.asarray(layers)[idx])

    radius_nodes = max(float(params.corridor_face_fraction) * face_rad / node_rad, 1.5)
    water_radius = max(float(params.water_corridor_face_fraction) * face_rad / node_rad, 1.5)

    def lines(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(np.array([s0[i], s1[i]]), np.array([t0[i], t1[i]])) for i in idx[mask]]

    reach = corridor_reach(lines((kinds == _LAND) | (kinds == _PORT)), lo, step, n, radius_nodes + 2.0)
    reach |= corridor_reach(lines((kinds == _WATER) | (kinds == _PORT)), lo, step, n, water_radius + 2.0)
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

    def straight(a: np.ndarray, b: np.ndarray, f: float) -> RouteSegment:
        return (np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32), f)

    land_edges: list[tuple[int, int, float]] = []
    water_edges: list[tuple[int, int, float]] = []
    pinned_nodes: set[int] = set()
    sea: list[RouteSegment] = []
    for j, i in enumerate(idx):
        a, b, f = starts[i], ends[i], float(flows[i])
        kind = kinds[j]
        if kind == _LAND:
            land_edges.append((node(a), node(b), f))
        elif kind == _PORT:
            land_end, sea_end, sea_face = (a, b, f1[j]) if land0[j] else (b, a, f0[j])
            # From the port as drawn: a settlement on refined water is moved ashore.
            port = snap_to_land(fields, land_end[None], land_reach)[0]
            coast = _coast_point(raster, fields, port, sea_end)
            if coast is None:
                land_edges.append((node(a), node(b), f))
                continue
            if coast is port:        # the water starts at the port: no road
                start = node(land_end)
            else:
                land_edges.append((node(land_end), node(coast), f))
                start = node(coast)
            pinned_nodes.add(start)
            if ocean[sea_face]:
                sea.append(straight(coast, sea_end, f))
            else:
                water_edges.append((start, node(sea_end), f))
        elif kind == _WATER:
            water_edges.append((node(a), node(b), f))
        elif kind == _SEA:
            sea.append(straight(a, b, f))

    river_flow, _ = _river_nodes(rivers, fields, raster)
    node_km = node_rad * EARTH_RADIUS_KM
    edge_cost = _road_cost(raster, river_flow, land, danger_scale, node_km, face_rad * EARTH_RADIUS_KM)

    pts = np.array(points).reshape(-1, 3)
    ends_on_land = snap_to_land(fields, pts, land_reach)
    segments: list[RouteSegment] = list(sea)
    out = RegionRoads(region_id=fields.region_id, segments=segments, undrawn=undrawn, sea=sea)
    smoothing = int(params.smoothing_iterations)
    # Routes pass through their settlements: chains end at them (a routed chain only lands on its ends).
    settlements = globe.get_feature("settlements") or []
    settled = {_key(np.asarray(c.position, dtype=np.float64)) for c in settlements}
    at_settlement = np.array([k in settled for k in keys], dtype=bool)
    for chain, chain_flows in _chains(land_edges, len(points), at_settlement):
        for path, edge_flows, _ in route_polyline(
            fields, raster, ends_on_land[chain], np.asarray(chain_flows), radius_nodes, edge_cost, smoothing,
        ):
            out.polylines.append(path)
            out.polyline_flows.append(np.asarray(edge_flows, dtype=np.float64))
            segments.extend(_segments(path, edge_flows))

    if water_edges:
        # A water leg is pinned to the land at a port (a settlement) or at a port hop's coast point.
        pinned = at_settlement.copy()
        pinned[list(pinned_nodes)] = True
        verts = np.where(pinned[:, None], ends_on_land, pts)
        wet = _wet_nodes(raster, river_flow)
        water_cost = _water_cost(wet, float(params.water_land_cost))
        for chain, chain_flows in _chains(water_edges, len(points), pinned):
            for path, edge_flows, pins in route_polyline(
                fields, raster, verts[chain], np.asarray(chain_flows), water_radius, water_cost, smoothing,
                wet=wet, pin=(bool(pinned[chain[0]]), bool(pinned[chain[-1]])),
            ):
                out.water_polylines.append(path)
                out.water_flows.append(np.asarray(edge_flows, dtype=np.float64))
                out.water_pins.append(pins)
                segments.extend(_segments(path, edge_flows))
    return out


def _segments(path: np.ndarray, edge_flows: np.ndarray) -> list[RouteSegment]:
    """A routed polyline as world-form ``(start, end, flow)`` segments."""
    return [(path[k].astype(np.float32), path[k + 1].astype(np.float32), float(edge_flows[k]))
            for k in range(len(path) - 1)]


def route_polyline(
    fields: RegionFields,
    raster: Raster,
    verts: np.ndarray,
    flows: np.ndarray,
    radius_nodes: float,
    edge_cost,
    smoothing_iterations: int,
    *,
    wet: Optional[np.ndarray] = None,
    pin: tuple[bool, bool] = (True, True),
) -> list[tuple[np.ndarray, np.ndarray, tuple[bool, bool]]]:
    """Re-route one polyline ``verts`` (K, 3) over ``raster``; ``(path, edge flows, pinned ends)`` per stretch.

    A road (``wet`` None) is routed between the dry nodes nearest its ends and
    ends exactly on its points; a stretch with no dry way through keeps the given
    line.  A water leg (``wet`` the ``(n²,)`` nodes a boat can use) starts from a
    ``pin``-ned end's own point and lands on it, and from the nearest wet node at
    any other end (a junction on the water, or the raster edge).
    """
    hi = raster.lo + raster.step * (raster.n - 1)
    s, t, _ = fields.geometry.ecef_to_local(verts)
    out = []
    for st in _stretches(s, t, verts, flows, raster.lo, hi):
        pins = (bool(st.real[0]) and pin[0], bool(st.real[-1]) and pin[1])
        anchors = np.where(st.real[:, None], st.ecef, fields.geometry.local_to_ecef(st.s, st.t))
        routed = _route_stretch(raster, st.s, st.t, radius_nodes, edge_cost, wet, pins)
        if routed is None:
            out.append((anchors, st.flows, pins))
            continue
        if wet is not None:
            routed = _pull_string(raster, routed, wet)
        path = raster.positions[routed].copy()
        if pins[0]:
            path[0] = st.ecef[0]
        if pins[1]:
            path[-1] = st.ecef[-1]
        path = np.stack(laplacian_smooth_sphere(list(path), iterations=smoothing_iterations)).astype(np.float64)
        out.append((path, _assign_flows(path, anchors, st.flows), pins))
    return out


def _line_nodes(raster: Raster, a: int, b: int) -> np.ndarray:
    """Raster nodes on the straight line from node ``a`` to node ``b``, two samples per step."""
    (ra, ca), (rb, cb) = divmod(a, raster.n), divmod(b, raster.n)
    m = max(int(np.ceil(2.0 * np.hypot(rb - ra, cb - ca))), 1)
    frac = np.arange(m + 1) / m
    return np.rint(ra + frac * (rb - ra)).astype(np.int64) * raster.n + np.rint(ca + frac * (cb - ca)).astype(np.int64)


def _pull_string(raster: Raster, nodes: np.ndarray, wet: np.ndarray) -> np.ndarray:
    """``nodes`` with every run a straight line over ``wet`` nodes can replace cut short.

    Grid paths cross open water in 45° doglegs; the boat sails straight.  A
    meandering river keeps its bends (the straight line would cross land).  The
    shortcuts are re-sampled one node apart, so smoothing only rounds the corners.
    """
    def clear(a: int, b: int) -> bool:
        return bool(wet[_line_nodes(raster, a, b)].all())

    out = [np.asarray(nodes[:1])]
    i, last = 0, len(nodes) - 1
    while i < last:
        j = last
        while j > i + 1 and not clear(int(nodes[i]), int(nodes[j])):
            j -= 1
        line = _line_nodes(raster, int(nodes[i]), int(nodes[j]))
        out.append(line[1:][np.r_[np.diff(line) != 0]])
        i = j
    return np.concatenate(out)


def _route_stretch(
    raster: Raster, s: np.ndarray, t: np.ndarray, radius_nodes: float, edge_cost,
    wet: Optional[np.ndarray] = None, pins: tuple[bool, bool] = (True, True),
) -> Optional[np.ndarray]:
    """Raster nodes of the least-cost path along one stretch.

    A road's ends move to the nearest dry node; a water leg's unpinned ends to the nearest wet one.
    """
    ends, _ = raster.index(np.array([s[0], s[-1]]), np.array([t[0], t[-1]]))
    allowed = raster.sampled & ~raster.water if wet is None else wet
    a, b = (int(e) if wet is not None and pinned else _nearest_node(raster, int(e), radius_nodes, allowed)
            for e, pinned in zip(ends, pins))
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
