"""Delta distributaries — the channels a big river splits into over its delta.

The world's delta stage (:mod:`worldgen.delta`) builds each delta as a plain of
global texels with one river carried across it to the sea; at ~14 km a texel
that is all the globe can say.  A region is fine enough to show the river
fanning out, so here the delta river's re-routed stem gets *distributaries*:

- **How many.**  ``base_count`` mouths in all at a mouth flow of 1000, plus
  ``per_decade`` for each tenfold more flow, at most ``max_count``.
- **Where they reach the sea.**  The plain's seaward edge (refined sea next to
  the refined ``delta`` plain) is split by its bearing from the apex, within
  ``max_angle_deg`` of the fan's axis, into that many equal sectors.  The main
  stem's own mouth holds one; each other sector gets a mouth near its middle.
- **Where they leave.**  Mouths are taken outward from the main stem on each
  side, each branching off the channel nearest it on that side — the main stem
  for the first, then the branch before it — at a random
  ``junction_min``–``junction_max`` of its distance from the apex.  Outer
  branches thus split off further upstream and the fan nests like a real one.
- **How they run.**  The least-cost path from junction to mouth on the
  rivers' raster, through a corridor around a gently bent guide line
  (``wander``): low ground is preferred like any river, water is avoided until
  the mouth, and ground beside another channel costs more (``separation``), so
  branches spread out instead of doubling one another.

A branch carries its mouth's share of the apex flow — ``main_share`` stays with
the main stem, the rest is split evenly — plus the shares of the branches that
leave it downstream.  The stem keeps its full path and flow; branches are extra
paths whose first vertex lies on their parent, so a local map re-routes each on
its own like any other river.  Deterministic per world seed, region and delta.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.ndimage import distance_transform_edt

from core.params import field, param_field as _f, paramclass
from region.corridor import Raster, node_spacing_rad, raster_frame, route_corridor, sample_raster
from region.upsample import EARTH_RADIUS_KM, RegionFields

if TYPE_CHECKING:
    from core.world import World
    from region.rivers import RegionRiverParams, RegionRivers, RiverPath

# Edge cost multiplier for a branch over water before its mouth.
_WATER_COST = 50.0
# A branch's guide is bent through one control point this far along it.
_BEND_AT = 0.5
# Mouths closer than this many raster nodes to one already taken are dropped.
_MIN_MOUTH_GAP = 3.0
# Two branches never leave within this many raster nodes of each other.
_MIN_JUNCTION_GAP = 2.5


@paramclass
class DistributaryParams:
    """Tuning for delta distributaries on regional maps."""

    enabled: bool = field(default=True, metadata={"label": "Distributaries"})
    # Mouths in all (main stem included) at mouth flow 1000, and more per tenfold flow.
    base_count: int = _f(3, label="Mouths at flow 1000", min=1, max=12)
    per_decade: float = _f(2.0, label="More mouths per 10× flow", min=0.0, max=6.0, step=0.5)
    max_count: int = _f(8, label="Max mouths", min=1, max=16)
    # Bearings from the apex a mouth may take, either side of the fan's axis.
    max_angle_deg: float = _f(70.0, label="Max mouth angle (°)", min=10.0, max=90.0, step=5.0)
    # Share of the apex flow the main stem keeps.
    main_share: float = _f(0.5, label="Main stem share", min=0.0, max=1.0, step=0.05)
    # Where a branch leaves, as a fraction of its mouth's distance from the apex.
    junction_min: float = _f(0.15, label="Junction min (apex→mouth)", min=0.0, max=1.0, step=0.05)
    junction_max: float = _f(0.6, label="Junction max (apex→mouth)", min=0.0, max=1.0, step=0.05)
    # Bend of the guide line (× branch length) and corridor half-width around it.
    wander: float = _f(0.15, label="Wander (× length)", min=0.0, max=0.5, step=0.05)
    corridor: float = _f(0.3, label="Corridor (× length)", min=0.05, max=1.0, step=0.05)
    # Extra cost beside another channel (decaying over ~2 raster nodes).
    separation: float = _f(3.0, label="Channel separation", min=0.0, max=10.0, step=0.5)
    valley_seeking: float = _f(1.0, label="Valley seeking", min=0.0, max=6.0, step=0.25)


@dataclass
class _Channel:
    nodes: np.ndarray        # (V,) raster nodes, apex/junction → mouth
    mouth_share: float       # its own mouth's share of the apex flow
    parent: int              # index into the channel list, -1 for the main stem
    children: list[int]


def mouth_count(flow: float, params: DistributaryParams) -> int:
    k = float(params.base_count) + float(params.per_decade) * np.log10(max(flow, 1.0) / 1000.0)
    return int(np.clip(round(k), 1, int(params.max_count)))


def _grid_rc(raster: Raster, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.divmod(np.asarray(nodes, dtype=np.int64), raster.n)
    return rows.astype(np.float64), cols.astype(np.float64)


def _path_nodes(raster: Raster, fields: RegionFields, positions: np.ndarray) -> np.ndarray:
    """Raster nodes along a polyline (consecutive repeats dropped; -1 off the raster)."""
    s, t, _ = fields.geometry.ecef_to_local(np.asarray(positions, dtype=np.float64))
    node, inside = raster.index(s, t)
    node = np.where(inside, node, -1)
    keep = np.concatenate([[True], node[1:] != node[:-1]])
    return node[keep]


def _front(raster: Raster, plain: np.ndarray) -> np.ndarray:
    """Sea nodes beside a plain node: where a distributary can reach the sea."""
    n = raster.n
    land = (plain & ~raster.water & raster.sampled).reshape(n, n)
    beside = np.zeros((n, n), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                beside |= np.roll(np.roll(land, dr, axis=0), dc, axis=1)
    return (beside.reshape(-1) & raster.ocean & raster.sampled)


def _pick_mouths(
    raster: Raster, front: np.ndarray, apex_rc: np.ndarray, axis: np.ndarray, main_mouth: int,
    k: int, radius_nodes: float, params: DistributaryParams, rng: np.random.Generator,
) -> list[int]:
    """Up to ``k − 1`` front nodes spread by bearing from the apex, away from the main mouth."""
    cand = np.flatnonzero(front)
    if cand.size == 0 or k < 2:
        return []
    rows, cols = _grid_rc(raster, cand)
    d = np.stack([cols - apex_rc[1], rows - apex_rc[0]], axis=1)
    dist = np.hypot(d[:, 0], d[:, 1])
    bearing = np.arctan2(axis[0] * d[:, 1] - axis[1] * d[:, 0], d @ axis)
    ok = (np.abs(bearing) <= np.deg2rad(float(params.max_angle_deg))) & (dist >= 0.25 * radius_nodes)
    cand, bearing = cand[ok], bearing[ok]
    if cand.size == 0:
        return []
    lo, hi = float(bearing.min()), float(bearing.max())
    width = (hi - lo) / k
    if width <= 0:
        return []
    mr, mc = _grid_rc(raster, np.array([main_mouth]))
    md = np.array([mc[0] - apex_rc[1], mr[0] - apex_rc[0]])
    main_bin = int(np.clip((np.arctan2(axis[0] * md[1] - axis[1] * md[0], md @ axis) - lo) // width, 0, k - 1))
    taken: list[int] = [main_mouth]
    for b in range(k):
        if b == main_bin:
            continue
        centre = lo + (b + 0.5 + rng.uniform(-0.25, 0.25)) * width
        pick = int(cand[np.argmin(np.abs(bearing - centre))])
        pr, pc = _grid_rc(raster, np.array(taken))
        qr, qc = _grid_rc(raster, np.array([pick]))
        if np.hypot(pr - qr[0], pc - qc[0]).min() >= _MIN_MOUTH_GAP:
            taken.append(pick)
    return taken[1:]


def _node_cost(raster: Raster, plain: np.ndarray, params: DistributaryParams) -> np.ndarray:
    """Per-node cost: valley-seeking over the plain, water dear."""
    on = raster.sampled & plain
    anomaly = raster.anomaly
    if on.any():
        mean, spread = float(anomaly[on].mean()), max(float(anomaly[on].std()), 1e-6)
    else:
        mean, spread = 0.0, 1.0
    cost = np.exp(np.clip(float(params.valley_seeking) * (anomaly - mean) / spread, -20.0, 20.0))
    return np.where(raster.water, cost * _WATER_COST, cost)


def _route_branch(
    raster: Raster, start: int, mouth: int, base_cost: np.ndarray, occupied: np.ndarray,
    params: DistributaryParams, rng: np.random.Generator,
) -> Optional[np.ndarray]:
    n = raster.n
    (sr,), (sc,) = _grid_rc(raster, np.array([start]))
    (mr,), (mc,) = _grid_rc(raster, np.array([mouth]))
    length = float(np.hypot(mr - sr, mc - sc))
    if length < 2.0:
        return None
    # A guide bent sideways through one control point.
    bend = float(params.wander) * length * rng.uniform(-1.0, 1.0)
    nr, nc = -(mc - sc) / length, (mr - sr) / length
    cr = sr + _BEND_AT * (mr - sr) + bend * nr
    cc = sc + _BEND_AT * (mc - sc) + bend * nc
    rows = np.array([sr, cr, mr])
    cols = np.array([sc, cc, mc])
    s = raster.lo + raster.step * cols
    t = raster.lo + raster.step * rows

    near = np.exp(-np.asarray(distance_transform_edt(~occupied.reshape(n, n))).reshape(-1) / 2.0)
    gr, gc = np.divmod(np.arange(n * n), n)
    leaving = np.hypot(gr - sr, gc - sc) <= 3.0           # free to leave its parent
    cost = base_cost * (1.0 + float(params.separation) * np.where(leaving, 0.0, near))

    def edge_cost(sub: np.ndarray, a: np.ndarray, b: np.ndarray, step: np.ndarray) -> np.ndarray:
        c = cost[sub]
        ok = raster.sampled[sub]
        out = step * 0.5 * (c[a] + c[b])
        return np.where(ok[a] & ok[b], out, np.inf)

    sinks = raster.ocean & raster.sampled
    radius = max(float(params.corridor) * length, 2.0)
    return route_corridor(raster, s, t, radius, edge_cost, sinks=sinks, mouth_radius=2.0)


def _fan_branches(
    fields: RegionFields, globe: "World", rivers: "RegionRivers", fan, fan_index: int,
    lo: float, step: float, n: int, params: "RegionRiverParams",
) -> list["RiverPath"]:
    from region.rivers import _to_path

    dp: DistributaryParams = params.distributaries
    node_km = node_spacing_rad(fields, step) * EARTH_RADIUS_KM
    radius_nodes = float(fan.radius_km) / max(node_km, 1e-9)
    s_a, t_a, valid = fields.geometry.ecef_to_local(np.asarray(fan.apex, dtype=np.float64)[None, :])
    if not valid[0]:
        return []
    apex_rc = np.array([(t_a[0] - lo) / step, (s_a[0] - lo) / step])
    reach = radius_nodes * 1.5 + 3.0
    gr, gc = np.divmod(np.arange(n * n), n)
    sampled = np.hypot(gr - apex_rc[0], gc - apex_rc[1]) <= reach
    if not sampled.any():
        return []
    raster = sample_raster(fields, globe, lo, step, n, sampled, keys=("delta",))
    plain = raster.values.get("delta", np.zeros(n * n)) > 0.5

    # The delta river: the largest-flow path passing the apex and ending at sea.
    best, best_flow, best_vertex = -1, 0.0, 0
    # A re-routed stem stays within a corridor (one global texel) of its world line.
    gw = globe.mesh.face_textures["elevation"].shape[2]
    global_texel = float(np.sqrt(np.mean(globe.mesh.face_areas))) / max(gw - 1, 1)
    for i, (pos, flows, _) in enumerate(rivers.paths):
        pts = np.asarray(pos, dtype=np.float64)
        gap = np.linalg.norm(pts - np.asarray(fan.apex, dtype=np.float64)[None, :], axis=1)
        j = int(np.argmin(gap))
        if gap[j] > 2.0 * global_texel:
            continue
        if float(np.max(flows)) > best_flow:
            best, best_flow, best_vertex = i, float(np.max(flows)), j
    if best < 0:
        return []
    stem_pos = np.asarray(rivers.paths[best][0], dtype=np.float64)[best_vertex:]
    stem = _path_nodes(raster, fields, stem_pos)
    stem = stem[stem >= 0]
    if stem.size < 3 or not raster.ocean[stem[-1]]:
        return []
    apex_flow = float(np.asarray(rivers.paths[best][1])[min(best_vertex, len(rivers.paths[best][1]) - 1)])

    k = mouth_count(apex_flow, dp)
    sr, sc = _grid_rc(raster, stem[[-1]])
    axis = np.array([sc[0] - apex_rc[1], sr[0] - apex_rc[0]])
    if np.linalg.norm(axis) < 1e-9:
        return []
    axis /= np.linalg.norm(axis)
    seed = (int(globe.seed or 0) * 1_000_003 + fields.region_id * 7_919 + fan_index) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    mouths = _pick_mouths(raster, _front(raster, plain), apex_rc, axis, int(stem[-1]), k, radius_nodes, dp, rng)
    if not mouths:
        return []

    # Each side's mouths outward from the main stem, branching off the channel before.
    def bearing(node: int) -> float:
        r, c = _grid_rc(raster, np.array([node]))
        d = np.array([c[0] - apex_rc[1], r[0] - apex_rc[0]])
        return float(np.arctan2(axis[0] * d[1] - axis[1] * d[0], d @ axis))

    main_b = bearing(int(stem[-1]))
    share = (1.0 - float(dp.main_share)) / max(len(mouths), 1)
    channels = [_Channel(nodes=stem, mouth_share=float(dp.main_share), parent=-1, children=[])]
    occupied = np.zeros(n * n, dtype=bool)
    for pos, _, _ in rivers.paths:
        nodes = _path_nodes(raster, fields, pos)
        occupied[nodes[nodes >= 0]] = True
    base_cost = _node_cost(raster, plain, dp)
    junctions: list[int] = []
    for side in (-1.0, 1.0):
        ordered = sorted((m for m in mouths if (bearing(m) - main_b) * side > 0),
                         key=lambda m: abs(bearing(m) - main_b))
        # This side's outermost channel, apex to sea: the main stem up to the
        # last junction, then the last branch — and which channel owns each node.
        trunk, owner = stem.copy(), np.zeros(len(stem), dtype=np.int64)
        for m in ordered:
            tr, tc = _grid_rc(raster, trunk)
            mr, mc = _grid_rc(raster, np.array([m]))
            target = rng.uniform(float(dp.junction_min), float(dp.junction_max)) * np.hypot(
                mr[0] - apex_rc[0], mc[0] - apex_rc[1])
            score = np.abs(np.hypot(tr - apex_rc[0], tc - apex_rc[1]) - target)
            # Never at the very end of the trunk (a branch must leave over land),
            # nor on another junction.
            score[max(len(trunk) - 3, 1):] = np.inf
            if junctions:
                jr, jc = _grid_rc(raster, np.array(junctions))
                gap = np.hypot(tr[:, None] - jr[None, :], tc[:, None] - jc[None, :]).min(axis=1)
                score[gap < _MIN_JUNCTION_GAP] = np.inf
            if not np.isfinite(score).any():
                continue
            j = int(np.argmin(score))
            path = _route_branch(raster, int(trunk[j]), m, base_cost, occupied, dp, rng)
            if path is None or len(path) < 3 or not raster.ocean[path[-1]]:
                continue
            parent = int(owner[j])
            channels.append(_Channel(nodes=path, mouth_share=share, parent=parent, children=[]))
            channels[parent].children.append(len(channels) - 1)
            occupied[path] = True
            junctions.append(int(trunk[j]))
            trunk = np.concatenate([trunk[:j], path])
            owner = np.concatenate([owner[:j], np.full(len(path), len(channels) - 1)])

    def served(i: int) -> float:
        return channels[i].mouth_share + sum(served(c) for c in channels[i].children)

    # Drawn polylines, each branch starting exactly on its parent's line.
    drawn: list[np.ndarray] = [stem_pos]
    out: list["RiverPath"] = []
    for i, ch in enumerate(channels[1:], start=1):
        flows = np.full(len(ch.nodes) - 1, apex_flow * served(i), dtype=np.float32)
        pts, f, in_water = _to_path(raster, ch.nodes, flows, params)
        parent_line = drawn[ch.parent]
        pts = pts.copy()
        pts[0] = parent_line[int(np.argmin(np.linalg.norm(parent_line - pts[0][None, :], axis=1)))]
        drawn.append(pts.astype(np.float64))
        out.append((pts, f, in_water))
    return out


def add_distributaries(
    fields: RegionFields, globe: "World", rivers: "RegionRivers", params: "RegionRiverParams",
) -> "RegionRivers":
    """``rivers`` with the distributaries of every delta whose apex is on this map."""
    from region.rivers import RegionRivers

    fans = globe.get_feature("delta_fans")
    if not params.distributaries.enabled or not fans or "delta" not in fields.textures or not rivers.paths:
        return rivers
    lo, step, n = raster_frame(fields, params.raster_texels, params.margin)
    hi = lo + step * (n - 1)
    apexes = np.stack([np.asarray(f.apex, dtype=np.float64) for f in fans])
    s, t, valid = fields.geometry.ecef_to_local(apexes)
    here = valid & (s >= lo) & (s <= hi) & (t >= lo) & (t <= hi)
    branches: list["RiverPath"] = []
    for i in np.flatnonzero(here):
        branches.extend(_fan_branches(fields, globe, rivers, fans[int(i)], int(i), lo, step, n, params))
    if not branches:
        return rivers
    ends = np.array([b[0][-1] for b in branches], dtype=np.float64).reshape(-1, 3)
    return RegionRivers(
        region_id=rivers.region_id,
        paths=list(rivers.paths) + branches,
        confluences=rivers.confluences,
        mouths=np.concatenate([np.asarray(rivers.mouths, dtype=np.float64).reshape(-1, 3), ends]),
        mouth_flows=np.concatenate([np.asarray(rivers.mouth_flows, dtype=np.float64),
                                    np.array([float(b[1][-1]) for b in branches])]),
        streams=rivers.streams,
    )
