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
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import KDTree

from core.params import param_field as _f, paramclass
from region.grid import RegionHexGrid
from region.sampling import GlobalSampler
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

# D8 steps, (drow, dcol).
_D8 = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


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


# ── the raster ───────────────────────────────────────────────────────────────


@dataclass
class _Raster:
    """A uniform ``n × n`` node grid over local ``[lo, lo + (n-1)·step]²``.

    Row 0 is the smallest ``t``.  All per-node arrays are flat ``(n²,)``, but
    only the ``sampled`` nodes hold data — the ones a corridor can reach.
    """

    lo: float
    step: float
    n: int
    sampled: np.ndarray          # (P,) bool
    positions: np.ndarray        # (P, 3) unit sphere
    anomaly: np.ndarray          # (P,) refined − coarse elevation, km
    ocean: np.ndarray            # (P,) bool
    lake: np.ndarray             # (P,) bool

    def index(self, s: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nearest ``(flat node, inside)`` for local points."""
        col = np.rint((np.asarray(s) - self.lo) / self.step).astype(np.int64)
        row = np.rint((np.asarray(t) - self.lo) / self.step).astype(np.int64)
        inside = (col >= 0) & (col < self.n) & (row >= 0) & (row < self.n)
        return np.where(inside, row * self.n + col, -1), inside


def _hex_texel_lookup(
    grid: RegionHexGrid, tex_size: int, s: np.ndarray, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(hex, row, col, covered)`` of the nearest hex texel to each local point.

    Unlike :meth:`RegionHexGrid.texels_at` this reaches past the cell edge,
    through the hexagons' over-hanging textures; ``covered`` is False only where
    no texture reaches.
    """
    _, hexes = KDTree(grid.centers).query(np.column_stack([s, t]), k=1, workers=-1)
    hexes = np.asarray(hexes, dtype=np.int64)
    half = grid.radius
    ds = s - (grid.centers[hexes, 0] - half)
    dt = (grid.centers[hexes, 1] + half) - t
    covered = (ds >= 0.0) & (ds <= 2.0 * half) & (dt >= 0.0) & (dt <= 2.0 * half)
    cols = np.clip((ds / (2.0 * half) * tex_size).astype(np.int64), 0, tex_size - 1)
    rows = np.clip((dt / (2.0 * half) * tex_size).astype(np.int64), 0, tex_size - 1)
    return hexes, rows, cols, covered


def _raster_frame(fields: RegionFields, params: RegionRiverParams) -> tuple[float, float, int]:
    """``(lo, step, n)`` of the raster: regional texel spacing, ``margin`` past the cell."""
    step = float(params.raster_texels) * 2.0 * fields.grid.radius / fields.tex_size
    lo = -float(params.margin)
    return lo, step, int(np.ceil((1.0 - 2.0 * lo) / step)) + 1


def _sample_raster(
    fields: RegionFields, globe: "World", lo: float, step: float, n: int, sampled: np.ndarray,
) -> _Raster:
    """Read the refined relief and water onto the ``sampled`` raster nodes."""
    grid, tex = fields.grid, fields.tex_size
    idx = np.flatnonzero(sampled)
    rows, cols = np.divmod(idx, n)
    s, t = lo + step * cols, lo + step * rows
    pts = fields.geometry.local_to_ecef(s, t)

    mesh = globe.mesh
    sampler = GlobalSampler(mesh)
    faces, u, v = sampler.locate(pts)
    hexes, hrows, hcols, covered = _hex_texel_lookup(grid, tex, s, t)

    def read(key: str) -> Optional[np.ndarray]:
        """Refined value where a hex texture covers the node, else the global one."""
        if key not in mesh.face_textures:
            return None
        out = sampler.sample(mesh.face_textures[key], faces, u, v)[:, 0]
        if key in fields.textures:
            fine = fields.textures[key][hexes, hrows, hcols, 0].astype(np.float64)
            out = np.where(covered, fine, out)
        return out

    coarse = sampler.sample(mesh.face_textures["elevation"], faces, u, v)[:, 0]
    elevation = read("elevation")
    assert elevation is not None
    ocean_r, lake_r = read("ocean_mask"), read("lake_mask")

    size = n * n
    positions = np.zeros((size, 3))
    positions[idx] = pts
    anomaly = np.zeros(size)
    anomaly[idx] = elevation - coarse
    ocean = np.zeros(size, dtype=bool)
    ocean[idx] = ocean_r > 0.5 if ocean_r is not None else elevation <= 0.0
    lake = np.zeros(size, dtype=bool)
    if lake_r is not None:
        lake[idx] = lake_r > 0.5
    return _Raster(lo=lo, step=step, n=n, sampled=sampled, positions=positions,
                   anomaly=anomaly, ocean=ocean, lake=lake)


# ── re-routing ───────────────────────────────────────────────────────────────


def _centreline(
    lo: float, step: float, n: int, s: np.ndarray, t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Grid ``(rows, cols)`` of a local polyline, densified to one sample per node."""
    x = (s - lo) / step
    y = (t - lo) / step
    seg = np.hypot(np.diff(x), np.diff(y))
    steps = np.maximum(np.ceil(seg * 2.0).astype(np.int64), 1)
    frac = np.concatenate([np.arange(k) / k for k in steps] + [np.zeros(1)])
    base = np.concatenate([np.full(k, i) for i, k in enumerate(steps)] + [np.array([len(x) - 1])])
    nxt = np.minimum(base + 1, len(x) - 1)
    xs = x[base] + frac * (x[nxt] - x[base])
    ys = y[base] + frac * (y[nxt] - y[base])
    return (np.clip(np.rint(ys), 0, n - 1).astype(np.int64),
            np.clip(np.rint(xs), 0, n - 1).astype(np.int64))


def _route_run(
    raster: _Raster,
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
    """
    n = raster.n
    rows, cols = _centreline(raster.lo, raster.step, n, s, t)
    mouth_radius = 2.0 * radius_nodes
    pad = int(np.ceil(mouth_radius)) + 1
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, n)
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, n)
    h, w = r1 - r0, c1 - c0

    line = np.ones((h, w), dtype=bool)
    line[rows - r0, cols - c0] = False
    corridor = np.asarray(distance_transform_edt(line)) <= radius_nodes
    # The stem this river joins was re-routed within its own corridor, so it can
    # sit up to a corridor width off the world confluence: widen around the mouth.
    gr, gc = np.mgrid[r0:r1, c0:c1]
    near_mouth = np.hypot(gr - rows[-1], gc - cols[-1]) <= mouth_radius
    if ends_here:
        corridor |= near_mouth
    rr, cc = np.nonzero(corridor)
    sub = (rr + r0) * n + (cc + c0)                       # corridor → raster node
    local = np.full((h, w), -1, dtype=np.int64)
    local[rr, cc] = np.arange(len(sub))

    # Node cost: exponential in the sub-grid anomaly, normalised by its spread
    # here, so the river seeks the valley on a plain as much as in the mountains.
    anomaly = raster.anomaly[sub]
    spread = max(float(anomaly.std()), 1e-6)
    cost = np.exp(np.clip(float(params.valley_seeking) * (anomaly - anomaly.mean()) / spread, -20.0, 20.0))

    srcs, dsts, lens = [], [], []
    for dr, dc in _D8:
        rr2, cc2 = rr + dr, cc + dc
        ok = (rr2 >= 0) & (rr2 < h) & (cc2 >= 0) & (cc2 < w)
        ok[ok] = corridor[rr2[ok], cc2[ok]]
        a = local[rr[ok], cc[ok]]
        b = local[rr2[ok], cc2[ok]]
        srcs.append(a)
        dsts.append(b)
        lens.append(np.hypot(dr, dc) * 0.5 * (cost[a] + cost[b]))
    graph = csr_matrix(
        (np.concatenate(lens), (np.concatenate(srcs), np.concatenate(dsts))), shape=(len(sub), len(sub))
    )

    start = local[rows[0] - r0, cols[0] - c0]
    end = local[rows[-1] - r0, cols[-1] - c0]
    dist_to, pred = dijkstra(graph, indices=int(start), return_predecessors=True)
    # A river whose world mouth is here ends on whatever it drains into near
    # that mouth; only when nothing is in reach does it stop at the world point.
    targets = np.flatnonzero(near_mouth[rr, cc] & sinks[sub]) if ends_here else np.empty(0, np.int64)
    targets = targets[np.isfinite(dist_to[targets])]
    if targets.size == 0:
        targets = np.array([end])
    goal = int(targets[np.argmin(dist_to[targets])])
    if not np.isfinite(dist_to[goal]):
        return None

    path = [goal]
    while path[-1] != start:
        prev = int(pred[path[-1]])
        if prev < 0:
            return None
        path.append(prev)
    path.reverse()
    local_path = np.asarray(path, dtype=np.int64)
    nodes = sub[local_path]
    # Stop at the first sink the path touches near its mouth, however it got
    # there; a lake or stem passed upstream of that is flowed through.
    if ends_here:
        hit = np.flatnonzero((sinks[nodes] & near_mouth[rr[local_path], cc[local_path]])[1:])
        if hit.size:
            nodes = nodes[:hit[0] + 2]
    return nodes


def _run_flows(
    raster: _Raster, nodes: np.ndarray, s: np.ndarray, t: np.ndarray, flows: np.ndarray
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


def _corridor_reach(
    paths: list[_StreamPath], lo: float, step: float, n: int, radius_nodes: float,
) -> np.ndarray:
    """``(n²,)`` mask of the raster nodes any path's corridor or mouth disc can use."""
    far = np.ones((n, n), dtype=bool)
    for p in paths:
        rows, cols = _centreline(lo, step, n, p.s, p.t)
        far[rows, cols] = False
    if far.all():
        return np.zeros(n * n, dtype=bool)
    return (np.asarray(distance_transform_edt(far)) <= 2.0 * radius_nodes + 2.0).ravel()


# ── entry point ──────────────────────────────────────────────────────────────


def _fill_short_gaps(flags: np.ndarray, max_gap: int) -> np.ndarray:
    """``flags`` with runs of up to ``max_gap`` False entries between two True ones set."""
    out = flags.copy()
    true_at = np.flatnonzero(flags)
    for a, b in zip(true_at[:-1], true_at[1:]):
        if 1 < b - a <= max_gap + 1:
            out[a + 1:b] = True
    return out


def _to_path(raster: _Raster, nodes: np.ndarray, flows: np.ndarray, params: RegionRiverParams) -> RiverPath:
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
    lo, step, n = _raster_frame(fields, params)

    # Corridor half-width in raster nodes: global texel spacing over node spacing.
    mesh = globe.mesh
    gw = mesh.face_textures["elevation"].shape[2]
    global_spacing = float(np.sqrt(np.mean(mesh.face_areas))) / max(gw - 1, 1)
    pair = fields.geometry.local_to_ecef(np.array([0.5, 0.5 + step]), np.array([0.5, 0.5]))
    node_spacing = float(np.linalg.norm(pair[1] - pair[0]))
    radius_nodes = max(float(params.corridor_global_texels) * global_spacing / node_spacing, 1.5)

    paths = _stream_paths(streams, fields, lo, step, n, int(params.min_global_nodes))
    raster = _sample_raster(fields, globe, lo, step, n, _corridor_reach(paths, lo, step, n, radius_nodes))

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
    return RegionRivers(
        region_id=fields.region_id,
        paths=[_to_path(raster, nodes, flows, params) for nodes, flows in routed],
        confluences=raster.positions[ends].reshape(-1, 3),
    )


def river_length_km(path: RiverPath) -> float:
    """Great-circle length of a river polyline, km."""
    pts = np.asarray(path[0], dtype=np.float64)
    dots = np.clip(np.einsum("ij,ij->i", pts[:-1], pts[1:]), -1.0, 1.0)
    return float(np.arccos(dots).sum()) * EARTH_RADIUS_KM
