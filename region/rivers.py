"""Regional rivers — the global network re-routed over the refined relief, plus
the small tributaries too minor for the world map.

The globe's rivers are facts at the global resolution: which basins drain where,
and how much water each stem carries.  A region keeps all of that and refines only
what the finer grid can actually see:

1. **Global rivers, re-routed.**  Each global river is re-traced through a
   *corridor* around its world path — about one global texel either side, the
   uncertainty the global grid leaves — as the least-cost path through the
   refined relief, where cost rises with the sub-grid elevation anomaly.  The
   river therefore follows the refined valley floor instead of the smoothed world
   polyline, but can never wander into another valley or change basin.  Stems are
   routed before their tributaries, and a tributary ends where it reaches its
   (already re-routed) stem, so confluences stay joined; a river ending at the
   sea ends at the *refined* coast.  Flow is copied from the nearest point of the
   world river.

2. **Sub-threshold tributaries.**  The refined relief is drained by a carving
   Priority-Flood (:func:`worldgen.hydrology.routing.priority_flood_tree`) with
   the sea, lakes, the re-routed rivers and the raster's rim as outlets, and
   accumulated precipitation-weighted in the *global* flow units — so
   ``tributary_min_flow`` reads on the same scale as the world's
   ``river_flow_threshold``.  Channels above it are cut into main stems the way
   the global stage does it (:func:`worldgen.hydrology.rivers.main_stem_paths`),
   each running down to the river, lake or coast it drains into.

Everything happens on one uniform raster over the region's local ``(s, t)``
frame, extended ``margin`` past the cell so streams crossing the cell edge have
some catchment behind them.  Inside the hexagons' textures the raster reads the
refined fields (nearest texel); beyond them it falls back to the global fields
read bilinearly.  Rivers leaving the raster are simply cut at its edge; the view
clips to the cell long before that.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import KDTree

from core.params import param_field as _f, paramclass
from region.grid import RegionHexGrid
from region.sampling import GlobalSampler
from region.upsample import EARTH_RADIUS_KM, RegionFields
from worldgen.hydrology.routing import accumulate_flow, priority_flood_tree
from worldgen.hydrology.rivers import main_stem_paths
from worldgen.hydrology.types import laplacian_smooth_sphere

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
    # Minimum accumulated flow for a regional tributary, in the world's flow units
    # (the world's default river threshold is 500).
    tributary_min_flow: float = _f(40.0, label="Tributary min flow", min=1.0, max=500.0)
    # Tributaries shorter than this many raster nodes are dropped as stubs.
    tributary_min_nodes: int = _f(6, label="Tributary min length (nodes)", min=2, max=50)
    smoothing_iterations: int = _f(8, label="Path smoothing iterations", min=0, max=20)


@dataclass
class RegionRivers:
    """Regional river polylines, in the world ``"rivers"`` feature's form."""

    region_id: int
    rivers: list[RiverPath]          # the global rivers, re-routed
    tributaries: list[RiverPath]     # regional channels below the world threshold

    @property
    def paths(self) -> list[RiverPath]:
        return self.rivers + self.tributaries


# ── the raster ───────────────────────────────────────────────────────────────


@dataclass
class _Raster:
    """A uniform ``n × n`` node grid over local ``[lo, lo + (n-1)·step]²``.

    Row 0 is the smallest ``t``.  All per-node arrays are flat ``(n²,)``.
    """

    lo: float
    step: float
    n: int
    positions: np.ndarray        # (P, 3) unit sphere
    elevation: np.ndarray        # (P,) refined, km
    anomaly: np.ndarray          # (P,) refined − coarse, km
    ocean: np.ndarray            # (P,) bool
    lake: np.ndarray             # (P,) bool
    weight: np.ndarray           # (P,) flow contributed, in global flow units

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
    _, hexes = KDTree(grid.centers).query(np.column_stack([s, t]), k=1)
    hexes = np.asarray(hexes, dtype=np.int64)
    half = grid.radius
    ds = s - (grid.centers[hexes, 0] - half)
    dt = (grid.centers[hexes, 1] + half) - t
    covered = (ds >= 0.0) & (ds <= 2.0 * half) & (dt >= 0.0) & (dt <= 2.0 * half)
    cols = np.clip((ds / (2.0 * half) * tex_size).astype(np.int64), 0, tex_size - 1)
    rows = np.clip((dt / (2.0 * half) * tex_size).astype(np.int64), 0, tex_size - 1)
    return hexes, rows, cols, covered


def _build_raster(fields: RegionFields, globe: "World", params: RegionRiverParams) -> _Raster:
    grid, tex = fields.grid, fields.tex_size
    step = float(params.raster_texels) * 2.0 * grid.radius / tex
    lo = -float(params.margin)
    n = int(np.ceil((1.0 - 2.0 * lo) / step)) + 1
    axis = lo + step * np.arange(n)
    tt, ss = np.meshgrid(axis, axis, indexing="ij")
    s, t = ss.ravel(), tt.ravel()
    positions = fields.geometry.local_to_ecef(s, t)

    mesh = globe.mesh
    sampler = GlobalSampler(mesh)
    faces, u, v = sampler.locate(positions)
    hexes, rows, cols, covered = _hex_texel_lookup(grid, tex, s, t)

    def read(key: str, channels_sum: bool = False) -> Optional[np.ndarray]:
        """Refined value where a hex texture covers the node, else the global one."""
        if key not in mesh.face_textures:
            return None
        coarse = sampler.sample(mesh.face_textures[key], faces, u, v)
        out = coarse.sum(axis=1) if channels_sum else coarse[:, 0]
        if key in fields.textures:
            fine = fields.textures[key][hexes, rows, cols].astype(np.float64)
            fine = fine.sum(axis=1) if channels_sum else fine[:, 0]
            out = np.where(covered, fine, out)
        return out

    coarse_elev = sampler.sample(mesh.face_textures["elevation"], faces, u, v)[:, 0]
    elevation = read("elevation")
    assert elevation is not None
    ocean_r, lake_r = read("ocean_mask"), read("lake_mask")
    ocean = ocean_r > 0.5 if ocean_r is not None else elevation <= 0.0
    lake = lake_r > 0.5 if lake_r is not None else np.zeros_like(ocean)

    # Flow in the global stage's units: each global texel contributes
    # precip / (mean global precip), so a raster node contributes that times its
    # share of a global texel's area.
    precip = read("precipitation", channels_sum=True)
    if precip is not None:
        mean_precip = max(float(np.asarray(mesh.face_textures["precipitation"]).sum(axis=-1).mean()), 1e-6)
        wetness = precip / mean_precip
    else:
        wetness = np.ones_like(elevation)
    frames = mesh.face_tangent_frames()
    gh = mesh.face_textures["elevation"].shape[1]
    global_texel = float(np.mean(np.asarray(frames[4]) * np.asarray(frames[5]))) / max(gh - 1, 1) ** 2
    spacing = np.linalg.norm(np.diff(positions.reshape(n, n, 3), axis=1), axis=-1).mean()
    node_area = float(spacing) ** 2

    return _Raster(
        lo=lo, step=step, n=n, positions=positions,
        elevation=elevation, anomaly=elevation - coarse_elev,
        ocean=ocean, lake=lake,
        weight=(wetness * node_area / global_texel).astype(np.float32),
    )


def _d8_csr(n: int) -> tuple[np.ndarray, np.ndarray]:
    """``(indptr, neighbours)`` D8 adjacency of an ``n × n`` grid, as CSR."""
    rows, cols = np.divmod(np.arange(n * n, dtype=np.int64), n)
    srcs, dsts = [], []
    for dr, dc in _D8:
        ok = (rows + dr >= 0) & (rows + dr < n) & (cols + dc >= 0) & (cols + dc < n)
        src = np.flatnonzero(ok)
        srcs.append(src)
        dsts.append(src + dr * n + dc)
    src, dst = np.concatenate(srcs), np.concatenate(dsts)
    indptr = np.zeros(n * n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n * n), out=indptr[1:])
    return indptr, dst[np.argsort(src, kind="stable")]


# ── re-routing the global rivers ─────────────────────────────────────────────


def _in_raster_runs(raster: _Raster, fields: RegionFields, positions: np.ndarray) -> list[tuple[int, int]]:
    """``[start, stop)`` vertex ranges of a polyline that lie on the raster."""
    s, t, valid = fields.geometry.ecef_to_local(np.asarray(positions, dtype=np.float64))
    _, inside = raster.index(s, t)
    inside &= valid
    runs: list[tuple[int, int]] = []
    start = None
    for i, ok in enumerate(inside.tolist() + [False]):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= 2:
                runs.append((start, i))
            start = None
    return runs


def _centreline(raster: _Raster, s: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grid ``(rows, cols)`` of a local polyline, densified to one sample per node."""
    x = (s - raster.lo) / raster.step
    y = (t - raster.lo) / raster.step
    seg = np.hypot(np.diff(x), np.diff(y))
    steps = np.maximum(np.ceil(seg * 2.0).astype(np.int64), 1)
    frac = np.concatenate([np.arange(k) / k for k in steps] + [np.zeros(1)])
    base = np.concatenate([np.full(k, i) for i, k in enumerate(steps)] + [np.array([len(x) - 1])])
    nxt = np.minimum(base + 1, len(x) - 1)
    xs = x[base] + frac * (x[nxt] - x[base])
    ys = y[base] + frac * (y[nxt] - y[base])
    return (np.clip(np.rint(ys), 0, raster.n - 1).astype(np.int64),
            np.clip(np.rint(xs), 0, raster.n - 1).astype(np.int64))


def _route_run(
    raster: _Raster,
    s: np.ndarray,
    t: np.ndarray,
    radius_nodes: float,
    sinks: np.ndarray,
    ends_here: bool,
    params: RegionRiverParams,
) -> Optional[np.ndarray]:
    """Least-cost node path along one in-raster stretch of a global river.

    The path is confined to nodes within ``radius_nodes`` of the world polyline.
    It starts at the stretch's first vertex and ends at its last — or, when the
    river's world mouth is in this stretch (``ends_here``), at the cheapest
    ``sinks`` node near that mouth (its re-routed stem, a lake, the refined sea).
    """
    n = raster.n
    rows, cols = _centreline(raster, s, t)
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


def _reroute_global_rivers(
    raster: _Raster,
    fields: RegionFields,
    world_rivers: Sequence,
    radius_nodes: float,
    params: RegionRiverParams,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Re-route every global river crossing the raster, stems first.

    Returns ``[(nodes, edge_flows)]`` and the mask of nodes they occupy.
    """
    pieces = []
    for entry in world_rivers:
        positions = np.asarray(entry[0], dtype=np.float64)
        flows = np.asarray(entry[1], dtype=np.float64)
        if len(positions) < 2:
            continue
        for a, b in _in_raster_runs(raster, fields, positions):
            s, t, _ = fields.geometry.ecef_to_local(positions[a:b])
            run_flows = flows[a:max(b - 1, a + 1)]
            pieces.append((float(run_flows.max()), s, t, run_flows, b == len(positions)))
    # Largest flow first, so a tributary's stem already exists when it arrives.
    pieces.sort(key=lambda p: -p[0])

    occupied = np.zeros(raster.n * raster.n, dtype=bool)
    routed: list[tuple[np.ndarray, np.ndarray]] = []
    for _, s, t, run_flows, ends_here in pieces:
        sinks = occupied | raster.ocean | raster.lake
        nodes = _route_run(raster, s, t, radius_nodes, sinks, ends_here, params)
        if nodes is None or len(nodes) < 2:
            continue
        routed.append((nodes, _run_flows(raster, nodes, s, t, run_flows)))
        occupied[nodes] = True
    return routed, occupied


# ── regional tributaries ─────────────────────────────────────────────────────


def _tributaries(
    raster: _Raster, outlets: np.ndarray, params: RegionRiverParams
) -> tuple[list[np.ndarray], np.ndarray]:
    """Channel paths below the world threshold, and the flow accumulation they use."""
    n = raster.n
    indptr, nbr_flat = _d8_csr(n)

    # Carving flood: water drains along the flood tree, so a depression in the
    # refined relief empties down its lowest route to the spill point rather
    # than straight across a filled flat.
    receiver, rank = priority_flood_tree(raster.elevation, outlets, indptr, nbr_flat)
    # accumulate_flow visits nodes in descending key order; latest-settled first
    # is upstream first.
    accum = accumulate_flow(
        receiver.astype(np.int32),
        np.full(n * n, -1, dtype=np.int32),
        np.ones(n * n, dtype=np.float32),
        np.zeros(n * n, dtype=np.float32),
        rank.astype(np.float64),
        raster.weight,
    )
    is_channel = (accum >= float(params.tributary_min_flow)) & ~outlets
    paths = main_stem_paths(receiver, accum, is_channel, outlets, int(params.tributary_min_nodes))
    return [np.asarray(p, dtype=np.int64) for p in paths], accum


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
    """Re-route the world's rivers through ``fields``' refined relief and add
    the region's sub-threshold tributaries.

    Needs the world's ``"rivers"`` feature for the first part; without it the
    region gets tributaries only.  Deterministic, like the refinement itself.
    """
    params = params or RegionRiverParams()
    raster = _build_raster(fields, globe, params)

    # Corridor half-width in raster nodes: global texel spacing over node spacing.
    mesh = globe.mesh
    gw = mesh.face_textures["elevation"].shape[2]
    global_spacing = float(np.sqrt(np.mean(mesh.face_areas))) / max(gw - 1, 1)
    node_spacing = float(np.linalg.norm(raster.positions[1] - raster.positions[0]))
    radius_nodes = max(float(params.corridor_global_texels) * global_spacing / node_spacing, 1.5)

    world_rivers = globe.get_feature("rivers") if globe.has_feature("rivers") else []
    routed, occupied = _reroute_global_rivers(raster, fields, world_rivers or [], radius_nodes, params)

    rim = np.zeros((raster.n, raster.n), dtype=bool)
    rim[[0, -1], :] = True
    rim[:, [0, -1]] = True
    outlets = raster.ocean | raster.lake | occupied | rim.ravel()
    trib_nodes, accum = _tributaries(raster, outlets, params)

    rivers = [_to_path(raster, nodes, flows, params) for nodes, flows in routed]
    tributaries = [
        _to_path(raster, nodes, np.maximum.accumulate(accum[nodes[:-1]]), params)
        for nodes in trib_nodes
    ]
    return RegionRivers(region_id=fields.region_id, rivers=rivers, tributaries=tributaries)


def river_length_km(path: RiverPath) -> float:
    """Great-circle length of a river polyline, km."""
    pts = np.asarray(path[0], dtype=np.float64)
    dots = np.clip(np.einsum("ij,ij->i", pts[:-1], pts[1:]), -1.0, 1.0)
    return float(np.arccos(dots).sum()) * EARTH_RADIUS_KM
