"""Corridor routing on a region raster — shared by the regional rivers and roads.

A uniform raster over the region's local ``(s, t)`` frame, ``margin`` past the
cell, sampled only where some corridor reaches; per world line, the least-cost
8-connected path within a set distance of its centreline.  The caller supplies
the cost (:func:`route_corridor`'s ``edge_cost``).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt, label
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import KDTree

from region.grid import RegionHexGrid
from region.sampling import GlobalSampler
from region.upsample import RegionFields

if TYPE_CHECKING:
    from core.world import World

# D8 steps, (drow, dcol).
_D8 = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))

# ``edge_cost(sub, a, b, length)``: per-edge cost of the corridor graph, where
# ``sub`` maps corridor node → raster node, ``a``/``b`` are the edges' corridor
# nodes and ``length`` their length in raster steps (1 or √2).  Non-finite = no edge.
EdgeCost = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass
class Raster:
    """A uniform ``n × n`` node grid over local ``[lo, lo + (n-1)·step]²``.

    Row 0 is the smallest ``t``.  Per-node arrays are flat ``(n²,)``; only
    ``sampled`` nodes hold data.
    """

    lo: float
    step: float
    n: int
    sampled: np.ndarray          # (P,) bool
    positions: np.ndarray        # (P, 3) unit sphere
    values: dict[str, np.ndarray]  # (P,) per sampled key
    anomaly: np.ndarray          # (P,) refined − coarse elevation, km
    ocean: np.ndarray            # (P,) bool
    lake: np.ndarray             # (P,) bool

    @property
    def water(self) -> np.ndarray:
        return self.ocean | self.lake

    @cached_property
    def water_bodies(self) -> np.ndarray:
        """``(n²,)`` connected water body (8-connected, over sampled nodes) of each node; 0 on land."""
        labelled = label(self.water.reshape(self.n, self.n), structure=np.ones((3, 3), dtype=bool))
        return np.asarray(labelled[0]).ravel()  # type: ignore[index]

    def index(self, s: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nearest ``(flat node, inside)`` for local points."""
        col = np.rint((np.asarray(s) - self.lo) / self.step).astype(np.int64)
        row = np.rint((np.asarray(t) - self.lo) / self.step).astype(np.int64)
        inside = (col >= 0) & (col < self.n) & (row >= 0) & (row < self.n)
        return np.where(inside, row * self.n + col, -1), inside


def raster_frame(fields: RegionFields, raster_texels: float, margin: float) -> tuple[float, float, int]:
    """``(lo, step, n)`` of a raster at ``raster_texels`` regional texels, ``margin`` past the cell."""
    step = float(raster_texels) * 2.0 * fields.grid.radius / fields.tex_size
    lo = -float(margin)
    return lo, step, int(np.ceil((1.0 - 2.0 * lo) / step)) + 1


def node_spacing_rad(fields: RegionFields, step: float) -> float:
    """Angular spacing of raster nodes ``step`` apart, at the cell centre."""
    pair = fields.geometry.local_to_ecef(np.array([0.5, 0.5 + step]), np.array([0.5, 0.5]))
    return float(np.linalg.norm(pair[1] - pair[0]))


def _hex_texel_lookup(
    grid: RegionHexGrid, tex_size: int, s: np.ndarray, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(hex, row, col, covered)`` of the nearest hex texel to each local point.

    Unlike :meth:`RegionHexGrid.texels_at` this reaches past the cell edge via
    the over-hang; ``covered`` is False where no texture reaches.
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


def sample_raster(
    fields: RegionFields, globe: "World", lo: float, step: float, n: int, sampled: np.ndarray,
    keys: Sequence[str] = (),
) -> Raster:
    """Read the refined relief, water and ``keys`` onto the ``sampled`` raster nodes.

    Nearest hex texel where a texture covers the node, else the globe (bilinear);
    region-only keys always read the nearest texel.  Multi-channel keys are
    summed (seasonal precipitation → annual).
    """
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
        fine = (fields.textures[key][hexes, hrows, hcols].astype(np.float64).sum(axis=-1)
                if key in fields.textures else None)
        if key not in mesh.face_textures:
            return fine
        out = sampler.sample(mesh.face_textures[key], faces, u, v).sum(axis=1)
        return out if fine is None else np.where(covered, fine, out)

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
    values: dict[str, np.ndarray] = {}
    for key in ("elevation", *keys):
        got = elevation if key == "elevation" else read(key)
        if got is None:
            continue
        full = np.zeros(size)
        full[idx] = got
        values[key] = full
    return Raster(lo=lo, step=step, n=n, sampled=sampled, positions=positions, values=values,
                  anomaly=anomaly, ocean=ocean, lake=lake)


def centreline(
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


def corridor_reach(
    lines: Sequence[tuple[np.ndarray, np.ndarray]], lo: float, step: float, n: int, reach_nodes: float,
) -> np.ndarray:
    """``(n²,)`` mask of the raster nodes within ``reach_nodes`` of any ``(s, t)`` line."""
    far = np.ones((n, n), dtype=bool)
    for s, t in lines:
        rows, cols = centreline(lo, step, n, s, t)
        far[rows, cols] = False
    if far.all():
        return np.zeros(n * n, dtype=bool)
    return (np.asarray(distance_transform_edt(far)) <= reach_nodes).ravel()


def route_corridor(
    raster: Raster,
    s: np.ndarray,
    t: np.ndarray,
    radius_nodes: float,
    edge_cost: EdgeCost,
    *,
    sinks: Optional[np.ndarray] = None,
    mouth_radius: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Least-cost raster-node path along one in-raster stretch of a world line.

    Confined to nodes within ``radius_nodes`` of ``(s, t)``; ends at the last
    vertex, or given ``sinks`` at the cheapest sink within ``mouth_radius`` of
    it (truncated at the first such sink touched).  None when no path exists.
    """
    n = raster.n
    rows, cols = centreline(raster.lo, raster.step, n, s, t)
    reach = float(mouth_radius) if sinks is not None and mouth_radius is not None else radius_nodes
    pad = int(np.ceil(max(reach, radius_nodes))) + 1
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, n)
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, n)
    h, w = r1 - r0, c1 - c0

    line = np.ones((h, w), dtype=bool)
    line[rows - r0, cols - c0] = False
    corridor = np.asarray(distance_transform_edt(line)) <= radius_nodes
    gr, gc = np.mgrid[r0:r1, c0:c1]
    near_mouth = np.hypot(gr - rows[-1], gc - cols[-1]) <= reach
    if sinks is not None:
        corridor |= near_mouth
    rr, cc = np.nonzero(corridor)
    sub = (rr + r0) * n + (cc + c0)                       # corridor → raster node
    local = np.full((h, w), -1, dtype=np.int64)
    local[rr, cc] = np.arange(len(sub))

    srcs, dsts, costs = [], [], []
    for dr, dc in _D8:
        rr2, cc2 = rr + dr, cc + dc
        ok = (rr2 >= 0) & (rr2 < h) & (cc2 >= 0) & (cc2 < w)
        ok[ok] = corridor[rr2[ok], cc2[ok]]
        a = local[rr[ok], cc[ok]]
        b = local[rr2[ok], cc2[ok]]
        c = np.asarray(edge_cost(sub, a, b, np.full(len(a), np.hypot(dr, dc))), dtype=np.float64)
        keep = np.isfinite(c)
        srcs.append(a[keep])
        dsts.append(b[keep])
        # Dijkstra drops zero-weight edges; keep them as a hair above zero.
        costs.append(np.maximum(c[keep], 1e-12))
    graph = csr_matrix(
        (np.concatenate(costs), (np.concatenate(srcs), np.concatenate(dsts))), shape=(len(sub), len(sub))
    )

    start = local[rows[0] - r0, cols[0] - c0]
    end = local[rows[-1] - r0, cols[-1] - c0]
    dist_to, pred = dijkstra(graph, indices=int(start), return_predecessors=True)
    targets = (np.flatnonzero(near_mouth[rr, cc] & sinks[sub]) if sinks is not None
               else np.empty(0, np.int64))
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
    # Stop at the first sink touched near the end; ones further back are travelled through.
    if sinks is not None:
        hit = np.flatnonzero((sinks[nodes] & near_mouth[rr[local_path], cc[local_path]])[1:])
        if hit.size:
            nodes = nodes[:hit[0] + 2]
    return nodes
