"""A town's or city's main streets: least-cost routes on a grid over the site, the secondary arterials
that fill the gaps, the wall street, and where the main streets aim (a town's long axis, a city's radials)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points

from settlegen.layout.geometry import _axes, _multi, chaikin
from settlegen.layout.params import SettlementLayoutParams


def _grid_edges(cost: np.ndarray, step_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(src, dst, weight)`` of an 8-connected grid over ``cost`` (per metre)."""
    n_r, n_c = cost.shape
    idx = np.arange(n_r * n_c).reshape(n_r, n_c)
    src, dst, w = [], [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r0, r1 = 0, n_r - dr
        c0, c1 = max(0, -dc), n_c - max(0, dc)
        a = idx[r0:r1, c0:c1]
        b = idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        e = 0.5 * (cost[r0:r1, c0:c1] + cost[r0 + dr:r1 + dr, c0 + dc:c1 + dc]) * math.hypot(dr, dc) * step_m
        src.append(a.reshape(-1))
        dst.append(b.reshape(-1))
        w.append(e.reshape(-1))
    return np.concatenate(src), np.concatenate(dst), np.concatenate(w)


def _csr(src: np.ndarray, dst: np.ndarray, w: np.ndarray, n: int):  # type: ignore[no-untyped-def]
    return coo_matrix((w, (src, dst)), shape=(n, n)).tocsr()


def _grid_graph(cost: np.ndarray, step_m: float):  # type: ignore[no-untyped-def]
    return _csr(*_grid_edges(cost, step_m), cost.size)


@dataclass
class RouteGrid:
    """The arterial routing grid: ``n`` × ``n`` cells of ``step`` m over the ``2·half`` frame
    (row 0 at the top), with a per-metre ``cost`` (slope, water, off the footprint, along roads)."""

    half: float
    n: int
    step: float
    xx: np.ndarray
    yy: np.ndarray
    cost: np.ndarray
    on_road: np.ndarray
    wet: np.ndarray

    @property
    def centres(self) -> np.ndarray:
        return np.column_stack([self.xx.reshape(-1), self.yy.reshape(-1)])

    def cell(self, p: np.ndarray) -> int:
        r = int(np.clip((self.half - p[1]) / self.step, 0, self.n - 1))
        q = int(np.clip((p[0] + self.half) / self.step, 0, self.n - 1))
        return r * self.n + q

    def cells_on(self, lines: Sequence) -> np.ndarray:  # type: ignore[type-arg]
        """``(n, n)`` bool: the cells ``lines`` (LineStrings or (k, 2) arrays) pass through."""
        out = np.zeros((self.n, self.n), dtype=bool)
        for xy in lines:
            line = xy if isinstance(xy, LineString) else LineString(np.asarray(xy, dtype=np.float64)) \
                if len(xy) >= 2 else None
            if line is None or line.is_empty:
                continue
            pts = shapely.get_coordinates(shapely.line_interpolate_point(
                line, np.append(np.arange(0.0, line.length, 0.4 * self.step), line.length)))
            r = ((self.half - pts[:, 1]) / self.step).astype(int)
            q = ((pts[:, 0] + self.half) / self.step).astype(int)
            ok = (r >= 0) & (r < self.n) & (q >= 0) & (q < self.n)
            out[r[ok], q[ok]] = True
        return out


def route_grid(
    half: float, slope: np.ndarray, wet: MultiPolygon, footprint: MultiPolygon, roads: Sequence[np.ndarray],
    params: SettlementLayoutParams,
) -> RouteGrid:
    """The :class:`RouteGrid` of a ``2·half`` frame (``slope``: the frame's (T, T) slope)."""
    step = float(params.route_step_m)
    n = max(int(math.ceil(2.0 * half / step)), 2)
    step = 2.0 * half / n
    c = -half + (np.arange(n) + 0.5) * step
    xx, yy = np.meshgrid(c, c[::-1])
    t = slope.shape[0]
    rows = np.clip(((half - yy) / (2.0 * half) * t).astype(int), 0, t - 1)
    cols = np.clip(((xx + half) / (2.0 * half) * t).astype(int), 0, t - 1)
    cost = 1.0 + float(params.slope_weight) * slope[rows, cols]
    grid = RouteGrid(half, n, step, xx, yy, cost, np.zeros((n, n), dtype=bool), np.zeros((n, n), dtype=bool))
    grid.on_road = on_road = grid.cells_on([xy for xy in roads if len(xy) >= 2])
    if not wet.is_empty:
        grid.wet = is_wet = shapely.contains_xy(wet, xx, yy)
        cost = np.where(is_wet & ~on_road, cost * float(params.water_cost), cost)
    if not footprint.is_empty:
        cost = np.where(shapely.contains_xy(footprint, xx, yy), cost, cost * float(params.outside_cost))
    grid.cost = np.where(on_road, cost * float(params.road_follow), cost)
    return grid


def _smooth_path(xy: np.ndarray, step: float) -> LineString:
    return LineString(chaikin(np.asarray(LineString(xy).simplify(0.6 * step).coords), 3))


def route_arterials(
    half: float, slope: np.ndarray, wet: MultiPolygon, footprint: MultiPolygon, roads: Sequence[np.ndarray],
    source: np.ndarray, targets: Sequence[np.ndarray], params: SettlementLayoutParams,
) -> list[LineString]:
    """Least-cost paths from ``source`` to each target over a ``route_step_m`` grid of the
    ``2·half`` frame (``slope``: the frame's (T, T) slope), merged into one smoothed network."""
    if not targets:
        return []
    grid = route_grid(half, slope, wet, footprint, roads, params)
    step, cell = grid.step, grid.cell
    graph = _grid_graph(grid.cost, step)

    s = cell(source)
    _, pred = dijkstra(graph, directed=False, indices=s, return_predecessors=True)
    centres = grid.centres
    segs: list[LineString] = []
    for target in targets:
        k = cell(target)
        path = [k]
        while path[-1] != s and pred[path[-1]] >= 0:
            path.append(int(pred[path[-1]]))
        if path[-1] != s:
            continue
        xy = centres[path]
        segs.append(LineString([target, xy[0]]))
        segs.extend(LineString([xy[i], xy[i + 1]]) for i in range(len(xy) - 1))
    segs.append(LineString([centres[s], source]))
    segs = [g for g in segs if g.length > 1e-6]
    if not segs:
        return []
    merged = shapely.line_merge(shapely.union_all(segs))
    # The source and targets stay ends (a merge can run a line through one, and smoothing would cut it).
    pins = np.asarray([source, *targets], dtype=np.float64).reshape(-1, 2)
    out = []
    for line in getattr(merged, "geoms", [merged]):
        xy = np.asarray(line.coords)
        hit = np.flatnonzero(np.min(np.linalg.norm(xy[1:-1, None, :] - pins[None], axis=2), axis=1) < 1e-6) + 1 \
            if len(xy) > 2 else np.zeros(0, dtype=np.int64)
        for a, b in zip([0, *hit], [*hit, len(xy) - 1]):
            piece = LineString(xy[a:b + 1])
            if piece.length > 1e-6:
                out.append(_smooth_path(np.asarray(piece.coords), step))
    return out


def _chain(pred: np.ndarray, k: int) -> list[int]:
    """Cells from ``k`` back to the Dijkstra source."""
    path = [k]
    while pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    return path


def infill_arterials(
    grid: RouteGrid, network: Sequence[LineString], serve: MultiPolygon, footprint: MultiPolygon, spacing: float,
    walls: Optional[MultiLineString] = None, gates: Optional[np.ndarray] = None,
    avoid: Optional[MultiPolygon] = None, max_streets: int = 400,
) -> list[LineString]:
    """Through streets filling the gaps in ``network`` until every cell of ``serve`` lies
    within ``spacing / 2`` of an arterial.  Each starts at the cell farthest from one, runs
    least-cost to the nearest arterial and on, the other way (> 110° off), to another
    arterial or out to ``footprint``'s edge.  Dry ground only; through ``walls`` only at
    ``gates``; ``avoid`` (citadel, precinct) dear."""
    from scipy.ndimage import binary_dilation, distance_transform_edt

    n, step = grid.n, grid.step
    centres = grid.centres
    if serve.is_empty or not network:
        return []
    wet = (grid.wet & ~grid.on_road).reshape(-1)
    cost = grid.cost
    if avoid is not None and not avoid.is_empty:
        cost = np.where(shapely.contains_xy(avoid, grid.xx, grid.yy), 4.0 * cost, cost)
    src, dst, w = _grid_edges(cost, step)
    keep = ~(wet[src] | wet[dst])
    covered = grid.cells_on(network).reshape(-1)
    walled = walls is not None and not walls.is_empty
    if walled:
        near = np.asarray(shapely.dwithin(walls, shapely.points(centres), 1.5 * step), dtype=bool)
        # A cell is on an arterial across the wall from it only by rasterising: not a way in.
        test = np.flatnonzero(covered & near)
        links = shapely.shortest_line(shapely.points(centres[test]), shapely.union_all(list(network)))
        covered[test[shapely.intersects(links, walls)]] = False
        test = np.flatnonzero(keep & (near[src] | near[dst]))
        segs = shapely.linestrings(np.stack([centres[src[test]], centres[dst[test]]], axis=1))
        cross = shapely.intersects(segs, walls)
        if gates is not None and len(gates):
            mid = 0.5 * (centres[src[test]] + centres[dst[test]])
            at_gate = np.min(np.linalg.norm(mid[:, None, :] - gates[None], axis=2), axis=1) < 1.5 * step
            cross &= ~at_gate
        keep[test[cross]] = False
    graph = _csr(src[keep], dst[keep], w[keep], n * n)

    in_foot = np.asarray(shapely.contains_xy(footprint, grid.xx, grid.yy), dtype=bool)
    grown = np.asarray(binary_dilation(in_foot, structure=np.ones((3, 3), dtype=bool)), dtype=bool)
    edge = (grown & ~in_foot).reshape(-1) & ~wet
    todo = (shapely.contains_xy(serve, grid.xx, grid.yy).reshape(-1) & ~wet)
    limit = 4.0 * spacing
    cos_apart = math.cos(math.radians(110.0))
    paths: list[tuple[list[int], Optional[bool]]] = []    # cells, far end on an arterial (None: a dead end)
    for _ in range(max_streets):
        dist = np.asarray(distance_transform_edt(~covered.reshape(n, n))).reshape(-1) * step
        score = np.where(todo, dist, -1.0)
        k = int(np.argmax(score))
        if score[k] < 0.5 * spacing:
            break
        d, pred = dijkstra(graph, directed=False, indices=k, limit=limit, return_predecessors=True)
        reach = np.isfinite(d)
        net = np.flatnonzero(covered & reach)
        if not len(net):
            todo &= np.linalg.norm(centres - centres[k], axis=1) > 0.5 * spacing
            continue
        a = int(net[np.argmin(d[net])])
        cells = _chain(pred, a)                      # a … k
        va = centres[a] - centres[k]
        ends = np.flatnonzero((covered | edge) & reach)
        v = centres[ends] - centres[k]
        norm = np.linalg.norm(v, axis=1)
        ok = ((norm > 2.0 * step) | edge[ends]) & ((v @ va) < cos_apart * norm * float(np.linalg.norm(va))) \
            & (np.linalg.norm(centres[ends] - centres[a], axis=1) > 0.5 * spacing)   # not a loop
        on_net: Optional[bool] = None
        if ok.any():
            b = int(ends[ok][np.argmin(d[ends[ok]])])
            leg = _chain(pred, b)[::-1]              # k … b, cut where it first meets an arterial or the edge
            stop = next(i for i in range(1, len(leg)) if covered[leg[i]] or edge[leg[i]])
            cells += leg[1:stop + 1]
            on_net = bool(covered[leg[stop]])
        if (len(cells) - 1) * step < 0.3 * spacing:
            todo &= np.linalg.norm(centres - centres[k], axis=1) > 0.5 * spacing
            continue
        paths.append((cells, on_net))
        covered[cells] = True

    out: list[LineString] = []
    net_geom = shapely.union_all(list(network))
    edge_line = footprint.boundary
    for cells, on_net in paths:
        xy = centres[cells].copy()
        xy[0] = shapely.get_coordinates(nearest_points(net_geom, Point(xy[0]))[0])[0]
        if on_net is not None:
            far = net_geom if on_net else edge_line
            xy[-1] = shapely.get_coordinates(nearest_points(far, Point(xy[-1]))[0])[0]
        line = _smooth_path(xy, step)
        if line.length > 1e-6:
            out.append(line)
            net_geom = net_geom.union(line)
    return out


def wall_street(core: MultiPolygon, walls: MultiLineString, offset: float, citadel: MultiPolygon,
                min_m: float = 40.0) -> list[LineString]:
    """The street just inside the walls: the core's outline ``offset`` m in, where there
    is a wall beside it (not along the water), round the citadel's outside."""
    if core.is_empty or walls.is_empty:
        return []
    inner = _multi(core.buffer(-offset, quad_segs=8))
    loops = MultiLineString([LineString(g.exterior.coords) for g in inner.geoms])
    line = loops.intersection(walls.buffer(offset + 2.0))
    if not citadel.is_empty:
        line = line.difference(citadel.buffer(offset))
    parts = shapely.get_parts(shapely.line_merge(shapely.union_all(
        [g for g in shapely.get_parts(line) if isinstance(g, LineString)] or [LineString()])))
    return [g for g in parts if isinstance(g, LineString) and g.length >= min_m]


def main_axis(
    part: Polygon, centre: np.ndarray, gates: Sequence[np.ndarray], max_deg: float,
    axis: Optional[np.ndarray] = None,
) -> np.ndarray:
    """The footprint's long axis (or ``axis``, the way of a through road) — or the way to
    a gate within ``max_deg`` of it."""
    u = np.asarray(axis, dtype=np.float64) if axis is not None else _axes(part)[1]
    best: Optional[tuple[float, np.ndarray]] = None
    for g in gates:
        d = np.asarray(g, dtype=np.float64) - centre
        n = float(np.linalg.norm(d))
        if n < 1.0:
            continue
        c = abs(float(d @ u)) / n
        if c >= math.cos(math.radians(max_deg)) and (best is None or c > best[0]):
            best = (c, d / n)
    return best[1] if best is not None else u


def radial_targets(
    part: Polygon, centre: np.ndarray, taken: Sequence[np.ndarray], n: int,
    shore: Optional[MultiPolygon] = None, min_spacing_m: float = 500.0, shore_m: float = 100.0,
) -> list[np.ndarray]:
    """Points on ``part``'s landward edge (farther than ``shore_m`` from ``shore``), each
    as far along it as can be from the ways already out (``taken``, then the points
    chosen), until ``n`` ways leave or none is ``max(landward edge / 2n, min_spacing_m)``
    from the rest.  With none taken, the first is the long axis's end."""
    radius = math.sqrt(part.area / math.pi)
    edge: BaseGeometry = part.exterior
    if shore is not None and not shore.is_empty:
        edge = edge.difference(shore.buffer(shore_m))
    lines = [g for g in shapely.get_parts(edge) if isinstance(g, LineString) and g.length > 0]
    length = sum(g.length for g in lines)
    if not lines or n <= 0:
        return []
    step = max(length / 2000.0, 10.0)
    cand = np.concatenate([np.asarray([(p.x, p.y) for p in shapely.line_interpolate_point(
        g, np.arange(0.5 * step, g.length, step))]).reshape(-1, 2) for g in lines])
    cand = cand[np.linalg.norm(cand - centre, axis=1) > 0.3 * radius]
    # Greedy farthest-point picks along a curve fall short of an even split: half of it.
    spacing = max(0.5 * length / n, float(min_spacing_m))
    ways = [np.asarray(p, dtype=np.float64) for p in taken]
    out: list[np.ndarray] = []
    while len(cand) and len(ways) < n:
        if ways:
            d = np.min(np.linalg.norm(cand[:, None, :] - np.asarray(ways)[None], axis=2), axis=1)
            k = int(np.argmax(d))
            if d[k] < spacing:
                break
        else:
            _, u, _, _, _ = _axes(part)
            k = int(np.argmax(np.abs((cand - centre) @ u)))
        ways.append(cand[k])
        out.append(cand[k])
    return out
