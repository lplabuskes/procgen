"""Borders that follow big rivers, run by run.

The plain polity borders (the label field before any river handling) are traced
into chains between junctions. Along each chain, a run of border points that
stays within reach of one big river is replaced by that river's stretch: "border
points 13–20 become river points 1–5". The river is picked per run — the
longest stretch of the chain a single river shadows wins, the rest of the chain
is free for another — so a border never flips between two parallel rivers.

Within a run, a border point within ``snap_fraction × reach`` of the river is on
it (the **core**: between the first and last such point the new line is the
river itself); further out it is pulled part way, ``w = (reach − d) / ((1 − g) ·
reach)``, held non-decreasing toward the core, so the line eases onto and off
the river along the border's own course instead of on a straight connector.

Each run claims the land between its border run and the new line (the polygon
of the two, even–odd, so a river that crosses the plain border gives alternating
lobes), a margin of ``edge_km`` round the old border, and a band of ``clean_km``
either side of the new line. There, a point of either of the run's two polities
takes the polity on its side of the new line: the one that was on the border's
left, or its right. The margin hides any mismatch between the traced border and
the field's own boundary; the band clears enclaves of the field beside the river,
which a trade of labels inside the polygon only would cut in two. Third
polities are never touched. That is a pure function of position, so the fill and
the traced lines agree and face seams match. A point is claimed by the first run
holding it only.

The plain border is the same at every tier; only the river differs. Region and
local rivers carry the index of the world river they re-route (``RegionRivers.
path_rivers``), so a tier swaps in its own drawing of the very river each run
follows (``RiverSwaps.with_rivers``).

A run is dropped when its polygon would take in a third polity, and split around
any settlement it would hand to the neighbour.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import KDTree

from realmgen.refine.barriers import flow_ramp
from worldgen.cost.surfaces import EARTH_RADIUS_KM

# Nearest river segments examined per point.
_K_NEAREST = 16
# Cells per side of a swap polygon's acceleration grid.
_GRID = 128
# Points per relabel call above which a swap builds its side grid.
_GRID_MIN_POINTS = 20_000
# How far either side of the border run its two polities are probed (which is left).
_SIDE_PROBE_KM = 2.0


@dataclass
class RiverFollowParams:
    reach_km: float = 70.0          # a border point this close to a full-strength river may follow it
    min_flow: float = 300.0         # reach ramps (log) from 0 at min_flow …
    full_flow: float = 4000.0       # … to reach_km at full_flow
    snap_fraction: float = 0.6      # within this share of the reach a point is on the river
    min_run_km: float = 40.0        # shortest river stretch (core) a border is moved onto
    parallel_ratio: float = 0.5     # core river stretch ≥ this × its ends' distance apart (else it's a crossing)
    max_gap: int = 2                # out-of-reach border points tolerated inside a run
    max_third_share: float = 0.05   # share of a polygon's land held by a third polity that drops a run
    sample_km: float = 4.0          # polygon sampling step for the third-polity check
    settlement_clear_km: float = 25.0  # a run is split this far either side of a settlement it would flip
    view_reach_km: float = 30.0     # the tier's drawing of a river counts at a core end / blend foot this close
    border_step_km: float = 1.0     # the swap's border edge is densified to this and bisected onto the boundary
    edge_km: float = 8.0            # margin round the old border taken by side (~ a lattice edge: what tracing cut off)
    clean_km: float = 20.0          # band either side of the new line taken by side (clears enclaves)


# ── geometry ─────────────────────────────────────────────────────────────────


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-15)


def _tangent_basis(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([0.0, 0.0, 1.0]) if abs(c[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = _unit(np.cross(c, ref))
    return e1, np.cross(c, e1)


def _gnomonic(p: np.ndarray, c: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    q = p / np.maximum(p @ c, 1e-9)[:, None]
    return np.column_stack([q @ e1, q @ e2])


def _even_odd(pts: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """``(P,)`` inside test of 2-D ``pts`` against the closed 2-D ``ring`` (even–odd crossings)."""
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    return np.asarray(MplPath(ring).contains_points(pts), dtype=bool)


class _RingIndex:
    """A grid over a ring's bounding box: cells clear of every edge (and their neighbours'
    edges) take their centre's inside-ness; only points in the rest are tested exactly."""

    def __init__(self, ring: np.ndarray, n: int = _GRID) -> None:
        self.ring = ring
        self.lo = ring.min(axis=0)
        span = np.maximum(ring.max(axis=0) - self.lo, 1e-15)
        self.n = n
        self.cell = span / n
        # Cells an edge passes through, sampled at a quarter cell, then dilated by one.
        a, b = ring, np.roll(ring, -1, axis=0)
        k = np.maximum(1, np.ceil(np.abs((b - a) / self.cell).max(axis=1) * 4).astype(np.int64))
        t = np.concatenate([np.arange(m) / m for m in k])
        e = np.repeat(np.arange(len(a)), k)
        samples = a[e] + t[:, None] * (b[e] - a[e])
        ij = self._cells(samples)
        edge = np.zeros((n + 2, n + 2), dtype=bool)
        edge[ij[:, 0] + 1, ij[:, 1] + 1] = True
        near = edge.copy()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                near[1:-1, 1:-1] |= edge[1 + di:n + 1 + di, 1 + dj:n + 1 + dj]
        self.near = near[1:-1, 1:-1]
        centres = self.lo + (np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), -1) + 0.5) * self.cell
        self.inside = _even_odd(centres.reshape(-1, 2), ring).reshape(n, n)

    def _cells(self, pts: np.ndarray) -> np.ndarray:
        return np.clip(np.floor((pts - self.lo) / self.cell).astype(np.int64), 0, self.n - 1)

    def contains(self, pts: np.ndarray) -> np.ndarray:
        out = np.zeros(len(pts), dtype=bool)
        hi = self.lo + self.cell * self.n
        box = np.flatnonzero(np.all((pts >= self.lo) & (pts <= hi), axis=1))
        if box.size == 0:
            return out
        ij = self._cells(pts[box])
        exact = self.near[ij[:, 0], ij[:, 1]]
        out[box[~exact]] = self.inside[ij[~exact, 0], ij[~exact, 1]]
        out[box[exact]] = _even_odd(pts[box[exact]], self.ring)
        return out


def _arc_lengths(pos: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])


def _polyline_query(x: np.ndarray, line: np.ndarray, tree: KDTree):
    """Nearest segment of ``line`` per point: ``(seg, t, dist)`` (k nearest midpoints, exact test)."""
    m = len(line) - 1
    k = min(_K_NEAREST, m)
    _, nn = tree.query(x, k=k)
    nn = np.asarray(nn).reshape(len(x), k)
    a, b = line[nn], line[nn + 1]
    ab = b - a
    t = np.clip(((x[:, None, :] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-30), 0.0, 1.0)
    dist = np.linalg.norm(x[:, None, :] - (a + t[..., None] * ab), axis=-1)
    j = np.argmin(dist, axis=1)
    rows = np.arange(len(x))
    return nn[rows, j], t[rows, j], dist[rows, j]


def _polyline_distance(x: np.ndarray, line: np.ndarray, tree: KDTree) -> np.ndarray:
    return _polyline_query(x, line, tree)[2]


def _polyline_side(x: np.ndarray, line: np.ndarray, tree: KDTree):
    """``(dist, side, interior)`` of points against an open polyline on the sphere.

    ``side`` > 0 on the left (walking the line); at a vertex the two segments'
    normals are averaged (a pseudo-normal, so corners don't flip sides).
    ``interior`` is False where the nearest point is an end of the line, past which
    no side is defined.
    """
    seg, t, dist = _polyline_query(x, line, tree)
    normal = _unit(np.cross(line[:-1], line[1:]))
    m = len(normal)
    n = normal[seg].copy()
    at_start = (t <= 0.0) & (seg > 0)
    at_end = (t >= 1.0) & (seg < m - 1)
    n[at_start] = normal[seg[at_start]] + normal[seg[at_start] - 1]
    n[at_end] = normal[seg[at_end]] + normal[seg[at_end] + 1]
    vertex = np.where(t[:, None] >= 1.0, line[seg + 1], line[seg])
    side = ((x - vertex) * n).sum(-1)
    side = np.where(at_start | at_end, side, (x * normal[seg]).sum(-1))
    interior = ~(((seg == 0) & (t <= 0.0)) | ((seg == m - 1) & (t >= 1.0)))
    return dist, side, interior


class _SideGrid:
    """A grid over ``[lo, hi]`` (gnomonic, the points being labelled), classified at its corners.

    A cell whose four corners agree, that no line, old border or polygon edge
    crosses and whose corners all sit more than a cell diagonal from every
    threshold (band, edge, the line itself) answers for every point inside it;
    the rest are classified exactly.
    """

    def __init__(self, sw: "RiverSwap", lo: np.ndarray, hi: np.ndarray, n: int = 2 * _GRID) -> None:
        self.lo = lo
        span = np.maximum(hi - lo, 1e-15)
        self.n = n
        self.cell = span / n
        g = np.stack(np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij"), -1).reshape(-1, 2)
        uv = self.lo + g * self.cell
        x = _unit(sw.centre[None, :] + uv[:, :1] * sw.e1[None, :] + uv[:, 1:2] * sw.e2[None, :])
        claim, left, d = sw._classify(x)
        diag = float(np.hypot(*self.cell)) * 1.5      # gnomonic stretch margin
        clear = ((np.abs(d["line"] - sw.band) > diag) & (np.abs(d["border"] - sw.edge) > diag)
                 & (d["line"] > diag)).reshape(n + 1, n + 1)
        claim = claim.reshape(n + 1, n + 1)
        left = left.reshape(n + 1, n + 1)
        interior = d["interior"].reshape(n + 1, n + 1)

        def corners(a: np.ndarray) -> tuple[np.ndarray, ...]:
            return a[:-1, :-1], a[1:, :-1], a[:-1, 1:], a[1:, 1:]

        def agree(a: np.ndarray) -> np.ndarray:
            c = corners(a)
            return (c[0] == c[1]) & (c[0] == c[2]) & (c[0] == c[3])

        ok = agree(claim) & agree(left) & agree(interior) & np.logical_and.reduce(corners(clear))
        # Polygon edges crossing a cell (sampled at a quarter cell, dilated by one).
        ring = sw.ring2d
        a, b = ring, np.roll(ring, -1, axis=0)
        k = np.maximum(1, np.ceil(np.abs((b - a) / self.cell).max(axis=1) * 4).astype(np.int64))
        t = np.concatenate([np.arange(m) / m for m in k])
        e = np.repeat(np.arange(len(a)), k)
        ij = self._cells(a[e] + t[:, None] * (b[e] - a[e]))
        edge = np.zeros((n + 2, n + 2), dtype=bool)
        edge[ij[:, 0] + 1, ij[:, 1] + 1] = True
        near = edge.copy()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                near[1:-1, 1:-1] |= edge[1 + di:n + 1 + di, 1 + dj:n + 1 + dj]
        self.ok = ok & ~near[1:-1, 1:-1]
        self.claim = claim[:-1, :-1]
        self.left = left[:-1, :-1]

    def _cells(self, uv: np.ndarray) -> np.ndarray:
        return np.clip(np.floor((uv - self.lo) / self.cell).astype(np.int64), 0, self.n - 1)

    def lookup(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(exact, claim, left)``: ``exact`` marks points the grid can't answer."""
        hi = self.lo + self.cell * self.n
        inside = np.all((uv >= self.lo) & (uv <= hi), axis=1)
        ij = self._cells(uv)
        ok = inside & self.ok[ij[:, 0], ij[:, 1]]
        claim = np.where(ok, self.claim[ij[:, 0], ij[:, 1]], False)
        left = np.where(ok, self.left[ij[:, 0], ij[:, 1]], False)
        return ~ok, claim, left


# ── swaps ────────────────────────────────────────────────────────────────────


@dataclass
class RiverSwap:
    """One border run replaced by a river stretch: ``left`` / ``right`` of the new line.

    The new line is the border points before ``ia`` blended toward their river
    ``foot`` by ``weight``, the ``river`` core, then the points after ``ib`` blended.
    """

    left: int                 # polity left of the border run (walking its order), so of the new line
    right: int
    band: float               # radians: clean band either side of the new line
    edge: float               # radians: margin round the old border
    depth: int
    border: np.ndarray        # (n, 3) the plain border run
    weight: np.ndarray        # (n,) blend toward the foot, 1 on the core [ia, ib]
    foot: np.ndarray          # (n, 3) nearest point on the river of each border point
    ia: int
    ib: int
    river: np.ndarray         # (m, 3) the core's river stretch, from foot[ia] to foot[ib]
    river_id: int             # index of that river in the world ``"rivers"``

    def __post_init__(self) -> None:
        w = self.weight[:, None]
        blend = _unit((1.0 - w) * self.border + w * self.foot)
        self.line = np.vstack([blend[:self.ia], self.river, blend[self.ib + 1:]])
        ring = np.vstack([self.border, self.line[::-1]])
        self.centre = _unit(ring.mean(axis=0))
        self.e1, self.e2 = _tangent_basis(self.centre)
        self.ring2d = _gnomonic(ring, self.centre, self.e1, self.e2)
        self.radius = float(np.linalg.norm(ring - self.centre, axis=1).max())
        self.reach = self.radius + max(self.band, self.edge)
        self._index: Optional[_RingIndex] = None
        self._line_tree: Optional[KDTree] = None
        self._border_tree: Optional[KDTree] = None

    def inside(self, points: np.ndarray) -> np.ndarray:
        """``(P,)`` points inside the polygon of the border run and the new line."""
        pts = np.asarray(points, dtype=np.float64)
        near = np.linalg.norm(pts - self.centre, axis=1) <= self.radius
        out = np.zeros(len(pts), dtype=bool)
        idx = np.flatnonzero(near)
        if idx.size:
            if self._index is None:
                self._index = _RingIndex(self.ring2d)
            out[idx] = self._index.contains(_gnomonic(pts[idx], self.centre, self.e1, self.e2))
        return out

    def _classify(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """Exact ``(claim, left, distances)`` for points ``x`` (3-D)."""
        if self._line_tree is None:
            self._line_tree = KDTree(0.5 * (self.line[:-1] + self.line[1:]))
            self._border_tree = KDTree(0.5 * (self.border[:-1] + self.border[1:]))
        assert self._border_tree is not None
        d_line, side, interior = _polyline_side(x, self.line, self._line_tree)
        d_border = _polyline_distance(x, self.border, self._border_tree)
        claim = interior & ((d_line < self.band) | (d_border < self.edge) | self.inside(x))
        return claim, side > 0.0, {"line": d_line, "border": d_border, "interior": interior}

    def relabel(self, points: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(labels, claimed)``: the run's polities by side of the new line where it claims a point."""
        pts = np.asarray(points, dtype=np.float64)
        out = np.asarray(labels).copy()
        claimed = np.zeros(len(pts), dtype=bool)
        idx = np.flatnonzero((np.linalg.norm(pts - self.centre, axis=1) <= self.reach)
                             & ((out == self.left) | (out == self.right)))
        if idx.size == 0:
            return out, claimed
        x = pts[idx]
        claim = np.zeros(idx.size, dtype=bool)
        left = np.zeros(idx.size, dtype=bool)
        exact = np.ones(idx.size, dtype=bool)
        if idx.size > _GRID_MIN_POINTS:
            uv = _gnomonic(x, self.centre, self.e1, self.e2)
            exact, claim, left = _SideGrid(self, uv.min(axis=0), uv.max(axis=0)).lookup(uv)
        if exact.any():
            c, lf, _ = self._classify(x[exact])
            claim[exact], left[exact] = c, lf
        rows = idx[claim]
        out[rows] = np.where(left[claim], self.left, self.right)
        claimed[rows] = True
        return out, claimed

    def moved(self, river: np.ndarray, foot: np.ndarray) -> "RiverSwap":
        return RiverSwap(self.left, self.right, self.band, self.edge, self.depth, self.border, self.weight, foot,
                         self.ia, self.ib, river, self.river_id)


class RiverSwaps:
    """The swaps of a world (or a tier's re-projection of them), applied to point labels."""

    def __init__(self, swaps: Sequence[RiverSwap], centroids: np.ndarray, params: RiverFollowParams) -> None:
        self.swaps = list(swaps)
        self.params = params
        self._centroids = centroids
        self._index_faces()

    def _index_faces(self) -> None:
        """Per swap, the owner faces whose points it may hold (centroid within its cap + a face)."""
        n = len(self._centroids)
        face_r = 2.0 * np.sqrt(4.0 * np.pi / max(n, 1))
        tree = KDTree(self._centroids)
        self._faces = np.zeros((len(self.swaps), n), dtype=bool)
        for i, sw in enumerate(self.swaps):
            self._faces[i, tree.query_ball_point(sw.centre, sw.reach + face_r)] = True
        self._any: np.ndarray = np.asarray(self._faces.any(axis=0))

    def __len__(self) -> int:
        return len(self.swaps)

    def apply(self, points: np.ndarray, owner: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """``labels`` with each swap's claims applied (the first swap claiming a point wins)."""
        out = labels.copy()
        cand = np.flatnonzero(self._any[owner])
        if cand.size == 0:
            return out
        pts, own = points[cand], owner[cand]
        done = np.zeros(cand.size, dtype=bool)
        for i, sw in enumerate(self.swaps):
            sel = np.flatnonzero(self._faces[i, own] & ~done)
            if sel.size == 0:
                continue
            new, claimed = sw.relabel(pts[sel], out[cand[sel]])
            done[sel[claimed]] = True
            out[cand[sel]] = new
        return out

    def with_rivers(self, segments: Optional[np.ndarray]) -> "RiverSwaps":
        """These swaps with their river replaced by the tier's drawing of it.

        ``segments`` are the tier's packed rivers with their world river ids
        (``pack_river_segments(..., river_ids=...)``, column 7). Each swap's core
        becomes its own river's drawn path between its nearest points to the core's
        ends (``_drawn_core``); the blend feet move onto that river too. Without ids, or
        where the river isn't drawn, a swap keeps the world line.
        """
        if segments is None or len(segments) == 0 or not self.swaps:
            return self
        seg = np.asarray(segments, dtype=np.float64)
        if seg.shape[1] < 8:
            return self
        a, b = _unit(seg[:, :3]), _unit(seg[:, 3:6])
        rid = seg[:, 7].astype(np.int64)
        joined = np.zeros(len(a), dtype=bool)
        joined[1:] = np.linalg.norm(a[1:] - b[:-1], axis=1) < 1e-9
        chain = np.cumsum(~joined)
        by_river = {int(r): np.flatnonzero(rid == r) for r in np.unique(rid[rid >= 0])}
        reach = self.params.view_reach_km / EARTH_RADIUS_KM
        out = []
        for sw in self.swaps:
            idx = by_river.get(sw.river_id)
            river = None if idx is None else _drawn_core(sw.river, a, b, chain, idx, reach)
            if river is None:
                out.append(sw)
                continue
            foot = _onto(sw.foot, a[idx], b[idx], reach)
            foot[sw.ia] = river[0]
            foot[sw.ib] = river[-1]
            out.append(sw.moved(river, foot))
        other = RiverSwaps.__new__(RiverSwaps)
        other.swaps, other.params, other._centroids = out, self.params, self._centroids
        other._index_faces()
        return other


def _param_on(line: np.ndarray, arc: np.ndarray, tree: KDTree,
              x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(u, inside, dist)``: arc position of each point's nearest point on ``line``
    (``inside`` False past its ends) and the distance to it."""
    seg, t, dist = _polyline_query(x, line, tree)
    u = arc[seg] + t * (arc[seg + 1] - arc[seg])
    m = len(line) - 1
    inside = ~(((seg == 0) & (t <= 0.0)) | ((seg == m - 1) & (t >= 1.0)))
    return u, inside, dist


def _drawn_core(core: np.ndarray, a: np.ndarray, b: np.ndarray, chain: np.ndarray,
                idx: np.ndarray, reach: float) -> Optional[np.ndarray]:
    """The world ``core`` replaced by its river's drawn path ``idx`` (segments of one river).

    The drawn path alongside the core (segments within ``reach`` of it, projecting
    inside it; the path with most such, from its first to its last) is cut at its
    nearest points to the core's two ends: "core points 1–9 become drawn points
    40–95". An end the drawing doesn't come within ``reach`` of (off the view)
    is cut at the path's own end instead, and the core carries on from there on the
    world line.
    """
    if len(core) < 2:
        return None
    arc = _arc_lengths(core)
    tree = KDTree(0.5 * (core[:-1] + core[1:]))
    _, in_a, d_a = _param_on(core, arc, tree, a[idx])
    _, in_b, d_b = _param_on(core, arc, tree, b[idx])
    beside = idx[in_a & in_b & (d_a <= reach) & (d_b <= reach)]
    if beside.size == 0:
        return None
    ids, counts = np.unique(chain[beside], return_counts=True)
    on = beside[chain[beside] == ids[np.argmax(counts)]]
    segs = np.arange(on.min(), on.max() + 1)
    path = np.vstack([a[segs], b[segs[-1]][None]])
    u_ends, _, _ = _param_on(core, arc, tree, path[[0, -1]])
    if u_ends[0] > u_ends[1]:
        path, u_ends = path[::-1], u_ends[::-1]
    path_arc = _arc_lengths(path)
    path_tree = KDTree(0.5 * (path[:-1] + path[1:]))
    seg, t, dist = _polyline_query(core[[0, -1]], path, path_tree)
    at = path_arc[seg] + t * (path_arc[seg + 1] - path_arc[seg])
    head = dist[0] > reach                     # the drawing starts after the core does
    tail = dist[1] > reach
    s0 = 0.0 if head else float(at[0])
    s1 = float(path_arc[-1]) if tail else float(at[1])
    if s1 <= s0:
        return None
    inner = np.flatnonzero((path_arc > s0) & (path_arc < s1))
    ss = np.concatenate([[s0], path_arc[inner], [s1]])
    drawn = np.column_stack([np.interp(ss, path_arc, path[:, k]) for k in range(3)])
    out = [core[arc < u_ends[0]]] if head else []
    out.append(drawn)
    if tail:
        out.append(core[arc > u_ends[1]])
    return _unit(np.vstack(out))


def _onto(x: np.ndarray, a: np.ndarray, b: np.ndarray, reach: float) -> np.ndarray:
    """``x`` moved to its nearest point on segments ``a→b`` when within ``reach``."""
    line_tree = KDTree(0.5 * (a + b))
    k = min(_K_NEAREST, len(a))
    _, nn = line_tree.query(x, k=k)
    nn = np.asarray(nn).reshape(len(x), k)
    sa, sb = a[nn], b[nn]
    ab = sb - sa
    t = np.clip(((x[:, None, :] - sa) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-30), 0.0, 1.0)
    foot = sa + t[..., None] * ab
    dist = np.linalg.norm(x[:, None, :] - foot, axis=-1)
    j = np.argmin(dist, axis=1)
    rows = np.arange(len(x))
    near = dist[rows, j] <= reach
    return np.where(near[:, None], _unit(foot[rows, j]), x)


# ── run selection ────────────────────────────────────────────────────────────


@dataclass
class _Near:
    """One river beside a chain, per chain point."""

    mask: np.ndarray      # within reach
    s: np.ndarray         # arc length of the foot along the river
    d: np.ndarray         # distance (inf out of reach)
    reach: np.ndarray     # reach at the foot
    foot: np.ndarray      # (K, 3)


class _Rivers:
    """World river segments eligible to carry a border, with per-river arc length."""

    def __init__(self, rivers: Optional[Iterable], p: RiverFollowParams) -> None:
        self.pos: List[np.ndarray] = []
        self.feature_index: List[int] = []     # index into the world "rivers" of each kept river
        self.arc: List[np.ndarray] = []
        a_parts, b_parts, rid_parts, sid_parts, reach_parts = [], [], [], [], []
        for fi, path in enumerate(rivers or []):
            pos = _unit(np.asarray(path[0], dtype=np.float64))
            flow = np.asarray(path[1], dtype=np.float64)
            in_lake = np.asarray(path[2], dtype=bool) if len(path) > 2 else np.zeros(flow.size, dtype=bool)
            m = min(flow.size, len(pos) - 1)
            if m <= 0:
                continue
            r = len(self.pos)
            self.feature_index.append(fi)
            self.pos.append(pos[:m + 1])
            self.arc.append(_arc_lengths(pos[:m + 1]))
            ramp = flow_ramp(flow[:m], min_flow=p.min_flow, full_flow=p.full_flow)
            reach = p.reach_km / EARTH_RADIUS_KM * ramp
            idx = np.flatnonzero((reach > 0.0) & ~in_lake[:m])
            a_parts.append(pos[idx])
            b_parts.append(pos[idx + 1])
            rid_parts.append(np.full(idx.size, r))
            sid_parts.append(idx)
            reach_parts.append(reach[idx])
        if a_parts and sum(len(x) for x in a_parts):
            self.a = np.concatenate(a_parts)
            self.b = np.concatenate(b_parts)
            self.rid = np.concatenate(rid_parts)
            self.sid = np.concatenate(sid_parts)
            self.reach = np.concatenate(reach_parts)
        else:
            self.a = self.b = np.zeros((0, 3))
            self.rid = self.sid = np.zeros(0, dtype=np.int64)
            self.reach = np.zeros(0)
        self.tree = KDTree(0.5 * (self.a + self.b)) if len(self.a) else None
        self.half = 0.5 * float(np.linalg.norm(self.b - self.a, axis=1).max(initial=0.0))
        self.max_reach = float(self.reach.max(initial=0.0))

    def near(self, x: np.ndarray) -> dict:
        """``{river: _Near}`` for the rivers within reach of any of the points ``x``."""
        if self.tree is None:
            return {}
        k = min(_K_NEAREST, len(self.a))
        _, nn = self.tree.query(x, k=k, distance_upper_bound=self.max_reach + self.half)
        nn = np.asarray(nn).reshape(len(x), k)
        valid = nn < len(self.a)
        s = np.where(valid, nn, 0)
        a, b = self.a[s], self.b[s]
        ab = b - a
        t = np.clip(((x[:, None, :] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-30), 0.0, 1.0)
        foot = a + t[..., None] * ab
        dist = np.linalg.norm(x[:, None, :] - foot, axis=-1)
        holds = valid & (dist < self.reach[s])
        out: dict = {}
        rid = self.rid[s]
        rows = np.arange(len(x))
        for r in np.unique(rid[holds]):
            d = np.where(holds & (rid == r), dist, np.inf)
            j = np.argmin(d, axis=1)
            dd = d[rows, j]
            inside = np.isfinite(dd)
            seg = s[rows, j]
            arc = self.arc[int(r)]
            sid = np.where(inside, self.sid[seg], 0)    # rows out of reach: unused
            out[int(r)] = _Near(
                mask=inside, s=arc[sid] + t[rows, j] * (arc[sid + 1] - arc[sid]), d=dd,
                reach=np.where(inside, self.reach[seg], 0.0), foot=_unit(foot[rows, j]),
            )
        return out

    def stretch(self, r: int, s0: float, s1: float) -> np.ndarray:
        """River ``r`` from arc ``s0`` to ``s1`` (either direction)."""
        pos, arc = self.pos[r], self.arc[r]
        lo, hi = min(s0, s1), max(s0, s1)
        inner = np.flatnonzero((arc > lo) & (arc < hi))
        ss = np.concatenate([[lo], arc[inner], [hi]])
        pts = _unit(np.column_stack([np.interp(ss, arc, pos[:, k]) for k in range(3)]))
        return pts if s0 <= s1 else pts[::-1]


def _runs(mask: np.ndarray, max_gap: int) -> List[tuple[int, int]]:
    """Inclusive ``(i0, i1)`` runs of ``mask``, bridging gaps of up to ``max_gap``; ends on True."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    out = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        if i - prev - 1 > max_gap:
            out.append((start, prev))
            start = int(i)
        prev = int(i)
    out.append((start, prev))
    return out


def _bump(st: dict, key: str) -> None:
    st[key] = st.get(key, 0) + 1


@dataclass
class _Run:
    """A candidate: chain points ``[i0, i1]`` onto ``river``, core ``[ia, ib]``."""

    river: int
    i0: int
    i1: int
    ia: int
    ib: int
    span: float


def _core(nr: _Near, i0: int, i1: int, g: float) -> Optional[tuple[int, int]]:
    full = np.flatnonzero(nr.mask[i0:i1 + 1] & (nr.d[i0:i1 + 1] <= g * nr.reach[i0:i1 + 1]))
    if full.size == 0:
        return None
    return i0 + int(full[0]), i0 + int(full[-1])


def build_river_swaps(
    chains: Sequence[tuple],
    rivers: Optional[Iterable],
    plain_labels: Callable[[np.ndarray], np.ndarray],
    is_water_at: Callable[[np.ndarray], np.ndarray],
    settlements: np.ndarray,
    centroids: np.ndarray,
    params: Optional[RiverFollowParams] = None,
    stats: Optional[dict] = None,
) -> RiverSwaps:
    """Swaps for the plain border ``chains`` (``trace_lattice_chains``) onto ``rivers``.

    ``plain_labels``/``is_water_at`` label sphere points; ``settlements`` is
    ``(S, 3)`` positions no swap may hand to the neighbour. ``stats`` (if given)
    counts candidates and why they were dropped.
    """
    p = params or RiverFollowParams()
    st = stats if stats is not None else {}
    riv = _Rivers(rivers, p)
    min_run = p.min_run_km / EARTH_RADIUS_KM
    settle = np.asarray(settlements, dtype=np.float64).reshape(-1, 3)
    settle_lab = plain_labels(settle) if len(settle) else np.zeros(0, dtype=np.int64)
    settle_tree = KDTree(settle) if len(settle) else None
    clear = p.settlement_clear_km / EARTH_RADIUS_KM
    ctx = (riv, p, min_run, plain_labels, is_water_at, settle, settle_lab, settle_tree, clear, st)

    accepted: List[tuple] = []
    for pts, (la, lb), depth in chains:
        pts = np.asarray(pts, dtype=np.float64)
        if la < 0 or lb < 0 or len(pts) < 3:
            continue
        near = riv.near(pts)
        if not near:
            continue
        runs: List[_Run] = []
        for r, nr in near.items():
            for i0, i1 in _runs(nr.mask, p.max_gap):
                run = _candidate(r, nr, i0, i1, pts, p, min_run, st)
                if run is not None:
                    runs.append(run)
        runs.sort(key=lambda c: -c.span)
        taken = np.zeros(len(pts), dtype=bool)
        for c in runs:
            if taken[c.i0:c.i1 + 1].any():
                _bump(st, "overlap")
                continue
            _bump(st, "candidates")
            for sw, i0, i1 in _settle(c, pts, near[c.river], la, lb, depth, ctx):
                taken[i0:i1 + 1] = True
                accepted.append((depth, -c.span, len(accepted), sw))
    accepted.sort(key=lambda t: t[:3])
    return RiverSwaps([t[3] for t in accepted], centroids, p)


def _candidate(r: int, nr: _Near, i0: int, i1: int, pts: np.ndarray, p: RiverFollowParams,
               min_run: float, st: dict) -> Optional[_Run]:
    core = _core(nr, i0, i1, p.snap_fraction)
    if core is None:
        _bump(st, "no_core")
        return None
    ia, ib = core
    span = abs(nr.s[ib] - nr.s[ia])
    if span < min_run:
        _bump(st, "short")
        return None
    # Against the core's straight extent, not its length: a border wandering between
    # two rivers is far longer than either and should still follow one.
    if span < p.parallel_ratio * float(np.linalg.norm(pts[ib] - pts[ia])):
        _bump(st, "not_parallel")
        return None
    return _Run(r, i0, i1, ia, ib, span)


def _sides(border: np.ndarray, la: int, lb: int, plain_labels: Callable, probe: float) -> tuple[int, int]:
    """``(left, right)``: which of ``la``/``lb`` lies left of the border run (majority of probes)."""
    normal = _unit(np.cross(border, np.gradient(border, axis=0)))
    left = plain_labels(_unit(border + probe * normal))
    right = plain_labels(_unit(border - probe * normal))
    votes_a = int(np.sum((left == la) & (right == lb)))
    votes_b = int(np.sum((left == lb) & (right == la)))
    return (la, lb) if votes_a >= votes_b else (lb, la)


def _make_swap(riv: _Rivers, c: _Run, pts: np.ndarray, nr: _Near, la: int, lb: int, depth: int,
               p: RiverFollowParams, plain_labels: Callable) -> RiverSwap:
    g = p.snap_fraction
    i0, i1, ia, ib = c.i0, c.i1, c.ia, c.ib
    sl = slice(i0, i1 + 1)
    w = np.where(nr.mask[sl], np.clip((nr.reach[sl] - nr.d[sl]) / np.maximum((1.0 - g) * nr.reach[sl], 1e-15),
                                      0.0, 1.0), 0.0)
    a, b = ia - i0, ib - i0
    w[a:b + 1] = 1.0
    w[:a + 1] = np.maximum.accumulate(w[:a + 1])
    w[b:] = np.maximum.accumulate(w[b:][::-1])[::-1]
    river = riv.stretch(c.river, nr.s[ia], nr.s[ib])
    left, right = _sides(pts[sl], la, lb, plain_labels, _SIDE_PROBE_KM / EARTH_RADIUS_KM)
    return RiverSwap(left, right, p.clean_km / EARTH_RADIUS_KM, p.edge_km / EARTH_RADIUS_KM, depth, pts[sl], w,
                     nr.foot[sl].copy(), a, b, river, riv.feature_index[c.river])


def _settle(c: _Run, pts: np.ndarray, nr: _Near, la: int, lb: int, depth: int, ctx: tuple,
            _depth: int = 0) -> List[tuple]:
    """The swap for run ``c``, split around settlements it would flip; dropped on a third polity."""
    riv, p, min_run, plain_labels, is_water_at, settle, settle_lab, settle_tree, clear, st = ctx
    sw = _make_swap(riv, c, pts, nr, la, lb, depth, p, plain_labels)
    if settle_tree is not None:
        idx = np.asarray(settle_tree.query_ball_point(sw.centre, sw.reach), dtype=np.int64)
        if idx.size:
            idx = idx[np.isin(settle_lab[idx], (la, lb))]
        if idx.size:
            idx = idx[sw.relabel(settle[idx], settle_lab[idx])[0] != settle_lab[idx]]
        if idx.size:
            if _depth >= 4:
                _bump(st, "settlement_drop")
                return []
            _bump(st, "settlement_split")
            # Split the run around the settlements' places along the river.
            at = riv.near(settle[idx]).get(c.river)
            keep = nr.mask.copy()
            if at is not None:
                for s_at in at.s[at.mask]:
                    keep &= np.abs(nr.s - s_at) > clear
            keep[:c.i0] = False
            keep[c.i1 + 1:] = False
            sub = _Near(keep, nr.s, nr.d, nr.reach, nr.foot)
            if at is None or not at.mask.any():
                # Not beside this river: drop the points nearest to it instead.
                for x in settle[idx]:
                    keep &= np.linalg.norm(pts - x, axis=1) > clear + p.reach_km / EARTH_RADIUS_KM
            out: List[tuple] = []
            for j0, j1 in _runs(keep, p.max_gap):
                run = _candidate(c.river, sub, j0, j1, pts, p, min_run, st)
                if run is not None:
                    out.extend(_settle(run, pts, sub, la, lb, depth, ctx, _depth + 1))
            return out
    # A third polity inside drops the run.
    lo, hi = sw.ring2d.min(axis=0), sw.ring2d.max(axis=0)
    step = p.sample_km / EARTH_RADIUS_KM
    gx = np.arange(lo[0], hi[0] + step, step)
    gy = np.arange(lo[1], hi[1] + step, step)
    if gx.size * gy.size > 200_000:
        _bump(st, "too_big")
        return []
    xx, yy = np.meshgrid(gx, gy)
    flat = np.column_stack([xx.ravel(), yy.ravel()])
    flat = flat[_even_odd(flat, sw.ring2d)]
    if len(flat):
        q = _unit(sw.centre[None, :] + flat[:, :1] * sw.e1[None, :] + flat[:, 1:2] * sw.e2[None, :])
        land = ~is_water_at(q)
        if land.any():
            third = ~np.isin(plain_labels(q[land]), (la, lb))
            if third.mean() > p.max_third_share:
                _bump(st, "third_polity")
                return []
    _bump(st, "accepted")
    return [(_on_boundary(sw, plain_labels, p.border_step_km / EARTH_RADIUS_KM), c.i0, c.i1)]


def _on_boundary(sw: RiverSwap, plain_labels: Callable, step: float, bisect_steps: int = 14) -> RiverSwap:
    """``sw`` with its border run densified to ``step`` and every point bisected onto the plain boundary.

    The polygon's border edge must match the field's own ``a | b`` boundary, or the
    swap leaves slivers of the wrong polity along it. The traced chain is only
    bracketed to a fraction of a lattice edge and joined by chords; this brings it
    to metres. A point without an ``a``/``b`` bracket across it stays put.
    """
    border = sw.border
    seg = np.linalg.norm(np.diff(border, axis=0), axis=1)
    cuts = np.maximum(1, np.ceil(seg / step).astype(np.int64))
    idx = np.concatenate([[0.0], np.cumsum(cuts)])                       # original point → new index
    t = np.concatenate([i + np.arange(k) / k for i, k in enumerate(cuts)] + [[len(border) - 1.0]])
    lo = np.floor(t).astype(np.int64).clip(0, len(border) - 2)
    f = (t - lo)[:, None]

    def lerp(x: np.ndarray) -> np.ndarray:
        return (1.0 - f) * x[lo] + f * x[lo + 1]

    pts = _unit(lerp(border))
    foot = _unit(lerp(sw.foot))
    weight = lerp(sw.weight[:, None])[:, 0]
    ia, ib = int(idx[sw.ia]), int(idx[sw.ib])
    weight[ia:ib + 1] = 1.0

    # Bracket each point across the local tangent, then bisect.
    tangent = np.gradient(pts, axis=0)
    normal = _unit(np.cross(pts, tangent))
    h = 2.0 * step
    p0 = _unit(pts - h * normal)
    p1 = _unit(pts + h * normal)
    l0, l1 = plain_labels(p0), plain_labels(p1)
    ok = np.flatnonzero(((l0 == sw.left) & (l1 == sw.right)) | ((l0 == sw.right) & (l1 == sw.left)))
    if ok.size:
        lo_p, hi_p, l_lo = p0[ok], p1[ok], l0[ok]
        for _ in range(bisect_steps):
            mid = _unit(lo_p + hi_p)
            same = plain_labels(mid) == l_lo
            lo_p = np.where(same[:, None], mid, lo_p)
            hi_p = np.where(same[:, None], hi_p, mid)
        pts[ok] = _unit(lo_p + hi_p)
    return RiverSwap(sw.left, sw.right, sw.band, sw.edge, sw.depth, pts, weight, foot, ia, ib, sw.river,
                     sw.river_id)
