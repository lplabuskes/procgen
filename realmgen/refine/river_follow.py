"""Borders that follow big rivers, run by run.

The plain polity borders (the label field before any river handling) are traced
into chains between junctions. Along each chain, a run of border points that
stays within reach of one big river is replaced by that river's stretch: "border
points 13–20 become river points 1–5". The river is picked per run — the
longest stretch of the chain a single river shadows wins, the rest of the chain
is free for another — so a border never flips between two parallel rivers. Two
runs of a chain along one river, the river running on the same way from one to the
other, are one: the border follows the river past a bulge it makes in between. Runs
are taken across all chains, sovereign borders first, and a river stretch carries
one border only: both borders of a strip a river runs down may be within reach
of it, and the longer run gets it.

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

from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import KDTree

from realmgen.refine.barriers import flow_ramp
from utils.simplex import fbm_sum, make_perm
from worldgen.cost.surfaces import EARTH_RADIUS_KM

# Nearest river segments examined per point.
_K_NEAREST = 16
# Cells per side of a swap polygon's acceleration grid.
_GRID = 128
# Points per relabel call above which a swap builds its side grid.
_GRID_MIN_POINTS = 20_000
# How far either side of the border run its two polities are probed (which is left).
_SIDE_PROBE_KM = 2.0
# The far side of a ``JunctionSwap``: whoever holds the land there without its polity.
OTHER = -3
# Junctions a ``JunctionSwap`` walks through at most.
_JUNCTION_HOPS = 4
# Bridgehead edge: radius × (1 + roughness · fBm(position)), octaves at these frequencies
# (unit sphere; ~40 / 20 / 10 km wavelengths) and weights.
_BRIDGEHEAD_ROUGHNESS = 0.35
_BRIDGEHEAD_FREQS = np.array([160.0, 320.0, 640.0])
_BRIDGEHEAD_AMPS = np.array([1.0, 0.5, 0.25]) / 1.75
_BRIDGEHEAD_PERM = make_perm(7919)
# A run end starting this far into reach (blend weight) joins the old border's end
# on a wavering connector, not the polygon's straight closing edge.
_JOIN_MIN_WEIGHT = 0.05
# Connector: points every 2 km, swung sideways by up to this share of its length (fBm
# at the bridgehead octaves, eased to 0 at both ends).
_JOIN_STEP_KM = 2.0
_JOIN_ROUGHNESS = 0.25
_JOIN_PERM = make_perm(4271)


@dataclass
class RiverFollowParams:
    reach_km: float = 120.0         # a border point this close to a full-strength river may follow it
    min_flow: float = 200.0         # reach ramps (log) from 0 at min_flow …
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
    ridge_reach_km: float = 120.0   # as reach_km, for a full-relief ridge (0 = ridges off)
    ridge_min_relief_km: float = 0.2   # ridge reach ramps (linear) from 0 at this relief …
    ridge_full_relief_km: float = 1.0  # … to ridge_reach_km here


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


def _connector(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """``p`` → ``q`` (``q`` left out) every ``_JOIN_STEP_KM``, wavering sideways like a plain border."""
    gap = float(np.linalg.norm(q - p))
    n = max(1, int(np.ceil(gap * EARTH_RADIUS_KM / _JOIN_STEP_KM)))
    t = np.arange(n) / n
    base = _unit((1.0 - t)[:, None] * p + t[:, None] * q)
    side = _unit(np.cross(base, q - p))
    off = _JOIN_ROUGHNESS * gap * np.sin(np.pi * t) * fbm_sum(base, _JOIN_PERM, _BRIDGEHEAD_FREQS, _BRIDGEHEAD_AMPS)
    return _unit(base + off[:, None] * side)


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
    river_id: int             # the line's id: the rivers passed (world ``"rivers"``, stream stems), then ridges
    # Ends cut short at a junction swap: past them the polygon is still claimed.
    open_start: bool = False
    open_end: bool = False
    # Ends starting deep in reach: the line runs from the old border's end on a connector.
    join_start: bool = False
    join_end: bool = False

    def __post_init__(self) -> None:
        w = self.weight[:, None]
        blend = _unit((1.0 - w) * self.border + w * self.foot)
        body = np.vstack([blend[:self.ia], self.river, blend[self.ib + 1:]])
        head = _connector(self.border[0], body[0]) if self.join_start and not self.open_start else body[:0]
        tail = _connector(self.border[-1], body[-1])[::-1] if self.join_end and not self.open_end else body[:0]
        self.line = np.vstack([head, body, tail])
        # How far the line proper (without connectors) starts / ends off the old border's ends.
        self.jumps = (float(np.linalg.norm(body[0] - self.border[0])),
                      float(np.linalg.norm(body[-1] - self.border[-1])))
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
        inside = self.inside(x)
        claim = interior & ((d_line < self.band) | (d_border < self.edge) | inside)
        if self.open_start or self.open_end:
            # Past an open end, inside the polygon (where a border cut short at a junction
            # swap bulges past the line's end): the side of the end segment, extended.
            near_start = np.linalg.norm(x - self.line[0], axis=1) < np.linalg.norm(x - self.line[-1], axis=1)
            beyond = inside & ~interior & np.where(near_start, self.open_start, self.open_end)
            if beyond.any():
                xb = x[beyond]
                n0 = np.cross(self.line[0], self.line[1])
                n1 = np.cross(self.line[-2], self.line[-1])
                side[beyond] = np.where(near_start[beyond], xb @ n0, xb @ n1)
                claim |= beyond
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
        claim, left = self._claim_left(pts[idx])
        rows = idx[claim]
        out[rows] = np.where(left[claim], self.left, self.right)
        claimed[rows] = True
        return out, claimed

    def _claim_left(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(claim, left)`` per point ``x``: claimed by this swap, and left of the new line."""
        claim = np.zeros(len(x), dtype=bool)
        left = np.zeros(len(x), dtype=bool)
        exact = np.ones(len(x), dtype=bool)
        if len(x) > _GRID_MIN_POINTS:
            uv = _gnomonic(x, self.centre, self.e1, self.e2)
            exact, claim, left = _SideGrid(self, uv.min(axis=0), uv.max(axis=0)).lookup(uv)
        if exact.any():
            c, lf, _ = self._classify(x[exact])
            claim[exact], left[exact] = c, lf
        return claim, left

    def moved(self, river: np.ndarray, foot: np.ndarray) -> "RiverSwap":
        return replace(self, river=river, foot=foot)


@dataclass
class JunctionSwap(RiverSwap):
    """A river stretch between two runs of different borders, through the junctions between them.

    ``border`` is ``polity``'s boundary from one run's core end to the other's
    (across the junctions where its neighbours meet), the new line the river between
    them; ``left``/``right`` is ``polity`` on its side and ``OTHER`` on the far side.
    Where it claims a point, ``polity`` takes the land of ``others`` on its side and
    gives its own land on the far side to ``without(points, polity)``: the neighbour
    that would hold it were ``polity`` not there. So the junctions move onto the river.

    Its polygon lies between ``polity``'s old boundary and the river, so all of it is
    far bank, past the line's ends too (else strips of ``polity`` stay where the
    boundary leaves the river beyond the stretch).

    A settlement within ``keep_radius`` of the river that it would hand over keeps a
    **bridgehead** instead (``keep_pos`` / ``keep_lab``): its polity's land on its bank
    within its distance from the river + ``keep_radius`` of its nearest point on the
    river (a half-disc on the bank, its edge roughened by noise) stays. One further
    from the river drops the swap.
    Measured against the current line, so the pocket follows a tier's drawn river.
    """

    polity: int = -1
    others: tuple = ()
    without: Optional[Callable[[np.ndarray, int], np.ndarray]] = None
    keep_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    keep_lab: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    keep_radius: float = 0.0

    def relabel(self, points: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(points, dtype=np.float64)
        out = np.asarray(labels).copy()
        claimed = np.zeros(len(pts), dtype=bool)
        mine = out == self.polity
        idx = np.flatnonzero((np.linalg.norm(pts - self.centre, axis=1) <= self.reach)
                             & (mine | np.isin(out, self.others)))
        if idx.size == 0:
            return out, claimed
        claim, left = self._claim_left(pts[idx])
        claimed[idx[claim]] = True
        claim &= ~self._held(pts[idx], out[idx])
        mine_side = left if self.left == self.polity else ~left
        take = idx[claim & mine_side]
        give = idx[claim & ~mine_side & mine[idx]]
        out[take] = self.polity
        if give.size and self.without is not None:
            out[give] = self.without(pts[give], self.polity)
        return out, claimed

    def _classify(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        claim, left, d = super()._classify(x)
        beyond = claim & ~d["interior"]                            # inside, past an end: far bank
        left = np.where(beyond, self.left != self.polity, left)
        return claim, left, d

    def _held(self, x: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Points of a kept settlement's polity in its bridgehead (see the class doc)."""
        held = np.zeros(len(x), dtype=bool)
        if len(self.keep_pos) == 0 or len(x) == 0:
            return held
        if self._line_tree is None:
            self._classify(x[:1])                      # builds the line tree
        tree = self._line_tree
        assert tree is not None
        line = self.line
        _, side, _ = _polyline_side(x, line, tree)
        seg, t, sd = _polyline_query(self.keep_pos, line, tree)
        foot = _unit(line[seg] + t[:, None] * (line[seg + 1] - line[seg]))
        mine_left = self.left == self.polity
        # A ragged edge, not a drawn circle: a pure function of position, so tiers agree.
        rough = 1.0 + _BRIDGEHEAD_ROUGHNESS * fbm_sum(x, _BRIDGEHEAD_PERM, _BRIDGEHEAD_FREQS, _BRIDGEHEAD_AMPS)
        for k in range(len(self.keep_pos)):
            # The pocket's bank by owner (a city on the river has no side): ``polity``'s
            # city on the far bank, a neighbour's on ``polity``'s.
            pocket_left = mine_left != (self.keep_lab[k] == self.polity)
            held |= ((labels == self.keep_lab[k]) & ((side > 0.0) == pocket_left)
                     & (np.linalg.norm(x - foot[k], axis=1) <= (sd[k] + self.keep_radius) * rough))
        return held


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
    cid = ids[np.argmax(counts)]
    on = beside[chain[beside] == cid]
    # Carry on along the drawing up to ``reach`` past the segments beside the core, so
    # the cut is the nearest point on the river itself: two swaps meeting at one
    # world point (a run and a junction swap) cut the drawing at the same point.
    members = np.flatnonzero(chain == cid)
    length = np.linalg.norm(b - a, axis=1)
    lo, hi = int(on.min()), int(on.max())
    reach_lo = reach_hi = 0.0
    while lo > members[0] and reach_lo < reach:
        lo -= 1
        reach_lo += float(length[lo])
    while hi < members[-1] and reach_hi < reach:
        hi += 1
        reach_hi += float(length[hi])
    segs = np.arange(lo, hi + 1)
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
    # Points another line in reach is far closer to (``_shadowed``): never in this one's core.
    shadowed: Optional[np.ndarray] = None


def relief_ramp(relief: np.ndarray, *, min_relief: float, full_relief: float) -> np.ndarray:
    """``clip((relief − min_relief) / (full_relief − min_relief), 0, 1)``."""
    span = max(float(full_relief) - float(min_relief), 1e-9)
    return np.clip((np.asarray(relief, dtype=np.float64) - float(min_relief)) / span, 0.0, 1.0)


def _ridge_pieces(on: np.ndarray, max_gap: int = 1) -> List[tuple[int, int]]:
    """``[(i0, i1)]`` segment ranges of the ``on`` stretches, gaps of ≤ ``max_gap`` bridged."""
    at = np.flatnonzero(on)
    if at.size == 0:
        return []
    cut = np.flatnonzero(np.diff(at) > max_gap + 1)
    starts = np.concatenate([[at[0]], at[cut + 1]])
    ends = np.concatenate([at[cut], [at[-1]]]) + 1
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


class _Rivers:
    """Line segments eligible to carry a border — world rivers, then ridges — with per-line arc length."""

    def __init__(self, rivers: Optional[Iterable], p: RiverFollowParams,
                 ridges: Optional[Iterable] = None) -> None:
        self.pos: List[np.ndarray] = []
        # Per kept line its id: index into ``rivers``, then ``len(rivers) + k`` for ridge k.
        self.feature_index: List[int] = []
        self.arc: List[np.ndarray] = []
        rivers = list(rivers or [])
        ridge_reach = p.ridge_reach_km / EARTH_RADIUS_KM
        lines = [(fi, path, False) for fi, path in enumerate(rivers)]
        if ridge_reach > 0.0:
            lines += [(len(rivers) + k, path, True) for k, path in enumerate(ridges or [])]
        self._parts: tuple[list, ...] = ([], [], [], [], [])    # a, b, line, segment, reach
        for fi, path, is_ridge in lines:
            pos = _unit(np.asarray(path[0], dtype=np.float64))
            strength = np.asarray(path[1], dtype=np.float64)
            m = min(strength.size, len(pos) - 1)
            if m <= 0:
                continue
            if is_ridge:
                reach = ridge_reach * relief_ramp(strength[:m], min_relief=p.ridge_min_relief_km,
                                                  full_relief=p.ridge_full_relief_km)
                # Relief comes and goes along a crest: each stretch above the minimum is a
                # line of its own, so no run (or bridge) carries a border across a low pass.
                pieces = _ridge_pieces(reach > 0.0)
            else:
                reach = p.reach_km / EARTH_RADIUS_KM * flow_ramp(strength[:m], min_flow=p.min_flow,
                                                                 full_flow=p.full_flow)
                pieces = [(0, m)] if np.any(reach > 0.0) else []
            for i0, i1 in pieces:
                self._add(fi, pos[i0:i1 + 1], reach[i0:i1])
        a_parts, b_parts, rid_parts, sid_parts, reach_parts = self._parts
        if sum(len(x) for x in a_parts):
            self.a = np.concatenate(a_parts)
            self.b = np.concatenate(b_parts)
            self.rid = np.concatenate(rid_parts)
            self.sid = np.concatenate(sid_parts)
            self.reach = np.concatenate(reach_parts)
        else:
            self.a = self.b = np.zeros((0, 3))
            self.rid = self.sid = np.zeros(0, dtype=np.int64)
            self.reach = np.zeros(0)
        self.n_seg = np.array([len(q) - 1 for q in self.pos], dtype=np.int64)
        self.tree = KDTree(0.5 * (self.a + self.b)) if len(self.a) else None
        self.half = 0.5 * float(np.linalg.norm(self.b - self.a, axis=1).max(initial=0.0))
        self.max_reach = float(self.reach.max(initial=0.0))

    def _add(self, fi: int, pos: np.ndarray, reach: np.ndarray) -> None:
        """Keep line ``pos`` (id ``fi``) with per-segment ``reach``; its segments in reach are indexed."""
        a_parts, b_parts, rid_parts, sid_parts, reach_parts = self._parts
        r = len(self.pos)
        self.feature_index.append(fi)
        self.pos.append(pos)
        self.arc.append(_arc_lengths(pos))
        # Through lakes too: a run reaching a lake shore carries on along the river's
        # course across it (only land is labelled), instead of stopping at the outlet
        # with the border still beside the river and jumping onto it on a straight edge.
        idx = np.flatnonzero(reach > 0.0)
        a_parts.append(pos[idx])
        b_parts.append(pos[idx + 1])
        rid_parts.append(np.full(idx.size, r))
        sid_parts.append(idx)
        reach_parts.append(reach[idx])

    def near(self, x: np.ndarray) -> dict:
        """``{river: _Near}`` for the rivers within reach of any of the points ``x``.

        Every segment within the largest reach is tested against its own reach (a
        fixed number of nearest segments missed a big river's lower course looping
        back past its own small headwaters, which crowded the nearest ones).
        """
        if self.tree is None or len(x) == 0:
            return {}
        hits = self.tree.query_ball_point(x, self.max_reach + self.half)
        lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(x))
        if lens.sum() == 0:
            return {}
        pt = np.repeat(np.arange(len(x)), lens)
        sg = np.fromiter((k for h in hits for k in h), dtype=np.int64, count=int(lens.sum()))
        a, b = self.a[sg], self.b[sg]
        ab = b - a
        t_raw = ((x[pt] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-30)
        t = np.clip(t_raw, 0.0, 1.0)
        foot = a + t[:, None] * ab
        dist = np.linalg.norm(x[pt] - foot, axis=-1)
        # Past a line's very end (its nearest segment the first / last, the foot clamped
        # to its end point) a point isn't beside it: every point beyond a tributary's
        # confluence would otherwise cluster onto it.
        rid_all = self.rid[sg]
        last = self.n_seg[rid_all] - 1
        past = ((self.sid[sg] == 0) & (t_raw < 0.0)) | ((self.sid[sg] == last) & (t_raw > 1.0))
        key = pt * len(self.pos) + rid_all
        order = np.lexsort((dist, key))
        first = np.ones(len(order), dtype=bool)
        first[1:] = key[order][1:] != key[order][:-1]
        past_keys = np.unique(key[order][first & past[order]])
        hold = (dist < self.reach[sg]) & ~np.isin(key, past_keys)
        pt, sg, t, foot, dist = pt[hold], sg[hold], t[hold], foot[hold], dist[hold]
        rid = self.rid[sg]
        # Per (point, river), the nearest segment holding the point.
        order = np.lexsort((dist, rid, pt))
        pt, sg, t, foot, dist, rid = pt[order], sg[order], t[order], foot[order], dist[order], rid[order]
        first = np.ones(len(pt), dtype=bool)
        first[1:] = (pt[1:] != pt[:-1]) | (rid[1:] != rid[:-1])
        pt, sg, t, foot, dist, rid = pt[first], sg[first], t[first], foot[first], dist[first], rid[first]
        out: dict = {}
        n = len(x)
        for r in np.unique(rid):
            m = rid == r
            rows = pt[m]
            arc = self.arc[int(r)]
            sid = self.sid[sg[m]]
            mask = np.zeros(n, dtype=bool)
            mask[rows] = True
            s_ = np.zeros(n)
            s_[rows] = arc[sid] + t[m] * (arc[sid + 1] - arc[sid])
            d = np.full(n, np.inf)
            d[rows] = dist[m]
            reach = np.zeros(n)
            reach[rows] = self.reach[sg[m]]
            f = np.zeros((n, 3))
            f[rows] = _unit(foot[m])
            out[int(r)] = _Near(mask=mask, s=s_, d=d, reach=reach, foot=f)
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


def _bridged_runs(nr: "_Near", p: RiverFollowParams) -> List[tuple[int, int]]:
    """``_runs`` of ``nr.mask``, consecutive ones joined when the river carries on between them.

    Two runs along one river, the river running the same way through both and on
    from the first's core to the second's, are one run: the border leaving the river
    between them (round a bulge, beyond reach) follows it there too.
    """
    out: List[tuple[int, int]] = []
    prev_core: Optional[tuple[int, int]] = None
    for i0, i1 in _runs(nr.mask, p.max_gap):
        core = _core(nr, i0, i1, p.snap_fraction)
        if out and prev_core is not None and core is not None:
            a0, a1 = prev_core
            dirs = np.sign([nr.s[a1] - nr.s[a0], nr.s[core[0]] - nr.s[a1], nr.s[core[1]] - nr.s[core[0]]])
            dirs = dirs[dirs != 0]
            if dirs.size and np.all(dirs == dirs[0]):
                out[-1] = (out[-1][0], i1)
                prev_core = (a0, core[1])
                continue
        out.append((i0, i1))
        prev_core = core
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
    full = nr.mask[i0:i1 + 1] & (nr.d[i0:i1 + 1] <= g * nr.reach[i0:i1 + 1])
    if nr.shadowed is not None:
        full &= ~nr.shadowed[i0:i1 + 1]
    full = np.flatnonzero(full)
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
    labels_without: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    ridges: Optional[Iterable] = None,
) -> RiverSwaps:
    """Swaps for the plain border ``chains`` (``trace_lattice_chains``) onto ``rivers`` and ``ridges``.

    ``rivers`` are ``(positions, flows, ...)`` polylines, ``ridges`` ``(positions,
    relief_km)`` (``ridges.ridge_lines``), one pool: a run takes whichever line it
    shadows longest. A ridge's swap id is ``len(rivers) + k``.

    ``plain_labels``/``is_water_at`` label sphere points; ``settlements`` is
    ``(S, 3)`` positions no swap may hand to the neighbour. ``stats`` (if given)
    counts candidates and why they were dropped. With ``labels_without``
    (``PolityLabelField.labels_without``) runs of different borders along one river
    are joined through the junctions between them (``JunctionSwap``).
    """
    p = params or RiverFollowParams()
    st = stats if stats is not None else {}
    riv = _Rivers(rivers, p, ridges)
    min_run = p.min_run_km / EARTH_RADIUS_KM
    settle = np.asarray(settlements, dtype=np.float64).reshape(-1, 3)
    settle_lab = plain_labels(settle) if len(settle) else np.zeros(0, dtype=np.int64)
    settle_tree = KDTree(settle) if len(settle) else None
    clear = p.settlement_clear_km / EARTH_RADIUS_KM
    ctx = (riv, p, min_run, plain_labels, is_water_at, settle, settle_lab, settle_tree, clear, st)

    # Candidates from every chain, taken sovereign first, then longest river stretch.
    cands: List[tuple] = []
    per_chain: List[tuple] = []
    for k, (pts, (la, lb), depth) in enumerate(chains):
        pts = np.asarray(pts, dtype=np.float64)
        if la < 0 or lb < 0 or len(pts) < 3:
            continue
        near = riv.near(pts)
        if not near:
            continue
        _shadow(near)
        ci = len(per_chain)
        # Per chain point: the accepted run holding it (-1 none), and whether in its core.
        per_chain.append((pts, near, la, lb, depth, np.full(len(pts), -1, dtype=np.int64), k,
                          np.zeros(len(pts), dtype=bool)))
        for r, nr in near.items():
            for i0, i1 in _bridged_runs(nr, p):
                run = _candidate(r, nr, i0, i1, pts, p, min_run, st)
                if run is not None:
                    cands.append((depth, -run.span, len(cands), ci, run))
    cands.sort(key=lambda t: t[:3])
    # Per river, the arc intervals already carrying a border: a stretch carries one
    # (two borders of a strip both within reach of its river would fight over it).
    river_taken: dict = {}
    accepted: List[tuple] = []
    for depth, _, _, ci, c in cands:
        pts, near, la, lb, _, owner, _, core = per_chain[ci]
        nr = near[c.river]
        if core[c.i0:c.i1 + 1].any():
            # Another run's core lies in it (where two rivers both shadow the border, e.g.
            # a tributary's run reaching past its confluence): keep the longest free stretch.
            c = _free_part(c, core, nr, pts, p, min_run, st)
            if c is None:
                _bump(st, "overlap")
                continue
        lo, hi = sorted((float(nr.s[c.ia]), float(nr.s[c.ib])))
        if any(lo < t_hi and t_lo < hi for t_lo, t_hi in river_taken.get(c.river, ())):
            _bump(st, "river_taken")
            continue
        _bump(st, "candidates")
        for sw, piece in _settle(c, pts, nr, la, lb, depth, ctx):
            # Earlier runs' lead-outs this one takes: they end where it starts.
            cut: Optional[tuple] = (sw, piece)
            for n in np.unique(owner[piece.i0:piece.i1 + 1]):
                if n >= 0 and cut is not None:
                    cut = _cut_back(accepted, int(n), cut[1], cut[0], per_chain, ctx)
            if cut is None:
                continue
            sw, piece = cut
            owner[piece.i0:piece.i1 + 1] = len(accepted)
            core[piece.ia:piece.ib + 1] = True
            river_taken.setdefault(c.river, []).append((lo, hi))
            accepted.append((depth, -c.span, len(accepted), sw, ci, piece))
    if labels_without is not None:
        accepted = _junction_swaps(accepted, chains, per_chain, labels_without, ctx)
    accepted.sort(key=lambda t: t[:3])
    return RiverSwaps([t[3] for t in accepted], centroids, p)


# A point is in no line's core where another line in reach is under this share of its distance.
_SHADOW_RATIO = 0.5


def _shadow(near: dict) -> None:
    """Mark each line's points that another line is far closer to (``_Near.shadowed``).

    Past a tributary's confluence the border runs beside the stem, still in reach of
    the tributary's lower course: those points go to the stem. Two parallel lines at
    similar distances shadow neither, so a border doesn't flip between them.
    """
    if len(near) < 2:
        return
    d = np.stack([nr.d for nr in near.values()])          # (R, P), inf out of reach
    order = np.sort(d, axis=0)
    for k, nr in enumerate(near.values()):
        # The closest other line: the best, or the second best where this one is the best.
        other = np.where(d[k] == order[0], order[1], order[0])
        nr.shadowed = nr.mask & (other < _SHADOW_RATIO * d[k])


def _free_part(c: _Run, core: np.ndarray, nr: _Near, pts: np.ndarray, p: RiverFollowParams,
               min_run: float, st: dict) -> Optional[_Run]:
    """The longest candidate in ``c``'s points clear of accepted runs' ``core`` points."""
    free = np.zeros(len(core), dtype=bool)
    free[c.i0:c.i1 + 1] = ~core[c.i0:c.i1 + 1] & nr.mask[c.i0:c.i1 + 1]
    best = None
    for j0, j1 in _runs(free, 0):
        run = _candidate(c.river, nr, j0, j1, pts, p, min_run, st)
        if run is not None and (best is None or run.span > best.span):
            best = run
    return best


def _cut_back(accepted: List[tuple], n: int, piece: _Run, sw_new: RiverSwap, per_chain: List[tuple],
              ctx: tuple) -> Optional[tuple]:
    """Runs ``n`` and ``piece`` (``sw_new``, same chain, overlapping on lead-outs) split between their cores.

    The split is the chain point between the cores where the two rivers' feet are
    closest (a tributary's run meets its stem's at the confluence); without one in
    reach of both, at ``n``'s core end. Both meet open (no connector back to the old
    border between two rivers). Returns ``piece``'s new ``(swap, run)``, or None when
    the cores overlap.
    """
    riv, p, _, plain_labels = ctx[:4]
    step = p.border_step_km / EARTH_RADIUS_KM
    depth, span, order, sw_old, ci, run = accepted[n]
    pts, near, la, lb = per_chain[ci][:4]
    nr_old, nr_new = near[run.river], near[piece.river]
    after = run.ib < piece.ia                  # ``piece`` comes after ``n`` along the chain
    lo, hi = (run.ib, piece.ia) if after else (piece.ib, run.ia)
    if hi - lo < 1:
        return None
    k = np.arange(lo, hi)                      # split after k: [.., k] to the earlier run
    both = nr_old.mask[k] & nr_old.mask[k + 1] & nr_new.mask[k] & nr_new.mask[k + 1]
    if both.any():
        gap = np.linalg.norm(nr_old.foot[k] - nr_new.foot[k], axis=1)
        split = int(k[both][np.argmin(gap[both])])
    else:
        split = lo if after else hi - 1
    if after:
        cut = _Run(run.river, run.i0, split, run.ia, run.ib, run.span)
        grown = _Run(piece.river, split + 1, piece.i1, piece.ia, piece.ib, piece.span)
    else:
        cut = _Run(run.river, split + 1, run.i1, run.ia, run.ib, run.span)
        grown = _Run(piece.river, piece.i0, split, piece.ia, piece.ib, piece.span)
    sw = _make_swap(riv, cut, pts, nr_old, la, lb, depth, p, plain_labels)
    sw = replace(sw, open_start=sw_old.open_start or not after, open_end=sw_old.open_end or after)
    accepted[n] = (depth, span, order, _on_boundary(sw, plain_labels, step), ci, cut)
    if grown != piece:
        sw_new = replace(_on_boundary(_make_swap(riv, grown, pts, nr_new, la, lb, sw_new.depth, p, plain_labels),
                                      plain_labels, step),
                         open_start=sw_new.open_start, open_end=sw_new.open_end)
    _bump(ctx[-1], "cut_back")
    sw_new = replace(sw_new, open_start=sw_new.open_start or after, open_end=sw_new.open_end or not after)
    return sw_new, grown


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
                     nr.foot[sl].copy(), a, b, river, riv.feature_index[c.river],
                     join_start=bool(w[0] > _JOIN_MIN_WEIGHT), join_end=bool(w[-1] > _JOIN_MIN_WEIGHT))


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
            sub = _Near(keep, nr.s, nr.d, nr.reach, nr.foot, nr.shadowed)
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
    # A third polity inside drops the run, bar the corner where the run starts or ends at
    # a junction with it (a third polity's land is never relabelled; the check is for a
    # polygon wrapping round one).
    sampled = _polygon_land(sw, p, plain_labels, is_water_at)
    if sampled is None:
        _bump(st, "too_big")
        return []
    q, inside = sampled
    if inside.size:
        corner = p.clean_km / EARTH_RADIUS_KM
        off_ends = ((np.linalg.norm(q - sw.border[0], axis=1) > corner)
                    & (np.linalg.norm(q - sw.border[-1], axis=1) > corner))
        if np.mean(~np.isin(inside, (la, lb)) & off_ends) > p.max_third_share:
            _bump(st, "third_polity")
            return []
    _bump(st, "accepted")
    return [(_on_boundary(sw, plain_labels, p.border_step_km / EARTH_RADIUS_KM), c)]


def _polygon_land(sw: RiverSwap, p: RiverFollowParams, plain_labels: Callable,
                  is_water_at: Callable) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """``(points, plain labels)`` of land sampled every ``sample_km`` inside ``sw``'s polygon;
    None if it's too big."""
    lo, hi = sw.ring2d.min(axis=0), sw.ring2d.max(axis=0)
    step = p.sample_km / EARTH_RADIUS_KM
    gx = np.arange(lo[0], hi[0] + step, step)
    gy = np.arange(lo[1], hi[1] + step, step)
    if gx.size * gy.size > 200_000:
        return None
    xx, yy = np.meshgrid(gx, gy)
    flat = np.column_stack([xx.ravel(), yy.ravel()])
    flat = flat[_even_odd(flat, sw.ring2d)]
    q = _unit(sw.centre[None, :] + flat[:, :1] * sw.e1[None, :] + flat[:, 1:2] * sw.e2[None, :])
    q = q[~is_water_at(q)] if len(q) else np.zeros((0, 3))
    return q, (plain_labels(q) if len(q) else np.zeros(0, dtype=np.int64))


# ── junctions ────────────────────────────────────────────────────────────────


def _end_key(x: np.ndarray) -> tuple:
    return tuple(np.round(np.asarray(x, dtype=np.float64) * 1e9).astype(np.int64).tolist())


def _walk(chains: Sequence[tuple], ends: dict, polity: int, k_a: int, from_end: bool, start: int,
          k_b: int, b_run: "_Run", b_gap: int) -> Optional[tuple[np.ndarray, set, list]]:
    """``polity``'s boundary from point ``start`` of chain ``k_a`` (walking to its end if
    ``from_end`` else to its start) through junctions to point ``b_gap`` of chain ``k_b``.

    ``(path, polities met, [(chain, first, last) point ranges walked])``, or None when
    the walk doesn't arrive within ``_JUNCTION_HOPS`` junctions (a coast, or the
    boundary running elsewhere).
    """
    pts = np.asarray(chains[k_a][0], dtype=np.float64)
    path = [pts[start:] if from_end else pts[start::-1]]
    spans = [(k_a, start, len(pts) - 1) if from_end else (k_a, 0, start)]
    met = set(chains[k_a][1])
    cur, cur_end = k_a, 1 if from_end else 0
    for _ in range(_JUNCTION_HOPS):
        at = np.asarray(chains[cur][0])[-1 if cur_end else 0]
        nxt = [(k, e) for k, e in ends.get(_end_key(at), ())
               if (k, e) != (cur, cur_end) and polity in chains[k][1]]
        if len(nxt) != 1:
            return None
        k, e = nxt[0]
        q = np.asarray(chains[k][0], dtype=np.float64)
        q = q if e == 0 else q[::-1]
        met |= set(chains[k][1])
        if k == k_b:
            # Entering B's chain, its gap-side core end must come first.
            if b_gap != (b_run.ia if e == 0 else b_run.ib):
                return None
            j = b_gap if e == 0 else len(q) - 1 - b_gap
            path.append(q[1:j + 1])
            spans.append((k, 0, b_gap) if e == 0 else (k, b_gap, len(q) - 1))
            return _unit(np.vstack(path)), met, spans
        path.append(q[1:])
        spans.append((k, 0, len(q) - 1))
        cur, cur_end = k, 1 - e
    return None


def _walk_to_river(chains: Sequence[tuple], ends: dict, polity: int, k_a: int, from_end: bool, start: int,
                   riv: "_Rivers", r: int, s_from: float, onward: float,
                   near_km: float) -> Optional[tuple[np.ndarray, set, list, float]]:
    """``polity``'s boundary from point ``start`` of chain ``k_a`` (whose end is a junction)
    through junctions until it comes within ``near_km`` (radians) of river ``r`` again,
    further along it (arc beyond ``s_from`` in the sign of ``onward``).

    ``(path, polities met, point ranges walked, arc where it returns)``; None when it
    doesn't within ``_JUNCTION_HOPS`` junctions.
    """
    pts = np.asarray(chains[k_a][0], dtype=np.float64)
    path = [pts[start:] if from_end else pts[start::-1]]
    spans = [(k_a, start, len(pts) - 1) if from_end else (k_a, 0, start)]
    met = set(chains[k_a][1])
    cur, cur_end = k_a, 1 if from_end else 0
    for _ in range(_JUNCTION_HOPS):
        at = np.asarray(chains[cur][0])[-1 if cur_end else 0]
        nxt = [(k, e) for k, e in ends.get(_end_key(at), ())
               if (k, e) != (cur, cur_end) and polity in chains[k][1]]
        if len(nxt) != 1:
            return None
        k, e = nxt[0]
        q = np.asarray(chains[k][0], dtype=np.float64)
        q = q if e == 0 else q[::-1]
        met |= set(chains[k][1])
        nr = riv.near(q).get(r)
        if nr is not None:
            back = np.flatnonzero(nr.mask & (nr.d <= near_km) & ((nr.s - s_from) * onward > 0.0))
            back = back[back > 0]
            if back.size:
                j = int(back[0])
                path.append(q[1:j + 1])
                spans.append((k, 0, j) if e == 0 else (k, len(q) - 1 - j, len(q) - 1))
                return _unit(np.vstack(path)), met, spans, float(nr.s[j])
        path.append(q[1:])
        spans.append((k, 0, len(q) - 1))
        cur, cur_end = k, 1 - e
    return None


def _checked_junction(pol: int, pol_left: bool, path: np.ndarray, met: set, r: int, s0: float, s1: float,
                      depth: int, labels_without: Callable, ctx: tuple) -> Optional[RiverSwap]:
    """The ``JunctionSwap`` of ``pol``'s boundary ``path`` onto river ``r`` from arc ``s0``
    to ``s1`` (``pol`` left of that line if ``pol_left``), or None when it fails the
    checks: a third polity in it, or a settlement it would hand over that isn't a
    river city (those keep a bridgehead)."""
    riv, p, min_run, plain_labels, is_water_at, settle, settle_lab, settle_tree, clear, st = ctx
    others = tuple(sorted(met - {pol}))
    left, right = (pol, OTHER) if pol_left else (OTHER, pol)
    n = len(path)
    sw = JunctionSwap(left, right, p.clean_km / EARTH_RADIUS_KM, p.edge_km / EARTH_RADIUS_KM, depth, path,
                      np.ones(n), path.copy(), 0, n - 1, riv.stretch(r, s0, s1), riv.feature_index[r],
                      open_start=True, open_end=True, polity=pol, others=others, without=labels_without)
    sampled = _polygon_land(sw, p, plain_labels, is_water_at)
    if sampled is None:
        _bump(st, "junction_too_big")
        return None
    inside = sampled[1]
    if inside.size and np.mean(~np.isin(inside, (pol,) + others)) > p.max_third_share:
        _bump(st, "junction_third_polity")
        return None
    if settle_tree is not None:
        # A settlement it would hand over keeps a bridgehead round it.
        idx = np.asarray(settle_tree.query_ball_point(sw.centre, sw.reach), dtype=np.int64)
        if idx.size:
            flip = idx[sw.relabel(settle[idx], settle_lab[idx])[0] != settle_lab[idx]]
            line_tree = KDTree(0.5 * (sw.line[:-1] + sw.line[1:]))
            if flip.size and np.any(_polyline_distance(settle[flip], sw.line, line_tree) > clear):
                _bump(st, "junction_settlement")      # only a river city keeps a bridgehead
                return None
            if flip.size:
                sw = replace(sw, keep_pos=settle[flip], keep_lab=settle_lab[flip], keep_radius=clear)
                if np.any(sw.relabel(settle[idx], settle_lab[idx])[0] != settle_lab[idx]):
                    _bump(st, "junction_settlement")
                    return None
                _bump(st, "junction_bridgehead")
    return _on_boundary(sw, plain_labels, p.border_step_km / EARTH_RADIUS_KM)


def _junction_swaps(accepted: List[tuple], chains: Sequence[tuple], per_chain: List[tuple],
                    labels_without: Callable, ctx: tuple) -> List[tuple]:
    """``accepted`` plus a ``JunctionSwap`` between each two runs of different borders
    that follow one river with a polity on the same bank, the runs' lead-outs into
    the gap dropped."""
    riv, p, min_run, plain_labels, is_water_at, settle, settle_lab, settle_tree, clear, st = ctx
    ends: dict = {}
    for k, (pts, _, _) in enumerate(chains):
        if len(pts) >= 2:
            ends.setdefault(_end_key(pts[0]), []).append((k, 0))
            ends.setdefault(_end_key(pts[-1]), []).append((k, 1))
    # Which accepted run holds each chain point: a junction swap may not walk through
    # another run's border (its line would fight that run's, e.g. on a parallel river).
    owner: dict = {}
    for n, (_, _, _, _, ci, c) in enumerate(accepted):
        k = per_chain[ci][6]
        own = owner.setdefault(k, np.full(len(chains[k][0]), -1, dtype=np.int64))
        own[c.i0:c.i1 + 1] = n
    by_river: dict = {}
    for n, (_, _, _, _, ci, c) in enumerate(accepted):
        nr = per_chain[ci][1][c.river]
        s_a, s_b = float(nr.s[c.ia]), float(nr.s[c.ib])
        by_river.setdefault(c.river, []).append((min(s_a, s_b), max(s_a, s_b), n))
    trunc: dict = {}
    out = list(accepted)
    for r, items in by_river.items():
        items.sort()
        for (_, hi_a, na), (lo_b, _, nb) in zip(items, items[1:]):
            if hi_a >= lo_b:
                continue
            depth_a, _, _, sw_a, ci_a, ca = accepted[na]
            depth_b, _, _, sw_b, ci_b, cb = accepted[nb]
            pc_a, pc_b = per_chain[ci_a], per_chain[ci_b]
            if pc_a[6] == pc_b[6]:
                continue
            nr_a, nr_b = pc_a[1][r], pc_b[1][r]
            up_a = nr_a.s[ca.ib] > nr_a.s[ca.ia]          # the river runs with the chain along A
            up_b = nr_b.s[cb.ib] > nr_b.s[cb.ia]
            ga = ca.ib if up_a else ca.ia                 # A's core end at hi_a, B's at lo_b
            gb = cb.ia if up_b else cb.ib
            for pol in sorted({pc_a[2], pc_a[3]} & {pc_b[2], pc_b[3]}):
                # The polity left of the river (downstream of arc) on both runs, or right on both.
                left_a = (sw_a.left == pol) == up_a
                left_b = (sw_b.left == pol) == up_b
                if left_a != left_b:
                    continue
                walk = _walk(chains, ends, pol, pc_a[6], ga == ca.ib, ga, pc_b[6], cb, gb)
                if walk is None:
                    _bump(st, "junction_walk")
                    continue
                path, met, spans = walk
                if any(np.any(~np.isin(owner[k][i:j + 1], (-1, na, nb))) for k, i, j in spans if k in owner):
                    _bump(st, "junction_overlap")
                    continue
                sw = _checked_junction(pol, left_a, path, met, r, float(nr_a.s[ga]), float(nr_b.s[gb]),
                                       min(depth_a, depth_b), labels_without, ctx)
                if sw is None:
                    continue
                _bump(st, "junction")
                trunc.setdefault(na, [ca.i0, ca.i1])[1 if up_a else 0] = ga
                trunc.setdefault(nb, [cb.i0, cb.i1])[0 if up_b else 1] = gb
                out.append((min(depth_a, depth_b), -abs(lo_b - hi_a), len(out), sw, -1, None))
                break
    # A run ending at a junction (its chain's end) while still beside the river: the
    # junction moves onto the river too, along the boundary of the run's polity that
    # carries on round it back to the river.
    for na, (depth_a, _, _, sw_a, ci_a, ca) in enumerate(accepted):
        pc_a = per_chain[ci_a]
        pts_a, k_a, r = pc_a[0], pc_a[6], ca.river
        nr_a = pc_a[1][r]
        up_a = nr_a.s[ca.ib] > nr_a.s[ca.ia]
        for hi in (False, True):
            done = trunc.get(na)
            if done is not None and done[1 if hi else 0] != (ca.i1 if hi else ca.i0):
                continue                                  # carried on by a junction swap already
            if (ca.i1 != len(pts_a) - 1) if hi else (ca.i0 != 0):
                continue
            jump = sw_a.jumps[1 if hi else 0]
            if jump * EARTH_RADIUS_KM < p.edge_km:
                continue
            g = ca.ib if hi else ca.ia
            s_g = float(nr_a.s[g])
            onward = (1.0 if up_a else -1.0) * (1.0 if hi else -1.0)     # the river's way on past this end
            if len(ends.get(_end_key(pts_a[-1 if hi else 0]), ())) == 1:
                continue                                  # a coast end: no junction to move
            for pol in (pc_a[2], pc_a[3]):
                walk = _walk_to_river(chains, ends, pol, k_a, hi, g, riv, r, s_g, onward,
                                      p.clean_km / EARTH_RADIUS_KM)
                if walk is None:
                    _bump(st, "junction_end_walk")
                    continue
                path, met, spans, s_end = walk
                if any(np.any(~np.isin(owner[k][i:j + 1], (-1, na))) for k, i, j in spans if k in owner):
                    _bump(st, "junction_overlap")
                    continue
                lo, hi_s = sorted((s_g, s_end))
                if any(lo < t_hi and t_lo < hi_s for t_lo, t_hi, m in by_river.get(r, ()) if m != na):
                    _bump(st, "junction_river_taken")
                    continue
                # ``pol`` left of the river walking up its arc, so of the line if it runs up.
                left_up = (sw_a.left == pol) == up_a
                sw = _checked_junction(pol, left_up == (s_end > s_g), path, met, r, s_g, s_end, depth_a,
                                       labels_without, ctx)
                if sw is None:
                    continue
                _bump(st, "junction_end")
                trunc.setdefault(na, [ca.i0, ca.i1])[1 if hi else 0] = g
                out.append((depth_a, -abs(s_end - s_g), len(out), sw, -1, None))
                break
    # Runs ending at a junction swap end on the river: no lead-out into the gap.
    for n, (i0, i1) in trunc.items():
        depth, span, order, _, ci, c = out[n]
        pts, near, la, lb = per_chain[ci][:4]
        run = _Run(c.river, i0, i1, c.ia, c.ib, c.span)
        sw = _make_swap(riv, run, pts, near[c.river], la, lb, depth, p, plain_labels)
        sw = replace(sw, open_start=i0 != c.i0 or out[n][3].open_start, open_end=i1 != c.i1 or out[n][3].open_end)
        out[n] = (depth, span, order, _on_boundary(sw, plain_labels, p.border_step_km / EARTH_RADIUS_KM), ci, run)
    return out


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
    if isinstance(sw, JunctionSwap):
        def key(x: np.ndarray) -> np.ndarray:      # its polity | anyone else
            return (plain_labels(x) == sw.polity).astype(np.int64)
        k0, k1 = key(p0), key(p1)
        ok = np.flatnonzero(k0 != k1)
    else:
        key = plain_labels
        k0, k1 = key(p0), key(p1)
        ok = np.flatnonzero(((k0 == sw.left) & (k1 == sw.right)) | ((k0 == sw.right) & (k1 == sw.left)))
    if ok.size:
        lo_p, hi_p, l_lo = p0[ok], p1[ok], k0[ok]
        for _ in range(bisect_steps):
            mid = _unit(lo_p + hi_p)
            same = key(mid) == l_lo
            lo_p = np.where(same[:, None], mid, lo_p)
            hi_p = np.where(same[:, None], hi_p, mid)
        pts[ok] = _unit(lo_p + hi_p)
    return replace(sw, border=pts, weight=weight, foot=foot, ia=ia, ib=ib)
