"""River barriers for the polity label field: borders near a big river snap onto it.

Within ``reach`` of a river a point at distance ``d`` takes the plain label of
the point ``g·reach + (1 − g)·d`` out from the river on its own bank (straight
away from the nearest segment), ``g = snap_fraction``. A border within ``g·reach``
therefore lies on the river; one between there and ``reach`` is stretched onto
the band, so it leaves the river along its own course rather than at a right
angle; one crossing the river keeps its crossing; outside the band nothing
moves. ``reach`` ramps with flow from
``min_flow`` (0) to ``full_flow`` (``reach_km``) on a log scale, so streams never
move a border and great rivers pull from furthest. A pure function of position
and the river polylines, so face seams agree. The region and local tiers swap in
their re-routed rivers (``PolityLabelField.with_rivers``) so the line follows the
river they draw.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree

from worldgen.cost.surfaces import EARTH_RADIUS_KM

# Nearest segments examined per point (by midpoint) before the exact distance test.
_K_NEAREST = 32
# Crossing tests run on the rivers coarsened to about this segment length (radians):
# they only decide which bank's candidates a cell may use; the snap places the line.
_COARSE_SEGMENT = 10.0 / EARTH_RADIUS_KM


def _coarsen(a: np.ndarray, b: np.ndarray, strength: np.ndarray, step: float):
    """Merge runs of contiguous segments (``b[i] == a[i+1]``) into ``~step``-long ones (max strength)."""
    m = a.shape[0]
    if m == 0:
        return a, b, strength
    joined = np.zeros(m, dtype=bool)
    joined[1:] = np.all(np.abs(a[1:] - b[:-1]) < 1e-12, axis=1)
    length = np.linalg.norm(b - a, axis=1)
    ca, cb, cs = [], [], []
    start, acc, peak = 0, 0.0, 0.0
    for i in range(m):
        if i > 0 and not joined[i] and start < i:   # a new chain: close the open run
            ca.append(a[start])
            cb.append(b[i - 1])
            cs.append(peak)
            start, acc, peak = i, 0.0, 0.0
        elif i > 0 and not joined[i]:
            start, acc, peak = i, 0.0, 0.0
        acc += length[i]
        peak = max(peak, strength[i])
        if acc >= step or i == m - 1:
            ca.append(a[start])
            cb.append(b[i])
            cs.append(peak)
            start, acc, peak = i + 1, 0.0, 0.0
    return np.asarray(ca), np.asarray(cb), np.asarray(cs)


def pack_river_segments(rivers: Optional[Iterable], *, min_flow: float = 0.0) -> np.ndarray:
    """``(M, 7)`` ``[a, b, flow]`` rows of the land segments of ``rivers`` above ``min_flow``.

    ``rivers`` is the ``(positions, flows, in_lake)`` polyline form (world
    ``"rivers"``, region / local paths); in-lake segments are dropped.
    """
    a, b, flow = _river_segments(rivers, min_flow=min_flow)
    return np.concatenate([a, b, flow[:, None]], axis=1)


def _river_segments(rivers: Optional[Iterable], *, min_flow: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_parts, b_parts, f_parts = [], [], []
    for path in rivers or []:
        pos = np.asarray(path[0], dtype=np.float64)
        flow = np.asarray(path[1], dtype=np.float64)
        in_lake = np.asarray(path[2], dtype=bool) if len(path) > 2 else np.zeros(flow.size, dtype=bool)
        m = min(flow.size, pos.shape[0] - 1)
        if m <= 0:
            continue
        keep = (flow[:m] > float(min_flow)) & ~in_lake[:m]
        if keep.any():
            a_parts.append(pos[:m][keep])
            b_parts.append(pos[1:m + 1][keep])
            f_parts.append(flow[:m][keep])
    if not a_parts:
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, empty.copy(), np.zeros(0, dtype=np.float64)
    a = np.concatenate(a_parts)
    b = np.concatenate(b_parts)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return a, b, np.concatenate(f_parts)


def flow_ramp(flow: np.ndarray, *, min_flow: float, full_flow: float) -> np.ndarray:
    """``clip(ln(flow / min_flow) / ln(full_flow / min_flow), 0, 1)``."""
    lo = max(float(min_flow), 1e-9)
    span = np.log(max(float(full_flow), lo * (1.0 + 1e-6)) / lo)
    return np.clip(np.log(np.maximum(np.asarray(flow, dtype=np.float64), 1e-12) / lo) / span, 0.0, 1.0)


class RiverBarriers:
    """River segments with a per-segment snap reach (radians on the unit sphere)."""

    def __init__(self, a: np.ndarray, b: np.ndarray, reach: np.ndarray, snap_fraction: float = 0.6,
                 strength: Optional[np.ndarray] = None) -> None:
        self.snap_fraction = float(np.clip(snap_fraction, 0.0, 1.0))
        # Coarsened copy with per-segment strength (flow ramp, 0–1) for ``crossings``.
        s = np.ones(len(a)) if strength is None else np.asarray(strength, dtype=np.float64)
        ca, cb, self.strength = _coarsen(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
                                         s, _COARSE_SEGMENT)
        self.ca, self.cb = ca, cb
        cn = np.cross(ca, cb)
        self.cnormal = cn / np.maximum(np.linalg.norm(cn, axis=1, keepdims=True), 1e-15)
        self._chalf = 0.5 * float(np.linalg.norm(cb - ca, axis=1).max(initial=0.0))
        self._ctree = KDTree(0.5 * (ca + cb))
        self.a = np.asarray(a, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.reach = np.asarray(reach, dtype=np.float64)
        normal = np.cross(self.a, self.b)
        self.normal = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-15)
        self._half = 0.5 * float(np.linalg.norm(self.b - self.a, axis=1).max(initial=0.0))
        self._max_reach = float(self.reach.max(initial=0.0))
        self._tree = KDTree(0.5 * (self.a + self.b))

    def segments_near(self, centres: np.ndarray, radius: np.ndarray) -> list:
        """Per centre, the coarse segments whose midpoint is within ``radius`` + half a segment (None if none)."""
        hits = self._ctree.query_ball_point(np.asarray(centres), np.asarray(radius) + self._chalf)
        return [np.asarray(h, dtype=np.int64) if h else None for h in hits]

    def crossings(self, segs: np.ndarray, points: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """``(p,)`` summed strength of the coarse segments ``segs`` the chord ``points[i] → targets[i]`` crosses.

        Arcs ``x→c`` and ``a→b`` (short, same hemisphere) cross when ``a, b`` lie on
        opposite sides of the circle through ``x, c`` and ``x, c`` on opposite sides
        of the circle through ``a, b``.
        """
        x = np.asarray(points, dtype=np.float64)
        t = np.asarray(targets, dtype=np.float64)
        n1 = np.cross(x, t)                                            # (p, 3)
        side_a = n1 @ self.ca[segs].T                                  # (p, m)
        side_b = n1 @ self.cb[segs].T
        n2 = self.cnormal[segs]                                        # (m, 3)
        side_x = x @ n2.T
        side_t = t @ n2.T                                              # (p, m)
        cross = (side_a * side_b < 0.0) & (side_x * side_t < 0.0)
        return cross.astype(np.float64) @ self.strength[segs]

    def snap_targets(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``(idx, targets (q, 3))``: the points within reach of a river and where to read their label.

        A target is ``g·reach + (1 − g)·d`` out from the nearest segment, on the point's side.
        The "nearest" is the nearest river whose band holds the point.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        m = self.a.shape[0]
        if m == 0 or pts.shape[0] == 0:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 3))
        k = min(_K_NEAREST, m)
        _, nn = self._tree.query(pts, k=k, distance_upper_bound=self._max_reach + self._half, workers=-1)
        nn = np.asarray(nn).reshape(pts.shape[0], k)
        near = np.flatnonzero(nn[:, 0] < m)
        if near.size == 0:
            return near, np.zeros((0, 3))
        x = pts[near]
        seg = nn[near]
        valid = seg < m
        s = np.where(valid, seg, 0)
        a, b = self.a[s], self.b[s]                                    # (q, k, 3)
        ab = b - a
        t = np.clip(((x[:, None, :] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-30), 0.0, 1.0)
        foot = a + t[..., None] * ab
        dist = np.where(valid, np.linalg.norm(x[:, None, :] - foot, axis=-1), np.inf)
        # The nearest river whose band holds the point (a stream's narrow band only
        # beside it), so a border on a tributary snaps to the tributary.
        holds = dist < self.reach[s]
        j = np.argmin(np.where(holds, dist, np.inf), axis=1)
        rows = np.arange(near.size)
        best, d = s[rows, j], dist[rows, j]
        inside = holds[rows, j]
        if not inside.any():
            return np.zeros(0, dtype=np.int64), np.zeros((0, 3))
        best, foot_in, x_in, d_in = best[inside], foot[rows, j][inside], x[inside], d[inside]
        # Straight away from the river: the segment normal beside a segment, fanning
        # out round its ends so bends don't fold the band (normal on the river itself).
        side = np.where((x_in * self.normal[best]).sum(-1) >= 0.0, 1.0, -1.0)
        away = x_in - foot_in
        norm = np.linalg.norm(away, axis=1, keepdims=True)
        away = np.where(norm > 1e-12, away / np.maximum(norm, 1e-15), side[:, None] * self.normal[best])
        g = self.snap_fraction
        out = g * self.reach[best] + (1.0 - g) * d_in
        target = foot_in + out[:, None] * away
        target /= np.linalg.norm(target, axis=1, keepdims=True)
        return near[inside], target


def build_river_barriers(segments: Optional[np.ndarray], *, reach_km: float, min_flow: float,
                         full_flow: float, snap_fraction: float = 0.6) -> Optional[RiverBarriers]:
    """``RiverBarriers`` over packed ``(M, 7)`` segments; None when off or none qualifies."""
    if reach_km <= 0.0 or segments is None or len(segments) == 0:
        return None
    seg = np.asarray(segments, dtype=np.float64)
    seg = seg[seg[:, 6] > float(min_flow)]
    if seg.shape[0] == 0:
        return None
    a, b, flow = seg[:, :3], seg[:, 3:6], seg[:, 6]
    ramp = flow_ramp(flow, min_flow=min_flow, full_flow=full_flow)
    reach = float(reach_km) / EARTH_RADIUS_KM * ramp
    keep = reach > 0.0
    if not keep.any():
        return None
    return RiverBarriers(a[keep], b[keep], reach[keep], snap_fraction, strength=ramp[keep])
