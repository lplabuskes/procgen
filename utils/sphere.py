"""Geometry on the unit sphere: positions are ECEF unit vectors, distances km on the Earth-sized globe."""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

EARTH_RADIUS_KM = 6371.0


def unit(v: np.ndarray) -> np.ndarray:
    """``v`` scaled to unit length along its last axis (zero stays zero)."""
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-15)


def tangent_basis(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit ``e1`` and ``e2 = p × e1`` spanning the tangent plane at unit vector ``p``
    (referenced to the pole, or to x near the poles)."""
    ref = np.array([0.0, 0.0, 1.0]) if abs(p[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = unit(np.cross(p, ref))
    return e1, np.cross(p, e1)


def km_to_rad(km: float) -> float:
    """Arc angle (radians on the unit sphere) of ``km`` of ground."""
    return float(km) / EARTH_RADIUS_KM


def km_to_chord(km: float) -> float:
    """Unit-sphere chord of an arc ``km`` long."""
    return 2.0 * np.sin(0.5 * km / EARTH_RADIUS_KM)


def arc_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Great-circle distance between unit vectors (rows), km."""
    c = np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1)
    return 2.0 * np.arcsin(np.clip(0.5 * c, 0.0, 1.0)) * EARTH_RADIUS_KM


def great_circle_km(a: np.ndarray, b: np.ndarray) -> float:
    """Great-circle ground distance (km) between two unit vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    angle = math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.dot(a, b)))
    return angle * EARTH_RADIUS_KM


def polyline_length_km(points: np.ndarray) -> float:
    """Great-circle length (km) of a unit-sphere polyline."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(arc_km(pts[:-1], pts[1:]).sum())


def cumulative_chord(pos: np.ndarray) -> np.ndarray:
    """``(P,)`` chord length along a ``(P, 3)`` polyline up to each vertex (0 at the first)."""
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])


def densify_polyline(points: np.ndarray, step_km: float) -> np.ndarray:
    """Unit-sphere polyline resampled to at most ``step_km`` between points (each re-normalised)."""
    pts = np.asarray(points, dtype=np.float64)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        k = max(int(np.ceil(float(arc_km(a, b)) / max(step_km, 1e-9))), 1)
        for j in range(1, k + 1):
            out.append(unit(a + (b - a) * (j / k)))
    return np.asarray(out)


def densify_polylines(polylines, max_step: float) -> Optional[np.ndarray]:
    """All vertices of ``polylines`` plus interpolated points so no gap exceeds
    ``max_step`` (chord). ``None`` when there are no points.

    Per polyline: its vertices, then each split segment's interior points
    ``1/n … (n-1)/n`` in turn.  Vectorised over all polylines at once.
    """
    lines = [p for p in (np.asarray(x, dtype=np.float64) for x in polylines) if p.ndim == 2 and p.shape[0] > 0]
    if not lines:
        return None
    sizes = np.array([len(p) for p in lines], dtype=np.int64)
    pts = np.concatenate(lines, axis=0)
    if max_step <= 0 or len(pts) < 2:
        return pts
    line_of = np.repeat(np.arange(len(lines)), sizes)
    # Segments between consecutive vertices of one polyline.
    seg_ok = line_of[:-1] == line_of[1:]
    a, b = pts[:-1], pts[1:]
    n_sub = np.ceil(np.linalg.norm(b - a, axis=1) / max_step).astype(np.int64)
    count = np.where(seg_ok & (n_sub > 1), n_sub - 1, 0)
    if not count.any():
        return pts
    # Output rows: each polyline's vertices, then its interior points.
    interior = np.bincount(line_of[:-1], weights=count, minlength=len(lines)).astype(np.int64)
    start = np.cumsum(sizes + interior) - (sizes + interior)
    out = np.empty((int((sizes + interior).sum()), 3))
    out[start[line_of] + (np.arange(len(pts)) - np.repeat(np.cumsum(sizes) - sizes, sizes))] = pts
    split = np.flatnonzero(count)
    seg = np.repeat(split, count[split])
    step = np.arange(len(seg)) - np.repeat(np.cumsum(count[split]) - count[split], count[split]) + 1
    t = (step / n_sub[seg])[:, None]
    # Rank of each interior point within its polyline's interior block.
    rank = np.arange(len(seg)) - (np.cumsum(interior) - interior)[line_of[seg]]
    out[start[line_of[seg]] + sizes[line_of[seg]] + rank] = a[seg] + t * (b[seg] - a[seg])
    return out


def laplacian_smooth_sphere(
    polyline: List[np.ndarray],
    iterations: int = 8,
    alpha: float = 0.5,
) -> List[np.ndarray]:
    """Laplacian smoothing on the unit sphere, re-projecting each iteration; endpoints pinned."""
    if len(polyline) < 3:
        return [p.astype(np.float32, copy=True) for p in polyline]

    pts = np.stack(polyline).astype(np.float64)
    for _ in range(iterations):
        mids = 0.5 * (pts[:-2] + pts[2:])
        pts[1:-1] += alpha * (mids - pts[1:-1])
        norms = np.linalg.norm(pts[1:-1], axis=1, keepdims=True)
        np.divide(pts[1:-1], norms, out=pts[1:-1], where=norms > 0)
    return [pts[i].astype(np.float32) for i in range(len(pts))]


def slerp_points(a: np.ndarray, b: np.ndarray, t: np.ndarray, angle: float) -> np.ndarray:
    """``(len(t), 3)`` points at fractions ``t`` along the great circle from ``a`` to ``b``, ``angle`` (rad) apart.

    (Near-)antipodal ends have no unique great circle; then just ``[a, b]``.
    """
    s = math.sin(angle)
    if s < 1e-9:
        return np.vstack([a, b])
    t = np.asarray(t, dtype=np.float64)[:, None]
    pts = (np.sin((1.0 - t) * angle) * a + np.sin(t * angle) * b) / s
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)
