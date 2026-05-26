"""Hydrological routing algorithms.

Priority-Flood depression filling (Planchon & Darboux, 2001) and
D-infinity multi-directional flow routing (Tarboton, 1997).

Public API:
    priority_flood(elevations, ocean_mask, neighbors_fn) -> (eff, was_raised)
    compute_dinf_flow(eff, n_nodes, n_faces, pixels_per_face, resolution)
        -> (receiver1, receiver2, frac1, frac2, flow_angle)
    inter_face_fallback(receiver1, receiver2, frac1, frac2, flow_angle, eff,
                        inter_adj, ocean_mask, positions, u_axes, v_axes,
                        pixels_per_face) -> None
    accumulate_flow(receiver1, receiver2, frac1, frac2, eff,
                    precip_weights=None) -> flow_accum
"""
from __future__ import annotations

import heapq
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def priority_flood(
    elevations: np.ndarray,
    ocean_mask: np.ndarray,
    neighbors_fn: Callable[[int], List[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Planchon-Darboux depression filling. Returns (eff, was_raised).

    was_raised[i] is True for land nodes inside a topographic depression.
    Per-step epsilon tilts the filled surface toward the spill point so the
    bilinear gradient doesn't go to zero inside lakes.
    """
    n = len(elevations)
    eff = elevations.astype(np.float32, copy=True)
    visited = np.zeros(n, dtype=bool)
    was_raised = np.zeros(n, dtype=bool)
    eps = 1e-6

    heap: List[Tuple[float, int]] = []
    for idx in np.where(ocean_mask)[0].tolist():
        heapq.heappush(heap, (float(elevations[idx]), idx))
        visited[idx] = True

    while heap:
        eff_elev, key = heapq.heappop(heap)
        for nb in neighbors_fn(key):
            if visited[nb]:
                continue
            visited[nb] = True
            nb_elev = float(elevations[nb])
            # Node is inside a depression: its natural elevation is below
            # the fill level being propagated from the spill point.
            if nb_elev < eff_elev:
                was_raised[nb] = True
            # Imposed-gradient fill: force a tiny downhill from key to nb
            # whenever the real elevation would create a flat or sink.
            nb_eff = max(nb_elev, eff_elev + eps)
            eff[nb] = nb_eff
            heapq.heappush(heap, (nb_eff, nb))
    return eff, was_raised


def compute_dinf_flow(
    eff: np.ndarray,
    n_nodes: int,
    n_faces: int,
    pixels_per_face: int,
    resolution: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tarboton D-infinity flow direction on each face's UV grid.

    8 triangular facets per node; steepest descending facet wins; flow splits
    between cardinal and diagonal vertices. Returns (r1, r2, f1, f2, flow_angle).
    Boundary nodes with no intra-face descent need inter_face_fallback.
    """
    receiver1 = np.full(n_nodes, -1, dtype=np.int32)
    receiver2 = np.full(n_nodes, -1, dtype=np.int32)
    frac1 = np.zeros(n_nodes, dtype=np.float32)
    frac2 = np.zeros(n_nodes, dtype=np.float32)
    flow_angle = np.full(n_nodes, np.nan, dtype=np.float32)
    best_slope = np.full(n_nodes, -np.inf, dtype=np.float32)

    res = resolution
    d_inv = float(res - 1)                # 1 / (UV grid cell spacing)
    inv_root2 = 1.0 / float(np.sqrt(2.0))
    pi_4 = float(np.pi / 4.0)

    face_offsets = (np.arange(n_faces, dtype=np.int64) * pixels_per_face)[:, None, None]

    # Each facet: (cardinal_dr, cardinal_dc, diag_dr, diag_dc, base, sign).
    # +V (row+1) is north, +U (col+1) is east. `base` is the global UV angle
    # of the cardinal direction; `sign` indicates whether the diagonal lies
    # CCW (+1) or CW (-1) from the cardinal in the (U east, V north) frame.
    facets = [
        (0, +1, +1, +1, 0.0,            +1.0),   # E + NE
        (+1, 0, +1, +1, np.pi / 2,      -1.0),   # N + NE
        (+1, 0, +1, -1, np.pi / 2,      +1.0),   # N + NW
        (0, -1, +1, -1, np.pi,          -1.0),   # W + NW
        (0, -1, -1, -1, np.pi,          +1.0),   # W + SW
        (-1, 0, -1, -1, -np.pi / 2,     -1.0),   # S + SW
        (-1, 0, -1, +1, -np.pi / 2,     +1.0),   # S + SE
        (0, +1, -1, +1, 0.0,            -1.0),   # E + SE
    ]

    for c_dr, c_dc, d_dr, d_dc, base, sign in facets:
        r_lo = max(0, -c_dr, -d_dr)
        r_hi = res - max(0, c_dr, d_dr)
        col_lo = max(0, -c_dc, -d_dc)
        col_hi = res - max(0, c_dc, d_dc)
        if r_lo >= r_hi or col_lo >= col_hi:
            continue

        rr = np.arange(r_lo, r_hi)
        cc = np.arange(col_lo, col_hi)
        R, C = np.meshgrid(rr, cc, indexing='ij')
        src_local = (R * res + C)[None, :, :]
        card_local = ((R + c_dr) * res + (C + c_dc))[None, :, :]
        diag_local = ((R + d_dr) * res + (C + d_dc))[None, :, :]

        src_flat = (face_offsets + src_local).reshape(-1)
        card_flat = (face_offsets + card_local).reshape(-1)
        diag_flat = (face_offsets + diag_local).reshape(-1)

        e0 = eff[src_flat]
        e1 = eff[card_flat]
        e2 = eff[diag_flat]

        s1 = (e0 - e1) * d_inv           # cardinal slope
        s2 = (e1 - e2) * d_inv           # orthogonal slope (toward diagonal)
        diag_slope = (e0 - e2) * d_inv * inv_root2

        r = np.arctan2(s2, s1)
        # Outside [0, pi/4]: clamp angle and use the bounding-edge slope.
        slope = np.where(
            r < 0.0,
            s1,
            np.where(r > pi_4, diag_slope, np.hypot(s1, s2)),
        )
        r_clamped = np.clip(r, 0.0, pi_4)

        better = (slope > 0.0) & (slope > best_slope[src_flat])
        if not better.any():
            continue
        idx = src_flat[better]
        sl = slope[better].astype(np.float32)
        rc = r_clamped[better]

        best_slope[idx] = sl
        f2 = (rc / pi_4).astype(np.float32)
        f1 = (1.0 - f2).astype(np.float32)
        receiver1[idx] = card_flat[better]
        receiver2[idx] = diag_flat[better]
        frac1[idx] = f1
        frac2[idx] = f2
        flow_angle[idx] = (base + sign * rc).astype(np.float32)

    return receiver1, receiver2, frac1, frac2, flow_angle


def inter_face_fallback(
    receiver1: np.ndarray,
    receiver2: np.ndarray,
    frac1: np.ndarray,
    frac2: np.ndarray,
    flow_angle: np.ndarray,
    eff: np.ndarray,
    inter_adj: Dict[int, List[int]],
    ocean_mask: np.ndarray,
    positions: np.ndarray,
    u_axes: np.ndarray,
    v_axes: np.ndarray,
    pixels_per_face: int,
) -> None:
    """Route boundary nodes with no descending intra-face facet to their
    steepest inter-face neighbour (D8 single receiver). Updates in place.
    """
    if not inter_adj:
        return
    for src, neighbors in inter_adj.items():
        if receiver1[src] != -1:
            continue
        e0 = float(eff[src])
        if ocean_mask[src]:
            continue
        best_slope = 0.0
        best_nb = -1
        for nb in neighbors:
            delta = e0 - float(eff[nb])
            if delta <= 0.0:
                continue
            diff = positions[src] - positions[nb]
            d = float(np.sqrt(diff @ diff))
            if d <= 0.0:
                continue
            slope = delta / d
            if slope > best_slope:
                best_slope = slope
                best_nb = nb
        if best_nb < 0:
            continue
        face = src // pixels_per_face
        direction = positions[best_nb] - positions[src]
        du = float(direction @ u_axes[face])
        dv = float(direction @ v_axes[face])
        receiver1[src] = best_nb
        receiver2[src] = -1
        frac1[src] = 1.0
        frac2[src] = 0.0
        flow_angle[src] = float(np.arctan2(dv, du))


def accumulate_flow(
    receiver1: np.ndarray,
    receiver2: np.ndarray,
    frac1: np.ndarray,
    frac2: np.ndarray,
    eff: np.ndarray,
    precip_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Multi-receiver accumulation in eff-descending order."""
    n = len(eff)
    if precip_weights is not None:
        flow_accum = precip_weights.copy()
    else:
        flow_accum = np.ones(n, dtype=np.float32)
    order = np.argsort(-eff)
    for key in order.tolist():
        r1 = int(receiver1[key])
        if r1 < 0:
            continue
        contrib = float(flow_accum[key])
        r2 = int(receiver2[key])
        if r2 < 0:
            flow_accum[r1] += contrib
        else:
            flow_accum[r1] += contrib * float(frac1[key])
            flow_accum[r2] += contrib * float(frac2[key])
    return flow_accum
