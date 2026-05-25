"""Great-circle arc crossing detection and river crossing resolution.

Public API:
    crossing_pairs(p1_a, p2_a, p1_b, p2_b) -> list of (si_a, si_b, t_a, t_b, cross_pt)
    resolve_self_crossings(rivers) -> List[River]
    resolve_crossings(rivers) -> List[River]
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from worldgen.hydrology.types import River, RiverNode, RiverSegment


def _arc_param(p1: np.ndarray, p2: np.ndarray, c: np.ndarray) -> float:
    """Parameter t in [0,1] for point c on arc p1->p2 (c must lie on the arc)."""
    total = float(np.arccos(np.clip(float(p1 @ p2), -1.0, 1.0)))
    if total < 1e-10:
        return 0.0
    return float(np.clip(np.arccos(np.clip(float(p1 @ c), -1.0, 1.0)) / total, 0.0, 1.0))


def crossing_pairs(
    p1_a: np.ndarray, p2_a: np.ndarray,
    p1_b: np.ndarray, p2_b: np.ndarray,
) -> List[Tuple[int, int, float, float, np.ndarray]]:
    """Find all crossing segment pairs between two rivers using vectorized NumPy.

    For each pair (i from river A, j from river B) the great-circle planes
    intersect at two antipodal candidates; a candidate is a real crossing only
    if it lies within both arcs. We test both candidates in one broadcast pass.

    Returns list of (si_a, si_b, t_a, t_b, cross_pt).
    """
    n_a = np.cross(p1_a, p2_a)                        # (Na, 3)
    n_b = np.cross(p1_b, p2_b)                        # (Nb, 3)
    d = np.cross(n_a[:, None, :], n_b[None, :, :])    # (Na, Nb, 3)
    d_len = np.linalg.norm(d, axis=2)                 # (Na, Nb)
    valid = d_len > 1e-10                              # non-coplanar pairs

    hits: List[Tuple[int, int, float, float, np.ndarray]] = []
    for sign in (1.0, -1.0):
        safe_len = np.where(valid, d_len, 1.0)
        c = sign * d / safe_len[:, :, None]            # (Na, Nb, 3)

        # c on arc A: (p1_a × c)·n_a ≥ 0  AND  (c × p2_a)·n_a ≥ 0
        p1a_x_c = np.cross(p1_a[:, None, :], c)       # (Na, Nb, 3)
        c_x_p2a = np.cross(c, p2_a[:, None, :])       # (Na, Nb, 3)
        on_a = (
            np.sum(p1a_x_c * n_a[:, None, :], axis=2) >= -1e-6
        ) & (
            np.sum(c_x_p2a * n_a[:, None, :], axis=2) >= -1e-6
        )

        # c on arc B: (p1_b × c)·n_b ≥ 0  AND  (c × p2_b)·n_b ≥ 0
        p1b_x_c = np.cross(p1_b[None, :, :], c)       # (Na, Nb, 3)
        c_x_p2b = np.cross(c, p2_b[None, :, :])       # (Na, Nb, 3)
        on_b = (
            np.sum(p1b_x_c * n_b[None, :, :], axis=2) >= -1e-6
        ) & (
            np.sum(c_x_p2b * n_b[None, :, :], axis=2) >= -1e-6
        )

        hit_mask = valid & on_a & on_b                 # (Na, Nb)
        if not hit_mask.any():
            continue
        for i, j in zip(*np.where(hit_mask)):
            i, j = int(i), int(j)
            cp = c[i, j].copy()
            t_a = _arc_param(p1_a[i], p2_a[i], cp)
            t_b = _arc_param(p1_b[j], p2_b[j], cp)
            # Exclude endpoint touches: t ≈ 0 or t ≈ 1 on either arc means the
            # candidate is a tributary junction (one river ends at the other),
            # not a genuine interior crossing that needs truncation.
            if not (1e-3 < t_a < 1 - 1e-3 and 1e-3 < t_b < 1 - 1e-3):
                continue
            hits.append((i, j, t_a, t_b, cp))
    return hits


def resolve_self_crossings(rivers: List[River]) -> List[River]:
    """Repair rivers that fold back on themselves after Laplacian smoothing.

    For each river, finds the earliest non-adjacent segment pair (i, j) whose
    great-circle arcs cross at point C, then excises the loop by reconnecting
    upstream (segments 0..i up to C) directly to downstream (from C through
    segments j+1..N), skipping the detour i+1..j.  Applied iteratively until
    no self-crossings remain; each splice removes at least one segment so
    iteration always terminates.
    """
    result: List[River] = []
    for river in rivers:
        for _ in range(len(river.segments)):
            n = len(river.segments)
            if n <= 2:
                break

            p1 = np.array([s.start.position for s in river.segments], dtype=np.float64)
            p2 = np.array([s.end.position for s in river.segments], dtype=np.float64)
            p1 /= np.maximum(np.linalg.norm(p1, axis=1, keepdims=True), 1e-12)
            p2 /= np.maximum(np.linalg.norm(p2, axis=1, keepdims=True), 1e-12)

            best: Optional[Tuple[int, int, float, float, np.ndarray]] = None
            for i in range(n - 2):
                for _si_a, si_b, t_i, t_j, cp in crossing_pairs(
                    p1[i:i+1], p2[i:i+1], p1[i+2:], p2[i+2:]
                ):
                    j = i + 2 + si_b
                    if best is None or i < best[0] or (i == best[0] and t_i < best[1]):
                        best = (i, j, t_i, t_j, cp)

            if best is None:
                break

            i, j, t_i, t_j, cp = best
            junction = RiverNode(
                position=cp.astype(np.float32),
                elevation=0.0,
                face_id=river.segments[i].start.face_id,
                uv=(0.0, 0.0),
            )
            new_segs: List[RiverSegment] = list(river.segments[:i])
            if t_i > 1e-3:
                new_segs.append(RiverSegment(
                    start=river.segments[i].start,
                    end=junction,
                    flow_rate=river.segments[i].flow_rate,
                    in_lake=river.segments[i].in_lake,
                ))
            if t_j < 1 - 1e-3:
                new_segs.append(RiverSegment(
                    start=junction,
                    end=river.segments[j].end,
                    flow_rate=river.segments[j].flow_rate,
                    in_lake=river.segments[j].in_lake,
                ))
            new_segs.extend(river.segments[j + 1:])
            river.segments = new_segs
            if new_segs:
                river.source = new_segs[0].start
                river.mouth = new_segs[-1].end

        if len(river.segments) > 2:
            result.append(river)
    return result


def resolve_crossings(rivers: List[River]) -> List[River]:
    """Terminate lower-flow rivers at crossing points with higher-flow rivers.

    For each river pair whose axis-aligned bounding boxes overlap, runs a
    fully vectorized great-circle arc intersection test across all segment
    pairs simultaneously. The lower-flow river at each crossing is truncated
    at the crossing point; the higher-flow river is unchanged.
    """
    if len(rivers) < 2:
        return rivers

    r_p1: List[np.ndarray] = []
    r_p2: List[np.ndarray] = []
    r_fr: List[np.ndarray] = []
    for river in rivers:
        p1 = np.array([s.start.position for s in river.segments], dtype=np.float64)
        p2 = np.array([s.end.position for s in river.segments], dtype=np.float64)
        p1 /= np.maximum(np.linalg.norm(p1, axis=1, keepdims=True), 1e-12)
        p2 /= np.maximum(np.linalg.norm(p2, axis=1, keepdims=True), 1e-12)
        r_p1.append(p1)
        r_p2.append(p2)
        r_fr.append(np.array([s.flow_rate for s in river.segments], dtype=np.float64))

    bounds: List[Tuple[np.ndarray, np.ndarray]] = []
    for p1, p2 in zip(r_p1, r_p2):
        pts = np.vstack([p1, p2])
        bounds.append((pts.min(axis=0), pts.max(axis=0)))

    truncations: Dict[int, Tuple[int, float, np.ndarray]] = {}

    for ri_a in range(len(rivers)):
        for ri_b in range(ri_a + 1, len(rivers)):
            lo_a, hi_a = bounds[ri_a]
            lo_b, hi_b = bounds[ri_b]
            if np.any(lo_a > hi_b) or np.any(lo_b > hi_a):
                continue  # bounding boxes are disjoint

            for si_a, si_b, t_a, t_b, cross_pt in crossing_pairs(
                r_p1[ri_a], r_p2[ri_a],
                r_p1[ri_b], r_p2[ri_b],
            ):
                fr_at_a = float(r_fr[ri_a][si_a])
                fr_at_b = float(r_fr[ri_b][si_b])
                if fr_at_a <= fr_at_b:
                    low_ri, low_si, low_t = ri_a, si_a, t_a
                else:
                    low_ri, low_si, low_t = ri_b, si_b, t_b
                prev = truncations.get(low_ri)
                if (prev is None
                        or low_si < prev[0]
                        or (low_si == prev[0] and low_t < prev[1])):
                    truncations[low_ri] = (low_si, low_t, cross_pt)

    if not truncations:
        return rivers

    kept: List[River] = []
    for ri, river in enumerate(rivers):
        if ri not in truncations:
            kept.append(river)
            continue
        seg_idx, t, cross_pt = truncations[ri]
        new_segs = list(river.segments[:seg_idx])
        # Include a partial segment up to the crossing point when the
        # crossing isn't right at the start of that segment.
        if seg_idx < len(river.segments) and t > 1e-3:
            orig = river.segments[seg_idx]
            junction = RiverNode(
                position=cross_pt.astype(np.float32),
                elevation=0.0,
                face_id=orig.start.face_id,
                uv=(0.0, 0.0),
            )
            new_segs.append(RiverSegment(
                start=orig.start,
                end=junction,
                flow_rate=orig.flow_rate,
                in_lake=orig.in_lake,
            ))
        if len(new_segs) > 2:
            river.segments = new_segs
            river.mouth = new_segs[-1].end
            kept.append(river)
    return kept
