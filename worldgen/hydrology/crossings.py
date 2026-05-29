"""Great-circle arc crossing detection and river crossing resolution.

Public API:
    crossing_pairs(p1_a, p2_a, p1_b, p2_b) -> list of (si_a, si_b, t_a, t_b, cross_pt)
    resolve_self_crossings(rivers) -> List[River]
    resolve_crossings(rivers) -> List[River]
"""
from __future__ import annotations

from typing import List, Optional, Tuple

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
    """Vectorized great-circle crossing detection.

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
            # Exclude endpoint touches (tributary junctions, not crossings).
            if not (1e-3 < t_a < 1 - 1e-3 and 1e-3 < t_b < 1 - 1e-3):
                continue
            hits.append((i, j, t_a, t_b, cp))
    return hits


def resolve_self_crossings(rivers: List[River]) -> List[River]:
    """Remove loops caused by Laplacian smoothing.

    Finds the earliest crossing, excises the detour by splicing upstream of i
    to downstream of j at the crossing point. Iterates until none remain.
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
    """Terminate lower-flow rivers at their crossing point with higher-flow rivers.

    Processed in descending flow order so terminus targets are finalized first.
    Already-truncated rivers are re-checked against current geometry.
    """
    if len(rivers) < 2:
        return rivers

    # Mutable per-river segment arrays; updated in-place as rivers are truncated.
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

    def _aabb(ri: int) -> Tuple[np.ndarray, np.ndarray]:
        pts = np.vstack([r_p1[ri], r_p2[ri]])
        return pts.min(axis=0), pts.max(axis=0)

    def _update_arrays(ri: int) -> None:
        segs = rivers[ri].segments
        p1 = np.array([s.start.position for s in segs], dtype=np.float64)
        p2 = np.array([s.end.position for s in segs], dtype=np.float64)
        p1 /= np.maximum(np.linalg.norm(p1, axis=1, keepdims=True), 1e-12)
        p2 /= np.maximum(np.linalg.norm(p2, axis=1, keepdims=True), 1e-12)
        r_p1[ri] = p1
        r_p2[ri] = p2
        r_fr[ri] = np.array([s.flow_rate for s in segs], dtype=np.float64)

    # Collect every crossing: (flow_low, si_low, t_low, ri_low, ri_high, cross_pt).
    all_crossings: List[Tuple] = []
    for ri_a in range(len(rivers)):
        lo_a, hi_a = _aabb(ri_a)
        for ri_b in range(ri_a + 1, len(rivers)):
            lo_b, hi_b = _aabb(ri_b)
            if np.any(lo_a > hi_b) or np.any(lo_b > hi_a):
                continue
            for si_a, si_b, t_a, t_b, cross_pt in crossing_pairs(
                r_p1[ri_a], r_p2[ri_a], r_p1[ri_b], r_p2[ri_b],
            ):
                fr_a = float(r_fr[ri_a][si_a])
                fr_b = float(r_fr[ri_b][si_b])
                if fr_a <= fr_b:
                    all_crossings.append((fr_a, si_a, t_a, ri_a, ri_b, cross_pt))
                else:
                    all_crossings.append((fr_b, si_b, t_b, ri_b, ri_a, cross_pt))

    if not all_crossings:
        return rivers

    # Deduplicate: for each unique river pair, keep only the crossing with the
    # minimum flow_low. This gives each pair exactly one truncation event and
    # eliminates role-flip double-truncations.
    pair_best: dict = {}
    for crossing in all_crossings:
        ri_low, ri_high = crossing[3], crossing[4]
        pair = (min(ri_low, ri_high), max(ri_low, ri_high))
        if pair not in pair_best or crossing[0] < pair_best[pair][0]:
            pair_best[pair] = crossing
    all_crossings = list(pair_best.values())

    # Descending flow: terminus targets finalized before the rivers that truncate on them.
    all_crossings.sort(key=lambda x: (-x[0], x[1], x[2]))

    truncated: set = set()
    dropped: set = set()

    for _flow_low, si_low, t_low, ri_low, ri_high, cross_pt in all_crossings:
        if ri_low in dropped:
            continue

        if ri_low in truncated or ri_high in truncated:  # re-verify against current geometry
            if ri_high in dropped:
                continue
            lo_low, hi_low = _aabb(ri_low)
            lo_high, hi_high = _aabb(ri_high)
            if np.any(lo_low > hi_high) or np.any(lo_high > hi_low):
                continue
            best: Optional[Tuple] = None
            for new_si_a, new_si_b, new_t_a, _new_t_b, new_cp in crossing_pairs(
                r_p1[ri_low], r_p2[ri_low], r_p1[ri_high], r_p2[ri_high],
            ):
                fr_low2 = float(r_fr[ri_low][new_si_a])
                fr_high2 = float(r_fr[ri_high][new_si_b])
                if fr_low2 > fr_high2:
                    continue  # roles flipped; skip
                if (best is None
                        or new_si_a < best[0]
                        or (new_si_a == best[0] and new_t_a < best[1])):
                    best = (new_si_a, new_t_a, new_cp)
            if best is None:
                continue
            si_low, t_low, cross_pt = best

        # Truncate rivers[ri_low] at (si_low, t_low).
        river = rivers[ri_low]
        new_segs = list(river.segments[:si_low])
        if si_low < len(river.segments) and t_low > 1e-3:
            orig = river.segments[si_low]
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
        if len(new_segs) <= 2:
            dropped.add(ri_low)
            continue
        river.segments = new_segs
        river.mouth = new_segs[-1].end
        _update_arrays(ri_low)
        truncated.add(ri_low)

    return [r for i, r in enumerate(rivers) if i not in dropped]
