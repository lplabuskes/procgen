"""Great-circle arc crossing detection and self-crossing removal.

Public API:
    crossing_pairs(p1_a, p2_a, p1_b, p2_b) -> list of (si_a, si_b, t_a, t_b, cross_pt)
    resolve_self_crossings(rivers) -> List[River]
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
