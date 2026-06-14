"""Inter-polity border tension, scored on the final sovereign partition."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def compute_border_tension(
    face_adjacency: List[List[int]],
    polity_of_face: np.ndarray,
    polity_power: np.ndarray,
    land_edge_costs: List[np.ndarray],
    face_flow: np.ndarray,
    *,
    barrier_scale: float,
    flow_ref: float,
    artery_weight: float = 0.6,
) -> Tuple[Dict[Tuple[int, int], float], np.ndarray]:
    """Score tension at border crossings — where a road punches the frontier.

    For each border edge ``(i, j)``::

        permeability = exp(-edge_cost / barrier_scale)     # cheap gap -> ~1
        balance      = min(P_a, P_b) / max(P_a, P_b)       # matched -> ~1
        artery       = (1 - artery_weight) + artery_weight * tanh(mean_flow / flow_ref)
        crossing     = permeability * balance * artery

    Evenly matched powers sharing a permeable gap are in tension even without
    trade (the artery term keeps a ``1 - artery_weight`` baseline; trade amplifies
    toward 1). A border's tension is its hottest crossing; a face's tension is the
    hottest crossing on its own edges, concentrating heat on chokepoints.

    Returns ``(pair_tension {(a<b): t in [0,1]}, face_tension (N,) float64)``.
    """
    polity_of_face = np.asarray(polity_of_face)
    flow = np.asarray(face_flow, dtype=np.float64)
    n = len(face_adjacency)

    bscale = max(float(barrier_scale), 1e-9)
    fref = max(float(flow_ref), 1e-9)
    aw = float(np.clip(artery_weight, 0.0, 1.0))
    power = np.asarray(polity_power, dtype=np.float64)

    def _balance(a: int, b: int) -> float:
        pa = float(power[a]) if a < power.size else 0.0
        pb = float(power[b]) if b < power.size else 0.0
        hi = max(pa, pb)
        return (min(pa, pb) / hi) if hi > 0.0 else 0.0

    pair_tension: Dict[Tuple[int, int], float] = {}
    face_tension = np.zeros(n, dtype=np.float64)

    for i in range(n):
        pi = int(polity_of_face[i])
        if pi < 0:
            continue
        costs = land_edge_costs[i]
        best_face = 0.0
        for idx, j in enumerate(face_adjacency[i]):
            pj = int(polity_of_face[j])
            if pj < 0 or pj == pi:
                continue
            c = float(costs[idx])
            if not np.isfinite(c):
                continue
            permeability = float(np.exp(-c / bscale))
            artery = (1.0 - aw) + aw * float(np.tanh(0.5 * (flow[i] + flow[j]) / fref))
            key = (pi, pj) if pi < pj else (pj, pi)
            crossing = permeability * _balance(*key) * artery
            best_face = max(best_face, crossing)
            if crossing > pair_tension.get(key, 0.0):
                pair_tension[key] = crossing
        face_tension[i] = best_face

    return pair_tension, face_tension
