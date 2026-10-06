"""Inter-polity border tension, scored on the final sovereign partition."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def border_crossings(
    face_adjacency: List[List[int]],
    polity_of_face: np.ndarray,
    polity_power: np.ndarray,
    land_edge_costs: List[np.ndarray],
    face_flow: np.ndarray,
    *,
    barrier_scale: float,
    flow_ref: float,
    artery_weight: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every passable border edge, once from each side, scored::

        permeability = exp(-edge_cost / barrier_scale)     # cheap gap -> ~1
        balance      = min(P_a, P_b) / max(P_a, P_b)       # matched -> ~1
        artery       = (1 - artery_weight) + artery_weight * tanh(mean_flow / flow_ref)
        crossing     = permeability * balance * artery

    Returns ``(from_face (E,), to_face (E,), crossing (E,) in [0, 1])``.
    """
    polity_of_face = np.asarray(polity_of_face)
    flow = np.asarray(face_flow, dtype=np.float64)
    bscale = max(float(barrier_scale), 1e-9)
    fref = max(float(flow_ref), 1e-9)
    aw = float(np.clip(artery_weight, 0.0, 1.0))
    power = np.asarray(polity_power, dtype=np.float64)

    def _balance(a: int, b: int) -> float:
        pa = float(power[a]) if a < power.size else 0.0
        pb = float(power[b]) if b < power.size else 0.0
        hi = max(pa, pb)
        return (min(pa, pb) / hi) if hi > 0.0 else 0.0

    src: List[int] = []
    dst: List[int] = []
    score: List[float] = []
    for i, adj in enumerate(face_adjacency):
        pi = int(polity_of_face[i])
        if pi < 0:
            continue
        costs = land_edge_costs[i]
        for idx, j in enumerate(adj):
            pj = int(polity_of_face[j])
            if pj < 0 or pj == pi:
                continue
            c = float(costs[idx])
            if not np.isfinite(c):
                continue
            permeability = float(np.exp(-c / bscale))
            artery = (1.0 - aw) + aw * float(np.tanh(0.5 * (flow[i] + flow[j]) / fref))
            src.append(i)
            dst.append(int(j))
            score.append(permeability * _balance(min(pi, pj), max(pi, pj)) * artery)
    return (np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64),
            np.array(score, dtype=np.float64))


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

    Crossings as :func:`border_crossings`. A border's tension is its hottest
    crossing; a face's is the hottest crossing on its own edges, concentrating
    heat on chokepoints.

    Returns ``(pair_tension {(a<b): t in [0,1]}, face_tension (N,) float64)``.
    """
    polity_of_face = np.asarray(polity_of_face)
    src, dst, crossing = border_crossings(
        face_adjacency, polity_of_face, polity_power, land_edge_costs, face_flow,
        barrier_scale=barrier_scale, flow_ref=flow_ref, artery_weight=artery_weight,
    )
    face_tension = np.zeros(len(face_adjacency), dtype=np.float64)
    np.maximum.at(face_tension, src, crossing)
    pair_tension: Dict[Tuple[int, int], float] = {}
    for i, j, t in zip(src.tolist(), dst.tolist(), crossing.tolist()):
        pi, pj = int(polity_of_face[i]), int(polity_of_face[j])
        key = (pi, pj) if pi < pj else (pj, pi)
        if t > pair_tension.get(key, 0.0):
            pair_tension[key] = t
    return pair_tension, face_tension
