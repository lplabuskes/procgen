"""Polity formation — pure functions over the land cost graph.

Stage 5 of the realm layer.  Four responsibilities, each a pure function that
takes pre-extracted arrays (no Globe coupling) so it can be tested against a
lightweight stub graph, mirroring :mod:`realmgen.population.calibration`:

1. **Capital seeding** (:func:`seed_capitals`) — pick which settlements found
   polities: walk settlements by population descending and keep one only if it
   is beyond ``min_separation`` cost-distance from every already-chosen capital
   and above ``min_pop_fraction`` of the peak population.  Emergent count,
   spatially spread (central-place spacing on the land surface).

2. **Territory growth** (:func:`grow_territory`) — a *power-weighted* Voronoi on
   the land cost graph.  Each face joins the capital with the smallest
   *political* distance ``cost_dist − offset``, where a more powerful capital
   gets a larger negative head-start ``offset``.  Because cost-distance grows
   fast across rugged terrain and is infinite across water, borders settle on
   natural barriers and coastlines, and unreachable islands stay outside any
   territory.

3. **Power scoring** (:func:`compute_polity_power`) — each polity's strength is
   the total population of the settlements inside its territory.

4. **Border tension** (:func:`compute_border_tension`) — along every shared
   border, tension rises with *permeability* (low land cost across the border),
   *trade-artery value* (trade flow on the border faces), and *power balance*
   (how evenly matched the two polities are).  A strong mountain barrier or a
   lopsided power gap suppresses tension; a permeable, high-traffic border
   between matched rivals is the fortification candidate.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra


# ── Capital seeding ─────────────────────────────────────────────────────────────


def seed_capitals(
    settlement_faces: Sequence[int],
    populations: np.ndarray,
    cost_graph: csr_matrix,
    *,
    min_separation: float,
    min_pop_fraction: float,
) -> List[int]:
    """Choose which settlements found polities — population + cost-distance spacing.

    Settlements are considered in descending population order.  A candidate
    becomes a capital only if its population is at least
    ``min_pop_fraction × peak`` and its face is at least ``min_separation``
    cost-distance (undirected, on the land graph) from every capital already
    chosen.  The running minimum distance is updated with one Dijkstra per
    accepted capital, exactly like the settlement seeder's exclusion mask.

    Args:
        settlement_faces: face index of each settlement (length S).
        populations: (S,) float — settlement populations, index-aligned.
        cost_graph: (N, N) land-cost CSR matrix — used undirected.
        min_separation: minimum cost-distance between two capitals.
        min_pop_fraction: minimum population as a fraction of the peak to qualify.

    Returns:
        Indices into ``settlement_faces`` of the chosen capitals, in descending
        population order (deterministic; population ties broken by index).
    """
    faces = np.asarray(settlement_faces, dtype=np.int64)
    pop = np.asarray(populations, dtype=np.float64)
    if faces.size == 0:
        return []

    n_faces = cost_graph.shape[0]
    peak = float(pop.max())
    threshold = float(min_pop_fraction) * peak
    sep = max(float(min_separation), 0.0)

    order = np.argsort(-pop, kind="stable")
    min_dist = np.full(n_faces, np.inf, dtype=np.float64)
    chosen: List[int] = []

    for idx in order.tolist():
        if pop[idx] < threshold:
            break  # remaining settlements are all smaller → none qualify
        face = int(faces[idx])
        if min_dist[face] < sep:
            continue  # too close to an existing capital
        chosen.append(idx)
        d = sp_dijkstra(cost_graph, directed=False, indices=face, return_predecessors=False)
        np.minimum(min_dist, d, out=min_dist)

    return chosen


# ── Territory growth ────────────────────────────────────────────────────────────


def grow_territory(
    n_faces: int,
    capital_faces: Sequence[int],
    capital_powers: np.ndarray,
    cost_graph: csr_matrix,
    *,
    power_bias: float,
    max_reach: float,
) -> np.ndarray:
    """Power-weighted Voronoi territory assignment on the land cost graph.

    Runs one undirected multi-source Dijkstra from all capital faces, then
    assigns each face to the capital ``k`` minimising the *political distance*::

        offset[k]       = power_bias · log(power[k] / max_power)      # ≤ 0
        pol_dist[k, i]  = cost_dist[k, i] − offset[k]

    The strongest capital has ``offset 0`` (no handicap); weaker capitals carry
    a positive handicap proportional to ``power_bias`` and the log power gap, so
    they cede contested ground to stronger neighbours.  A face whose *raw*
    cost-distance to its nearest capital exceeds ``max_reach`` — or which no
    capital can reach overland at all — is left unclaimed (``-1``).  Each
    capital's own face is forced to its own polity.

    Args:
        n_faces: number of mesh faces (N).
        capital_faces: face index of each capital (length C).
        capital_powers: (C,) float — capital power (population) for the
            head-start, index-aligned with ``capital_faces``.
        cost_graph: (N, N) land-cost CSR matrix — used undirected.  Must encode
            water impassability (no finite edges to/from ocean ∪ lake faces) for
            coastlines to act as borders and islands to stay unclaimed; the
            caller is responsible for masking those edges.
        power_bias: cost-distance territory advantage conferred per e-fold of
            relative power; 0 → a pure cost-distance Voronoi.
        max_reach: raw cost-distance beyond which a capital claims nothing.

    Returns:
        (N,) int32 — polity id (0..C-1) per face, or -1 where unclaimed/water.
    """
    caps = np.asarray(capital_faces, dtype=np.int64)
    if caps.size == 0:
        return np.full(n_faces, -1, dtype=np.int32)

    power = np.asarray(capital_powers, dtype=np.float64)
    max_power = float(power.max())
    if max_power <= 0.0:
        offset = np.zeros(caps.size, dtype=np.float64)
    else:
        norm = np.clip(power / max_power, 1e-6, 1.0)
        offset = float(power_bias) * np.log(norm)  # ≤ 0

    # cost_dist[k, i] = undirected land cost-distance from capital k to face i.
    cost_dist = sp_dijkstra(
        cost_graph, directed=False, indices=caps, return_predecessors=False,
    )  # (C, N)

    pol_dist = cost_dist - offset[:, np.newaxis]            # (C, N)
    reachable = cost_dist <= max(float(max_reach), 0.0)     # raw-distance cap
    pol_dist = np.where(reachable, pol_dist, np.inf)

    best = np.argmin(pol_dist, axis=0)                      # (N,)
    claimed = np.isfinite(pol_dist.min(axis=0))
    polity_of_face = np.where(claimed, best, -1).astype(np.int32)

    # A capital always rules its own seat regardless of params.
    polity_of_face[caps] = np.arange(caps.size, dtype=np.int32)
    return polity_of_face


# ── Power scoring ───────────────────────────────────────────────────────────────


def compute_polity_power(
    polity_of_face: np.ndarray,
    settlement_faces: Sequence[int],
    populations: np.ndarray,
    n_polities: int,
) -> np.ndarray:
    """Total settlement population inside each polity's territory.

    Args:
        polity_of_face: (N,) int — polity id per face (-1 = unclaimed).
        settlement_faces: face index of each settlement (length S).
        populations: (S,) float — settlement populations, index-aligned.
        n_polities: number of polities (C).

    Returns:
        (C,) float64 — summed member-settlement population per polity.
    """
    polity_of_face = np.asarray(polity_of_face)
    faces = np.asarray(settlement_faces, dtype=np.int64)
    pop = np.asarray(populations, dtype=np.float64)
    power = np.zeros(int(n_polities), dtype=np.float64)
    if faces.size == 0:
        return power
    pids = polity_of_face[faces]
    valid = pids >= 0
    np.add.at(power, pids[valid], pop[valid])
    return power


# ── Border tension ──────────────────────────────────────────────────────────────


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
    """Score inter-polity tension at border crossings.

    Tension is fundamentally a property of a *crossing* — the place a road or
    trade route punches through the frontier, and the place a fort would stand.
    For every border edge ``(i, j)`` between two polities the crossing tension
    combines a barrier, a power-balance, and an artery term::

        permeability = exp(-edge_cost / barrier_scale)          # cheap gap → ~1
        balance      = min(P_a, P_b) / max(P_a, P_b)            # matched → ~1
        artery       = (1 - artery_weight) + artery_weight · tanh(crossing_flow / flow_ref)
        crossing     = permeability · balance · artery

    where ``crossing_flow`` is the mean trade flow of the two faces.  Two evenly
    matched powers sharing a permeable gap are in tension even with no trade
    crossing it (military, not merely economic), so the artery term keeps a
    baseline of ``1 - artery_weight`` and trade *amplifies* it toward 1 — rather
    than being a pure multiplier that can zero tension out wherever a border
    happens to fall in trade-dead wilderness.  A strong natural barrier or a
    lopsided power gap still pull tension toward zero; the cheapest, busiest
    crossing between evenly matched rivals scores highest — the fortification
    candidate.

    A border's overall tension is its *hottest* crossing (``max``), not the mean
    of its whole length, so a single artery through an otherwise quiet wilderness
    frontier still flags the border as contested.  The per-face tension takes,
    for each face, the hottest crossing on its own edges — concentrating the
    heat on the actual chokepoints for display and for Stage 6 fort placement.

    Args:
        face_adjacency: ``list[list[int]]`` neighbour indices, length N.
        polity_of_face: (N,) int — polity id per face (-1 = unclaimed/water).
        polity_power: (C,) float — per-polity power from
            :func:`compute_polity_power`.
        land_edge_costs: per-face edge-cost arrays (from
            ``build_land_edge_costs``), aligned with ``face_adjacency``.
        face_flow: (N,) float — trade flow per face (any non-negative scale;
            ``flow_ref`` is interpreted in the same units).
        barrier_scale: edge-cost e-fold for the permeability term.
        flow_ref: reference flow for the artery term (``tanh`` half-saturates
            near ``1.1 × flow_ref``).
        artery_weight: fraction of the artery term driven by trade (in [0, 1]).
            The remaining ``1 - artery_weight`` is the trade-independent baseline
            so a permeable, matched border is always at least mildly contested.

    Returns:
        ``(pair_tension, face_tension)`` where ``pair_tension`` maps
        ``(a, b)`` (a < b) → the hottest-crossing tension in [0, 1] and
        ``face_tension`` is an ``(N,)`` float64 per-face heat for display.
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
        adj = face_adjacency[i]
        costs = land_edge_costs[i]
        best_face = 0.0
        for idx, j in enumerate(adj):
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
            if crossing > best_face:
                best_face = crossing
            if crossing > pair_tension.get(key, 0.0):
                pair_tension[key] = crossing
        face_tension[i] = best_face

    return pair_tension, face_tension
