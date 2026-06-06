"""Population & Calibration — Stage 4 of the realm layer.

Two responsibilities:

1. **Hinterland assignment** — each land face is assigned to the settlement
   that can reach it most cheaply on the land cost graph.  Settlements with
   a lower ``influence`` offset claim wider territories.

2. **Iterative relaxation** — carrying capacity (hinterland fertility × area)
   and trade wealth (face flow at the settlement's face) combine into a per-
   settlement population estimate.  Running trade → population → trade for
   2–3 passes lets hub reinforcement stabilise without solving the loop
   analytically.  After the final pass, settlements whose population falls
   below a minimum fraction of the peak are culled.
"""
from __future__ import annotations

import dataclasses
from typing import List

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

from realmgen.placement.types import Settlement


def assign_hinterlands(
    n_faces: int,
    settlements: List[Settlement],
    cost_graph: csr_matrix,
) -> np.ndarray:
    """Assign each land face to its nearest settlement on the land cost graph.

    Runs undirected multi-source Dijkstra from all settlement nodes.  The
    effective claim cost for settlement *k* reaching face *i* is:

        dist[k, i] + settlements[k].influence

    The settlement with the smallest effective cost wins.  Settlements with
    lower ``influence`` (e.g. capitals at 0.0) claim wider territories when
    distances are otherwise equal.

    Args:
        n_faces: number of mesh faces (N).
        settlements: list of :class:`~realmgen.placement.types.Settlement`.
        cost_graph: (N, N) directed land-cost CSR matrix — used undirected.

    Returns:
        (N,) int32 — index into *settlements* for the owning settlement, or
        -1 for faces that are unreachable from any settlement.
    """
    if not settlements:
        return np.full(n_faces, -1, dtype=np.int32)

    s_nodes = np.array([s.face_id for s in settlements], dtype=np.int32)
    influence = np.array([s.influence for s in settlements], dtype=np.float64)

    # dist[k, i] = undirected cost-distance from settlements[k] to face i.
    dist = sp_dijkstra(
        cost_graph, directed=False, indices=s_nodes, return_predecessors=False,
    )
    # shape: (S, N)

    effective = dist + influence[:, np.newaxis]  # (S, N) — lower = better claim
    best_k = np.argmin(effective, axis=0).astype(np.int32)  # (N,)
    reachable = np.isfinite(effective[best_k, np.arange(n_faces)])

    return np.where(reachable, best_k, np.int32(-1))


def compute_carrying_capacity(
    hinterland: np.ndarray,
    fertility: np.ndarray,
    face_areas: np.ndarray,
    n_settlements: int,
) -> np.ndarray:
    """Sum fertility × face_area over each settlement's hinterland.

    Args:
        hinterland: (N,) int32 from :func:`assign_hinterlands` — settlement
            index per face, -1 for unassigned.
        fertility: (N,) float in [0, 1] — per-face ``fertility_mean``.
        face_areas: (N,) float64 — per-face solid-angle areas (unit-sphere).
        n_settlements: total settlement count S.

    Returns:
        (S,) float64 — raw carrying capacity (arbitrary units).
    """
    cap = np.zeros(n_settlements, dtype=np.float64)
    contribution = np.asarray(fertility, dtype=np.float64) * np.asarray(face_areas, dtype=np.float64)
    valid = hinterland >= 0
    np.add.at(cap, hinterland[valid], contribution[valid])
    return cap


def compute_trade_wealth(
    settlements: List[Settlement],
    face_flow: np.ndarray,
) -> np.ndarray:
    """Extract trade flow at each settlement's face as a wealth proxy.

    Args:
        settlements: list of :class:`~realmgen.placement.types.Settlement`.
        face_flow: (N,) float — cumulative trade flow per face.

    Returns:
        (S,) float64 — per-settlement trade wealth (non-negative).
    """
    flow = np.asarray(face_flow, dtype=np.float64)
    return np.array([float(flow[s.face_id]) for s in settlements], dtype=np.float64)


def compute_population(
    carrying_cap: np.ndarray,
    trade_wealth: np.ndarray,
    *,
    trade_weight: float = 1.5,
) -> np.ndarray:
    """Combine carrying capacity and trade wealth into population estimates.

    Formula::

        pop = carrying_cap × (1 + trade_weight × log1p(wealth / median_wealth))

    The log normalisation means large trade hubs are boosted significantly but
    are still grounded in the hinterland carrying capacity.  Settlements with
    zero carrying capacity (completely isolated) get zero population regardless
    of trade.

    Args:
        carrying_cap: (S,) from :func:`compute_carrying_capacity`.
        trade_wealth: (S,) from :func:`compute_trade_wealth`.
        trade_weight: scalar multiplier on the log-trade bonus (default 1.5).

    Returns:
        (S,) float64 — population (arbitrary scale; relative values meaningful).
    """
    cap = np.maximum(carrying_cap, 0.0)
    wealth = np.maximum(trade_wealth, 0.0)

    pos_wealth = wealth[wealth > 0.0]
    baseline = float(np.median(pos_wealth)) if pos_wealth.size > 0 else 1.0
    if baseline <= 0.0:
        baseline = 1.0

    trade_bonus = np.log1p(wealth / baseline)
    return cap * (1.0 + float(trade_weight) * trade_bonus)


def cull_settlements(
    settlements: List[Settlement],
    population: np.ndarray,
    *,
    min_pop_fraction: float = 0.005,
) -> tuple[list[Settlement], np.ndarray]:
    """Remove settlements below a minimum population fraction.

    Any settlement with ``population < min_pop_fraction × max_population`` is
    dropped.  Culling is unconditional — display tiers are re-derived from
    population at render time, so there is nothing to downgrade.

    Args:
        settlements: list of :class:`~realmgen.placement.types.Settlement`.
        population: (S,) float64 aligned with *settlements*.
        min_pop_fraction: minimum population as a fraction of the peak to
            survive (default 0.5 %).

    Returns:
        ``(settlements, population)`` — filtered list plus the aligned
        ``(S',)`` population array.
    """
    pop = np.asarray(population, dtype=np.float64)
    max_pop = float(pop.max()) if pop.size > 0 else 0.0
    threshold = float(min_pop_fraction) * max_pop

    kept_s: list[Settlement] = []
    kept_p: list[float] = []

    for s, p in zip(settlements, pop):
        if p >= threshold:
            kept_s.append(s)
            kept_p.append(float(p))

    return kept_s, np.array(kept_p, dtype=np.float64)


def apply_population_to_settlements(
    settlements: List[Settlement],
    population: np.ndarray,
) -> list[Settlement]:
    """Return a new list with ``Settlement.population`` set from *population*.

    Args:
        settlements: original list.
        population: (S,) float64 aligned with *settlements*.

    Returns:
        New list of :class:`Settlement` objects with ``population`` filled in.
    """
    pop = np.asarray(population, dtype=np.float64)
    return [dataclasses.replace(s, population=float(p)) for s, p in zip(settlements, pop)]
