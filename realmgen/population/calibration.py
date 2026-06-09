"""Population & Calibration — Stage 4 of the realm layer.

Two responsibilities:

1. **Hinterland assignment** — soft, fractional.  Each settlement bids for
   every face within ``max_distance`` cost-distance; the bid combines a
   linear proximity falloff with an influence-derived pull, and each face's
   carrying capacity is split among bidders in proportion to their bids.  A
   village beside a city therefore still captures a share of the surrounding
   land instead of being shut out, so it can survive in the city's shadow.
   The *displayed* hinterland is the single highest bidder per face.  When the
   caller supplies a *water-permeable* cost graph (the population stage does),
   a coastal settlement also bids for a *limited* band of nearby ocean/lake,
   pulling its fishing-grounds productivity (the aquatic component of
   ``fertility_mean``) into carrying capacity; the band stays small because
   water steps are made deliberately expensive.

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


def compute_hinterland_shares(
    n_faces: int,
    settlements: List[Settlement],
    cost_graph: csr_matrix,
    *,
    max_distance: float,
    influence_scale: float = 4.5,
    reach_scale: float = 9.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft hinterland assignment — fractional capacity shares + display owner.

    Runs undirected multi-source Dijkstra from all settlement nodes, then has
    each settlement *k* bid for every face *i*::

        influence[k]    = log(max_pop / pop[k])          # 0 for top settlement
        reach[k]        = max_distance * exp(-influence[k] / reach_scale)
        proximity[k, i] = relu(1 - dist[k, i] / reach[k])       # in [0, 1]
        pull[k]         = exp(-influence[k] / influence_scale)   # in (0, 1]
        bid[k, i]       = proximity[k, i] * pull[k]

    ``influence`` is derived from each settlement's current population: the
    highest-population settlement gets influence 0 (widest reach, strongest
    pull) and lower-population settlements get proportionally larger influence.
    This means hinterland priority emerges from population rather than from
    fixed tier labels.

    ``max_distance`` caps the largest market area (the top-population settlement
    at influence 0).  ``reach_scale`` controls how steeply catchment radius
    shrinks with influence; smaller values give more differentiation.
    ``influence_scale`` controls how steeply pull decays; smaller values
    widen the pull gap between large and small settlements.

    Each face's carrying capacity is divided among its bidders in proportion to
    their bids, so a low-pull village next to a high-pull city still claims a
    nonzero share of the nearby land (and can survive culling).

    Args:
        n_faces: number of mesh faces (N).
        settlements: list of :class:`~realmgen.placement.types.Settlement`
            (reads ``face_id`` and ``population``).
        cost_graph: (N, N) directed land-cost CSR matrix — used undirected. Pass
            a water-permeable variant to let coastal settlements claim a limited
            band of ocean/lake for fishing grounds.
        max_distance: hard cap (cost-distance) on the widest hinterland reach.
        influence_scale: e-folding scale for ``pull``; smaller widens the pull
            gap between settlements.
        reach_scale: e-folding scale for catchment radius; smaller gives higher-
            population centres a proportionally larger market area.

    Returns:
        ``(shares, owner)`` where:

        * ``shares`` is ``(S, N)`` float64 — fraction of face *i*'s capacity
          claimed by settlement *k*; each column sums to 1 where any settlement
          is in range, else all zero.
        * ``owner`` is ``(N,)`` int32 — index of the single highest bidder per
          face (for display), or -1 where no settlement is within range.
    """
    if not settlements:
        return (
            np.zeros((0, n_faces), dtype=np.float64),
            np.full(n_faces, -1, dtype=np.int32),
        )

    s_nodes = np.array([s.face_id for s in settlements], dtype=np.int32)
    populations = np.array([s.population for s in settlements], dtype=np.float64)
    max_pop = float(populations.max())
    if max_pop <= 0.0:
        influence = np.zeros_like(populations)
    else:
        min_pop = max_pop * 1e-3
        influence = np.log(max_pop / np.maximum(populations, min_pop))

    # dist[k, i] = undirected cost-distance from settlements[k] to face i
    # (+inf where unreachable).
    dist = sp_dijkstra(
        cost_graph, directed=False, indices=s_nodes, return_predecessors=False,
    )  # (S, N)

    md = max(float(max_distance), 1e-9)
    scale = max(float(influence_scale), 1e-9)
    reach = max(float(reach_scale), 1e-9)
    # Per-settlement catchment radius: higher-order centres (lower influence)
    # draw from a proportionally larger market area. reach_scale=inf → uniform md.
    md_k = md * np.exp(-influence / reach)                  # (S,)
    proximity = np.maximum(1.0 - dist / md_k[:, np.newaxis], 0.0)   # (S, N); inf → 0
    pull = np.exp(-influence / scale)                       # (S,)
    bid = proximity * pull[:, np.newaxis]                   # (S, N)

    total = bid.sum(axis=0)                                 # (N,)
    in_range = total > 0.0
    shares = np.zeros_like(bid)
    np.divide(bid, total, out=shares, where=in_range)       # broadcast over rows

    owner = np.where(in_range, np.argmax(bid, axis=0), -1).astype(np.int32)
    return shares, owner


def compute_carrying_capacity(
    shares: np.ndarray,
    fertility: np.ndarray,
    face_areas: np.ndarray,
) -> np.ndarray:
    """Per-settlement carrying capacity from fractional hinterland shares.

    ``cap[k] = Σ_i shares[k, i] · fertility[i] · face_area[i]``.

    Args:
        shares: (S, N) float64 from :func:`compute_hinterland_shares`.
        fertility: (N,) float in [0, 1] — per-face ``fertility_mean`` (already
            includes aquatic fishing productivity on coastal ocean/lake faces, so
            water claimed via a permeable hinterland graph contributes here).
        face_areas: (N,) float64 — per-face solid-angle areas (unit-sphere).

    Returns:
        (S,) float64 — raw carrying capacity (arbitrary units).
    """
    shares = np.asarray(shares, dtype=np.float64)
    if shares.size == 0:
        return np.zeros(shares.shape[0], dtype=np.float64)
    contribution = np.asarray(fertility, dtype=np.float64) * np.asarray(face_areas, dtype=np.float64)
    return shares @ contribution


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
