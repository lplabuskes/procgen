"""Capital seeding, defensibility, and per-polity scalar rolls (Stage 0).

Pure functions over pre-extracted arrays (no World coupling), so they test
against lightweight stub graphs.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

from realmgen.polity.types import Scalars


def compute_defensibility(
    land_base_cost: np.ndarray,
    situational_value: np.ndarray,
    land_mask: Optional[np.ndarray] = None,
    *,
    rugged_weight: float = 0.6,
    situational_weight: float = 0.8,
) -> np.ndarray:
    """Per-face defensibility multiplier (>= 1) from terrain.

    Rugged terrain (high ``land_base_cost``, normalized over land p50..p90) and
    situational value (harbors, passes, confluences) make a place easier to hold.
    Multiplies a settlement's population to form its local power.
    """
    cost = np.asarray(land_base_cost, dtype=np.float64)
    sv = np.clip(np.asarray(situational_value, dtype=np.float64), 0.0, 1.0)
    finite = np.isfinite(cost)
    if land_mask is not None:
        finite = finite & np.asarray(land_mask, dtype=bool)
    valid = cost[finite]
    lo = float(np.percentile(valid, 50.0)) if valid.size else 0.0
    hi = float(np.percentile(valid, 90.0)) if valid.size else 1.0
    span = max(hi - lo, 1e-9)
    rugged_norm = np.clip((cost - lo) / span, 0.0, 1.0)
    return 1.0 + rugged_weight * rugged_norm + situational_weight * sv


def seed_capitals(
    settlement_faces: Sequence[int],
    populations: np.ndarray,
    cost_graph: csr_matrix,
    *,
    weights: Optional[np.ndarray] = None,
    min_separation: float,
    min_pop_fraction: float,
) -> List[int]:
    """Pick which settlements found sovereign polities.

    Candidates are taken in descending ``weights`` order (caller passes
    population x trade x defensibility). A candidate qualifies if its population
    clears ``min_pop_fraction * peak`` and its face is at least ``min_separation``
    cost-distance from every capital already chosen (one Dijkstra per accepted
    capital maintains the running minimum). Returns indices into
    ``settlement_faces``, descending weight order, ties by index.
    """
    faces = np.asarray(settlement_faces, dtype=np.int64)
    pop = np.asarray(populations, dtype=np.float64)
    if faces.size == 0:
        return []
    w = pop if weights is None else np.asarray(weights, dtype=np.float64)

    threshold = float(min_pop_fraction) * float(pop.max())
    sep = max(float(min_separation), 0.0)
    min_dist = np.full(cost_graph.shape[0], np.inf, dtype=np.float64)  # type: ignore[union-attr]
    chosen: List[int] = []

    for idx in np.argsort(-w, kind="stable").tolist():
        if pop[idx] < threshold:
            continue  # ranked by weight, not population — skip, don't break
        face = int(faces[idx])
        if min_dist[face] < sep:
            continue  # too close to an existing capital
        chosen.append(idx)
        d = sp_dijkstra(cost_graph, directed=False, indices=face, return_predecessors=False)
        np.minimum(min_dist, d, out=min_dist)

    return chosen


def cohesion_to_gain(
    cohesion: np.ndarray,
    *,
    gain_floor: float = 0.5,
    gain_span: float = 1.0,
) -> np.ndarray:
    """Map cohesion in [0, 1] to an authority multiplier in [floor, floor+span].

    High cohesion lifts effective authority (hold directly — Song); low cohesion
    lowers it (shed vassals/independents — HRE).
    """
    c = np.clip(np.asarray(cohesion, dtype=np.float64), 0.0, 1.0)
    return float(gain_floor) + float(gain_span) * c


def roll_polity_scalars(
    capital_faces: Sequence[int],
    capital_pop_norm: np.ndarray,
    capital_trade_norm: np.ndarray,
    base_seed: int,
    *,
    cohesion_min: float = 0.4,
    infrastructure_min: float = 0.4,
    infrastructure_emergent: float = 0.5,
    legitimacy_spread: float = 0.3,
    maritime_scale: float = 1.5,
) -> Scalars:
    """Roll the four per-polity scalars, one (C,) array each.

    Each capital draws from an RNG substream keyed on its **face index**
    (``SeedSequence([base_seed, face])``), so its scalars are stable under
    seeding/iteration reorder and reproduce for a world seed.

    * ``cohesion`` ~ U[cohesion_min, 1].
    * ``infrastructure`` — random [infrastructure_min, 1] blended (weight
      ``infrastructure_emergent``) with mean(pop_norm, trade_norm): a big rich
      capital administers further. Drives projection range.
    * ``legitimacy`` ~ 1 +/- legitimacy_spread (field magnitude).
    * ``maritime_reach`` ~ maritime_scale * trade_norm * jitter (~0 if landlocked).
    """
    caps = list(capital_faces)
    c = len(caps)
    cohesion = np.empty(c, dtype=np.float64)
    infrastructure = np.empty(c, dtype=np.float64)
    legitimacy = np.empty(c, dtype=np.float64)
    maritime = np.empty(c, dtype=np.float64)
    pop_norm = np.asarray(capital_pop_norm, dtype=np.float64)
    trade_norm = np.asarray(capital_trade_norm, dtype=np.float64)
    emergent_w = float(np.clip(infrastructure_emergent, 0.0, 1.0))

    for k, face in enumerate(caps):
        rng = np.random.default_rng(np.random.SeedSequence([int(base_seed), int(face)]))
        cohesion[k] = cohesion_min + (1.0 - cohesion_min) * rng.random()
        infra_rand = infrastructure_min + (1.0 - infrastructure_min) * rng.random()
        emergent = 0.5 * (float(pop_norm[k]) + float(trade_norm[k]))
        infrastructure[k] = np.clip(
            (1.0 - emergent_w) * infra_rand + emergent_w * emergent, infrastructure_min, 1.0
        )
        legitimacy[k] = 1.0 + legitimacy_spread * (2.0 * rng.random() - 1.0)
        maritime[k] = maritime_scale * float(trade_norm[k]) * (0.5 + 0.5 * rng.random())

    return Scalars(cohesion, infrastructure, legitimacy, maritime)
