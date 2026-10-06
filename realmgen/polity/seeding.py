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

    Ruggedness is ``land_base_cost`` normalised over land p50..p90. Multiplies a
    settlement's population to form its local power.
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

    Greedy in descending ``weights``: a candidate qualifies if its population is
    ``>= min_pop_fraction * peak`` and it is ``>= min_separation`` cost-distance
    from every chosen capital. Returns indices into ``settlement_faces``.
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
    """Map cohesion in [0, 1] to an authority multiplier in [floor, floor+span]."""
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

    RNG is keyed on ``SeedSequence([base_seed, face])`` so scalars are stable
    under reordering.

    * ``cohesion`` ~ U[cohesion_min, 1].
    * ``infrastructure``: U[infrastructure_min, 1] blended (weight
      ``infrastructure_emergent``) with mean(pop_norm, trade_norm).
    * ``legitimacy`` ~ 1 +/- legitimacy_spread.
    * ``maritime_reach`` ~ maritime_scale * trade_norm * jitter.
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


def compute_settlement_defensibility(
    settlement_faces: Sequence[int],
    land_base_cost: np.ndarray,
    situational_value: np.ndarray,
    river_flow_mag: np.ndarray,
    trade_wealth: np.ndarray,
    danger: np.ndarray,
    land_mask: np.ndarray,
    base_seed: int,
    *,
    rugged_weight: float = 0.6,
    situational_weight: float = 0.8,
    river_weight: float = 0.4,
    wall_base_prob: float = 0.3,
    wall_danger_boost: float = 0.3,
    wall_wealth_boost: float = 0.3,
    wall_min_strength: float = 0.3,
    wall_max_strength: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-settlement defensibility, wall presence, and wall strength.

    Terrain + river barrier + walls (rolled per ``(base_seed, face)``, biased by
    danger and trade wealth). ``trade_wealth`` is ``(S,)``; the other arrays are
    per-face ``(N,)``; ``land_mask`` sets the normalisation percentiles.

    Returns ``(defensibility (S,) >= 1, has_walls (S,) bool, wall_strength (S,))``.
    """
    faces = np.asarray(settlement_faces, dtype=np.int64)
    s = len(faces)
    if s == 0:
        return np.empty(0, np.float64), np.empty(0, bool), np.empty(0, np.float64)

    cost = np.asarray(land_base_cost, dtype=np.float64)
    sv = np.clip(np.asarray(situational_value, dtype=np.float64), 0.0, 1.0)
    flow = np.asarray(river_flow_mag, dtype=np.float64)
    danger_arr = np.asarray(danger, dtype=np.float64)
    wealth = np.asarray(trade_wealth, dtype=np.float64)
    lmask = np.asarray(land_mask, dtype=bool)

    # Same normalisation as compute_defensibility.
    valid_cost = cost[lmask & np.isfinite(cost)]
    lo = float(np.percentile(valid_cost, 50.0)) if valid_cost.size else 0.0
    hi = float(np.percentile(valid_cost, 90.0)) if valid_cost.size else 1.0
    rugged_norm = np.clip((cost[faces] - lo) / max(hi - lo, 1e-9), 0.0, 1.0)

    land_flow = flow[lmask & (flow > 0.0)]
    flow_ref = float(np.percentile(land_flow, 90.0)) if land_flow.size else 1.0
    river_norm = np.clip(flow[faces] / max(flow_ref, 1e-9), 0.0, 1.0)

    land_danger = danger_arr[lmask & (danger_arr > 0.0)]
    danger_ref = float(np.percentile(land_danger, 90.0)) if land_danger.size else 1.0
    danger_norm = np.clip(danger_arr[faces] / max(danger_ref, 1e-9), 0.0, 1.0)

    wealth_max = float(wealth.max()) if wealth.size else 0.0
    wealth_norm = np.clip(wealth / max(wealth_max, 1e-9), 0.0, 1.0)

    terrain = 1.0 + rugged_weight * rugged_norm + situational_weight * sv[faces]

    river_bonus = river_weight * river_norm

    wall_prob = np.clip(
        wall_base_prob + wall_danger_boost * danger_norm + wall_wealth_boost * wealth_norm,
        0.0, 1.0,
    )
    has_walls, wall_str = roll_walls(
        wall_prob, 0.5 * (danger_norm + wealth_norm),
        [[int(base_seed), int(face)] for face in faces.tolist()],
        wall_min_strength, wall_max_strength,
    )
    return terrain + river_bonus + wall_str, has_walls, wall_str


def roll_walls(
    prob: np.ndarray, bias: np.ndarray, keys: Sequence[Sequence[int]],
    min_strength: float, max_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``(has_walls, wall_strength)``: walls with probability ``prob``, each from its
    own ``keys[k]`` RNG stream; quality half random, half ``bias`` (both in [0, 1])."""
    s = len(keys)
    has_walls = np.zeros(s, dtype=bool)
    wall_str = np.zeros(s, dtype=np.float64)
    for k, key in enumerate(keys):
        rng = np.random.default_rng(np.random.SeedSequence([int(x) for x in key]))
        if rng.random() < float(prob[k]):
            has_walls[k] = True
            quality = np.clip(0.5 * rng.random() + 0.5 * float(bias[k]), 0.0, 1.0)
            wall_str[k] = min_strength + (max_strength - min_strength) * quality
    return has_walls, wall_str
