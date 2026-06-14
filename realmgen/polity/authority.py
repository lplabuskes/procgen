"""Authority projection, claiming, and hierarchy classification (Stages 1-3).

The single decision that produces the tree texture is, at each settlement,
*projected authority* (how strongly the claiming capital reaches it across the
cost surface) vs *local power* (its population x terrain defensibility).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

from realmgen.polity.types import ControlClass


def compute_authority_fields(
    n_faces: int,
    capital_faces: Sequence[int],
    legitimacy: np.ndarray,
    infrastructure: np.ndarray,
    maritime_reach: np.ndarray,
    land_cost_graph: csr_matrix,
    hybrid_cost_graph: Optional[csr_matrix] = None,
    *,
    base_range: float = 40.0,
    range_floor: float = 0.5,
    sea_range_factor: float = 3.0,
) -> np.ndarray:
    """Per-capital authority field, ``(C, N)`` in roughly [0, legitimacy].

        land_range = base_range * (range_floor + infrastructure)
        auth_land  = legitimacy * exp(-land_dist / land_range)

    ``land_dist`` is the undirected multi-source cost-distance from all capitals.
    Cheap terrain carries the field far; rugged/dangerous terrain (baked into the
    edge costs) pinches it, creating the interior **saddles** where vassalage
    concentrates. With a hybrid land/sea graph, a gentler sea channel
    (``legitimacy * maritime_reach``, range x ``sea_range_factor``) is max-combined
    in, so a high-maritime coastal capital beads authority along coastlines.
    """
    caps = np.asarray(capital_faces, dtype=np.int64)
    if caps.size == 0:
        return np.zeros((0, n_faces), dtype=np.float64)

    leg = np.asarray(legitimacy, dtype=np.float64)
    infra = np.asarray(infrastructure, dtype=np.float64)
    mar = np.asarray(maritime_reach, dtype=np.float64)

    land_dist = sp_dijkstra(land_cost_graph, directed=False, indices=caps, return_predecessors=False)
    land_range = base_range * (range_floor + infra)
    auth_land = leg[:, None] * np.exp(-land_dist / land_range[:, None])

    if hybrid_cost_graph is not None:
        hybrid_dist = sp_dijkstra(
            hybrid_cost_graph, directed=False, indices=caps, return_predecessors=False,
        )[:, :n_faces]  # land-node slice (sea shortcuts through ports)
        sea_range = land_range * float(sea_range_factor)
        auth_sea = (leg * mar)[:, None] * np.exp(-hybrid_dist / sea_range[:, None])
        return np.maximum(auth_land, auth_sea)

    return auth_land


def claim_faces(
    field: np.ndarray,
    capital_faces: Sequence[int],
    eligible_mask: np.ndarray,
    *,
    hysteresis: float = 0.15,
    max_iters: int = 8,
    prev: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """Assign each eligible face to its strongest projector, ``-1`` if none.

    Pure ``argmax_k field[k, face]``, except the incumbent claimant's field gets a
    ``(1 + hysteresis)`` bonus so contested faces stick rather than flicker;
    iterated to a fixed point (fields are static, so 1-3 passes). Faces outside
    ``eligible_mask`` (water, or outside the parent territory during recursion) or
    reached by no capital stay ``-1``. Each capital always rules its own seat.
    ``prev`` carries an incumbent assignment across outer passes.

    Returns ``(polity_of_face (N,) int32, n_iters)``.
    """
    field = np.asarray(field, dtype=np.float64)
    c, n = field.shape
    if c == 0:
        return np.full(n, -1, dtype=np.int32), 0
    caps = np.asarray(capital_faces, dtype=np.int64)
    eligible = np.asarray(eligible_mask, dtype=bool)
    seats = np.arange(c, dtype=np.int32)
    reachable = (field.max(axis=0) > 1e-12) & eligible

    def _assign(scores: np.ndarray) -> np.ndarray:
        a = np.where(reachable, np.argmax(scores, axis=0).astype(np.int32), -1).astype(np.int32)
        a[caps] = seats  # a capital always rules its own seat
        return a

    cur = prev.copy().astype(np.int32) if prev is not None else _assign(field)
    cur[caps] = seats

    n_iters = 0
    for _ in range(max(1, int(max_iters))):
        n_iters += 1
        boosted = field.copy()
        valid = np.flatnonzero(cur >= 0)
        boosted[cur[valid], valid] *= (1.0 + float(hysteresis))
        nxt = _assign(boosted)
        churn = int(np.count_nonzero((nxt != cur) & reachable))
        cur = nxt
        if churn == 0:
            break

    return cur, n_iters


def classify_settlements(
    settlement_faces: Sequence[int],
    populations: np.ndarray,
    polity_of_face: np.ndarray,
    field: np.ndarray,
    projecting_power: np.ndarray,
    defensibility: np.ndarray,
    authority_gain: np.ndarray,
    is_capital: np.ndarray,
    *,
    direct_threshold: float,
    vassal_floor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Label each settlement DIRECT / VASSAL / INDEPENDENT, in the same currency::

        eff = projecting_power[claimant] * field[claimant, face] * authority_gain[claimant]
        loc = population * defensibility[face]
        r   = eff / loc   ->   r>=direct: DIRECT;  r>=vassal_floor: VASSAL;  else INDEPENDENT

    ``projecting_power`` is the claimant *capital's* own strength (not the whole
    domain) — so a sprawling realm with a modest capital cannot hold its far
    provinces directly. A capital is always DIRECT; an unclaimed face INDEPENDENT.

    Returns ``(control_class (S,) int32, ratio (S,) float64)``.
    """
    faces = np.asarray(settlement_faces, dtype=np.int64)
    pop = np.asarray(populations, dtype=np.float64)
    s = faces.size
    control = np.full(s, int(ControlClass.INDEPENDENT), dtype=np.int32)
    ratio = np.zeros(s, dtype=np.float64)
    if s == 0:
        return control, ratio

    claimant = np.asarray(polity_of_face)[faces]
    dom = np.asarray(projecting_power, dtype=np.float64)
    defn = np.asarray(defensibility, dtype=np.float64)
    gain = np.asarray(authority_gain, dtype=np.float64)

    valid = claimant >= 0
    vf, vc = faces[valid], claimant[valid]
    eff = dom[vc] * field[vc, vf] * gain[vc]
    loc = np.maximum(pop[valid] * defn[vf], 1e-9)
    r = eff / loc
    ratio[valid] = r
    control[valid] = np.where(
        r >= float(direct_threshold), int(ControlClass.DIRECT),
        np.where(r >= float(vassal_floor), int(ControlClass.VASSAL), int(ControlClass.INDEPENDENT)),
    ).astype(np.int32)

    cap = np.asarray(is_capital, dtype=bool)
    control[cap] = int(ControlClass.DIRECT)
    ratio[cap] = np.inf
    return control, ratio
