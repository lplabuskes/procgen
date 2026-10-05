"""Authority projection, claiming, and hierarchy classification (Stages 1-3).

The single decision that produces the tree texture is, at each settlement,
*projected authority* (how strongly the claiming capital reaches it across the
cost surface) vs *local power* (its population x terrain defensibility).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

from realmgen.polity.types import ControlClass


class ProjectionDistances:
    """Cost-distance rows from capital faces over the land and hybrid graphs, cached.

    The same capitals are projected repeatedly (per subtree, per epoch), so each
    face's Dijkstra runs once.
    """

    def __init__(self, land_graph: csr_matrix, hybrid_graph: Optional[csr_matrix], n_faces: int) -> None:
        self.land_graph = land_graph
        self.hybrid_graph = hybrid_graph
        self.n_faces = n_faces
        self._land: Dict[int, np.ndarray] = {}
        self._hybrid: Dict[int, np.ndarray] = {}

    def rows(self, faces: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """``(land (C, N), hybrid land-node slice (C, N) or None)`` for ``faces``."""
        missing = list(dict.fromkeys(int(f) for f in faces if int(f) not in self._land))
        if missing:
            land = sp_dijkstra(self.land_graph, directed=False, indices=missing, return_predecessors=False)
            self._land.update(zip(missing, land))
            if self.hybrid_graph is not None:
                hybrid = sp_dijkstra(
                    self.hybrid_graph, directed=False, indices=missing, return_predecessors=False,
                )[:, :self.n_faces]  # land-node slice (sea shortcuts through ports)
                self._hybrid.update(zip(missing, hybrid))
        land_rows = np.stack([self._land[int(f)] for f in faces])
        if self.hybrid_graph is None:
            return land_rows, None
        return land_rows, np.stack([self._hybrid[int(f)] for f in faces])


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
    sea_weight: float = 1.0,
    distances: Optional[ProjectionDistances] = None,
) -> np.ndarray:
    """Per-capital authority field, ``(C, N)`` in roughly [0, legitimacy].

        land_range = base_range * (range_floor + infrastructure)
        auth_land  = legitimacy * exp(-land_dist / land_range)

    Rugged/dangerous terrain (baked into edge costs) pinches the field, creating
    the interior saddles where vassalage concentrates. With a hybrid graph a sea
    channel (``sea_weight * legitimacy * maritime_reach``, range x
    ``sea_range_factor``) is max-combined in.

    Keep ``sea_range_factor`` modest and ``sea_weight <= 1``: a large sea range
    flattens ``auth_sea`` into a world-wide coastal plateau that wins the claiming
    ``argmax`` against every nearer land capital.

    ``distances`` caches Dijkstra rows; it must be over the same two graphs.
    """
    caps = np.asarray(capital_faces, dtype=np.int64)
    if caps.size == 0:
        return np.zeros((0, n_faces), dtype=np.float64)

    leg = np.asarray(legitimacy, dtype=np.float64)
    infra = np.asarray(infrastructure, dtype=np.float64)
    mar = np.asarray(maritime_reach, dtype=np.float64)

    if distances is None:
        distances = ProjectionDistances(land_cost_graph, hybrid_cost_graph, n_faces)
    land_dist, hybrid_dist = distances.rows(caps)
    land_range = base_range * (range_floor + infra)
    auth_land = leg[:, None] * np.exp(-land_dist / land_range[:, None])

    if hybrid_dist is not None:
        sea_range = land_range * float(sea_range_factor)
        auth_sea = float(sea_weight) * (leg * mar)[:, None] * np.exp(-hybrid_dist / sea_range[:, None])
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

    ``argmax_k field[k, face]`` with a ``(1 + hysteresis)`` bonus for the
    incumbent so contested faces don't flicker; iterated to a fixed point. Faces
    outside ``eligible_mask`` or unreached stay ``-1``; each capital rules its own
    seat. ``prev`` carries an incumbent assignment across outer passes.

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
    power_exponent: float = 1.0,
    loyalty_mult: Optional[np.ndarray] = None,
    force_independent: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Label each settlement DIRECT / VASSAL / INDEPENDENT, in the same currency::

        eff = projecting_power[claimant]**e * field[claimant, face] * authority_gain[claimant]
        loc = population**e * defensibility[face]
        r   = eff / loc * loyalty_mult  ->  r>=direct: DIRECT;  r>=vassal_floor: VASSAL;  else INDEPENDENT

    ``projecting_power`` is the claimant capital's own strength, not the domain's,
    so a sprawling realm with a modest capital can't hold far provinces directly.
    Capitals are always DIRECT; unclaimed faces INDEPENDENT.

    ``power_exponent`` < 1 compresses the ~70x Zipf population spread (``e=0.5``
    -> ~8x) so distance, not raw size, drives the decision.

    ``loyalty_mult`` ``(S,)`` scales the ratio (Stage-4 economic grip);
    ``force_independent`` ``(S,)`` bool forces INDEPENDENT (Stage-4 secession).
    Capitals are pinned DIRECT last, so neither lever can dislodge a seat.

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

    e = float(power_exponent)
    valid = claimant >= 0
    vf, vc = faces[valid], claimant[valid]
    eff = dom[vc] ** e * field[vc, vf] * gain[vc]
    loc = np.maximum(pop[valid] ** e * defn[vf], 1e-9)
    r = eff / loc
    if loyalty_mult is not None:
        r = r * np.asarray(loyalty_mult, dtype=np.float64)[valid]
    ratio[valid] = r
    control[valid] = np.where(
        r >= float(direct_threshold), int(ControlClass.DIRECT),
        np.where(r >= float(vassal_floor), int(ControlClass.VASSAL), int(ControlClass.INDEPENDENT)),
    ).astype(np.int32)

    if force_independent is not None:
        control[np.asarray(force_independent, dtype=bool)] = int(ControlClass.INDEPENDENT)

    cap = np.asarray(is_capital, dtype=bool)
    control[cap] = int(ControlClass.DIRECT)
    ratio[cap] = np.inf
    return control, ratio
