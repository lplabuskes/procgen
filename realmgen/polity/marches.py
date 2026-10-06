"""Stage 4 marcher autonomy — frontier vassals hold more loosely.

Border tension is scored on the final partition, so it can't steer the forest
that makes it. Instead :class:`~realmgen.polity.builder.PolityStage` scores
tension on its **first** forest's sovereign borders, spreads it inward
(:func:`frontier_exposure`), and grows the forest again; there a vassal whose
territory reaches a hot frontier gets its autonomy raised
(:func:`march_autonomy`). Fragmentation epochs shed the most autonomous vassal
first, so marches are the first to break away.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

from core.params import param_field as _f, paramclass

# scipy drops explicit zeros from sparse graphs; floor the source edge weights.
_MIN_EDGE = 1e-9


@paramclass
class MarchParams:
    """Marcher autonomy on the first pass's sovereign frontiers (consumed by PolityStage)."""
    weight: float = _f(0.5, label="March autonomy weight (0 = off)", min=0.0, max=1.0)
    reach: float = _f(6.0, label="March reach (cost-dist e-fold from the border)", min=0.1, max=50.0)
    tension_quantile: float = _f(
        0.9, label="March full-exposure tension (quantile of border faces)", min=0.1, max=1.0)


def frontier_exposure(
    land_graph: csr_matrix, face_tension: np.ndarray, *, reach: float, tension_quantile: float = 1.0,
) -> np.ndarray:
    """``(N,)`` in [0, 1]: ``max_b t[b] · exp(−d(b, f) / reach)`` over border faces ``b``.

    ``t`` is tension over its ``tension_quantile`` among border faces, clipped to 1
    (raw tension is small and world-dependent). One Dijkstra from a virtual source
    tied to each border face at cost ``−reach · ln(t[b])``.
    Unreachable faces get 0.
    """
    t = np.asarray(face_tension, dtype=np.float64)
    n = t.size
    border = np.flatnonzero(t > 0.0)
    if border.size == 0 or reach <= 0.0:
        return np.zeros(n, dtype=np.float64)
    t = np.clip(t / float(np.quantile(t[border], np.clip(tension_quantile, 0.0, 1.0))), 0.0, 1.0)
    g = land_graph.tocoo()
    src_w = np.maximum(-float(reach) * np.log(t[border]), _MIN_EDGE)
    aug = coo_matrix(
        (np.concatenate([g.data, src_w]),
         (np.concatenate([g.row, np.full(border.size, n)]), np.concatenate([g.col, border]))),
        shape=(n + 1, n + 1),
    ).tocsr()
    d = dijkstra(aug, directed=False, indices=n, return_predecessors=False)[:n]
    return np.where(np.isfinite(d), np.exp(-d / float(reach)), 0.0)


def march_autonomy(autonomy: float, exposure: float, weight: float) -> float:
    """Raise ``autonomy`` toward 1 by ``weight · exposure``: ``1 − (1 − a)(1 − w·e)``."""
    w = float(np.clip(weight, 0.0, 1.0)) * float(np.clip(exposure, 0.0, 1.0))
    return float(1.0 - (1.0 - float(autonomy)) * (1.0 - w))
