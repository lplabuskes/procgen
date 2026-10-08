"""Physical barriers that polity borders snap to: drainage divides (ridges).

Authority decays with land cost-distance, so a border sits where two capitals'
fields cross. An extra cost on the edges that cross a ridge puts a step in each
field there; a crossing within reach of the step slides onto the ridge.

A ridge is a **drainage divide** on the face graph: two adjacent land faces whose
downhill paths (priority-flood receivers) don't meet within ``min_detour`` hops.
Two faces on the same slope merge a step or two down, so only true divides count;
the step is scaled by the divide's relief over the ground ``floor_hops`` down
either side, so a flat watershed on a plain barely registers.

Rivers are not a claim cost: the refine stage moves drawn borders onto them
(:mod:`realmgen.refine.river_follow`).
"""
from __future__ import annotations

import heapq
from typing import List, Sequence

import numpy as np

from core.params import param_field as _f, paramclass
from worldgen.geology.channels import ELEVATION
from worldgen.hydrology.channels import FILLED_HEIGHT


@paramclass
class BarrierParams:
    """Ridge step in the polity cost surface."""
    ridge_weight: float = _f(
        8.0, label="Ridge crossing cost (at full relief)", min=0.0, max=40.0, step=0.5
    )
    ridge_relief_ref: float = _f(
        0.5, label="Ridge relief for full cost (km)", min=0.1, max=5.0, step=0.1
    )
    ridge_min_detour: int = _f(
        6, label="Ridge: min downhill hops to a common face", min=2, max=20, step=1, hidden=True
    )
    ridge_floor_hops: int = _f(
        2, label="Ridge: hops down to the relief floor", min=1, max=6, step=1, hidden=True
    )


def face_elevation(mesh) -> np.ndarray:
    """``(N,)`` mean depression-filled height (km), elevation without hydrology."""
    ch = FILLED_HEIGHT if FILLED_HEIGHT.present(mesh) else ELEVATION
    return np.asarray(ch.read(mesh), dtype=np.float64)[..., 0].mean(axis=(1, 2))


def face_drainage(face_adjacency: Sequence[Sequence[int]], elevation: np.ndarray,
                  is_water: np.ndarray) -> np.ndarray:
    """``(N,)`` downhill receiver per face by priority flood from the water, ``-1`` on water.

    Every land face drains to water through flats and pits; a landmass with no
    water drains to its lowest face.
    """
    elev = np.asarray(elevation, dtype=np.float64)
    water = np.asarray(is_water, dtype=bool)
    n = elev.size
    receiver = np.full(n, -1, dtype=np.int64)
    done = water.copy()
    heap = [(float(elev[f]), int(f)) for f in np.flatnonzero(water)]
    heapq.heapify(heap)
    while True:
        while heap:
            level, f = heapq.heappop(heap)
            for g in face_adjacency[f]:
                if not done[g]:
                    done[g] = True
                    receiver[g] = f
                    heapq.heappush(heap, (max(level, float(elev[g])), int(g)))
        rest = np.flatnonzero(~done)
        if rest.size == 0:
            return receiver
        lowest = int(rest[np.argmin(elev[rest])])   # a waterless landmass: its own sink
        done[lowest] = True
        heap = [(float(elev[lowest]), lowest)]


def divide_relief(face_adjacency: Sequence[Sequence[int]], elevation: np.ndarray,
                  is_water: np.ndarray, *, min_detour: int = 6, floor_hops: int = 2) -> List[np.ndarray]:
    """Per-edge divide relief (km), aligned with ``face_adjacency``; 0 off a divide.

    Edge ``(i, j)`` (both land) is a divide when the downhill paths from ``i`` and
    ``j`` meet only after ``>= min_detour`` hops in total (or never). Its relief is
    ``min(e_i, e_j) - max(floor_i, floor_j)``, the floors ``floor_hops`` down each
    path (sea level for water), clipped at 0.
    """
    elev = np.asarray(elevation, dtype=np.float64)
    water = np.asarray(is_water, dtype=bool)
    n = elev.size
    receiver = face_drainage(face_adjacency, elev, water)
    ground = np.where(water, np.maximum(elev, 0.0), elev)

    hops = max(int(min_detour), int(floor_hops) + 1)
    chain = np.empty((n, hops), dtype=np.int64)    # chain[f, k] = face k hops down from f
    chain[:, 0] = np.arange(n)
    for k in range(1, hops):
        prev = chain[:, k - 1]
        chain[:, k] = np.where(prev >= 0, receiver[np.maximum(prev, 0)], -1)
    floor_face = chain[:, int(floor_hops)]
    for k in range(int(floor_hops) - 1, -1, -1):   # path ended early: its last face
        floor_face = np.where(floor_face >= 0, floor_face, chain[:, k])
    floor = ground[floor_face]

    deg = np.array([len(a) for a in face_adjacency], dtype=np.int64)
    src = np.repeat(np.arange(n), deg)
    dst = np.fromiter((j for a in face_adjacency for j in a), dtype=np.int64, count=int(deg.sum()))
    land_edge = ~water[src] & ~water[dst]

    ci, cj = chain[src], chain[dst]                                   # (E, H)
    meet = (ci[:, :, None] == cj[:, None, :]) & (ci[:, :, None] >= 0)  # (E, H, H)
    steps = np.arange(hops)
    detour = np.where(meet, steps[:, None] + steps[None, :], hops * 2).min(axis=(1, 2))
    divide = land_edge & (detour >= int(min_detour))
    relief = np.minimum(elev[src], elev[dst]) - np.maximum(floor[src], floor[dst])
    flat = np.where(divide, np.maximum(relief, 0.0), 0.0)
    indptr = np.concatenate([[0], np.cumsum(deg)])
    return [flat[indptr[i]:indptr[i + 1]] for i in range(n)]


def add_ridge_costs(land_edge_costs: Sequence[np.ndarray], relief: Sequence[np.ndarray],
                    params: BarrierParams) -> List[np.ndarray]:
    """``land_edge_costs`` plus ``ridge_weight · min(relief / ridge_relief_ref, 1)`` per edge."""
    w, ref = float(params.ridge_weight), max(float(params.ridge_relief_ref), 1e-9)
    if w <= 0.0:
        return [np.asarray(c, dtype=np.float64) for c in land_edge_costs]
    return [np.asarray(c, dtype=np.float64) + w * np.minimum(np.asarray(r) / ref, 1.0)
            for c, r in zip(land_edge_costs, relief)]
