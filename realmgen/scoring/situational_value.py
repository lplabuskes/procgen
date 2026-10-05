"""Per-face static situational value in [0, 1] (water faces 0).

A normalised weighted sum of six components: harbor quality, river confluences,
navigable river heads (fall-line transshipment points), mountain passes, strait
adjacency, and delta apexes (the last firm ground before the marsh, commanding
every distributary — Cairo, New Orleans).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from realmgen.scoring.suitability import _positions_to_face_ids

if TYPE_CHECKING:
    from realmgen.scoring.builder import MountainPassParams, SituationalValueWeightParams


# ── Harbor quality ────────────────────────────────────────────────────────────


def compute_harbor_quality(
    mesh,
    is_ocean: np.ndarray,
    face_mean_elev_km: np.ndarray,
    *,
    depth_scale_km: float = 0.5,
) -> np.ndarray:
    """Harbor score in [0, 1] for coastal land faces, (N,) float32.

    ``concavity × depth``: concavity is the fraction of neighbours that are ocean;
    depth is ``clip(-mean_ocean_neighbor_elev / depth_scale_km, 0, 1)``.
    """
    n = len(mesh.faces)
    is_ocean = np.asarray(is_ocean, dtype=bool)
    elev = np.asarray(face_mean_elev_km, dtype=np.float64)

    score = np.zeros(n, dtype=np.float64)
    dscale = max(depth_scale_km, 1e-9)
    for i, adj in enumerate(mesh.face_adjacency):
        if is_ocean[i] or not adj:
            continue
        ocean_adj = [j for j in adj if is_ocean[j]]
        if not ocean_adj:
            continue
        concavity = len(ocean_adj) / len(adj)
        mean_depth = elev[np.array(ocean_adj, dtype=np.intp)].mean()
        depth_factor = float(np.clip(-mean_depth / dscale, 0.0, 1.0))
        score[i] = concavity * depth_factor
    return score.astype(np.float32)


# ── River confluences ─────────────────────────────────────────────────────────


def compute_river_confluences(
    rivers,
    face_centroids: np.ndarray,
    n_faces: int,
) -> np.ndarray:
    """``tanh(k - 1)`` for faces touched by k distinct rivers (non-lake segments), (N,) float32.

    Args:
        rivers: ``(positions (M,3), flows, in_lake (M-1,))`` tuples (the ``"rivers"``
            feature), or None.
        face_centroids: (N, 3) unit-sphere face centroids.
    """
    face_centroids = np.asarray(face_centroids, dtype=np.float64)
    face_count = np.zeros(n_faces, dtype=np.int32)
    for positions, _flows, in_lake in (rivers or []):
        fids = _positions_to_face_ids(positions, face_centroids)
        touched: set[int] = set()
        for k, lake in enumerate(in_lake):
            if not lake:
                touched.add(int(fids[k]))
                touched.add(int(fids[k + 1]))
        for fid in touched:
            face_count[fid] += 1

    score = np.tanh(np.maximum(face_count.astype(np.float64) - 1.0, 0.0))
    return score.astype(np.float32)


# ── Navigable river heads ─────────────────────────────────────────────────────


def compute_navigable_heads(
    rivers,
    face_centroids: np.ndarray,
    river_navigability: np.ndarray,
    n_faces: int,
) -> np.ndarray:
    """(N,) float32 mask of each river's navigable head (the fall-line point).

    The head is the first non-lake segment from the source whose face has
    ``river_navigability > 0`` (0 = rapids, 1 = placid).
    """
    face_centroids = np.asarray(face_centroids, dtype=np.float64)
    nav = np.asarray(river_navigability, dtype=np.float64)
    mask = np.zeros(n_faces, dtype=bool)

    for positions, _flows, in_lake in (rivers or []):
        fids = _positions_to_face_ids(positions, face_centroids)
        head_fid: int | None = None

        for k, lake in enumerate(in_lake):
            if lake:
                continue
            fid = int(fids[k])
            if nav[fid] > 0.0:
                head_fid = fid
                break

        if head_fid is None:
            continue
        mask[head_fid] = True

    return mask.astype(np.float32)


# ── Mountain passes ───────────────────────────────────────────────────────────


def compute_mountain_passes(
    mesh,
    land_base_cost: np.ndarray,
    is_water: np.ndarray,
    *,
    params: "MountainPassParams",
) -> np.ndarray:
    """Score low-cost corridors through high-cost terrain, (N,) float32 in [0, 1].

    A land face cheaper than its land neighbours' mean, where that mean exceeds
    ``min_neighbor_cost`` (ruling out plains), scores ``contrast / (contrast + cost_scale)``
    with ``contrast = mean_neighbor_cost − face_cost``.
    """
    p = params
    n = len(mesh.faces)
    cost = np.asarray(land_base_cost, dtype=np.float64)
    is_water = np.asarray(is_water, dtype=bool)

    score = np.zeros(n, dtype=np.float64)
    cscale = max(p.cost_scale, 1e-9)
    for i, adj in enumerate(mesh.face_adjacency):
        if is_water[i]:
            continue
        land_adj = [j for j in adj if not is_water[j]]
        if len(land_adj) < 2:
            continue
        neighbor_mean = cost[np.array(land_adj, dtype=np.intp)].mean()
        if neighbor_mean < p.min_neighbor_cost:
            continue
        contrast = neighbor_mean - cost[i]
        if contrast <= 0.0:
            continue
        score[i] = contrast / (contrast + cscale)
    return score.astype(np.float32)


# ── Strait adjacency ──────────────────────────────────────────────────────────


def compute_strait_adjacency(
    mesh,
    is_strait: np.ndarray,
    is_water: np.ndarray,
) -> np.ndarray:
    """(N,) float32: 1.0 for land faces bordering a strait-tagged face, else 0."""
    is_strait = np.asarray(is_strait, dtype=bool)
    is_water = np.asarray(is_water, dtype=bool)
    n = len(mesh.faces)

    score = np.zeros(n, dtype=np.float32)
    for i, adj in enumerate(mesh.face_adjacency):
        if is_water[i]:
            continue
        if any(is_strait[j] for j in adj):
            score[i] = 1.0
    return score


# ── Combination ───────────────────────────────────────────────────────────────


def compute_delta_apexes(apexes: np.ndarray | None, face_centroids: np.ndarray, n_faces: int) -> np.ndarray:
    """(N,) float32 mask of the faces holding a delta apex (``apexes``: (M, 3) or None)."""
    mask = np.zeros(n_faces, dtype=np.float32)
    if apexes is None or len(apexes) == 0:
        return mask
    mask[_positions_to_face_ids(np.asarray(apexes, dtype=np.float64), face_centroids)] = 1.0
    return mask


def combine_situational_value(
    is_water: np.ndarray,
    harbor: np.ndarray,
    confluence: np.ndarray,
    nav_head: np.ndarray,
    passes: np.ndarray,
    straits: np.ndarray,
    *,
    params: "SituationalValueWeightParams",
    delta: np.ndarray | None = None,
) -> np.ndarray:
    """Weight-normalised situational value, (N,) float32 in [0, 1]; water faces 0.

    Components are (N,) in [0, 1].  ``delta=None`` (no deltas) drops its weight
    from the normalisation.
    """
    p = params
    delta_w = float(p.delta_weight) if delta is not None else 0.0
    total_w = (
        p.harbor_weight + p.confluence_weight + p.nav_head_weight
        + p.pass_weight + p.strait_weight + delta_w
    )
    total_w = max(total_w, 1e-9)

    score = (
        p.harbor_weight * np.asarray(harbor, dtype=np.float64)
        + p.confluence_weight * np.asarray(confluence, dtype=np.float64)
        + p.nav_head_weight * np.asarray(nav_head, dtype=np.float64)
        + p.pass_weight * np.asarray(passes, dtype=np.float64)
        + p.strait_weight * np.asarray(straits, dtype=np.float64)
        + (delta_w * np.asarray(delta, dtype=np.float64) if delta is not None else 0.0)
    ) / total_w

    score = np.where(np.asarray(is_water, dtype=bool), 0.0, score)
    return np.clip(score, 0.0, 1.0).astype(np.float32)
