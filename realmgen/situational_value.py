"""Per-face static situational value scorer.

Five independent components are combined into one [0, 1] situational-value
scalar per land face.  Each component is a pure function that takes
pre-extracted arrays, making them easy to test against a lightweight stub mesh.

Components:

  * **harbor quality**   — shoreline concavity (fraction of neighbors that are
    ocean) × depth factor (mean ocean-neighbor elevation, normalized).
  * **river confluences** — faces touched by 2 or more distinct river polylines;
    score = tanh(count - 1) so each additional river raises the score.
  * **navigable river heads** — the most-upstream face of each river that still
    has ``river_navigability > 0`` (the fall-line transshipment point).
  * **mountain passes** — faces whose land cost is substantially below the mean
    cost of their high-cost land neighbors (cheap corridor through hard terrain).
  * **strait adjacency** — land faces that border a strait-tagged ocean face.

Combination is a weighted sum normalized by total weight so individual weights
need not sum to exactly 1; water faces are forced to 0.
"""
from __future__ import annotations

import numpy as np

from realmgen.suitability import _positions_to_face_ids


# ── Harbor quality ────────────────────────────────────────────────────────────


def compute_harbor_quality(
    mesh,
    is_ocean: np.ndarray,
    face_mean_elev_km: np.ndarray,
    *,
    depth_scale_km: float = 0.5,
) -> np.ndarray:
    """Shoreline-concavity × depth quality score in [0, 1].

    A harbor face is a land face adjacent to at least one ocean face.
    Two factors are multiplied together:

    * *Concavity* — fraction of the face's neighbors that are ocean.
      A face completely surrounded by ocean (islet) scores 1; a face with one
      ocean neighbor on an open coastline scores low.
    * *Depth factor* — ``clip(-mean_ocean_neighbor_elev / depth_scale_km, 0, 1)``.
      Deeper adjacent water gives a higher score (better anchorage).

    Non-coastal land faces score 0.

    Args:
        mesh: object with ``faces`` and ``face_adjacency``.
        is_ocean: (N,) bool — ocean faces.
        face_mean_elev_km: (N,) float — per-face mean elevation in km
            (ocean faces have negative values).
        depth_scale_km: depth at which the depth factor saturates to 1.0.

    Returns:
        (N,) float32 in [0, 1].
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
    """Score faces touched by multiple distinct river polylines.

    For each face, count how many distinct rivers have at least one non-in-lake
    segment endpoint mapped to that face.  A face with k rivers through it scores
    ``tanh(k - 1)`` (≈ 0 for k=1, ≈ 0.76 for k=2, ≈ 0.96 for k=3+).

    Args:
        rivers: list of ``(positions (M,3), flows, in_lake (M-1,))`` tuples as
            stored in the ``"rivers"`` globe feature, or None.
        face_centroids: (N, 3) float — unit-sphere centroid of each face; used
            to map polyline positions to face indices.
        n_faces: total face count.

    Returns:
        (N,) float32 in [0, 1].
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
    """Binary mask of navigable-river head faces (the fall-line points).

    For each river the navigable head is the first non-in-lake segment (walking
    from source toward mouth) whose mapped face has ``river_navigability > 0``.
    If the entire non-lake portion is navigable, the source face is the head.

    These are the transshipment points where overland routes switch to river
    transport (or vice versa) — historically high-value city sites.

    Args:
        rivers: list of ``(positions (M,3), flows, in_lake (M-1,))`` tuples.
        face_centroids: (N, 3) float — unit-sphere centroid of each face.
        river_navigability: (N,) float — per-face navigability taper (0 = rapids,
            1 = placid); from ``CostSurfaces.river_navigability``.
        n_faces: total face count.

    Returns:
        (N,) float32 — 1.0 at each river's navigable head face, 0 elsewhere.
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
                break  # first navigable non-lake face from source = head

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
    min_neighbor_cost: float = 3.0,
    cost_scale: float = 1.5,
) -> np.ndarray:
    """Score low-cost corridors through high-cost terrain (mountain passes).

    A pass candidate is a land face whose ``land_base_cost`` is below the mean
    of its land neighbors, AND whose land neighbors' mean cost exceeds
    ``min_neighbor_cost`` (ruling out flat plains where everything is cheap).
    Score = ``contrast / (contrast + cost_scale)`` where
    ``contrast = mean_neighbor_cost − face_cost``.

    Args:
        mesh: object with ``faces`` and ``face_adjacency``.
        land_base_cost: (N,) float — per-face isotropic land traversal cost.
        is_water: (N,) bool — water faces are excluded.
        min_neighbor_cost: neighbors' mean cost must exceed this for the face to
            qualify as a pass (prevents flat grasslands from scoring).
        cost_scale: half-saturation constant for the contrast sigmoid.

    Returns:
        (N,) float32 in [0, 1].
    """
    n = len(mesh.faces)
    cost = np.asarray(land_base_cost, dtype=np.float64)
    is_water = np.asarray(is_water, dtype=bool)

    score = np.zeros(n, dtype=np.float64)
    cscale = max(cost_scale, 1e-9)
    for i, adj in enumerate(mesh.face_adjacency):
        if is_water[i]:
            continue
        land_adj = [j for j in adj if not is_water[j]]
        if len(land_adj) < 2:
            continue
        neighbor_mean = cost[np.array(land_adj, dtype=np.intp)].mean()
        if neighbor_mean < min_neighbor_cost:
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
    """Binary score for land faces that border a strait-tagged ocean face.

    Strait-adjacent land faces can control the narrow passage, making them
    high-value military and trade sites regardless of their local suitability.

    Args:
        mesh: object with ``faces`` and ``face_adjacency``.
        is_strait: (N,) bool — strait-tagged ocean faces.
        is_water: (N,) bool — water faces score 0 (they are not settlement sites).

    Returns:
        (N,) float32 — 1.0 where adjacent to a strait, 0 elsewhere.
    """
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


def combine_situational_value(
    is_water: np.ndarray,
    harbor: np.ndarray,
    confluence: np.ndarray,
    nav_head: np.ndarray,
    passes: np.ndarray,
    straits: np.ndarray,
    *,
    harbor_weight: float = 0.25,
    confluence_weight: float = 0.20,
    nav_head_weight: float = 0.20,
    pass_weight: float = 0.20,
    strait_weight: float = 0.15,
) -> np.ndarray:
    """Weighted-average situational value in [0, 1]; water faces are forced to 0.

    Weights are normalized so they need not sum to exactly 1.  A face that is a
    confluence AND a navigable head scores doubly — the additive structure rewards
    faces with multiple strategic properties (fall-line city sites, natural harbors
    on rivers, etc.).

    Args:
        is_water: (N,) bool — open water; these faces receive 0.
        harbor:     (N,) float32 in [0, 1].
        confluence: (N,) float32 in [0, 1].
        nav_head:   (N,) float32 in [0, 1].
        passes:     (N,) float32 in [0, 1].
        straits:    (N,) float32 in [0, 1].

    Returns:
        (N,) float32 in [0, 1].
    """
    total_w = harbor_weight + confluence_weight + nav_head_weight + pass_weight + strait_weight
    total_w = max(total_w, 1e-9)

    score = (
        harbor_weight * np.asarray(harbor, dtype=np.float64)
        + confluence_weight * np.asarray(confluence, dtype=np.float64)
        + nav_head_weight * np.asarray(nav_head, dtype=np.float64)
        + pass_weight * np.asarray(passes, dtype=np.float64)
        + strait_weight * np.asarray(straits, dtype=np.float64)
    ) / total_w

    score = np.where(np.asarray(is_water, dtype=bool), 0.0, score)
    return np.clip(score, 0.0, 1.0).astype(np.float32)
