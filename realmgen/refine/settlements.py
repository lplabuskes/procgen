"""Texel-level settlement position refinement.

A settlement's face-level position is its face centroid.  This module snaps that
position to the texel within the same face that best matches a placement
preference, in priority order:

1. **Coast** — a land texel adjacent to water, preferring ocean over lake.
2. **River** — a land texel within a texel-spacing of a river polyline.
3. **Fertility** — among the remaining candidates, the highest per-face fertility
   (a no-op tiebreak within a single face, kept for correctness if candidates
   ever span faces).
4. **Nearest centroid** — deterministic final tiebreak; keeps the marker stable
   and central among equally-ranked candidates.

Positions are mutated in place on the ``Settlement`` objects.  The function
returns the ``{old_position_key: new_position}`` map so dependent geometry (trade
routes) can be remapped without re-deriving which settlement moved.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import KDTree


def position_key(pos: np.ndarray) -> tuple[float, float, float]:
    """Hashable key for a unit-sphere position, stable for centroid-derived points."""
    p = np.asarray(pos, dtype=np.float64).ravel()
    return (round(float(p[0]), 9), round(float(p[1]), 9), round(float(p[2]), 9))


def _neighbor_any(mask2d: np.ndarray) -> np.ndarray:
    """True where any 4-neighbour (no wrap) of a (H, W) boolean grid is True."""
    out = np.zeros_like(mask2d)
    out[1:, :] |= mask2d[:-1, :]
    out[:-1, :] |= mask2d[1:, :]
    out[:, 1:] |= mask2d[:, :-1]
    out[:, :-1] |= mask2d[:, 1:]
    return out


def _build_river_tree(rivers) -> Optional[KDTree]:
    """Build a KD-tree over all river polyline vertices, or None if unavailable."""
    if not rivers:
        return None
    chunks = []
    for entry in rivers:
        pts = np.asarray(entry[0], dtype=np.float64)
        if pts.ndim == 2 and pts.shape[0] > 0:
            chunks.append(pts)
    if not chunks:
        return None
    return KDTree(np.concatenate(chunks, axis=0))


def refine_settlement_positions(
    settlements,
    *,
    mesh,
    ocean_tex: np.ndarray,
    lake_tex: Optional[np.ndarray],
    fertility: np.ndarray,
    rivers,
    river_threshold_factor: float = 1.5,
) -> dict[tuple[float, float, float], np.ndarray]:
    """Snap each settlement to its best in-face texel; mutate ``position`` in place.

    Returns ``{old_position_key: new_position}`` for settlements that moved.
    """
    moved: dict[tuple[float, float, float], np.ndarray] = {}
    if not settlements:
        return moved

    n, h, w = ocean_tex.shape[0], ocean_tex.shape[1], ocean_tex.shape[2]
    positions = mesh.face_texture_positions(h, w)  # (N, H, W, 3)
    centroids = np.asarray(mesh.face_centroids, dtype=np.float64)
    fertility = np.asarray(fertility, dtype=np.float64)

    ocean = ocean_tex[..., 0] > 0.5                              # (N, H, W)
    lake = (lake_tex[..., 0] > 0.5) if lake_tex is not None else np.zeros_like(ocean)
    land = ~(ocean | lake)

    river_tree = _build_river_tree(rivers)

    for s in settlements:
        f = int(s.face_id)
        if f < 0 or f >= n:
            continue
        pos_f = positions[f].reshape(-1, 3).astype(np.float64)  # (T, 3)
        land2d = land[f]
        if not land2d.any():
            continue  # all-water texel grid; leave at centroid

        # Coast preference (ocean over lake).
        coast_ocean = (land2d & _neighbor_any(ocean[f])).reshape(-1)
        coast_lake = (land2d & _neighbor_any(lake[f])).reshape(-1)
        candidate = land2d.reshape(-1).copy()
        if coast_ocean.any():
            candidate = coast_ocean
        elif coast_lake.any():
            candidate = coast_lake

        # River preference among current candidates.
        if river_tree is not None and candidate.any():
            cand_pos = pos_f[candidate]
            spacing = _texel_spacing(positions[f])
            dists, _ = river_tree.query(cand_pos)
            near = dists < (river_threshold_factor * spacing)
            if near.any():
                idx = np.flatnonzero(candidate)[near]
                new_mask = np.zeros_like(candidate)
                new_mask[idx] = True
                candidate = new_mask

        cand_idx = np.flatnonzero(candidate)
        if cand_idx.size == 0:
            continue

        # Fertility tiebreak (per-face → constant within one face), then nearest
        # to the centroid as the deterministic final selector.
        cand_pos = pos_f[cand_idx]
        d_centroid = np.linalg.norm(cand_pos - centroids[f][None, :], axis=1)
        # All candidates share face f, so fertility is uniform; rank by distance.
        best = cand_idx[int(np.argmin(d_centroid))]
        new_pos = pos_f[best].copy()

        if not np.array_equal(new_pos, np.asarray(s.position, dtype=np.float64)):
            moved[position_key(s.position)] = new_pos
            s.position = new_pos

    return moved


def _texel_spacing(face_positions: np.ndarray) -> float:
    """Median 3-D distance between horizontally-adjacent texels of one face."""
    grid = np.asarray(face_positions, dtype=np.float64)  # (H, W, 3)
    diffs = np.linalg.norm(grid[:, 1:, :] - grid[:, :-1, :], axis=2)
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs))


def remap_routes(routes, moved: dict) -> list:
    """Return a copy of trade routes with endpoints moved to refined settlement
    positions.  Endpoints not matching a moved settlement are left unchanged."""
    if not routes:
        return routes
    out = []
    for start_pos, end_pos, flow in routes:
        new_start = moved.get(position_key(start_pos), start_pos)
        new_end = moved.get(position_key(end_pos), end_pos)
        out.append((np.asarray(new_start), np.asarray(new_end), flow))
    return out
