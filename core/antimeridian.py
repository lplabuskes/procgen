from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.mesh import PolygonMesh


def find_ideal_antimeridian(mesh: PolygonMesh, n_samples: int = 360) -> float:
    """
    Return the longitude (radians) for the map cut that minimizes land crossings.

    Among all candidate cut longitudes with the fewest land-face crossings,
    picks the one that maximizes the minimum angular distance to any land face
    centroid — i.e., the cut that is as far from land as possible.

    Falls back to π (the standard antimeridian) when the mesh has no
    is_ocean face metadata.
    """
    has_ocean_meta = any("is_ocean" in meta for meta in mesh.face_metadata)
    if not has_ocean_meta:
        return np.pi

    is_land = np.array(
        [not meta.get("is_ocean", False) for meta in mesh.face_metadata],
        dtype=bool,
    )
    if not np.any(is_land):
        return np.pi

    # Padded per-face vertex longitudes: (n_faces, max_verts), NaN for padding.
    verts = mesh.vertices
    n_faces = len(mesh.faces)
    max_n = max(len(f) for f in mesh.faces)
    face_lons = np.full((n_faces, max_n), np.nan, dtype=np.float64)
    for i, face in enumerate(mesh.faces):
        v = verts[np.array(face, dtype=np.int32)]
        face_lons[i, : len(face)] = np.arctan2(v[:, 1], v[:, 0])

    # Land centroid longitudes for the distance tie-breaker.
    c = mesh.face_centroids  # (n_faces, 3)
    land_centroid_lons = np.arctan2(c[is_land, 1], c[is_land, 0])

    candidates = np.linspace(-np.pi, np.pi, n_samples, endpoint=False)
    land_counts = np.empty(n_samples, dtype=np.int32)

    for ci, cut_lon in enumerate(candidates):
        # Shift all vertex lons so cut_lon maps to ±π (the standard antimeridian).
        # For small convex faces, max−min > π reliably detects a crossing.
        shifted = ((face_lons - cut_lon) % (2.0 * np.pi)) - np.pi
        max_s = np.nanmax(shifted, axis=1)
        min_s = np.nanmin(shifted, axis=1)
        land_counts[ci] = int(np.sum(((max_s - min_s) > np.pi) & is_land))

    min_land = int(land_counts.min())
    good = candidates[land_counts == min_land]

    # Among cuts with fewest land crossings, pick the one farthest from all land.
    best_cut = float(good[0])
    best_min_dist = -1.0
    for cut_lon in good:
        diffs = np.abs(land_centroid_lons - cut_lon)
        min_dist = float(np.minimum(diffs, 2.0 * np.pi - diffs).min())
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_cut = float(cut_lon)

    return best_cut
