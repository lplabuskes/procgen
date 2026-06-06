"""Per-face helpers for the resources stage.

Two cheap, robust signals derived from data already on the mesh:
  - `face_elevation_range`: pixel elevation span per face, the "ruggedness" proxy
    we use in place of unreliable per-pixel slope (see
    docs/resources_generation_spec.md for the slope rationale).
  - `face_hops_from_seeds`: BFS hop count on the face-adjacency graph from a
    boolean seed mask, used for water-proximity and volcanism-proximity terms.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from core.mesh import PolygonMesh
from worldgen.geology.channels import VOLCANIC_ACTIVITY


def face_elevation_range(elevation_tex: np.ndarray) -> np.ndarray:
    """Per-face pixel elevation span, in km.

    Args:
        elevation_tex: (N, H, W, 1) float32; channel 0 is elevation in km.

    Returns:
        (N,) float32 — `pixel_elev.max() - pixel_elev.min()` per face.
    """
    elev = elevation_tex[..., 0]
    return (elev.max(axis=(1, 2)) - elev.min(axis=(1, 2))).astype(np.float32)


def face_hops_from_seeds(
    mesh: PolygonMesh,
    seed_mask: np.ndarray,
    max_hops: int = 16,
) -> np.ndarray:
    """Per-face shortest-path hop count from any seed face, capped at `max_hops`.

    BFS on the face adjacency graph. Seed faces return 0. Faces unreachable
    within `max_hops` return `max_hops + 1`; callers using `exp(-hops * decay)`
    treat this naturally as a near-zero contribution.

    Args:
        mesh: PolygonMesh with populated `face_adjacency`.
        seed_mask: (N,) bool — True for seed faces.
        max_hops: BFS depth cap.

    Returns:
        (N,) int32 hop counts.
    """
    n = len(mesh.faces)
    if seed_mask.shape != (n,):
        raise ValueError(f"seed_mask shape {seed_mask.shape} != ({n},)")

    sentinel = max_hops + 1
    hops = np.full(n, sentinel, dtype=np.int32)
    seed_idx = np.flatnonzero(seed_mask)
    if seed_idx.size == 0:
        return hops
    hops[seed_idx] = 0

    q: deque[int] = deque(int(i) for i in seed_idx)
    while q:
        u = q.popleft()
        d = int(hops[u])
        if d >= max_hops:
            continue
        for v in mesh.face_adjacency[u]:
            if hops[v] > d + 1:
                hops[v] = d + 1
                q.append(int(v))
    return hops


def river_traversed_faces(rivers, n_faces: int) -> np.ndarray:
    """Bool mask of faces that any river segment passes through.

    Args:
        rivers: iterable of `River` objects (from `worldgen.hydrology.types`),
            or None.
        n_faces: total face count.

    Returns:
        (n_faces,) bool — True where a river segment endpoint references the face.
    """
    mask = np.zeros(n_faces, dtype=bool)
    if not rivers:
        return mask
    for river in rivers:
        for seg in river.segments:
            if seg.in_lake:
                continue
            mask[seg.start.face_id] = True
            mask[seg.end.face_id] = True
    return mask


def lake_containing_faces(lake_mask_tex: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Bool mask of faces whose lake-mask fraction exceeds `threshold`.

    Args:
        lake_mask_tex: (N, H, W, 1) float32; channel 0 is lake mask (1=lake).
        threshold: minimum area fraction of lake pixels to count the face.

    Returns:
        (N,) bool mask.
    """
    return (lake_mask_tex[..., 0].mean(axis=(1, 2)) >= threshold)


def volcanic_faces(mesh: PolygonMesh, threshold: float) -> np.ndarray:
    """Bool mask of faces with `volcanic_activity` >= threshold.

    Reads `volcanic_activity` from per-face metadata (written by the geology
    stage). Faces without the key are treated as non-volcanic.
    """
    return VOLCANIC_ACTIVITY.read(mesh) >= threshold


def coastal_ocean_faces(mesh: PolygonMesh, is_ocean: np.ndarray) -> np.ndarray:
    """Bool mask of ocean faces that border at least one non-ocean face.

    Used as BFS seeds for computing ocean depth (distance from coast into open
    water), which drives the coastal-proximity factor in aquatic fertility.
    """
    n = len(mesh.faces)
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if not is_ocean[i]:
            continue
        for nb in mesh.face_adjacency[i]:
            if not is_ocean[nb]:
                mask[i] = True
                break
    return mask
