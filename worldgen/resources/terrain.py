"""Per-face helpers for the resources stage: ruggedness (pixel elevation span, used instead of
noise-dominated per-pixel slope), BFS hop counts, and seed masks."""
from __future__ import annotations

from collections import deque

import numpy as np

from core.mesh import PolygonMesh
from worldgen.geology.channels import VOLCANIC_ACTIVITY


def face_elevation_range(elevation_tex: np.ndarray) -> np.ndarray:
    """(N,) float32 per-face pixel elevation span (km) from an (N, H, W, 1) elevation texture."""
    elev = elevation_tex[..., 0]
    return (elev.max(axis=(1, 2)) - elev.min(axis=(1, 2))).astype(np.float32)


def face_hops_from_seeds(
    mesh: PolygonMesh,
    seed_mask: np.ndarray,
    max_hops: int = 16,
) -> np.ndarray:
    """(N,) int32 BFS hop count from any seed face; faces beyond `max_hops` get `max_hops + 1`."""
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
    """(n_faces,) bool: faces referenced by a non-lake river segment endpoint."""
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
    """(N,) bool: faces whose lake pixel fraction is at least `threshold`."""
    return (lake_mask_tex[..., 0].mean(axis=(1, 2)) >= threshold)


def volcanic_faces(mesh: PolygonMesh, threshold: float) -> np.ndarray:
    """Bool mask of faces with `volcanic_activity` >= threshold."""
    return VOLCANIC_ACTIVITY.read(mesh) >= threshold


def coastal_ocean_faces(mesh: PolygonMesh, is_ocean: np.ndarray) -> np.ndarray:
    """Bool mask of ocean faces bordering a non-ocean face (seeds for ocean depth hops)."""
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
