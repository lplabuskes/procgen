import math
import random
from collections import deque
from typing import List, Optional

import numpy as np

from core.mesh import PolygonMesh


def _boundary_volcanic_intensity(
    my_type: str,
    btype: str,
    normal_avg: float,
    neighbor_types: set,
    rate_norm: float,
) -> float:
    """Volcanic activity intensity [0, 1] at a boundary face, before decay."""
    rate = min(1.0, abs(normal_avg) / rate_norm)

    if btype == "divergent":
        return (0.5 if my_type == "oceanic" else 0.25) * rate

    if btype == "convergent":
        if my_type == "continental" and "oceanic" in neighbor_types:
            return 0.9 * rate  # volcanic arc, oceanic subducts under continental
        if my_type == "oceanic" and "oceanic" in neighbor_types:
            return 0.4 * rate  # island arc, both sides

    return 0.0


def compute_volcanism(
    mesh: PolygonMesh,
    plate_types_by_face: List[str],
    boundary_data: List[dict],
    n_hotspots: int = 8,
    decay_rate: float = 0.6,
    max_hops: int = 6,
    rate_norm: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Compute volcanic activity per face.

    Sources are ridges/rifts, convergent arcs and random hotspots; activity
    spreads by BFS with exponential decay per hop, max contribution wins.

    Returns:
        (n_faces,) float64 activity in [0, 1]; mostly exactly 0.
    """
    n_faces = len(mesh.faces)
    adjacency = mesh.face_adjacency
    rng = random.Random(seed)

    source_intensity = np.zeros(n_faces, dtype=np.float64)

    for i in range(n_faces):
        bd = boundary_data[i]
        if bd["boundary_type"] in ("interior", "transform"):
            continue
        source_intensity[i] = _boundary_volcanic_intensity(
            my_type=plate_types_by_face[i],
            btype=bd["boundary_type"],
            normal_avg=bd["normal_avg"],
            neighbor_types=bd["neighbor_plate_types"],
            rate_norm=rate_norm,
        )

    for hf in rng.sample(range(n_faces), min(n_hotspots, n_faces)):
        source_intensity[hf] = max(source_intensity[hf], rng.uniform(0.6, 1.0))

    activity = np.zeros(n_faces, dtype=np.float64)
    for src in range(n_faces):
        intensity = source_intensity[src]
        if intensity == 0.0:
            continue
        visited: dict[int, int] = {src: 0}
        queue: deque[tuple[int, int]] = deque([(src, 0)])
        while queue:
            face_idx, hops = queue.popleft()
            contribution = intensity * math.exp(-decay_rate * hops)
            if contribution > activity[face_idx]:
                activity[face_idx] = contribution
            if hops < max_hops:
                for nb in adjacency[face_idx]:
                    if nb not in visited:
                        visited[nb] = hops + 1
                        queue.append((nb, hops + 1))

    return activity


def generate_volcano_positions(
    mesh: PolygonMesh,
    volcanism: np.ndarray,
    elevations: np.ndarray,
    threshold: float = 0.3,
) -> np.ndarray:
    """Return unit-sphere positions of above-sea-level volcano peaks.

    A peak has activity ≥ `threshold`, elevation ≥ 0 and no neighbour with higher activity.

    Returns:
        float32 (K, 3) array of unit-sphere positions (may be empty).
    """
    adjacency = mesh.face_adjacency
    centroids = mesh.face_centroids  # (N, 3)
    peaks = [
        i for i, act in enumerate(volcanism)
        if act >= threshold
        and elevations[i] >= 0.0
        and all(volcanism[nb] <= act for nb in adjacency[i])
    ]
    if not peaks:
        return np.empty((0, 3), dtype=np.float32)
    return centroids[np.array(peaks, dtype=np.int32)].astype(np.float32)
