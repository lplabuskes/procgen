import math
from collections import deque
from typing import List

import numpy as np

from mesh.mymesh import PolygonMesh

# Baseline elevation by plate type (km, roughly)
_BASELINE = {
    "oceanic": -4.0,
    "continental": 0.3,
}

def _boundary_delta(
    my_type: str,
    btype: str,
    normal_avg: float,
    neighbor_types: set,
    rate_norm: float,
) -> float:
    """Elevation delta (km) to apply at a boundary face, before spatial decay."""
    rate = abs(normal_avg) / rate_norm

    if btype == "convergent":
        if my_type == "continental" and "continental" in neighbor_types:
            # Continental-continental collision: mountain range on both sides
            return 4.0 * rate
        elif my_type == "continental" and "oceanic" in neighbor_types:
            # Oceanic subducts under continental: volcanic arc
            return 2.5 * rate
        elif my_type == "oceanic" and "continental" in neighbor_types:
            # Oceanic subducts: trench
            return -2.5 * rate
        else:
            # Oceanic-oceanic: trench on both sides (island arc is minor)
            return -1.5 * rate

    if btype == "divergent":
        if my_type == "oceanic":
            # Mid-ocean ridge: elevated above oceanic baseline
            return 2.0 * rate
        else:
            # Continental rift: depressed below continental baseline
            return -1.5 * rate

    # Transform boundaries and interior: no delta
    return 0.0


def compute_face_elevations(
    mesh: PolygonMesh,
    plate_types_by_face: List[str],
    boundary_data: List[dict],
    decay_rate: float = 0.5,
    max_hops: int = 8,
    rate_norm: float = 1.0,
) -> np.ndarray:
    """Compute elevation (km) for each face based on plate type and boundary effects.

    Baseline elevation is set by plate type (oceanic/continental). Each boundary face
    receives a delta based on boundary type, convergence rate, and neighboring plate
    type context. That delta is then propagated outward via BFS with exponential decay
    over hop distance.

    Args:
        mesh: The dual polygon mesh.
        plate_types_by_face: Per-face plate type string ('oceanic' or 'continental').
        boundary_data: Per-face dicts from classify_plate_boundaries(), containing
            boundary_type, normal_avg, and neighbor_plate_types.
        decay_rate: Controls how fast boundary effects decay with hop distance.
            elevation_contribution = delta * exp(-decay_rate * hops).
        max_hops: Maximum BFS hops each boundary face influences.

    Returns:
        float64 array of shape (n_faces,) with elevation in km.
    """
    n_faces = len(mesh.faces)
    adjacency = mesh.face_adjacency

    elevation = np.array(
        [_BASELINE.get(plate_types_by_face[i], 0.0) for i in range(n_faces)],
        dtype=np.float64,
    )

    # Compute the elevation delta at each boundary face.
    boundary_deltas = np.zeros(n_faces, dtype=np.float64)
    for i in range(n_faces):
        bd = boundary_data[i]
        if bd["boundary_type"] == "interior":
            continue
        boundary_deltas[i] = _boundary_delta(
            my_type=plate_types_by_face[i],
            btype=bd["boundary_type"],
            normal_avg=bd["normal_avg"],
            neighbor_types=bd["neighbor_plate_types"],
            rate_norm=rate_norm,
        )

    # Accumulate decayed contributions from each boundary face via BFS.
    delta_field = np.zeros(n_faces, dtype=np.float64)
    for src in range(n_faces):
        delta = boundary_deltas[src]
        if delta == 0.0:
            continue

        visited: dict[int, int] = {src: 0}
        queue: deque[tuple[int, int]] = deque([(src, 0)])
        while queue:
            face_idx, hops = queue.popleft()
            contribution = delta * math.exp(-decay_rate * hops)
            if abs(contribution) > abs(delta_field[face_idx]):
                delta_field[face_idx] = contribution
            if hops < max_hops:
                for nb in adjacency[face_idx]:
                    if nb not in visited:
                        visited[nb] = hops + 1
                        queue.append((nb, hops + 1))

    elevation += delta_field
    return elevation
