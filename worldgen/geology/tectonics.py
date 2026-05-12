import heapq
import random
from typing import List, Optional

import numpy as np

from mesh.mymesh import PolygonMesh


def build_face_adjacency(mesh: PolygonMesh) -> List[List[int]]:
    """Return adjacency[i] = list of face indices sharing an edge with face i."""
    edge_to_faces: dict = {}
    for face_idx, face in enumerate(mesh.faces):
        n = len(face)
        for i in range(n):
            edge = frozenset([face[i], face[(i + 1) % n]])
            edge_to_faces.setdefault(edge, []).append(face_idx)

    adjacency: List[List[int]] = [[] for _ in range(len(mesh.faces))]
    for neighbors in edge_to_faces.values():
        if len(neighbors) == 2:
            a, b = neighbors
            adjacency[a].append(b)
            adjacency[b].append(a)
    return adjacency


def generate_plates(
    mesh: PolygonMesh,
    n_plates: int,
    seed: Optional[int] = None,
    compactness: float = 0.9,
) -> List[int]:
    """Assign each face to a plate via per-plate randomized-priority flood fill.

    Returns a list of length len(mesh.faces) where entry i is the plate id (0-indexed)
    of face i.  Each step picks a plate uniformly at random (so frontier size gives no
    compounding advantage) and expands it one face using a distance-biased heap so
    that cells closer to the seed are preferred, producing compact plates.

    compactness in [0, 1]: 0 = fully random priority (snaky), 1 = pure distance
    (circular). Default 0.7 gives compact plates with irregular borders.
    """
    rng = random.Random(seed)
    n_faces = len(mesh.faces)
    n_plates = min(n_plates, n_faces)
    adjacency = build_face_adjacency(mesh)

    # Precompute normalized face centroids for distance calculations.
    face_centroids = np.zeros((n_faces, 3), dtype=np.float64)
    for i, face in enumerate(mesh.faces):
        pts = mesh.vertices[np.array(face)]
        c = np.mean(pts, axis=0)
        face_centroids[i] = c / np.linalg.norm(c)

    plate_ids: List[int] = [-1] * n_faces
    seeds = rng.sample(range(n_faces), n_plates)
    seed_centroids = face_centroids[seeds]

    def push(heap: list, plate_id: int, face_idx: int) -> None:
        # dist proxy: (1 - cos θ) / 2 normalizes angular distance to [0, 1]
        cos_theta = float(np.dot(seed_centroids[plate_id], face_centroids[face_idx]))
        dist = (1.0 - cos_theta) / 2.0
        priority = compactness * dist + (1.0 - compactness) * rng.random()
        heapq.heappush(heap, (priority, face_idx))

    heaps: List[list] = [[] for _ in range(n_plates)]
    for plate_id, face_idx in enumerate(seeds):
        plate_ids[face_idx] = plate_id
        for neighbor in adjacency[face_idx]:
            push(heaps[plate_id], plate_id, neighbor)

    # Each plate gets a random weight; weighted selection produces a realistic
    # spread of plate sizes rather than near-uniform areas.
    active = list(range(n_plates))
    active_weights = [rng.random() ** 0.25 for _ in range(n_plates)]
    while active:
        idx = rng.choices(range(len(active)), weights=active_weights, k=1)[0]
        plate_id = active[idx]
        heap = heaps[plate_id]

        while heap:
            _, face_idx = heapq.heappop(heap)
            if plate_ids[face_idx] == -1:
                plate_ids[face_idx] = plate_id
                for neighbor in adjacency[face_idx]:
                    if plate_ids[neighbor] == -1:
                        push(heap, plate_id, neighbor)
                break

        if not heap:
            active[idx] = active[-1]
            active_weights[idx] = active_weights[-1]
            active.pop()
            active_weights.pop()

    return plate_ids
