import heapq
import random
from typing import List, Optional

import numpy as np

from mesh.mymesh import PolygonMesh


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
    adjacency = mesh.face_adjacency
    face_centroids = mesh.face_centroids

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

    # Absorb any plate wholly surrounded by a single other plate.
    changed = True
    while changed:
        changed = False
        plate_neighbor_set: dict[int, set[int]] = {}
        for face_idx in range(n_faces):
            pid = plate_ids[face_idx]
            if pid not in plate_neighbor_set:
                plate_neighbor_set[pid] = set()
            for nb in adjacency[face_idx]:
                nb_pid = plate_ids[nb]
                if nb_pid != pid:
                    plate_neighbor_set[pid].add(nb_pid)
        for pid, neighbors in plate_neighbor_set.items():
            if len(neighbors) == 1:
                surrounding = next(iter(neighbors))
                for face_idx in range(n_faces):
                    if plate_ids[face_idx] == pid:
                        plate_ids[face_idx] = surrounding
                changed = True
                break  # recompute neighbor sets with updated plate_ids

    return plate_ids


def generate_plate_velocities(
    mesh: PolygonMesh,
    plate_ids: List[int],
    n_plates: int,
    drift_scale: float = 1.0,
    spin_scale: float = 0.1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return angular velocity vectors for each plate, shape (n_plates, 3).

    The velocity of a point p on the unit sphere belonging to plate k is omega[k] x p.
    Each plate's omega is a sum of a drift component (perpendicular to the plate centroid,
    translating the plate across the sphere) and a spin component (along the centroid,
    rotating the plate in place).  Real plates have much smaller spin than drift, so
    spin_scale defaults to a tenth of drift_scale.
    """
    rng = np.random.default_rng(seed)

    # Compute normalized centroid for each plate.
    plate_centroid_sum = np.zeros((n_plates, 3), dtype=np.float64)
    plate_count = np.zeros(n_plates, dtype=np.int64)
    for face_idx, pid in enumerate(plate_ids):
        pts = mesh.vertices[np.array(mesh.faces[face_idx])]
        plate_centroid_sum[pid] += np.mean(pts, axis=0)
        plate_count[pid] += 1
    norms = np.linalg.norm(plate_centroid_sum, axis=1, keepdims=True)
    plate_centroids = plate_centroid_sum / np.where(norms > 0, norms, 1.0)

    # Draw a random drift pole (uniform on S²) and a random speed for each plate.
    drift_axes = rng.standard_normal((n_plates, 3))
    drift_axes /= np.linalg.norm(drift_axes, axis=1, keepdims=True)
    drift_speeds = rng.exponential(scale=drift_scale, size=n_plates)
    drift_omega = drift_axes * drift_speeds[:, np.newaxis]

    # Project out the centroid component so drift is purely tangential.
    dot = np.einsum("ij,ij->i", drift_omega, plate_centroids)
    drift_omega -= dot[:, np.newaxis] * plate_centroids

    # Spin: random magnitude along the centroid axis.
    spin_speeds = rng.normal(scale=spin_scale, size=n_plates)
    spin_omega = spin_speeds[:, np.newaxis] * plate_centroids

    return (drift_omega + spin_omega).astype(np.float64)


def _tarjan_aps_iterative(nodes: set, adjacency: list) -> set:
    """Find articulation points in the subgraph induced by nodes using iterative Tarjan's."""
    if len(nodes) <= 2:
        return set()

    _ROOT = -1
    disc: dict[int, int] = {}
    low: dict[int, int] = {}
    parent: dict[int, int] = {}
    root_child_count: dict[int, int] = {}
    ap: set[int] = set()
    timer = [0]

    nbrs = {u: [v for v in adjacency[u] if v in nodes] for u in nodes}

    for start in nodes:
        if start in disc:
            continue

        parent[start] = _ROOT
        disc[start] = low[start] = timer[0]
        timer[0] += 1
        root_child_count[start] = 0

        stack = [(start, iter(nbrs[start]))]

        while stack:
            u, it = stack[-1]
            v = next(it, None)

            if v is None:
                stack.pop()
                if stack:
                    p = parent[u]
                    low[p] = min(low[p], low[u])
                    if parent[p] != _ROOT and low[u] >= disc[p]:
                        ap.add(p)
            elif v not in disc:
                parent[v] = u
                disc[v] = low[v] = timer[0]
                timer[0] += 1
                if parent[u] == _ROOT:
                    root_child_count[u] += 1
                stack.append((v, iter(nbrs[v])))
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for root, cnt in root_child_count.items():
        if cnt > 1:
            ap.add(root)

    return ap


def relax_plates(
    mesh: PolygonMesh,
    plate_ids: List[int],
    max_passes: int = 10,
) -> List[int]:
    """Iteratively remove appendages from plates using articulation point detection.

    An appendage is a set of faces reachable from the rest of its plate only through
    a single face (the articulation point). Each pass: for every plate, find articulation
    points, identify the faces stranded behind each one, and reassign those faces to
    whichever neighboring plate they border most. Repeats until stable or max_passes.
    """
    adjacency = mesh.face_adjacency
    face_centroids = mesh.face_centroids
    current = list(plate_ids)

    for _ in range(max_passes):
        plate_faces: dict[int, set] = {}
        for i, pid in enumerate(current):
            plate_faces.setdefault(pid, set()).add(i)

        # Representative face for each plate: closest to the plate's unit centroid.
        plate_reps: dict[int, int] = {}
        for pid, faces in plate_faces.items():
            face_list = list(faces)
            pts = face_centroids[face_list]
            centroid = np.mean(pts, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm
            plate_reps[pid] = face_list[int(np.argmax(pts @ centroid))]

        reassignments: dict[int, int] = {}

        for pid, faces in plate_faces.items():
            if len(faces) <= 1:
                continue
            rep = plate_reps[pid]
            aps = _tarjan_aps_iterative(faces, adjacency)
            if not aps:
                continue

            appendage_faces: set[int] = set()
            for ap in aps:
                # BFS from rep through the plate minus this AP and already-found appendages.
                remaining = faces - {ap} - appendage_faces
                if rep not in remaining:
                    continue
                visited: set[int] = {rep}
                queue = [rep]
                while queue:
                    node = queue.pop()
                    for nb in adjacency[node]:
                        if nb in remaining and nb not in visited:
                            visited.add(nb)
                            queue.append(nb)
                appendage_faces |= remaining - visited

            for f in appendage_faces:
                counts: dict[int, int] = {}
                for nb in adjacency[f]:
                    nb_pid = current[nb]
                    if nb_pid != pid:
                        counts[nb_pid] = counts.get(nb_pid, 0) + 1
                if counts:
                    reassignments[f] = max(counts, key=counts.__getitem__)

        if not reassignments:
            break
        for f, new_pid in reassignments.items():
            current[f] = new_pid

    return current


def assign_plate_types(
    n_plates: int,
    oceanic_fraction: float = 0.7,
    seed: Optional[int] = None,
) -> List[str]:
    """Classify each plate as 'oceanic' or 'continental'.

    Returns a list of length n_plates. oceanic_fraction controls the expected
    proportion of oceanic plates (default 0.7 matches Earth's rough ratio).
    """
    rng = random.Random(seed)
    return [
        "oceanic" if rng.random() < oceanic_fraction else "continental"
        for _ in range(n_plates)
    ]


def tag_face_metadata(
    mesh: PolygonMesh,
    plate_ids: List[int],
    plate_types: List[str],
) -> None:
    """Write plate_id and plate_type into each face's metadata dict in-place."""
    for face_idx, pid in enumerate(plate_ids):
        mesh.set_face_metadata(face_idx, plate_id=pid, plate_type=plate_types[pid])
