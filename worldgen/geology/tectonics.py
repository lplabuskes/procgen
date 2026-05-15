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


def compute_face_stress(
    mesh: PolygonMesh,
    plate_ids: List[int],
    plate_omegas: np.ndarray,
) -> np.ndarray:
    """Return stress at each face (shape: n_faces,).

    Stress is the maximum relative tangential velocity across any cross-plate edge:
        stress[i] = max over cross-plate neighbors j of |(omega_k - omega_m) x p_i|

    Interior faces (all neighbors on the same plate) get stress 0.
    """
    adjacency = mesh.face_adjacency
    face_centroids = mesh.face_centroids

    n_faces = len(mesh.faces)
    pids = np.asarray(plate_ids)
    stress = np.zeros(n_faces, dtype=np.float64)

    for i in range(n_faces):
        k = pids[i]
        p_i = face_centroids[i]
        max_rel = 0.0
        for j in adjacency[i]:
            m = pids[j]
            if m == k:
                continue
            delta_omega = plate_omegas[k] - plate_omegas[m]
            rel_vel = np.linalg.norm(np.cross(delta_omega, p_i))
            if rel_vel > max_rel:
                max_rel = rel_vel
        stress[i] = max_rel

    return stress


def _hypothetical_stress(
    face_idx: int,
    candidate_plate: int,
    plate_ids: List[int],
    plate_omegas: np.ndarray,
    face_centroids: np.ndarray,
    adjacency: List[List[int]],
) -> float:
    """Stress face_idx would have if assigned to candidate_plate."""
    p_i = face_centroids[face_idx]
    max_rel = 0.0
    for j in adjacency[face_idx]:
        m = plate_ids[j]
        if m == candidate_plate:
            continue
        delta_omega = plate_omegas[candidate_plate] - plate_omegas[m]
        rel_vel = float(np.linalg.norm(np.cross(delta_omega, p_i)))
        if rel_vel > max_rel:
            max_rel = rel_vel
    return max_rel


def _plate_remains_connected(
    face_idx: int,
    plate_id: int,
    plate_ids: List[int],
    adjacency: List[List[int]],
) -> bool:
    """Return True if plate_id stays connected after removing face_idx from it."""
    members = [i for i, pid in enumerate(plate_ids) if pid == plate_id and i != face_idx]
    if not members:
        return True
    in_plate = set(members)
    visited = {members[0]}
    stack = [members[0]]
    while stack:
        cur = stack.pop()
        for nb in adjacency[cur]:
            if nb in in_plate and nb not in visited:
                visited.add(nb)
                stack.append(nb)
    return len(visited) == len(members)


def apply_stress_reassignment(
    mesh: PolygonMesh,
    plate_ids: List[int],
    plate_omegas: np.ndarray,
    stress_percentile: float = 90.0,
) -> List[int]:
    """Reassign high-stress boundary faces to the neighboring plate that minimises their stress.

    Faces in the top `stress_percentile` percent of boundary stress are candidates.
    Each candidate is assigned to whichever adjacent plate would yield the lowest
    hypothetical stress, keeping the current plate if no neighbor improves things.
    """
    adjacency = mesh.face_adjacency
    face_centroids = mesh.face_centroids

    stress = compute_face_stress(mesh, plate_ids, plate_omegas)

    boundary_stress = stress[stress > 0.0]
    if boundary_stress.size == 0:
        return list(plate_ids)
    threshold = float(np.percentile(boundary_stress, stress_percentile))

    current = list(plate_ids)
    for i in range(len(plate_ids)):
        if stress[i] <= threshold:
            continue

        neighbor_plates = {current[j] for j in adjacency[i]} - {current[i]}
        if not neighbor_plates:
            continue

        best_plate = current[i]
        best_stress = _hypothetical_stress(i, current[i], current, plate_omegas, face_centroids, adjacency)

        for m in neighbor_plates:
            s = _hypothetical_stress(i, m, current, plate_omegas, face_centroids, adjacency)
            if s < best_stress:
                best_stress = s
                best_plate = m

        if best_plate != current[i] and _plate_remains_connected(i, current[i], current, adjacency):
            current[i] = best_plate

    return current


def relax_plates(
    mesh: PolygonMesh,
    plate_ids: List[int],
    plate_omegas: np.ndarray,
    stress_percentile: float = 90.0,
    max_passes: int = 10,
) -> List[int]:
    """Iteratively reassign high-stress faces until convergence or max_passes.

    Returns the stabilised plate assignment.
    """
    current = list(plate_ids)
    for _ in range(max_passes):
        updated = apply_stress_reassignment(
            mesh, current, plate_omegas, stress_percentile
        )
        if updated == current:
            break
        current = updated
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
