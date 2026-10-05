import heapq
import random
from typing import List, Optional

import numpy as np

from core.mesh import PolygonMesh


def generate_plates(
    mesh: PolygonMesh,
    n_plates: int,
    seed: Optional[int] = None,
    compactness: float = 0.9,
) -> List[int]:
    """Assign each face to a plate via per-plate randomized-priority flood fill.

    Each step picks a plate at random (not by frontier size, which would compound)
    and grows it one face from a distance-biased heap (compactness knob).
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
        # (1 - cos θ) / 2: angular distance proxy in [0, 1]
        cos_theta = float(np.dot(seed_centroids[plate_id], face_centroids[face_idx]))
        dist = (1.0 - cos_theta) / 2.0
        priority = compactness * dist + (1.0 - compactness) * rng.random()
        heapq.heappush(heap, (priority, face_idx))

    heaps: List[list] = [[] for _ in range(n_plates)]
    for plate_id, face_idx in enumerate(seeds):
        plate_ids[face_idx] = plate_id
        for neighbor in adjacency[face_idx]:
            push(heaps[plate_id], plate_id, neighbor)

    # Random per-plate weights spread plate sizes instead of near-uniform areas.
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


def compute_plate_centroids(
    mesh: PolygonMesh,
    plate_ids: List[int],
    n_plates: int,
) -> tuple[np.ndarray, List[int]]:
    """Return (centroids, centroid_faces): unit-sphere plate centroids and nearest-center face per plate."""
    face_centroids = mesh.face_centroids
    ids = np.array(plate_ids, dtype=np.int32)

    centroid_sum = np.zeros((n_plates, 3), dtype=np.float64)
    np.add.at(centroid_sum, ids, face_centroids)
    norms = np.linalg.norm(centroid_sum, axis=1, keepdims=True)
    centroids = (centroid_sum / np.where(norms > 0, norms, 1.0)).astype(np.float64)

    centroid_faces: List[int] = []
    for pid in range(n_plates):
        face_indices = np.where(ids == pid)[0]
        if len(face_indices) == 0:
            centroid_faces.append(-1)
        else:
            dots = face_centroids[face_indices] @ centroids[pid]
            centroid_faces.append(int(face_indices[np.argmax(dots)]))

    return centroids, centroid_faces


def generate_plate_velocities(
    mesh: PolygonMesh,
    plate_ids: List[int],
    n_plates: int,
    drift_scale: float = 1.0,
    spin_scale: float = 0.1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return angular velocity vectors for each plate, shape (n_plates, 3).

    Velocity at p on plate k is omega[k] x p.  omega = drift (perpendicular to the
    plate centroid) + spin (along it); real plates spin far less than they drift.
    """
    rng = np.random.default_rng(seed)

    plate_centroids, _ = compute_plate_centroids(mesh, plate_ids, n_plates)

    drift_axes = rng.standard_normal((n_plates, 3))
    drift_axes /= np.linalg.norm(drift_axes, axis=1, keepdims=True)
    drift_speeds = rng.exponential(scale=drift_scale, size=n_plates)
    drift_omega = drift_axes * drift_speeds[:, np.newaxis]

    dot = np.einsum("ij,ij->i", drift_omega, plate_centroids)
    drift_omega -= dot[:, np.newaxis] * plate_centroids

    spin_speeds = rng.normal(scale=spin_scale, size=n_plates)
    spin_omega = spin_speeds[:, np.newaxis] * plate_centroids

    plate_omegas = (drift_omega + spin_omega).astype(np.float64)

    # No-net-rotation: subtract the mass-weighted mean (area × density, or area
    # alone when plate_type metadata is absent).
    _DENSITY = {"oceanic": 3.0, "continental": 2.7}
    face_areas = mesh.face_areas
    sample_type = mesh.face_metadata[0].get("plate_type") if mesh.face_metadata else None
    if sample_type is not None:
        densities = np.array(
            [_DENSITY.get(mesh.face_metadata[i].get("plate_type", ""), 1.0)
             for i in range(len(plate_ids))],
            dtype=np.float64,
        )
        face_masses = face_areas * densities
    else:
        face_masses = face_areas
    plate_mass = np.zeros(n_plates, dtype=np.float64)
    np.add.at(plate_mass, np.array(plate_ids, dtype=np.int32), face_masses)
    omega_mean = (plate_mass[:, np.newaxis] * plate_omegas).sum(axis=0) / plate_mass.sum()
    plate_omegas -= omega_mean

    return plate_omegas


def compute_face_velocities(
    mesh: PolygonMesh,
    plate_ids: List[int],
    plate_omegas: np.ndarray,
) -> np.ndarray:
    """Return (n_faces, 3) surface velocities, omega x centroid."""
    face_centroids = mesh.face_centroids
    omegas = plate_omegas[np.array(plate_ids, dtype=np.int32)]
    return np.cross(omegas, face_centroids)


def _tarjan_aps_iterative(nodes: set, adjacency: list) -> set:
    """Articulation points of the subgraph induced by ``nodes`` (iterative Tarjan)."""
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

    An appendage is reachable from the rest of its plate only through one
    articulation point; its faces go to the neighbouring plate they border most.
    """
    adjacency = mesh.face_adjacency
    current = list(plate_ids)

    for _ in range(max_passes):
        plate_faces: dict[int, set] = {}
        for i, pid in enumerate(current):
            plate_faces.setdefault(pid, set()).add(i)

        n_plates_local = max(plate_faces.keys()) + 1
        _, centroid_faces = compute_plate_centroids(mesh, current, n_plates_local)
        plate_reps: dict[int, int] = {pid: centroid_faces[pid] for pid in plate_faces}

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
                remaining = faces - {ap} - appendage_faces
                if rep in remaining:
                    # Faces unreachable from rep are appendages.
                    visited: set[int] = {rep}
                    queue = [rep]
                    while queue:
                        node = queue.pop()
                        for nb in adjacency[node]:
                            if nb in remaining and nb not in visited:
                                visited.add(nb)
                                queue.append(nb)
                    appendage_faces |= remaining - visited
                else:
                    # rep is the AP itself: all but the largest component are appendages.
                    components: list[set[int]] = []
                    unvisited = set(remaining)
                    while unvisited:
                        start = next(iter(unvisited))
                        comp: set[int] = set()
                        stack = [start]
                        while stack:
                            node = stack.pop()
                            if node in comp:
                                continue
                            comp.add(node)
                            for nb in adjacency[node]:
                                if nb in remaining and nb not in comp:
                                    stack.append(nb)
                        unvisited -= comp
                        components.append(comp)
                    if len(components) > 1:
                        main = max(components, key=len)
                        for comp in components:
                            if comp is not main:
                                appendage_faces |= comp

            # Faces with external neighbours first, then propagate inward.
            boundary_queue: list[int] = []
            for f in appendage_faces:
                counts: dict[int, int] = {}
                for nb in adjacency[f]:
                    nb_pid = current[nb]
                    if nb_pid != pid:
                        counts[nb_pid] = counts.get(nb_pid, 0) + 1
                if counts:
                    reassignments[f] = max(counts, key=counts.__getitem__)
                    boundary_queue.append(f)

            while boundary_queue:
                f = boundary_queue.pop()
                for nb in adjacency[f]:
                    if nb in appendage_faces and nb not in reassignments:
                        reassignments[nb] = reassignments[f]
                        boundary_queue.append(nb)

        if not reassignments:
            break
        for f, new_pid in reassignments.items():
            current[f] = new_pid

        # Batch reassignment can disconnect plates; absorb minor components into
        # their neighbours from the boundary inward.
        repaired = True
        while repaired:
            repaired = False
            pf: dict[int, set] = {}
            for i, pid in enumerate(current):
                pf.setdefault(pid, set()).add(i)
            for pid, faces in pf.items():
                comps: list[set] = []
                unvisited = set(faces)
                while unvisited:
                    start = next(iter(unvisited))
                    comp: set[int] = set()
                    stk = [start]
                    while stk:
                        node = stk.pop()
                        if node in comp:
                            continue
                        comp.add(node)
                        for nb in adjacency[node]:
                            if nb in unvisited:
                                stk.append(nb)
                    unvisited -= comp
                    comps.append(comp)
                if len(comps) <= 1:
                    continue
                repaired = True
                main_comp = max(comps, key=len)
                for comp in comps:
                    if comp is main_comp:
                        continue
                    island = set(comp)
                    queue = [
                        f for f in island
                        if any(current[nb] != pid for nb in adjacency[f])
                    ]
                    while island:
                        nxt: list[int] = []
                        for f in queue:
                            if f not in island:
                                continue
                            counts: dict[int, int] = {}
                            for nb in adjacency[f]:
                                nb_pid = current[nb]
                                if nb_pid != pid:
                                    counts[nb_pid] = counts.get(nb_pid, 0) + 1
                            if counts:
                                current[f] = max(counts, key=counts.__getitem__)
                                island.discard(f)
                                for nb in adjacency[f]:
                                    if nb in island:
                                        nxt.append(nb)
                        queue = nxt
                        if not queue and island:
                            queue = list(island)
                break  # pf is stale; restart connectivity scan

    return current


def classify_plate_boundaries(
    mesh: PolygonMesh,
    plate_ids: List[int],
    plate_omegas: np.ndarray,
) -> List[dict]:
    """Classify each face by its mean normal vs shear relative velocity over inter-plate edges.

    Returns per-face dicts:
        boundary_type: 'interior' | 'convergent' | 'divergent' | 'transform'
        normal_avg: float — mean normal velocity (positive = diverging); 0 if interior.
        neighbor_plate_types: set[str] — plate types seen across inter-plate edges.
    """
    adjacency = mesh.face_adjacency
    face_centroids = mesh.face_centroids
    n_faces = len(mesh.faces)
    result: List[dict] = [
        {"boundary_type": "interior", "normal_avg": 0.0, "neighbor_plate_types": set()}
        for _ in range(n_faces)
    ]

    for face_idx in range(n_faces):
        pid_a = plate_ids[face_idx]
        normal_sum = 0.0
        shear_sum = 0.0
        n_boundary = 0
        neighbor_types: set[str] = set()

        for nb in adjacency[face_idx]:
            pid_b = plate_ids[nb]
            if pid_b == pid_a:
                continue

            nb_type = mesh.face_metadata[nb].get("plate_type", "")
            if nb_type:
                neighbor_types.add(nb_type)

            m = face_centroids[face_idx] + face_centroids[nb]
            m_norm = float(np.linalg.norm(m))
            if m_norm < 1e-10:
                continue
            m = m / m_norm

            v_rel = np.cross(plate_omegas[pid_a], m) - np.cross(plate_omegas[pid_b], m)

            # Normal: neighbour → this face, tangent at m. Positive v_n = diverging.
            d = face_centroids[face_idx] - face_centroids[nb]
            d = d - np.dot(d, m) * m
            d_norm = float(np.linalg.norm(d))
            if d_norm < 1e-10:
                continue
            d = d / d_norm

            v_n = float(np.dot(v_rel, d))
            v_s = float(np.linalg.norm(v_rel - v_n * d))
            normal_sum += v_n
            shear_sum += v_s
            n_boundary += 1

        if n_boundary == 0:
            continue

        normal_avg = normal_sum / n_boundary
        shear_avg = shear_sum / n_boundary
        if abs(normal_avg) >= shear_avg:
            btype = "divergent" if normal_avg > 0 else "convergent"
        else:
            btype = "transform"

        result[face_idx] = {
            "boundary_type": btype,
            "normal_avg": normal_avg,
            "neighbor_plate_types": neighbor_types,
        }

    return result


def assign_plate_types(
    n_plates: int,
    oceanic_fraction: float = 0.7,
    seed: Optional[int] = None,
) -> List[str]:
    """Classify each plate as 'oceanic' or 'continental' (expected oceanic share = oceanic_fraction)."""
    rng = random.Random(seed)
    return [
        "oceanic" if rng.random() < oceanic_fraction else "continental"
        for _ in range(n_plates)
    ]
