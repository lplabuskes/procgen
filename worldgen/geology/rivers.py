from __future__ import annotations

import numpy as np

from worldgen.mesh.mymesh import PolygonMesh


def compute_river_flow(
    mesh: PolygonMesh,
    elevations: np.ndarray,
    fill_iterations: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute D8 flow direction and flow accumulation on the face graph.

    Sinks (local elevation minima) are filled by iteratively raising each face
    to its lowest neighbour's elevation before routing is computed.

    Args:
        mesh: The polygon mesh.
        elevations: Per-face elevation in km, shape (n_faces,).
        fill_iterations: Maximum sink-fill passes (stops early if stable).

    Returns:
        flow: float64 (n_faces,) upstream face count per face.
        flow_to: int32 (n_faces,) downstream face index, -1 for sinks.
    """
    n = len(mesh.faces)
    adjacency = mesh.face_adjacency
    elev = elevations.copy()

    # Fill sinks: process low-to-high; raise any face below all its neighbours.
    for _ in range(fill_iterations):
        changed = False
        for i in np.argsort(elev):
            nbs = adjacency[i]
            if not nbs:
                continue
            min_nb = min(elev[nb] for nb in nbs)
            if elev[i] < min_nb:
                elev[i] = min_nb
                changed = True
        if not changed:
            break

    # Flow direction: each face routes to its steepest descent neighbour.
    flow_to = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        nbs = adjacency[i]
        if not nbs:
            continue
        lowest = min(nbs, key=lambda nb: elev[nb])
        if elev[lowest] < elev[i]:
            flow_to[i] = lowest

    # Flow accumulation: pass upstream flow to each face's downstream target.
    flow = np.ones(n, dtype=np.float64)
    for i in np.argsort(elev)[::-1]:
        dst = int(flow_to[i])
        if dst >= 0:
            flow[dst] += flow[i]

    return flow, flow_to


def extract_river_paths(
    flow: np.ndarray,
    flow_to: np.ndarray,
    threshold: float = 100.0,
    elevations: np.ndarray | None = None,
) -> list[list[int]]:
    """Extract river centre-lines as sequences of face indices.

    Each path starts at a river source (a river face with no upstream river
    face routing into it) and follows flow_to until leaving the river set or
    reaching a sink.

    Args:
        flow: Flow accumulation per face (output of compute_river_flow).
        flow_to: Downstream face index per face, -1 for sinks.
        threshold: Minimum flow accumulation for a face to be a river face.
        elevations: Optional per-face elevations (km). When provided, faces
            with elevation < 0 (ocean) are excluded from the river set so
            paths stay on land.

    Returns:
        List of paths; each is a list of face indices from source to mouth.
        Confluences cause downstream segments to appear in multiple paths.
    """
    river_set = set(int(i) for i in np.where(flow >= threshold)[0])
    if elevations is not None:
        river_set = {i for i in river_set if elevations[i] >= 0.0}

    has_river_upstream: set[int] = set()
    for i in river_set:
        dst = int(flow_to[i])
        if dst in river_set:
            has_river_upstream.add(dst)

    sources = river_set - has_river_upstream

    paths: list[list[int]] = []
    for src in sorted(sources):
        path = [src]
        cur = src
        while True:
            dst = int(flow_to[cur])
            if dst < 0 or dst not in river_set:
                break
            path.append(dst)
            cur = dst
        if len(path) > 1:
            paths.append(path)

    return paths
