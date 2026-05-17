import math
from collections import defaultdict, deque
from typing import List

import heapq

import numpy as np

from worldgen.geology.simplex import make_perm, simplex3_vec
from worldgen.mesh.mymesh import PolygonMesh


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


def compute_erosion(
    mesh: PolygonMesh,
    elevations: np.ndarray,
    hydraulic_iterations: int = 5,
    hydraulic_erosion_rate: float = 0.4,
    hydraulic_deposition_rate: float = 0.3,
    thermal_iterations: int = 10,
    thermal_talus: float = 0.5,
    thermal_erosion_rate: float = 0.5,
) -> np.ndarray:
    """Apply hydraulic then thermal erosion to face-level elevations.

    Hydraulic pass: iteratively fills local sinks (priority-queue fill to lowest
    outflow neighbor), then erodes high faces proportional to flow volume received
    from uphill neighbors, depositing sediment on the receiving face.

    Thermal pass: for each face, if the elevation difference to a neighbor exceeds
    `thermal_talus` (km), transfer `thermal_erosion_rate` of the excess to that
    neighbor (gravity-driven slumping).

    Args:
        mesh: The dual polygon mesh.
        elevations: Per-face elevations in km, shape (n_faces,).
        hydraulic_iterations: Number of full hydraulic fill+erode cycles.
        hydraulic_erosion_rate: Fraction of elevation removed per unit flow at eroding faces.
        hydraulic_deposition_rate: Fraction of transported sediment deposited per step.
        thermal_iterations: Number of thermal erosion sweeps.
        thermal_talus: Slope threshold (km) above which material slides.
        thermal_erosion_rate: Fraction of excess slope transferred per sweep.

    Returns:
        float64 array of shape (n_faces,) with eroded elevations.
    """
    elev = elevations.copy()
    adjacency = mesh.face_adjacency
    n_faces = len(mesh.faces)

    # --- Hydraulic erosion ---
    for _ in range(hydraulic_iterations):
        # Priority-queue fill: raise sinks to the level of their lowest outflow neighbor.
        # Process faces from lowest to highest; any face whose all neighbors are higher
        # gets filled to the min neighbor elevation.
        order = np.argsort(elev)
        for i in order:
            neighbors = adjacency[i]
            if not neighbors:
                continue
            nb_elevs = [elev[nb] for nb in neighbors]
            min_nb = min(nb_elevs)
            if elev[i] < min_nb:
                elev[i] = min_nb

        # Flow accumulation: route water downhill, accumulate volume at each face.
        flow = np.ones(n_faces, dtype=np.float64)
        order = np.argsort(elev)[::-1]  # highest first
        for i in order:
            neighbors = adjacency[i]
            lower = [nb for nb in neighbors if elev[nb] < elev[i]]
            if not lower:
                continue
            # Distribute flow proportionally by elevation drop.
            drops = np.array([elev[i] - elev[nb] for nb in lower], dtype=np.float64)
            drops /= drops.sum()
            for nb, frac in zip(lower, drops):
                flow[nb] += flow[i] * frac

        # Erode high faces, deposit on receiving faces.
        sediment = np.zeros(n_faces, dtype=np.float64)
        for i in order:
            neighbors = adjacency[i]
            lower = [nb for nb in neighbors if elev[nb] < elev[i]]
            if not lower:
                # Deposition at sinks
                elev[i] += sediment[i] * hydraulic_deposition_rate
                continue
            erosion = hydraulic_erosion_rate * math.log1p(flow[i]) * 0.01
            elev[i] -= erosion
            drops = np.array([elev[i] - elev[nb] for nb in lower if elev[nb] < elev[i]], dtype=np.float64)
            lower = [nb for nb in lower if elev[nb] < elev[i]]
            if len(drops) == 0:
                elev[i] += sediment[i] * hydraulic_deposition_rate
                continue
            drops /= drops.sum()
            for nb, frac in zip(lower, drops):
                sediment[nb] += (erosion + sediment[i]) * frac * (1.0 - hydraulic_deposition_rate)
            elev[i] += sediment[i] * hydraulic_deposition_rate

    # --- Thermal erosion ---
    for _ in range(thermal_iterations):
        # Single sweep; iterate faces in random-ish order (by elevation, high to low).
        order = np.argsort(elev)[::-1]
        for i in order:
            for nb in adjacency[i]:
                diff = elev[i] - elev[nb]
                if diff > thermal_talus:
                    transfer = thermal_erosion_rate * (diff - thermal_talus) * 0.5
                    elev[i] -= transfer
                    elev[nb] += transfer

    return elev


def compute_pixel_erosion(
    elevation_textures: np.ndarray,
    thermal_iterations: int = 5,
    thermal_talus: float = 0.05,
    thermal_erosion_rate: float = 0.5,
) -> np.ndarray:
    """Apply pixel-level thermal erosion to elevation textures.

    Smooths noise-induced sharp slopes within each face's texture by
    transferring material wherever adjacent-pixel elevation differences exceed
    `thermal_talus`. Vectorized across all faces simultaneously.

    Args:
        elevation_textures: float32 (N, H, W, 4); channel 0 is elevation (km).
        thermal_iterations: Number of full erosion sweeps.
        thermal_talus: Minimum pixel-to-pixel elevation difference (km) that
            triggers material transfer.
        thermal_erosion_rate: Fraction of excess slope transferred per sweep.

    Returns:
        Modified copy of elevation_textures (channel 0 updated; others unchanged).
    """
    result = elevation_textures.copy()
    elev = result[:, :, :, 0].astype(np.float64)  # (N, H, W)

    for _ in range(thermal_iterations):
        # W-axis: right neighbours
        diff = elev[:, :, :-1] - elev[:, :, 1:]
        transfer = np.where(diff > thermal_talus, thermal_erosion_rate * (diff - thermal_talus) * 0.5, 0.0)
        elev[:, :, :-1] -= transfer
        elev[:, :, 1:] += transfer

        diff = elev[:, :, 1:] - elev[:, :, :-1]
        transfer = np.where(diff > thermal_talus, thermal_erosion_rate * (diff - thermal_talus) * 0.5, 0.0)
        elev[:, :, 1:] -= transfer
        elev[:, :, :-1] += transfer

        # H-axis: lower neighbours
        diff = elev[:, :-1, :] - elev[:, 1:, :]
        transfer = np.where(diff > thermal_talus, thermal_erosion_rate * (diff - thermal_talus) * 0.5, 0.0)
        elev[:, :-1, :] -= transfer
        elev[:, 1:, :] += transfer

        diff = elev[:, 1:, :] - elev[:, :-1, :]
        transfer = np.where(diff > thermal_talus, thermal_erosion_rate * (diff - thermal_talus) * 0.5, 0.0)
        elev[:, 1:, :] -= transfer
        elev[:, :-1, :] += transfer

    result[:, :, :, 0] = elev.astype(np.float32)
    return result


def compute_noise_elevation_deltas(
    mesh: PolygonMesh,
    tex_size: int = 32,
    amplitude: float = 1.5,
    base_frequency: float = 1.5,
    octaves: int = 6,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
    seed: int = 0,
    nyquist_safety: float = 0.5,
    per_pixel_elevations: np.ndarray | None = None,
    coast_sigma: float = 1.5,
) -> np.ndarray:
    """Per-pixel fBm simplex noise at uniform amplitude across all faces.

    Fully vectorized — evaluates all pixels in one numpy pass per octave.

    The fBm sum is band-limited to the pixel grid: octaves whose spatial
    frequency exceeds `nyquist_safety / max_pixel_spacing` (cycles per unit)
    are skipped, where `max_pixel_spacing` is the largest adjacent-pixel 3D
    distance across the whole mesh. Without this, high-frequency octaves
    alias against the per-face pixel grids, and because adjacent faces sample
    the same continuous noise field on slightly different grids, bilinear
    reconstruction near a shared edge gives different values on each side —
    a visible seam. `nyquist_safety=0.5` keeps the highest octave at no more
    than ~4 samples per cycle.

    Returns:
        float32 array of shape (N_faces, tex_size, tex_size, 4).
        R channel = noise delta in km; G, B, A = 0.
    """
    n_faces = len(mesh.faces)
    positions = mesh.face_texture_positions(tex_size, tex_size)  # (N, H, W, 3)

    # Empirical pixel spacing in 3D: largest adjacent-pixel step across the mesh.
    dx = np.linalg.norm(np.diff(positions, axis=2), axis=-1)
    dy = np.linalg.norm(np.diff(positions, axis=1), axis=-1)
    max_pixel_spacing = float(max(dx.max(), dy.max()))
    max_freq = nyquist_safety / max_pixel_spacing if max_pixel_spacing > 0 else float("inf")

    flat_xyz = positions.reshape(-1, 3).astype(np.float64)       # (N*H*W, 3)

    perm = make_perm(seed)
    noise_sum = np.zeros(len(flat_xyz), dtype=np.float64)
    amp, freq, total_amp = 1.0, float(base_frequency), 0.0
    for _ in range(octaves):
        if freq > max_freq:
            break
        noise_sum += amp * simplex3_vec(flat_xyz * freq, perm)
        total_amp += amp
        amp *= persistence
        freq *= lacunarity

    if total_amp == 0.0:
        total_amp = 1.0  # all octaves above Nyquist — output is flat zero.

    noise_norm = (noise_sum / total_amp).reshape(n_faces, tex_size, tex_size)

    noise_delta = (noise_norm * amplitude).astype(np.float32)

    if per_pixel_elevations is not None:
        noise_delta *= np.exp(-np.abs(per_pixel_elevations) / coast_sigma)

    result = np.zeros((n_faces, tex_size, tex_size, 4), dtype=np.float32)
    result[:, :, :, 0] = noise_delta
    return result


def compute_elevation_textures(
    mesh: PolygonMesh,
    elevations: np.ndarray,
    tex_size: int = 32,
) -> np.ndarray:
    """Per-face elevation textures for smooth cross-face gradients.

    Each pixel stores the interpolated elevation in km, blended from the mean
    elevation of vertices shared with neighboring faces.

    Returns:
        float32 array of shape (N_faces, tex_size, tex_size, 4).
        R channel = elevation in km; G, B, A = 0.
    """
    n_faces = len(mesh.faces)
    n_verts = len(mesh.vertices)
    textures = np.zeros((n_faces, tex_size, tex_size, 4), dtype=np.float32)

    # Scatter-add to compute mean face elevation at each vertex (no Python loop).
    face_sizes = np.array([len(f) for f in mesh.faces])
    all_verts = np.concatenate([np.asarray(f) for f in mesh.faces])
    all_face_idx = np.repeat(np.arange(n_faces), face_sizes)
    vertex_sum = np.zeros(n_verts, dtype=np.float64)
    vertex_count = np.zeros(n_verts, dtype=np.float64)
    np.add.at(vertex_sum, all_verts, elevations[all_face_idx])
    np.add.at(vertex_count, all_verts, 1.0)
    vertex_elevations = vertex_sum / np.maximum(vertex_count, 1.0)

    # Pixel UV grid in [0, 1]^2 — identical for every face.
    u_coords = np.linspace(0.0, 1.0, tex_size, dtype=np.float64)
    v_coords = np.linspace(0.0, 1.0, tex_size, dtype=np.float64)
    uu, vv = np.meshgrid(u_coords, v_coords)
    pixel_uv = np.stack([uu.ravel(), vv.ravel()], axis=1)  # (P, 2)
    n_pix = pixel_uv.shape[0]

    # Group faces by degree so every batch has the same fan-triangle structure,
    # enabling fully vectorized barycentric solves across the batch.
    # For an icosphere dual this is just 2 groups: ~12 pentagons + N hexagons.
    degree_groups: dict[int, list[int]] = defaultdict(list)
    for i, face in enumerate(mesh.faces):
        degree_groups[len(face)].append(i)

    for degree, fidx_list in degree_groups.items():
        fidx = np.asarray(fidx_list)
        B = len(fidx)

        # (B, degree, 2) vertex UVs; (B, degree) elevation deltas from face center.
        verts_uv = np.stack([
            mesh._face_vertex_uvs[i].astype(np.float64) for i in fidx_list
        ])
        batch_verts = np.array([mesh.faces[i] for i in fidx_list])  # (B, degree)
        vert_elev = vertex_elevations[batch_verts]  # (B, degree)

        # Fan-triangle edge vectors from apex (vertex 0).
        v0_uv = verts_uv[:, 0, :]                                    # (B, 2)
        e1_uv = verts_uv[:, 1:-1, :] - v0_uv[:, np.newaxis, :]      # (B, n_tris, 2)
        e2_uv = verts_uv[:, 2:, :] - v0_uv[:, np.newaxis, :]        # (B, n_tris, 2)
        det = (
            e1_uv[:, :, 0] * e2_uv[:, :, 1]
            - e1_uv[:, :, 1] * e2_uv[:, :, 0]
        )  # (B, n_tris)
        det = np.where(np.abs(det) < 1e-12, 1.0, det)

        # Barycentric coordinates for all pixels across all triangles: (B, P, n_tris).
        pv = pixel_uv[np.newaxis, :, :] - v0_uv[:, np.newaxis, :]   # (B, P, 2)
        beta = (
            pv[:, :, 0:1] * e2_uv[:, np.newaxis, :, 1]
            - pv[:, :, 1:2] * e2_uv[:, np.newaxis, :, 0]
        ) / det[:, np.newaxis, :]
        gamma = (
            e1_uv[:, np.newaxis, :, 0] * pv[:, :, 1:2]
            - e1_uv[:, np.newaxis, :, 1] * pv[:, :, 0:1]
        ) / det[:, np.newaxis, :]

        best_tri = np.argmax(
            np.minimum(1.0 - beta - gamma, np.minimum(beta, gamma)), axis=2
        )  # (B, P)

        bi = np.arange(B)[:, np.newaxis]    # (B, 1)
        pi = np.arange(n_pix)[np.newaxis, :]  # (1, P)
        b_sel = beta[bi, pi, best_tri]      # (B, P)
        g_sel = gamma[bi, pi, best_tri]     # (B, P)

        delta_vals = (
            (1.0 - b_sel - g_sel) * vert_elev[:, 0:1]
            + b_sel * vert_elev[bi, best_tri + 1]
            + g_sel * vert_elev[bi, best_tri + 2]
        )  # (B, P)

        textures[fidx, :, :, 0] = delta_vals.reshape(B, tex_size, tex_size).astype(np.float32)

    return textures
