import math
from collections import deque
from typing import List

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
    face_elevations: np.ndarray | None = None,
    elevation_deltas: np.ndarray | None = None,
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

    if face_elevations is not None:
        per_pixel_elev = face_elevations[:, np.newaxis, np.newaxis].astype(np.float32)
        if elevation_deltas is not None:
            per_pixel_elev = per_pixel_elev + elevation_deltas
        noise_delta *= np.exp(-np.abs(per_pixel_elev) / coast_sigma)

    result = np.zeros((n_faces, tex_size, tex_size, 4), dtype=np.float32)
    result[:, :, :, 0] = noise_delta
    return result


def compute_elevation_delta_textures(
    mesh: PolygonMesh,
    elevations: np.ndarray,
    tex_size: int = 32,
) -> np.ndarray:
    """Per-face elevation delta textures for smooth cross-face gradients.

    Each pixel stores (elevation_at_point - face_center_elevation) in km.
    The face center has delta=0; edge pixels blend toward the mean elevation
    of faces sharing that vertex.

    Returns:
        float32 array of shape (N_faces, tex_size, tex_size, 4).
        R channel = delta in km; G, B, A = 0.
    """
    n_faces = len(mesh.faces)
    textures = np.zeros((n_faces, tex_size, tex_size, 4), dtype=np.float32)

    # vertex → list of face indices
    vertex_to_faces: dict[int, list[int]] = {}
    for face_idx, face in enumerate(mesh.faces):
        for v_idx in face:
            vertex_to_faces.setdefault(v_idx, []).append(face_idx)

    # vertex elevation = mean of adjacent face elevations
    n_verts = len(mesh.vertices)
    vertex_elevations = np.empty(n_verts, dtype=np.float64)
    for v_idx in range(n_verts):
        adj = vertex_to_faces.get(v_idx, [])
        vertex_elevations[v_idx] = elevations[adj].mean() if adj else 0.0

    # Barycentric interpolation within fan triangles — mirrors face_texture_positions.
    # At any polygon edge, the fan-apex barycentric weight is exactly 0, so the
    # interpolated elevation depends only on the two shared endpoint vertices, giving
    # the same absolute value from both adjacent faces (perfect edge continuity).
    u_coords = np.linspace(0.0, 1.0, tex_size, dtype=np.float64)
    v_coords = np.linspace(0.0, 1.0, tex_size, dtype=np.float64)
    uu, vv = np.meshgrid(u_coords, v_coords)
    pixel_uv = np.stack([uu.ravel(), vv.ravel()], axis=1)  # (H*W, 2)
    n_pix = pixel_uv.shape[0]
    pix_idx = np.arange(n_pix)

    for face_idx, face in enumerate(mesh.faces):
        face_arr = np.array(face)
        verts_uv = mesh._face_vertex_uvs[face_idx].astype(np.float64)  # (n_v, 2)
        vert_deltas = (vertex_elevations[face_arr] - float(elevations[face_idx]))  # (n_v,)

        v0_uv = verts_uv[0]
        e1_uv = verts_uv[1:-1] - v0_uv   # (n_tris, 2)
        e2_uv = verts_uv[2:] - v0_uv     # (n_tris, 2)

        det = e1_uv[:, 0] * e2_uv[:, 1] - e1_uv[:, 1] * e2_uv[:, 0]
        det = np.where(np.abs(det) < 1e-12, 1.0, det)

        pv = pixel_uv - v0_uv
        beta = (pv[:, 0:1] * e2_uv[None, :, 1] - pv[:, 1:2] * e2_uv[None, :, 0]) / det  # weight for face[i]
        gamma = (e1_uv[None, :, 0] * pv[:, 1:2] - e1_uv[None, :, 1] * pv[:, 0:1]) / det  # weight for face[i+1]
        alpha = 1.0 - beta - gamma  # weight for face[0]

        best_tri = np.argmax(np.minimum(alpha, np.minimum(beta, gamma)), axis=1)
        b_sel = beta[pix_idx, best_tri]
        g_sel = gamma[pix_idx, best_tri]

        delta_vals = (
            (1.0 - b_sel - g_sel) * vert_deltas[0]
            + b_sel * vert_deltas[1:-1][best_tri]
            + g_sel * vert_deltas[2:][best_tri]
        )
        textures[face_idx, :, :, 0] = delta_vals.reshape(tex_size, tex_size).astype(np.float32)

    return textures
