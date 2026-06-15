from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import List, TYPE_CHECKING

import numpy as np

from utils.simplex import fbm_sum, make_perm
from core.mesh import PolygonMesh

if TYPE_CHECKING:
    from worldgen.geology.elevation_builder import FaceErosionParams


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
    """Compute per-face elevation (km) from plate type and boundary effects.

    Boundary deltas are propagated outward via BFS with exponential decay over hop distance.
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
    params: "FaceErosionParams",
) -> np.ndarray:
    """Apply hydraulic then thermal erosion to face-level elevations.

    Hydraulic pass: fill sinks → accumulate flow downhill → erode high faces and deposit at sinks.
    Thermal pass: transfer elevation across edges exceeding the talus angle (gravity slumping).
    """
    hydraulic_iterations = params.hydraulic_iterations
    hydraulic_erosion_rate = params.hydraulic_erosion_rate
    hydraulic_deposition_rate = params.hydraulic_deposition_rate
    thermal_iterations = params.thermal_iterations
    thermal_talus = params.thermal_talus
    thermal_erosion_rate = params.thermal_erosion_rate

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

    Band-limited to the pixel grid: octaves above nyquist_safety/max_pixel_spacing are
    skipped to prevent aliasing across face UV boundaries (which appears as seams).
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
    amp, freq, total_amp = 1.0, float(base_frequency), 0.0
    freqs, amps = [], []
    for _ in range(octaves):
        if freq > max_freq:
            break
        freqs.append(freq)
        amps.append(amp)
        total_amp += amp
        amp *= persistence
        freq *= lacunarity
    noise_sum = fbm_sum(flat_xyz, perm, np.array(freqs), np.array(amps))

    if total_amp == 0.0:
        total_amp = 1.0  # all octaves above Nyquist — output is flat zero.

    noise_norm = (noise_sum / total_amp).reshape(n_faces, tex_size, tex_size)

    noise_delta = (noise_norm * amplitude).astype(np.float32)

    if per_pixel_elevations is not None:
        noise_delta *= np.exp(-np.abs(per_pixel_elevations) / coast_sigma)

    result = np.zeros((n_faces, tex_size, tex_size, 1), dtype=np.float32)
    result[:, :, :, 0] = noise_delta
    return result


def face_elevations_from_texture(elevation_textures: np.ndarray) -> np.ndarray:
    return elevation_textures[:, :, :, 0].mean(axis=(1, 2)).astype(np.float64)


def rasterize_face_values(
    mesh: PolygonMesh,
    face_values: np.ndarray,
    tex_size: int,
) -> np.ndarray:
    """Rasterize per-face scalars to per-pixel textures via vertex averaging + barycentric interp.

    Adjacent faces share vertex values, so rasterized values agree continuously across boundaries.
    Returns float32 (N, tex_size, tex_size, K); K=1 when face_values is 1-D.
    """
    values = np.asarray(face_values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    N, K = values.shape
    n_verts = len(mesh.vertices)

    # Scatter-add face values onto shared vertices, then average.
    face_sizes = np.array([len(f) for f in mesh.faces])
    all_verts = np.concatenate([np.asarray(f) for f in mesh.faces])
    all_face_idx = np.repeat(np.arange(N), face_sizes)
    vertex_sum = np.zeros((n_verts, K), dtype=np.float64)
    vertex_count = np.zeros(n_verts, dtype=np.float64)
    np.add.at(vertex_sum, all_verts, values[all_face_idx])
    np.add.at(vertex_count, all_verts, 1.0)
    vertex_vals = vertex_sum / np.maximum(vertex_count[:, None], 1.0)  # (n_verts, K)

    # Pixel UV grid in [0, 1]^2 — identical for every face.
    u_coords = np.linspace(0.0, 1.0, tex_size, dtype=np.float64)
    v_coords = np.linspace(0.0, 1.0, tex_size, dtype=np.float64)
    uu, vv = np.meshgrid(u_coords, v_coords)
    pixel_uv = np.stack([uu.ravel(), vv.ravel()], axis=1)  # (P, 2)
    n_pix = pixel_uv.shape[0]

    textures = np.zeros((N, tex_size, tex_size, K), dtype=np.float32)

    # Group faces by degree so every batch shares the same fan-triangle layout,
    # enabling fully vectorised barycentric solves.
    degree_groups: dict[int, list[int]] = defaultdict(list)
    for i, face in enumerate(mesh.faces):
        degree_groups[len(face)].append(i)

    for degree, fidx_list in degree_groups.items():
        fidx = np.asarray(fidx_list)
        B = len(fidx)

        verts_uv = np.stack([
            mesh._face_vertex_uvs[i].astype(np.float64) for i in fidx_list
        ])                                                               # (B, degree, 2)
        batch_verts = np.array([mesh.faces[i] for i in fidx_list])      # (B, degree)
        vert_v = vertex_vals[batch_verts]                                # (B, degree, K)

        v0_uv = verts_uv[:, 0, :]                                        # (B, 2)
        e1_uv = verts_uv[:, 1:-1, :] - v0_uv[:, np.newaxis, :]          # (B, n_tris, 2)
        e2_uv = verts_uv[:, 2:, :] - v0_uv[:, np.newaxis, :]            # (B, n_tris, 2)
        det = (
            e1_uv[:, :, 0] * e2_uv[:, :, 1]
            - e1_uv[:, :, 1] * e2_uv[:, :, 0]
        )                                                                # (B, n_tris)
        det = np.where(np.abs(det) < 1e-12, 1.0, det)

        pv = pixel_uv[np.newaxis, :, :] - v0_uv[:, np.newaxis, :]       # (B, P, 2)
        beta = (
            pv[:, :, 0:1] * e2_uv[:, np.newaxis, :, 1]
            - pv[:, :, 1:2] * e2_uv[:, np.newaxis, :, 0]
        ) / det[:, np.newaxis, :]                                        # (B, P, n_tris)
        gamma = (
            e1_uv[:, np.newaxis, :, 0] * pv[:, :, 1:2]
            - e1_uv[:, np.newaxis, :, 1] * pv[:, :, 0:1]
        ) / det[:, np.newaxis, :]

        best_tri = np.argmax(
            np.minimum(1.0 - beta - gamma, np.minimum(beta, gamma)), axis=2
        )                                                                # (B, P)

        bi = np.arange(B)[:, np.newaxis]
        pi = np.arange(n_pix)[np.newaxis, :]
        b_sel = beta[bi, pi, best_tri]                                   # (B, P)
        g_sel = gamma[bi, pi, best_tri]

        vals = (
            (1.0 - b_sel - g_sel)[:, :, None] * vert_v[:, 0:1, :]
            + b_sel[:, :, None] * vert_v[bi, best_tri + 1, :]
            + g_sel[:, :, None] * vert_v[bi, best_tri + 2, :]
        )                                                                # (B, P, K)

        textures[fidx] = vals.reshape(B, tex_size, tex_size, K).astype(np.float32)

    return textures


def smooth_texture_seams(
    mesh: PolygonMesh,
    textures: np.ndarray,
    n_passes: int = 1,
) -> np.ndarray:
    """Average matched boundary pixel pairs across face edges to close texture seams."""
    tex_size = textures.shape[1]
    fa, fb, ra, ca, rb, cb = mesh.boundary_pixel_pairs(tex_size)
    if fa.size == 0:
        return textures

    a_idx = (fa, ra, ca)
    b_idx = (fb, rb, cb)
    ones = np.ones(fa.size, dtype=np.int32)
    for _ in range(n_passes):
        partner_sum = np.zeros_like(textures)
        partner_count = np.zeros(textures.shape[:3], dtype=np.int32)
        # Each pixel collects its partner's value (both directions of the pair).
        np.add.at(partner_sum, a_idx, textures[b_idx])
        np.add.at(partner_sum, b_idx, textures[a_idx])
        np.add.at(partner_count, a_idx, ones)
        np.add.at(partner_count, b_idx, ones)
        mask = partner_count > 0
        cnt = (1 + partner_count[mask]).astype(np.float32)[:, None]
        textures[mask] = (textures[mask] + partner_sum[mask]) / cnt

    return textures


def compute_elevation_textures(
    mesh: PolygonMesh,
    elevations: np.ndarray,
    tex_size: int = 32,
) -> np.ndarray:
    """Rasterize per-face elevations; returns float32 (N, tex_size, tex_size, 1) in km."""
    return rasterize_face_values(mesh, elevations, tex_size)


def smooth_elevation_seams(
    mesh: PolygonMesh,
    elevation_textures: np.ndarray,
    n_passes: int = 1,
) -> np.ndarray:
    return smooth_texture_seams(mesh, elevation_textures, n_passes)


def compute_ocean_mask_texture(
    mesh: PolygonMesh,
    elevation_textures: np.ndarray,
    sea_level: float = 0.0,
    seed_level: float = -0.5,
) -> np.ndarray:
    """Pixel-level ocean flood fill using exact cross-face pixel pairs.

    Phase 1: faces with max elevation ≤ seed_level seed the BFS.
    Phase 2: face BFS expands to adjacent faces with max elevation ≤ sea_level.
    Phase 3: pixel BFS propagates 4-connected intra-face and cross-face via matched pairs.
    Isolated sub-sea-level depressions unreachable from any ocean seed remain land.
    """
    N, H, W, _ = elevation_textures.shape
    elevs = elevation_textures[:, :, :, 0]    # (N, H, W)
    elev_max = elevs.max(axis=(1, 2))          # (N,)

    # ---- Phase 1: seed faces ------------------------------------------------
    is_core = elev_max <= seed_level           # unambiguously ocean

    # ---- Phase 2: face BFS --------------------------------------------------
    face_queue: deque[int] = deque(int(i) for i in np.where(is_core)[0])
    while face_queue:
        fi = face_queue.popleft()
        for fj in mesh.face_adjacency[fi]:
            if not is_core[fj] and elev_max[fj] <= sea_level:
                is_core[fj] = True
                face_queue.append(fj)

    # Build cross-face pixel-pair adjacency, indexed by flat pixel id.
    cross_adj: List[List[int]] = [[] for _ in range(N * H * W)]
    fa, fb, ra, ca, rb, cb = mesh.boundary_pixel_pairs(H)
    flat_a = (fa.astype(np.int64) * H + ra) * W + ca
    flat_b = (fb.astype(np.int64) * H + rb) * W + cb
    for ia, ib in zip(flat_a.tolist(), flat_b.tolist()):
        cross_adj[ia].append(ib)
        cross_adj[ib].append(ia)

    ocean_mask = np.zeros((N, H, W), dtype=bool)
    pixel_queue: deque[int] = deque()

    # ---- Phase 3: pixel BFS -------------------------------------------------
    # Seed: all pixels of core ocean faces.
    for i in np.where(is_core)[0]:
        ocean_mask[i] = True
        base = i * H * W
        pixel_queue.extend(range(base, base + H * W))

    # BFS with intra- and cross-face propagation over a flat pixel index.
    elevs_flat = elevs.reshape(-1)
    mask_flat = ocean_mask.reshape(-1)
    HW = H * W
    while pixel_queue:
        pid = pixel_queue.popleft()
        fi, rem = divmod(pid, HW)
        r, c = divmod(rem, W)

        # Intra-face: 4-connected neighbours at or below sea level.
        if r > 0:
            n = pid - W
            if not mask_flat[n] and elevs_flat[n] <= sea_level:
                mask_flat[n] = True
                pixel_queue.append(n)
        if r + 1 < H:
            n = pid + W
            if not mask_flat[n] and elevs_flat[n] <= sea_level:
                mask_flat[n] = True
                pixel_queue.append(n)
        if c > 0:
            n = pid - 1
            if not mask_flat[n] and elevs_flat[n] <= sea_level:
                mask_flat[n] = True
                pixel_queue.append(n)
        if c + 1 < W:
            n = pid + 1
            if not mask_flat[n] and elevs_flat[n] <= sea_level:
                mask_flat[n] = True
                pixel_queue.append(n)

        # Cross-face: matched partner pixels at or below sea level.
        for n in cross_adj[pid]:
            if not mask_flat[n] and elevs_flat[n] <= sea_level:
                mask_flat[n] = True
                pixel_queue.append(n)

    return ocean_mask[:, :, :, np.newaxis].astype(np.uint8)
