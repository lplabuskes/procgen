from __future__ import annotations

import math
from collections import deque
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


# Kernel support of rasterize_face_values, in mean face-centroid spacings.  At
# 2.0 a face's centre keeps ~1/2 of its own value and takes the rest from its six
# neighbours — about as smooth as the old vertex-averaged raster (1/3), but
# without its creases.
RASTER_SUPPORT_SPACINGS = 2.0


def _wendland_c2(r: np.ndarray) -> np.ndarray:
    """Wendland's C² compact kernel on ``r`` = distance / support (0 for r ≥ 1)."""
    s = np.clip(1.0 - r, 0.0, None)
    return s ** 4 * (4.0 * r + 1.0)


def rasterize_face_values(
    mesh: PolygonMesh,
    face_values: np.ndarray,
    tex_size: int,
    support_spacings: float = RASTER_SUPPORT_SPACINGS,
) -> np.ndarray:
    """Rasterize per-face scalars to per-pixel textures as a smooth field over the sphere.

    Every texel takes the kernel-weighted mean of the face values whose centroids
    lie within ``support_spacings`` mean centroid spacings of its position
    (Wendland C², normalised).  The result is a function of position alone, so
    adjacent faces' textures — seams and over-hangs included — agree wherever
    they cover the same ground, and its gradient is continuous: no creases at
    face edges, unlike piecewise-linear interpolation over the polygons.
    Returns float32 (N, tex_size, tex_size, K); K=1 when face_values is 1-D.
    """
    from scipy.spatial import KDTree

    values = np.asarray(face_values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    N, K = values.shape

    centroids = np.asarray(mesh.face_centroids, dtype=np.float64)
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    # Hex lattice of area A: centre-to-centre spacing sqrt(2A / sqrt 3).
    spacing = math.sqrt(2.0 * float(np.mean(mesh.face_areas)) / math.sqrt(3.0))
    support = support_spacings * spacing
    # Enough neighbours to hold every centroid inside the support (π s² / A, ×2 margin).
    k = int(min(N, max(8, math.ceil(2.0 * math.pi * support_spacings ** 2 * 2.0 / math.sqrt(3.0)) + 1)))

    tree = KDTree(centroids)
    positions = mesh.face_texture_positions(tex_size, tex_size).reshape(-1, 3).astype(np.float64)
    out = np.empty((positions.shape[0], K), dtype=np.float64)
    chunk = 1 << 18
    for start in range(0, positions.shape[0], chunk):
        pts = positions[start:start + chunk]
        dist, idx = tree.query(pts, k=k, workers=-1)
        w = _wendland_c2(dist / support)
        total = w.sum(axis=1)
        # A texel farther than the support from every centroid (a badly
        # stretched face) falls back to its nearest face.
        bare = total <= 1e-12
        w[bare, 0] = 1.0
        total[bare] = 1.0
        out[start:start + chunk] = np.einsum("pk,pkc->pc", w, values[idx]) / total[:, None]

    return out.reshape(N, tex_size, tex_size, K).astype(np.float32)


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
