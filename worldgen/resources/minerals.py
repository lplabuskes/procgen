"""Mineral deposit placement — hybrid geology + noise model.

Geology gives each mineral type a candidate mask of land faces (convergent / divergent boundaries,
volcanism, sedimentary or evaporite basins); an independent fBm noise draw per type, sampled at face
centroids, makes candidates above the type's threshold deposits, with abundance the normalised excess.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from utils.simplex import fbm_sum, make_perm
from core.mesh import PolygonMesh
from worldgen.climate.channels import BIOME
from worldgen.geology.channels import (
    BOUNDARY_TYPE,
    ELEVATION_FACE,
    IS_OCEAN,
    PLATE_TYPE,
    VOLCANIC_ACTIVITY,
)
from worldgen.resources.terrain import face_hops_from_seeds

if TYPE_CHECKING:
    from worldgen.resources.builder import MineralParams


MINERAL_NAMES: list[str] = [
    "Iron",         # 0
    "Copper",       # 1
    "Gold",         # 2
    "Gems",         # 3
    "Coal",         # 4
    "Tin",          # 5
    "Rare Metals",  # 6
    "Salt",         # 7
    "Marble",       # 8
]
N_MINERALS = len(MINERAL_NAMES)

IRON, COPPER, GOLD, GEMS, COAL, TIN, RARE_METALS, SALT, MARBLE = range(N_MINERALS)


# RGBA palette — one colour per MINERAL_NAMES entry.
MINERAL_PALETTE = np.array([
    [0.55, 0.55, 0.58, 1.0],  # 0 Iron        — steel grey
    [0.75, 0.40, 0.10, 1.0],  # 1 Copper       — copper orange
    [0.95, 0.80, 0.10, 1.0],  # 2 Gold         — gold yellow
    [0.60, 0.10, 0.80, 1.0],  # 3 Gems         — gem purple
    [0.20, 0.15, 0.10, 1.0],  # 4 Coal         — dark charcoal
    [0.30, 0.75, 0.60, 1.0],  # 5 Tin          — pale teal
    [0.15, 0.60, 0.90, 1.0],  # 6 Rare Metals  — bright cyan-blue
    [0.96, 0.95, 0.88, 1.0],  # 7 Salt         — off-white
    [0.82, 0.76, 0.70, 1.0],  # 8 Marble       — warm cream stone
], dtype=np.float32)


def _fbm_noise(centroids: np.ndarray, frequency: float, seed: int, octaves: int = 3) -> np.ndarray:
    """(N,) float32 fBm simplex noise at unit-sphere centroids, normalized to [0, 1]."""
    perm = make_perm(seed % (2**31))
    xyz = centroids.astype(np.float64)
    amp, freq, total = 1.0, float(frequency), 0.0
    freqs, amps = [], []
    for _ in range(octaves):
        freqs.append(freq)
        amps.append(amp)
        total += amp
        amp *= 0.5
        freq *= 2.0
    noise = fbm_sum(xyz, perm, np.array(freqs), np.array(amps))
    noise /= total
    mn, mx = float(noise.min()), float(noise.max())
    if mx > mn:
        noise = (noise - mn) / (mx - mn)
    return noise.astype(np.float32)


def place_mineral_deposits(
    mesh: PolygonMesh,
    *,
    params: "MineralParams",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place mineral deposits on the globe using the hybrid geology+noise model.

    Returns:
        positions:   (M, 3) float32  — unit-sphere face centroids of deposits.
        kinds:       (M,)   int8     — MINERAL_* index.
        abundances:  (M,)   float32  — [0, 1], higher = richer deposit.
    """
    p = params
    n = len(mesh.faces)
    centroids = mesh.face_centroids.astype(np.float32)

    boundary_type = BOUNDARY_TYPE.read(mesh)
    plate_type = PLATE_TYPE.read(mesh)
    volcanic = VOLCANIC_ACTIVITY.read(mesh).astype(np.float32)
    elevation = ELEVATION_FACE.read(mesh).astype(np.float32)
    is_ocean = IS_OCEAN.read(mesh)
    biome = BIOME.read(mesh)

    is_convergent = boundary_type == "convergent"
    is_divergent = boundary_type == "divergent"
    is_continental = plate_type == "continental"

    # Coal: sedimentary basins — continental, low, far from any plate boundary.
    boundary_seed_mask = boundary_type != "interior"
    if boundary_seed_mask.any():
        boundary_hops = face_hops_from_seeds(
            mesh, boundary_seed_mask,
            max_hops=p.coal_min_boundary_hops + 6,
        )
    else:
        boundary_hops = np.full(n, p.coal_min_boundary_hops + 10, dtype=np.int32)

    coal_candidates = (
        is_continental
        & ~is_ocean
        & (elevation < p.coal_max_elevation_km)
        & (boundary_hops >= p.coal_min_boundary_hops)
    )

    # Salt: evaporite basins — continental, arid, low-to-moderate, tectonically quiet.
    _SEMI_ARID, _DESERT = 7, 8
    is_arid = (biome == _SEMI_ARID) | (biome == _DESERT)
    salt_candidates = (
        is_continental
        & ~is_ocean
        & is_arid
        & (elevation < 1.5)
        & (boundary_hops >= p.coal_min_boundary_hops)
    )

    # Marble: metamorphic rock formed at convergent boundaries.
    marble_candidates = is_continental & ~is_ocean & is_convergent

    land = ~is_ocean

    # (kind, candidate_mask, seed_offset, threshold); offsets keep noise draws independent.  Land only.
    dt = p.deposit_threshold
    specs: list[tuple[int, np.ndarray, int, float]] = [
        (IRON,        land & is_convergent,                                           0, p.iron_deposit_threshold),
        (COPPER,      land & is_convergent,                                           1, dt),
        (GOLD,        land & (volcanic >= p.volcanic_threshold_gold),                 2, p.gold_deposit_threshold),
        (GEMS,        land & (volcanic >= p.volcanic_threshold_gems),                 3, p.gem_deposit_threshold),
        (COAL,        coal_candidates,                                                4, p.coal_deposit_threshold),
        (TIN,         land & is_divergent,                                            5, dt),
        (RARE_METALS, land & (is_divergent | (volcanic >= p.volcanic_threshold_rare)),
         6, p.rare_metals_deposit_threshold),
        (SALT,        salt_candidates,                                               7, p.salt_deposit_threshold),
        (MARBLE,      marble_candidates,                                             8, p.marble_deposit_threshold),
    ]

    all_positions: list[np.ndarray] = []
    all_kinds:     list[np.ndarray] = []
    all_abund:     list[np.ndarray] = []

    for kind_idx, candidate_mask, offset, threshold in specs:
        if not candidate_mask.any():
            continue

        noise = _fbm_noise(centroids, p.noise_frequency, seed + offset * 1000)

        score = noise * candidate_mask.astype(np.float32)
        deposit_mask = (score > threshold) & candidate_mask

        if not deposit_mask.any():
            continue

        excess = score[deposit_mask] - threshold
        abundance = np.clip(excess / max(1.0 - threshold, 1e-6), 0.0, 1.0)

        all_positions.append(centroids[deposit_mask])
        all_kinds.append(np.full(int(deposit_mask.sum()), kind_idx, dtype=np.int8))
        all_abund.append(abundance.astype(np.float32))

    if not all_positions:
        empty: tuple[np.ndarray, np.ndarray, np.ndarray] = (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.int8),
            np.zeros(0, dtype=np.float32),
        )
        return empty

    positions = np.concatenate(all_positions, axis=0)
    kinds = np.concatenate(all_kinds, axis=0)
    abundances = np.concatenate(all_abund, axis=0)
    return positions, kinds, abundances


def place_mineral_sites(
    deposit_faces: np.ndarray,
    texel_positions: np.ndarray,
    land: np.ndarray,
    face_centroids: np.ndarray,
    *,
    min_sites: int,
    max_sites: int,
    share_concentration: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each deposit into the workings it is mined at, inside its face.

    Each deposit gets ``min_sites``–``max_sites`` workings on distinct owned land texels (nearest
    centroid is the deposit's face), interior before coastal, jittered within the texel.  Shares are
    Dirichlet(``share_concentration``), so a few workings carry most of it.

    Args:
        deposit_faces: (M,) face of each deposit.
        texel_positions: (N, H, W, 3) face texel centres.
        land: (N, H, W) bool, not ocean or lake.
        face_centroids: (N, 3).

    Returns:
        positions: (K, 3) float32 unit vectors of the workings.
        deposit:   (K,) int32 deposit index.
        share:     (K,) float32 share of its deposit (sums to 1 per deposit).
    """
    from scipy.spatial import KDTree

    deposit_faces = np.asarray(deposit_faces, dtype=np.int64)
    n_dep = deposit_faces.size
    out_pos: list[np.ndarray] = []
    out_dep: list[np.ndarray] = []
    out_share: list[np.ndarray] = []
    if n_dep == 0:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int32), np.zeros(0, np.float32)
    h, w = land.shape[1:]
    tree = KDTree(np.asarray(face_centroids, dtype=np.float64))
    lo, hi = max(1, int(min_sites)), max(1, int(min_sites), int(max_sites))
    for d, f in enumerate(deposit_faces.tolist()):
        rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, d])
        grid = np.asarray(texel_positions[f], dtype=np.float64)          # (H, W, 3)
        pos = grid.reshape(-1, 3)
        dry = np.asarray(land[f], dtype=bool)
        owned = (tree.query(pos, workers=-1)[1] == f).reshape(h, w)
        interior = dry.copy()
        interior[1:, :] &= dry[:-1, :]
        interior[:-1, :] &= dry[1:, :]
        interior[:, 1:] &= dry[:, :-1]
        interior[:, :-1] &= dry[:, 1:]
        # Interior texels first (in random order), then the coast.
        tiers = [np.flatnonzero((interior & owned).reshape(-1)),
                 np.flatnonzero((dry & owned & ~interior).reshape(-1))]
        cand = np.concatenate([rng.permutation(t) for t in tiers])
        if cand.size == 0:          # no owned land: the face's own centroid
            out_pos.append(np.asarray(face_centroids[f], dtype=np.float64)[None])
            out_dep.append(np.array([d], np.int32))
            out_share.append(np.ones(1, np.float32))
            continue
        k = min(int(rng.integers(lo, hi + 1)), cand.size)
        picked = cand[:k]
        # Jitter interior texels by up to half a spacing; coast texels keep their centre, off the sea.
        r, c = np.divmod(picked, w)
        du = rng.uniform(-0.5, 0.5, k)
        dv = rng.uniform(-0.5, 0.5, k)
        step_c = np.where((c + 1 < w)[:, None], grid[r, np.minimum(c + 1, w - 1)] - grid[r, c],
                          grid[r, c] - grid[r, np.maximum(c - 1, 0)])
        step_r = np.where((r + 1 < h)[:, None], grid[np.minimum(r + 1, h - 1), c] - grid[r, c],
                          grid[r, c] - grid[np.maximum(r - 1, 0), c])
        inner = interior.reshape(-1)[picked][:, None]
        p = grid[r, c] + np.where(inner, du[:, None] * step_c + dv[:, None] * step_r, 0.0)
        share = rng.dirichlet(np.full(k, max(float(share_concentration), 1e-3))) if k > 1 else np.ones(1)
        out_pos.append(p)
        out_dep.append(np.full(k, d, np.int32))
        out_share.append(share)
    positions = np.concatenate(out_pos)
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)
    return (positions.astype(np.float32), np.concatenate(out_dep).astype(np.int32),
            np.concatenate(out_share).astype(np.float32))


def site_workers(deposit: np.ndarray, share: np.ndarray, workers: np.ndarray) -> np.ndarray:
    """``(K,)`` miners at each working: its deposit's ``workers`` × its share."""
    return np.asarray(workers, dtype=np.float64)[np.asarray(deposit, dtype=np.int64)] * np.asarray(share, np.float64)


def mineral_workings(world) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``(positions (K, 3) unit float64, deposit (K,), share (K,))`` of ``world``'s mineral workings.

    Without ``mineral_sites`` each deposit is one working at its own position; None without deposits.
    """
    minerals = world.get_feature("minerals")
    if minerals is None or len(minerals[0]) == 0:
        return None
    sites = world.get_feature("mineral_sites")
    if sites is None:
        pos = np.asarray(minerals[0], dtype=np.float64)
        dep = np.arange(len(pos), dtype=np.int64)
        share = np.ones(len(pos))
    else:
        pos = np.asarray(sites[0], dtype=np.float64)
        dep = np.asarray(sites[1], dtype=np.int64)
        share = np.asarray(sites[2], dtype=np.float64)
    return pos / np.linalg.norm(pos, axis=1, keepdims=True), dep, share
