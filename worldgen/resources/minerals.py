"""Mineral deposit placement — hybrid geology + noise model.

Geology assigns candidate faces per mineral type; fBm simplex noise clusters
the candidates into deposits and sets abundance.

Mineral      Geological seed
-----------  ----------------------------------------------------------
Iron         convergent boundary faces
Copper       convergent boundary faces (independent noise draw)
Gold         volcanic faces  (volcanic_activity ≥ gold_volcanic_threshold)
Gems         volcanic faces  (volcanic_activity ≥ gem_volcanic_threshold; rarer)
Coal         sedimentary basins: continental, low elevation, far from boundaries
Tin          divergent boundary faces
Rare Metals  divergent + hot-spot faces (volcanic_activity ≥ rare_volcanic_threshold)

For each type a single-octave + 2-octave-harmonic simplex noise field is
sampled at face centroids.  Faces where noise > deposit_threshold AND the
geological candidate mask is True become deposit sites; abundance is the
normalised excess above the threshold.
"""
from __future__ import annotations

import numpy as np

from utils.simplex import fbm_sum, make_perm
from worldgen.mesh.mymesh import PolygonMesh
from worldgen.resources.terrain import face_hops_from_seeds


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
    """fBm simplex noise at unit-sphere centroids, normalized to [0, 1].

    Returns (N,) float32.
    """
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
    noise_frequency: float = 3.0,
    deposit_threshold: float = 0.55,
    iron_deposit_threshold: float = 0.35,
    gold_deposit_threshold: float = 0.70,
    gem_deposit_threshold: float = 0.78,
    coal_deposit_threshold: float = 0.68,
    salt_deposit_threshold: float = 0.58,
    marble_deposit_threshold: float = 0.60,
    rare_metals_deposit_threshold: float = 0.75,
    volcanic_threshold_gold: float = 0.25,
    volcanic_threshold_gems: float = 0.40,
    volcanic_threshold_rare: float = 0.15,
    coal_min_boundary_hops: int = 3,
    coal_max_elevation_km: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place mineral deposits on the globe using the hybrid geology+noise model.

    Args:
        mesh: fully-built dual mesh (geology + climate metadata present).
        noise_frequency: spatial frequency of the clustering noise field.
            Lower values = larger deposit clusters.
        deposit_threshold: fraction of the [0,1] noise range above which a
            candidate face becomes a deposit.  Higher = fewer deposits.
        volcanic_threshold_gold: minimum volcanic_activity for a face to be a
            gold/gems candidate.
        volcanic_threshold_gems: as above but for the rarer gems type.
        volcanic_threshold_rare: minimum volcanic_activity for the rare-metals
            hot-spot component.
        coal_min_boundary_hops: minimum BFS hops from any tectonic boundary
            face for a face to qualify as a sedimentary coal basin.
        coal_max_elevation_km: maximum face-level mean elevation (km) for coal
            candidates.  Sedimentary basins are low-lying.
        seed: base RNG seed; each mineral type uses seed + type_offset so their
            noise draws are spatially independent.

    Returns:
        positions:   (M, 3) float32  — unit-sphere face centroids of deposits.
        kinds:       (M,)   int8     — MINERAL_* index.
        abundances:  (M,)   float32  — [0, 1], higher = richer deposit.
    """
    n = len(mesh.faces)
    centroids = mesh.face_centroids.astype(np.float32)

    # ── read face metadata into flat arrays ─────────────────────────────
    boundary_type = np.array(
        [mesh.face_metadata[i].get("boundary_type", "interior") for i in range(n)]
    )
    plate_type = np.array(
        [mesh.face_metadata[i].get("plate_type", "oceanic") for i in range(n)]
    )
    volcanic = np.array(
        [float(mesh.face_metadata[i].get("volcanic_activity", 0.0)) for i in range(n)],
        dtype=np.float32,
    )
    elevation = np.array(
        [float(mesh.face_metadata[i].get("elevation", 0.0)) for i in range(n)],
        dtype=np.float32,
    )
    is_ocean = np.array(
        [bool(mesh.face_metadata[i].get("is_ocean", False)) for i in range(n)],
        dtype=bool,
    )
    biome = np.array(
        [int(mesh.face_metadata[i].get("biome", 0)) for i in range(n)],
        dtype=np.int32,
    )

    # ── geological candidate masks ───────────────────────────────────────
    is_convergent  = boundary_type == "convergent"
    is_divergent   = boundary_type == "divergent"
    is_continental = plate_type == "continental"

    # Coal: sedimentary basin proxy — continental, non-ocean, low elevation,
    # far from any active plate boundary.
    boundary_seed_mask = boundary_type != "interior"
    if boundary_seed_mask.any():
        boundary_hops = face_hops_from_seeds(
            mesh, boundary_seed_mask,
            max_hops=coal_min_boundary_hops + 6,
        )
    else:
        boundary_hops = np.full(n, coal_min_boundary_hops + 10, dtype=np.int32)

    coal_candidates = (
        is_continental
        & ~is_ocean
        & (elevation < coal_max_elevation_km)
        & (boundary_hops >= coal_min_boundary_hops)
    )

    # Salt: evaporite basins — continental, arid biome, low-to-moderate elevation,
    # tectonically quiet (reuses boundary_hops gate from coal).
    _SEMI_ARID, _DESERT = 7, 8
    is_arid = (biome == _SEMI_ARID) | (biome == _DESERT)
    salt_candidates = (
        is_continental
        & ~is_ocean
        & is_arid
        & (elevation < 1.5)
        & (boundary_hops >= coal_min_boundary_hops)
    )

    # Marble: metamorphic rock formed at convergent boundaries.
    marble_candidates = is_continental & ~is_ocean & is_convergent

    land = ~is_ocean

    # ── per-mineral (kind, candidate_mask, seed_offset, threshold) tuples ─
    # Offset keeps each type's noise draw spatially independent.
    # All types are gated on land — ocean deposits are not modelled.
    dt = deposit_threshold
    specs: list[tuple[int, np.ndarray, int, float]] = [
        (IRON,        land & is_convergent,                                         0, iron_deposit_threshold),
        (COPPER,      land & is_convergent,                                         1, dt),
        (GOLD,        land & (volcanic >= volcanic_threshold_gold),                  2, gold_deposit_threshold),
        (GEMS,        land & (volcanic >= volcanic_threshold_gems),                  3, gem_deposit_threshold),
        (COAL,        coal_candidates,                                               4, coal_deposit_threshold),
        (TIN,         land & is_divergent,                                           5, dt),
        (RARE_METALS, land & (is_divergent | (volcanic >= volcanic_threshold_rare)), 6, rare_metals_deposit_threshold),
        (SALT,        salt_candidates,                                               7, salt_deposit_threshold),
        (MARBLE,      marble_candidates,                                             8, marble_deposit_threshold),
    ]

    all_positions: list[np.ndarray] = []
    all_kinds:     list[np.ndarray] = []
    all_abund:     list[np.ndarray] = []

    for kind_idx, candidate_mask, offset, threshold in specs:
        if not candidate_mask.any():
            continue

        noise = _fbm_noise(centroids, noise_frequency, seed + offset * 1000)

        # Score = noise masked to candidates; deposit where score > threshold.
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

    positions  = np.concatenate(all_positions, axis=0)
    kinds      = np.concatenate(all_kinds,     axis=0)
    abundances = np.concatenate(all_abund,     axis=0)
    return positions, kinds, abundances
