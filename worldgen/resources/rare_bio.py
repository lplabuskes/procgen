"""Rare biological resource placement.

Ten rare biological resources are placed based on narrow ecological criteria
(biome, temperature, precipitation, elevation) plus independent fBm noise
draws per type.  A global cap on each type enforces world-level rarity — only
the top-scoring faces within the eligible zone are kept.

Land resources (excluded from ocean and lake faces):
  Spices     Tropical lowland, very hot + very wet
  Tea        Tropical/Temperate highland, cool + wet
  Silk       Temperate/warm, moderate climate (mulberry belt)
  Ivory      Savannah/tropical plains, warm + moderate rain
  Furs       Tundra/Taiga, cold boreal zone
  Incense    Arid highlands (frankincense / myrrh hillsides)
  Dyes       Warm subtropical, broad climate (indigo / madder)
  Amber      Coastal boreal/temperate forest, low elevation
  Rubber     Tropical wet lowland (Hevea brasiliensis)

Ocean resource:
  Pearls     Warm shallow coastal ocean (oyster beds), within 2 hops of land
"""
from __future__ import annotations

import numpy as np

from worldgen.geology.simplex import fbm_sum, make_perm
from worldgen.mesh.mymesh import PolygonMesh
from worldgen.resources.terrain import face_hops_from_seeds, lake_containing_faces


# Biome index constants — must match worldgen/climate/classification.py BIOME_NAMES order.
_ICECAP, _TUNDRA, _TAIGA, _TEMPERATE, _STEPPE, _SAVANNAH, _TROPICAL, _SEMI_ARID, _DESERT = range(9)


RARE_BIO_NAMES: list[str] = [
    "Spices",   # 0
    "Tea",      # 1
    "Silk",     # 2
    "Ivory",    # 3
    "Furs",     # 4
    "Incense",  # 5
    "Dyes",     # 6
    "Amber",    # 7
    "Rubber",   # 8
    "Pearls",   # 9
]
N_RARE_BIO = len(RARE_BIO_NAMES)

SPICES, TEA, SILK, IVORY, FURS, INCENSE, DYES, AMBER, RUBBER, PEARLS = range(N_RARE_BIO)


# RGBA palette — one colour per RARE_BIO_NAMES entry.
RARE_BIO_PALETTE = np.array([
    [0.90, 0.45, 0.10, 1.0],  # 0 Spices   — rust orange (paprika/cinnamon)
    [0.40, 0.70, 0.30, 1.0],  # 1 Tea       — leaf green
    [0.95, 0.88, 0.55, 1.0],  # 2 Silk      — pale champagne gold
    [0.93, 0.92, 0.80, 1.0],  # 3 Ivory     — ivory white-cream
    [0.50, 0.65, 0.80, 1.0],  # 4 Furs      — cool slate-blue
    [0.75, 0.60, 0.35, 1.0],  # 5 Incense   — sandy warm brown
    [0.45, 0.10, 0.60, 1.0],  # 6 Dyes      — deep indigo-purple
    [0.90, 0.65, 0.10, 1.0],  # 7 Amber     — honey amber
    [0.15, 0.50, 0.20, 1.0],  # 8 Rubber    — dark forest green
    [0.80, 0.92, 0.98, 1.0],  # 9 Pearls    — pale iridescent aqua
], dtype=np.float32)


def _fbm_noise(centroids: np.ndarray, frequency: float, seed: int, octaves: int = 3) -> np.ndarray:
    """fBm simplex noise at unit-sphere centroids, normalized to [0, 1].

    Returns (N,) float32.
    """
    perm = make_perm(seed % (2 ** 31))
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


def _cap_to_top_n(deposit_mask: np.ndarray, scores: np.ndarray, max_count: int) -> np.ndarray:
    """Return a new mask keeping only the top max_count faces by score."""
    n_deposits = int(deposit_mask.sum())
    if n_deposits <= max_count:
        return deposit_mask
    indices = np.where(deposit_mask)[0]
    order = np.argsort(scores[indices])[::-1]   # descending by score
    keep = indices[order[:max_count]]
    result = np.zeros_like(deposit_mask)
    result[keep] = True
    return result


def place_rare_bio_resources(
    mesh: PolygonMesh,
    *,
    noise_frequency: float = 4.0,
    base_threshold: float = 0.70,
    max_spices: int = 8,
    max_tea: int = 8,
    max_silk: int = 5,
    max_ivory: int = 6,
    max_furs: int = 10,
    max_incense: int = 6,
    max_dyes: int = 12,
    max_amber: int = 5,
    max_rubber: int = 6,
    max_pearls: int = 6,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place rare biological resource deposits on the globe.

    Args:
        mesh: fully-built dual mesh (geology + climate + hydrology metadata present).
        noise_frequency: spatial frequency of the clustering noise field.
            Lower values produce larger regional pockets; higher values scatter
            smaller spots more evenly.
        base_threshold: minimum [0,1] noise score for a candidate face to
            become a deposit site.  Higher = fewer total sites before the cap.
        max_*: global cap on the number of deposit faces for each resource type.
            The top-scoring candidates are kept; the rest are discarded even if
            they are above the threshold.
        seed: base RNG seed; each resource type gets seed + offset so their
            noise draws are spatially independent.

    Returns:
        positions:   (M, 3) float32  — unit-sphere face centroids of deposits.
        kinds:       (M,)   int8     — RARE_BIO_* index.
        abundances:  (M,)   float32  — [0, 1], normalised excess above threshold.
    """
    n = len(mesh.faces)
    centroids = mesh.face_centroids.astype(np.float32)

    # ── read face metadata into flat arrays ─────────────────────────────────
    biome = np.array(
        [int(mesh.face_metadata[i].get("biome", 0)) for i in range(n)],
        dtype=np.int32,
    )
    t_annual = np.array(
        [float(mesh.face_metadata[i].get("temperature_annual", 0.0)) for i in range(n)],
        dtype=np.float32,
    )
    p_annual = np.array(
        [float(mesh.face_metadata[i].get("precipitation_annual", 0.0)) for i in range(n)],
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
    dist_coast = np.array(
        [float(mesh.face_metadata[i].get("distance_to_coast", 9999.0)) for i in range(n)],
        dtype=np.float32,
    )

    # Exclude both ocean faces and lake faces from land-resource eligibility.
    lake_mask_tex = mesh.face_textures.get("lake_mask")
    if lake_mask_tex is not None:
        is_lake = lake_containing_faces(lake_mask_tex)
    else:
        is_lake = np.zeros(n, dtype=bool)

    land = ~is_ocean & ~is_lake

    # BFS hops from all non-ocean faces — used to find coastal ocean for Pearls.
    # Ocean faces adjacent to land have hops == 1; second ring == 2, etc.
    coastal_hops = face_hops_from_seeds(mesh, ~is_ocean, max_hops=3)

    # ── ecological candidate masks ───────────────────────────────────────────
    # Each mask gates a narrow but plausible habitat; independent noise then
    # clusters actual deposit sites within that habitat.

    # Spices (black pepper, cardamom, cinnamon, cloves): tropical lowland, very hot + wet.
    spices_mask = (
        land
        & (biome == _TROPICAL)
        & (t_annual > 22.0)
        & (p_annual > 1500.0)
        & (elevation < 1.0)
    )

    # Tea (Camellia sinensis): tropical/temperate highland band, cool-to-warm + wet.
    tea_mask = (
        land
        & ((biome == _TROPICAL) | (biome == _TEMPERATE))
        & (t_annual >= 13.0) & (t_annual <= 24.0)
        & (p_annual > 1200.0)
        & (elevation >= 0.5) & (elevation <= 2.5)
    )

    # Silk (silkworm + mulberry): temperate-to-warm, moderate climate.
    silk_mask = (
        land
        & ((biome == _TEMPERATE) | (biome == _TROPICAL) | (biome == _SAVANNAH))
        & (t_annual >= 14.0) & (t_annual <= 28.0)
        & (p_annual >= 400.0) & (p_annual <= 1800.0)
        & (elevation < 2.0)
    )

    # Ivory (elephant territories): savannah/tropical plains, warm + moderate rain.
    ivory_mask = (
        land
        & ((biome == _SAVANNAH) | (biome == _TROPICAL) | (biome == _STEPPE))
        & (t_annual > 18.0)
        & (p_annual >= 350.0) & (p_annual <= 1600.0)
        & (elevation < 1.5)
    )

    # Furs (sable, ermine, beaver): cold boreal/tundra zone.
    furs_mask = (
        land
        & ((biome == _TUNDRA) | (biome == _TAIGA))
        & (t_annual < 5.0)
    )

    # Incense (frankincense, myrrh): arid highland shrublands.
    incense_mask = (
        land
        & ((biome == _DESERT) | (biome == _SEMI_ARID))
        & (elevation >= 0.3) & (elevation <= 2.5)
    )

    # Dyes (indigo, madder, woad): warm subtropical, broad climate range.
    dyes_mask = (
        land
        & ((biome == _TROPICAL) | (biome == _SAVANNAH) | (biome == _TEMPERATE) | (biome == _STEPPE))
        & (t_annual >= 14.0) & (t_annual <= 30.0)
        & (p_annual >= 400.0) & (p_annual <= 2500.0)
        & (elevation < 2.5)
    )

    # Amber (fossilised resin): coastal boreal/temperate forest, low elevation.
    amber_mask = (
        land
        & ((biome == _TEMPERATE) | (biome == _TAIGA))
        & (t_annual >= -5.0) & (t_annual <= 15.0)
        & (dist_coast < 400.0)
        & (elevation < 0.8)
    )

    # Rubber (Hevea brasiliensis): tropical wet lowland, very warm + very wet.
    rubber_mask = (
        land
        & (biome == _TROPICAL)
        & (t_annual > 22.0)
        & (p_annual > 1800.0)
        & (elevation < 0.8)
    )

    # Pearls (Pinctada oysters): warm shallow coastal ocean, within 2 hops of land.
    pearls_mask = (
        is_ocean
        & (coastal_hops <= 2)
        & (t_annual > 18.0)
    )

    # ── (kind, candidate_mask, seed_offset, max_count) ──────────────────────
    specs: list[tuple[int, np.ndarray, int, int]] = [
        (SPICES,  spices_mask,  0, max_spices),
        (TEA,     tea_mask,     1, max_tea),
        (SILK,    silk_mask,    2, max_silk),
        (IVORY,   ivory_mask,   3, max_ivory),
        (FURS,    furs_mask,    4, max_furs),
        (INCENSE, incense_mask, 5, max_incense),
        (DYES,    dyes_mask,    6, max_dyes),
        (AMBER,   amber_mask,   7, max_amber),
        (RUBBER,  rubber_mask,  8, max_rubber),
        (PEARLS,  pearls_mask,  9, max_pearls),
    ]

    all_positions: list[np.ndarray] = []
    all_kinds: list[np.ndarray] = []
    all_abund: list[np.ndarray] = []

    for kind_idx, candidate_mask, offset, max_count in specs:
        if not candidate_mask.any():
            continue

        noise = _fbm_noise(centroids, noise_frequency, seed + offset * 1000 + 77777)

        deposit_mask = candidate_mask & (noise > base_threshold)

        if not deposit_mask.any():
            continue

        # Apply global cap: keep only the top-scoring faces.
        deposit_mask = _cap_to_top_n(deposit_mask, noise, max_count)

        excess = noise[deposit_mask] - base_threshold
        abundance = np.clip(excess / max(1.0 - base_threshold, 1e-6), 0.0, 1.0)

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
