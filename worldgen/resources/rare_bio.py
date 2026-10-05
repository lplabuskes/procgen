"""Rare biological resource placement.

Each type has a narrow ecological habitat mask (biome, temperature, precipitation, elevation; Pearls
in warm coastal ocean, the rest on land) and an independent fBm noise draw; a per-type global cap
keeps only the top-scoring faces.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from utils.simplex import fbm_sum, make_perm
from core.mesh import PolygonMesh
from worldgen.climate.channels import (
    BIOME,
    DISTANCE_TO_COAST,
    PRECIPITATION_ANNUAL,
    TEMPERATURE_ANNUAL,
)
from worldgen.geology.channels import ELEVATION_FACE, IS_OCEAN
from worldgen.hydrology.channels import LAKE_MASK
from worldgen.resources.terrain import face_hops_from_seeds, lake_containing_faces

if TYPE_CHECKING:
    from worldgen.resources.builder import RareBioParams


# Must match BIOME_NAMES order.
_ICECAP, _TUNDRA, _TAIGA, _TEMPERATE, _STEPPE, _SAVANNAH, _TROPICAL, _SEMI_ARID, _DESERT, _WETLAND = range(10)


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
    """(N,) float32 fBm simplex noise at unit-sphere centroids, normalized to [0, 1]."""
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
    """Mask keeping only the top ``max_count`` faces by score."""
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
    params: "RareBioParams",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place rare biological resource deposits on the globe.

    Returns:
        positions:   (M, 3) float32  — unit-sphere face centroids of deposits.
        kinds:       (M,)   int8     — RARE_BIO_* index.
        abundances:  (M,)   float32  — [0, 1], normalised excess above threshold.
    """
    p = params
    n = len(mesh.faces)
    centroids = mesh.face_centroids.astype(np.float32)

    biome = BIOME.read(mesh)
    t_annual = TEMPERATURE_ANNUAL.read(mesh).astype(np.float32)
    p_annual = PRECIPITATION_ANNUAL.read(mesh).astype(np.float32)
    elevation = ELEVATION_FACE.read(mesh).astype(np.float32)
    is_ocean = IS_OCEAN.read(mesh)
    dist_coast = (
        DISTANCE_TO_COAST.read(mesh).astype(np.float32)
        if DISTANCE_TO_COAST.present(mesh)
        else np.full(n, 9999.0, dtype=np.float32)
    )

    lake_mask_tex = LAKE_MASK.read(mesh) if LAKE_MASK.present(mesh) else None
    if lake_mask_tex is not None:
        is_lake = lake_containing_faces(lake_mask_tex)
    else:
        is_lake = np.zeros(n, dtype=bool)

    land = ~is_ocean & ~is_lake

    # Hops from land, for Pearls: ocean faces next to land are 1.
    coastal_hops = face_hops_from_seeds(mesh, ~is_ocean, max_hops=3)

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
        (SPICES,  spices_mask,  0, p.max_spices),
        (TEA,     tea_mask,     1, p.max_tea),
        (SILK,    silk_mask,    2, p.max_silk),
        (IVORY,   ivory_mask,   3, p.max_ivory),
        (FURS,    furs_mask,    4, p.max_furs),
        (INCENSE, incense_mask, 5, p.max_incense),
        (DYES,    dyes_mask,    6, p.max_dyes),
        (AMBER,   amber_mask,   7, p.max_amber),
        (RUBBER,  rubber_mask,  8, p.max_rubber),
        (PEARLS,  pearls_mask,  9, p.max_pearls),
    ]

    all_positions: list[np.ndarray] = []
    all_kinds: list[np.ndarray] = []
    all_abund: list[np.ndarray] = []

    for kind_idx, candidate_mask, offset, max_count in specs:
        if not candidate_mask.any():
            continue

        noise = _fbm_noise(centroids, p.noise_frequency, seed + offset * 1000 + 77777)

        deposit_mask = candidate_mask & (noise > p.base_threshold)

        if not deposit_mask.any():
            continue

        deposit_mask = _cap_to_top_n(deposit_mask, noise, max_count)

        excess = noise[deposit_mask] - p.base_threshold
        abundance = np.clip(excess / max(1.0 - p.base_threshold, 1e-6), 0.0, 1.0)

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
