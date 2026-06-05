"""Per-pixel forest density texture.

Forest density is biome-gated, then modulated by precipitation, capped by a
treeline cutoff, lifted slightly near rivers/lakes, and given a small noise
component for visual variation:

    forest = clip(biome_baseline × P_modulation × treeline + riparian + noise, 0, 1)

When `mesh` is supplied to `compute_forest_density`, the three per-face inputs
(biome baseline, biome precipitation reference, and the riparian bonus) are
rasterized via `rasterize_face_values` — averaged onto shared vertices and
barycentric-interpolated within fan triangles — so adjacent biomes blend
continuously instead of producing a blocky per-face appearance.

Ocean pixels are zeroed via `ocean_mask`.
"""
from __future__ import annotations

import numpy as np

from worldgen.geology.elevation import rasterize_face_values
from utils.simplex import fbm_sum, make_perm
from core.mesh import PolygonMesh


# Biome index → forest density baseline. Indices follow `BIOME_NAMES` in
# `worldgen/climate/classification.py`:
#   0 Icecap, 1 Tundra, 2 Taiga, 3 Temperate, 4 Steppe,
#   5 Savannah, 6 Tropical, 7 Semi-Arid, 8 Desert.
BIOME_FOREST_BASELINE = np.array([
    0.00,  # Icecap
    0.00,  # Tundra
    0.55,  # Taiga
    0.65,  # Temperate
    0.05,  # Steppe
    0.18,  # Savannah
    0.85,  # Tropical
    0.02,  # Semi-Arid
    0.00,  # Desert
], dtype=np.float32)

# Reference annual precipitation (mm) used to centre per-pixel modulation
# around each biome's "normal" wetness. Pixels with P notably above the
# reference get a small density boost; pixels notably below get a small cut.
BIOME_PRECIP_REFERENCE_MM = np.array([
    100.0,  # Icecap
    250.0,  # Tundra
    600.0,  # Taiga
    900.0,  # Temperate
    400.0,  # Steppe
    700.0,  # Savannah
    2000.0,  # Tropical
    300.0,  # Semi-Arid
    100.0,  # Desert
], dtype=np.float32)


def _value_noise(
    positions: np.ndarray,
    seed: int,
    octaves: int,
    base_frequency: float = 2.0,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
) -> np.ndarray:
    """fBm simplex noise sampled at `positions`, normalised to roughly [-1, 1].

    Args:
        positions: (N, H, W, 3) float32 unit-sphere positions.
        seed: int — controls the permutation table.
        octaves: number of fBm octaves to sum.
        base_frequency: spatial frequency of the lowest octave (cycles/unit).
        lacunarity: frequency multiplier between octaves.
        persistence: amplitude multiplier between octaves.

    Returns:
        (N, H, W) float32.
    """
    n, h, w, _ = positions.shape
    flat = positions.reshape(-1, 3).astype(np.float64)
    perm = make_perm(seed)
    amp, freq, total = 1.0, float(base_frequency), 0.0
    freqs, amps = [], []
    for _ in range(max(octaves, 1)):
        freqs.append(freq)
        amps.append(amp)
        total += amp
        amp *= persistence
        freq *= lacunarity
    out = fbm_sum(flat, perm, np.array(freqs), np.array(amps))
    if total == 0.0:
        total = 1.0
    return (out / total).reshape(n, h, w).astype(np.float32)


def compute_forest_density(
    *,
    precipitation_tex: np.ndarray,
    elevation_tex: np.ndarray,
    ocean_mask_tex: np.ndarray,
    face_biome: np.ndarray,
    riparian_hops: np.ndarray,
    pixel_positions: np.ndarray | None,
    precipitation_modulation_strength: float,
    treeline_elevation_km: float,
    treeline_falloff_km: float,
    riparian_bonus: float,
    riparian_decay_per_hop: float,
    riparian_max_hops: int,
    noise_amplitude: float,
    noise_octaves: int,
    seed: int,
    mesh: PolygonMesh | None = None,
) -> np.ndarray:
    """Compute per-pixel forest density in [0, 1].

    Args:
        precipitation_tex: (N, H, W, K) float32 mm/season.
        elevation_tex: (N, H, W, 1) float32 km.
        ocean_mask_tex: (N, H, W, 1) uint8; 1=ocean.
        face_biome: (N,) int — per-face biome index.
        riparian_hops: (N,) int — face-graph hops from any river/lake face.
        pixel_positions: (N, H, W, 3) float32 unit-sphere positions, or None
            to skip the noise component entirely.
        mesh: optional PolygonMesh. When provided, per-face baseline / P_ref /
            riparian-bonus values are rasterized via `rasterize_face_values`
            so adjacent biomes blend continuously instead of looking blocky.
            When omitted, each value is broadcast per face.

    Returns:
        (N, H, W, 1) float32 in [0, 1].
    """
    biome_idx = np.clip(face_biome.astype(np.int32), 0, len(BIOME_FOREST_BASELINE) - 1)
    face_baseline = BIOME_FOREST_BASELINE[biome_idx]                  # (N,)
    face_p_ref = BIOME_PRECIP_REFERENCE_MM[biome_idx]                 # (N,)

    p_annual = precipitation_tex.sum(axis=-1).astype(np.float32)     # (N, H, W) mm/yr

    h = riparian_hops.astype(np.float32)
    face_riparian = riparian_bonus * np.exp(-h * riparian_decay_per_hop)
    face_riparian[riparian_hops > riparian_max_hops] = 0.0
    face_riparian = face_riparian.astype(np.float32)                  # (N,)

    if mesh is not None:
        tex_size = p_annual.shape[1]
        baseline_tex = rasterize_face_values(mesh, face_baseline, tex_size=tex_size)[..., 0]
        p_ref_tex = rasterize_face_values(mesh, face_p_ref, tex_size=tex_size)[..., 0]
        riparian_tex = rasterize_face_values(mesh, face_riparian, tex_size=tex_size)[..., 0]
    else:
        baseline_tex = np.broadcast_to(
            face_baseline[:, None, None], p_annual.shape
        ).astype(np.float32)
        p_ref_tex = np.broadcast_to(
            face_p_ref[:, None, None], p_annual.shape
        ).astype(np.float32)
        riparian_tex = np.broadcast_to(
            face_riparian[:, None, None], p_annual.shape
        ).astype(np.float32)

    rel = (p_annual - p_ref_tex) / np.maximum(p_ref_tex, 1.0)
    p_modulation = np.clip(
        1.0 + precipitation_modulation_strength * np.tanh(rel), 0.7, 1.3
    ).astype(np.float32)                                              # (N, H, W)

    elev = elevation_tex[..., 0].astype(np.float32)                  # (N, H, W)
    falloff = max(treeline_falloff_km, 1e-3)
    treeline = np.clip(
        (treeline_elevation_km - elev) / falloff, 0.0, 1.0
    ).astype(np.float32)                                              # (N, H, W)

    if pixel_positions is not None and noise_amplitude > 0.0:
        noise = _value_noise(
            pixel_positions, seed=seed, octaves=noise_octaves
        ) * noise_amplitude
    else:
        noise = np.zeros_like(p_annual)

    density = baseline_tex * p_modulation * treeline + riparian_tex + noise
    density = np.clip(density, 0.0, 1.0)

    ocean = ocean_mask_tex[..., 0] > 0
    density = np.where(ocean, 0.0, density).astype(np.float32)

    return density[..., None]
