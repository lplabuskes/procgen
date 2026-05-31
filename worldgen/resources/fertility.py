"""Per-pixel fertility texture.

Fertility is a multiplicative product of five factors, each in roughly [0, 1+],
optionally docked by a small per-pixel forest penalty:
    fertility = clip((climate × terrain × water × geology × biome_mod)
                     × (1 - forest_penalty_strength × forest_density), 0, 1)

- `climate` (per-face): bell-curve in per-face mean growing-season T and
  log-precipitation. Computed as a per-face scalar so it can be rasterized
  (blended across face boundaries) instead of using raw per-pixel textures
  that are not continuous at face edges.
- `terrain` (per-face): penalty for rugged faces, using per-face pixel
  elevation range as a proxy for "is this a mountainous area". We deliberately
  avoid per-pixel slope from the elevation texture — at default resolution,
  pixel-spacing slope is dominated by the noise overlay rather than real
  terrain (see docs/resources_module_followups.md).
- `water` (per-face): floodplain bonus decayed by hop count on the face
  adjacency graph from river/lake faces.
- `geology` (per-face): volcanic-soil bonus decayed by hop count from faces
  with `volcanic_activity` above threshold.
- `biome_mod` (per-face): coarse multiplier reflecting that desert / tundra /
  icecap biomes don't produce fertile soil regardless of inputs.
- `forest_density` (per-pixel, optional): dense forest sequesters nutrients in
  biomass, leaving soils poor for agriculture once cleared (classic tropical
  rainforest problem). A small multiplicative penalty proportional to the
  forest-density texture from `compute_forest_density`.

All five per-face factors are combined into a single per-face multiplier. When
`mesh` is supplied to `compute_fertility`, that multiplier is rasterized via
`rasterize_face_values` — averaged onto shared vertices and barycentric-
interpolated within each face's fan triangles — so adjacent faces blend
continuously instead of producing a blocky per-face appearance. Without a
mesh (e.g. in unit tests with synthetic inputs) the multiplier is simply
broadcast across each face's pixels.

Ocean pixels are zeroed via `ocean_mask`.
"""
from __future__ import annotations

import numpy as np

from worldgen.geology.elevation import rasterize_face_values
from worldgen.mesh.mymesh import PolygonMesh


# Biome index → fertility multiplier. Indices follow `BIOME_NAMES` in
# `worldgen/climate/classification.py`:
#   0 Icecap, 1 Tundra, 2 Taiga, 3 Temperate, 4 Steppe,
#   5 Savannah, 6 Tropical, 7 Semi-Arid, 8 Desert.
BIOME_FERTILITY_MULT = np.array([
    0.00,  # Icecap
    0.20,  # Tundra
    0.60,  # Taiga
    1.00,  # Temperate
    0.50,  # Steppe
    0.80,  # Savannah
    0.95,  # Tropical
    0.30,  # Semi-Arid
    0.05,  # Desert
], dtype=np.float32)


def _growing_season_temperature_face(temperature_tex: np.ndarray) -> np.ndarray:
    """Per-face mean temperature over growing seasons (T > 5°C).

    Averages over H and W after selecting growing-season pixels, then falls
    back to the annual mean for faces where no season qualifies.

    Args:
        temperature_tex: (N, H, W, K) float32 °C.

    Returns:
        (N,) float32 °C.
    """
    t = temperature_tex.astype(np.float32)
    mask = t > 5.0                                               # (N, H, W, K)
    weight = mask.sum(axis=-1).astype(np.float32)               # (N, H, W)
    safe = np.where(weight > 0, weight, t.shape[-1])
    season_sum = np.where(mask, t, 0.0).sum(axis=-1)            # (N, H, W)
    annual_mean = t.mean(axis=-1)                               # (N, H, W)
    t_grow_px = np.where(weight > 0, season_sum / safe, annual_mean)  # (N, H, W)
    return t_grow_px.mean(axis=(1, 2)).astype(np.float32)       # (N,)


def _climate_term_face(
    temperature_tex: np.ndarray,
    precipitation_tex: np.ndarray,
    *,
    optimal_temperature_c: float,
    temperature_sigma_c: float,
    optimal_precipitation_mm: float,
    precipitation_sigma_log: float,
) -> np.ndarray:
    """Bell-curve climate suitability per face, in [0, 1].

    Uses per-face means so the result can be rasterized (blended across face
    boundaries) rather than being computed from raw per-pixel textures that are
    discontinuous at face edges.
    """
    t_grow_face = _growing_season_temperature_face(temperature_tex)   # (N,)
    t_term = np.exp(
        -0.5 * ((t_grow_face - optimal_temperature_c) / max(temperature_sigma_c, 1e-3)) ** 2
    )

    p_annual_face = precipitation_tex.sum(axis=-1).mean(axis=(1, 2)).astype(np.float32)
    log_p = np.log(np.maximum(p_annual_face, 1.0))
    log_p_opt = np.log(max(optimal_precipitation_mm, 1.0))
    p_term = np.exp(
        -0.5 * ((log_p - log_p_opt) / max(precipitation_sigma_log, 1e-3)) ** 2
    )

    return (t_term * p_term).astype(np.float32)                       # (N,)


def _hop_bonus(hops: np.ndarray, bonus: float, decay_per_hop: float, max_hops: int) -> np.ndarray:
    """Exponential-decay bonus from per-face hop counts.

    Args:
        hops: (N,) int — hop counts from `face_hops_from_seeds`.
        bonus: amplitude at hop=0.
        decay_per_hop: exponent so that `factor = 1 + bonus * exp(-hops * decay)`.
        max_hops: faces beyond this contribute zero.

    Returns:
        (N,) float32 multiplier ≥ 1.
    """
    h = hops.astype(np.float32)
    factor = 1.0 + bonus * np.exp(-h * decay_per_hop)
    factor[hops > max_hops] = 1.0
    return factor.astype(np.float32)


def compute_fertility(
    *,
    temperature_tex: np.ndarray,
    precipitation_tex: np.ndarray,
    ocean_mask_tex: np.ndarray,
    face_biome: np.ndarray,
    face_elev_range_km: np.ndarray,
    water_hops: np.ndarray,
    volcanism_hops: np.ndarray,
    optimal_temperature_c: float,
    temperature_sigma_c: float,
    optimal_precipitation_mm: float,
    precipitation_sigma_log: float,
    elevation_range_threshold_km: float,
    floodplain_bonus: float,
    floodplain_decay_per_hop: float,
    floodplain_max_hops: int,
    volcanic_bonus: float,
    volcanic_decay_per_hop: float,
    volcanic_max_hops: int,
    forest_density_tex: np.ndarray | None = None,
    forest_penalty_strength: float = 0.0,
    mesh: PolygonMesh | None = None,
) -> np.ndarray:
    """Compute per-pixel fertility in [0, 1].

    Args:
        temperature_tex: (N, H, W, K) float32 °C.
        precipitation_tex: (N, H, W, K) float32 mm/season.
        ocean_mask_tex: (N, H, W, 1) float32; 1=ocean.
        face_biome: (N,) int — per-face biome index (from `mesh.face_metadata`).
        face_elev_range_km: (N,) float32 — pixel elevation span per face.
        water_hops: (N,) int — face-graph hops from any river/lake face.
        volcanism_hops: (N,) int — face-graph hops from any volcanic face.
        mesh: optional PolygonMesh. When provided, the per-face multiplier is
            rasterized via `rasterize_face_values` so adjacent faces blend
            continuously instead of looking blocky. When omitted, the multiplier
            is broadcast as a single value per face.

    Returns:
        (N, H, W, 1) float32 in [0, 1].
    """
    face_climate = _climate_term_face(
        temperature_tex,
        precipitation_tex,
        optimal_temperature_c=optimal_temperature_c,
        temperature_sigma_c=temperature_sigma_c,
        optimal_precipitation_mm=optimal_precipitation_mm,
        precipitation_sigma_log=precipitation_sigma_log,
    )                                                                # (N,)

    face_terrain = (
        1.0 - np.clip(
            face_elev_range_km / max(elevation_range_threshold_km, 1e-3), 0.0, 1.0
        )
    ).astype(np.float32)                                             # (N,)

    face_water = _hop_bonus(
        water_hops, floodplain_bonus, floodplain_decay_per_hop, floodplain_max_hops
    )                                                                # (N,)
    face_geology = _hop_bonus(
        volcanism_hops, volcanic_bonus, volcanic_decay_per_hop, volcanic_max_hops
    )                                                                # (N,)

    biome_idx = np.clip(face_biome.astype(np.int32), 0, len(BIOME_FERTILITY_MULT) - 1)
    face_biome_mult = BIOME_FERTILITY_MULT[biome_idx]                # (N,)

    face_mult = (
        face_climate * face_terrain * face_water * face_geology * face_biome_mult
    ).astype(np.float32)                                             # (N,)

    if mesh is not None:
        tex_size = temperature_tex.shape[1]
        fertility = rasterize_face_values(
            mesh, face_mult, tex_size=tex_size
        )[..., 0]                                                    # (N, H, W)
    else:
        fertility = np.broadcast_to(
            face_mult[:, None, None],
            (len(face_mult), temperature_tex.shape[1], temperature_tex.shape[2]),
        ).copy().astype(np.float32)                                  # (N, H, W)

    if forest_density_tex is not None and forest_penalty_strength > 0.0:
        forest = forest_density_tex
        if forest.ndim == 4:
            forest = forest[..., 0]
        forest_factor = 1.0 - forest_penalty_strength * forest.astype(np.float32)
        fertility = fertility * forest_factor

    fertility = np.clip(fertility, 0.0, 1.0)

    ocean = ocean_mask_tex[..., 0] > 0.5                             # (N, H, W) bool
    fertility = np.where(ocean, 0.0, fertility).astype(np.float32)

    return fertility[..., None]
