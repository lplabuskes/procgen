"""Per-pixel fertility texture.

    fertility = clip((climate × terrain × water × geology × biome_mod)
                     × (1 - forest_penalty_strength × forest_density), 0, 1)

The first five factors are per-face:

- `climate`: bell curves in growing-season T and log-precipitation (per-face means, so the product can
  be rasterized smoothly rather than inheriting per-pixel seams).
- `terrain`: ruggedness penalty from the face's pixel elevation range (per-pixel slope is
  noise-dominated at default resolution).
- `water` / `geology`: floodplain / volcanic-soil bonuses decayed by face hops from river-lake /
  volcanic faces.
- `biome_mod`: desert, tundra and icecap soils are poor regardless.

`forest_density` is per-pixel: dense forest locks nutrients in biomass.  With a `mesh` the per-face
product is rasterized via `rasterize_face_values` so faces blend; without one (tests) it is broadcast.
Ocean pixels are zeroed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from worldgen.geology.elevation import rasterize_face_values
from core.mesh import PolygonMesh

if TYPE_CHECKING:
    from worldgen.resources.builder import FertilityParams


# Biome index (`BIOME_NAMES`) → fertility multiplier.
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


def _growing_season_temperature_pixel(temperature_tex: np.ndarray) -> np.ndarray:
    """(N, H, W) mean °C over growing seasons (T > 5°C), else the annual mean; input (N, H, W, K)."""
    t = temperature_tex.astype(np.float32)
    mask = t > 5.0                                               # (N, H, W, K)
    weight = mask.sum(axis=-1).astype(np.float32)               # (N, H, W)
    safe = np.where(weight > 0, weight, t.shape[-1])
    season_sum = np.where(mask, t, 0.0).sum(axis=-1)            # (N, H, W)
    return np.where(weight > 0, season_sum / safe, t.mean(axis=-1)).astype(np.float32)


def _growing_season_temperature_face(temperature_tex: np.ndarray) -> np.ndarray:
    """(N,) face mean of :func:`_growing_season_temperature_pixel`."""
    return _growing_season_temperature_pixel(temperature_tex).mean(axis=(1, 2)).astype(np.float32)


def _climate_term_face(
    temperature_tex: np.ndarray,
    precipitation_tex: np.ndarray,
    *,
    optimal_temperature_c: float,
    temperature_sigma_c: float,
    optimal_precipitation_mm: float,
    precipitation_sigma_log: float,
) -> np.ndarray:
    """Bell-curve climate suitability per face, in [0, 1], from per-face means."""
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
    """(N,) multiplier ``1 + bonus · exp(-hops · decay_per_hop)``; 1 beyond ``max_hops``."""
    h = hops.astype(np.float32)
    factor = 1.0 + bonus * np.exp(-h * decay_per_hop)
    factor[hops > max_hops] = 1.0
    return factor.astype(np.float32)


def compute_aquatic_fertility(
    *,
    temperature_tex: np.ndarray,
    is_ocean: np.ndarray,
    is_lake: np.ndarray,
    ocean_depth_hops: np.ndarray,
    river_hops: np.ndarray,
    current_temp_delta: np.ndarray | None,
    params: "FertilityParams",
) -> np.ndarray:
    """Per-face fishing productivity in [0, 1] for ocean and lake faces.

    - Ocean: coastal proximity (decays with hops from shore) × SST bell (peaks cold-temperate, ~8 °C)
      × (1 + upwelling from cold currents, i.e. negative ``current_temp_delta``).
    - Lake: base × river-fed bonus (decays with ``river_hops``) × temperature bell.

    Land faces return 0.  ``temperature_tex`` is (N, H, W, K) °C; ``current_temp_delta`` is None when
    currents are disabled.
    """
    p = params
    n = len(is_ocean)
    result = np.zeros(n, dtype=np.float32)

    t_annual = temperature_tex.mean(axis=-1).mean(axis=(1, 2)).astype(np.float32)

    if is_ocean.any():
        coastal = np.exp(
            -ocean_depth_hops.astype(np.float32) * p.ocean_coast_decay_per_hop
        )
        t_factor = np.exp(
            -0.5 * ((t_annual - p.ocean_optimal_temperature_c) /
                    max(p.ocean_temperature_sigma_c, 1e-3)) ** 2
        )
        if current_temp_delta is not None:
            cold_delta = np.maximum(0.0, -current_temp_delta.astype(np.float32))
            upwelling = p.ocean_upwelling_bonus * np.tanh(
                cold_delta / max(p.ocean_upwelling_scale_c, 1e-3)
            )
        else:
            upwelling = np.zeros(n, dtype=np.float32)
        ocean_fert = (coastal * t_factor * (1.0 + upwelling)).astype(np.float32)
        result = np.where(is_ocean, ocean_fert, result).astype(np.float32)

    if is_lake.any():
        river_prox = 1.0 + p.lake_river_bonus * np.exp(
            -river_hops.astype(np.float32) * p.lake_river_decay_per_hop
        )
        river_prox = river_prox.astype(np.float32)
        river_prox[river_hops > p.lake_river_max_hops] = 1.0
        t_factor_lake = np.exp(
            -0.5 * ((t_annual - p.lake_optimal_temperature_c) /
                    max(p.lake_temperature_sigma_c, 1e-3)) ** 2
        ).astype(np.float32)
        lake_fert = (p.lake_base_fertility * river_prox * t_factor_lake).astype(np.float32)
        result = np.where(is_lake, lake_fert, result).astype(np.float32)

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def neighbour_water_fertility(
    aquatic: np.ndarray,
    is_water_face: np.ndarray,
    adjacency: list[list[int]],
) -> np.ndarray:
    """``(N,)`` aquatic fertility for each face's water pixels.

    Water faces keep their own value; other faces take the mean of their water neighbours (the same
    fishing grounds), or 0.
    """
    out = np.where(is_water_face, aquatic, 0.0).astype(np.float32)
    for f in np.flatnonzero(~is_water_face):
        vals = [aquatic[nb] for nb in adjacency[f] if is_water_face[nb]]
        if vals:
            out[f] = float(np.mean(vals))
    return out


def compute_fertility(
    *,
    temperature_tex: np.ndarray,
    precipitation_tex: np.ndarray,
    ocean_mask_tex: np.ndarray,
    face_biome: np.ndarray,
    face_elev_range_km: np.ndarray,
    water_hops: np.ndarray,
    volcanism_hops: np.ndarray,
    params: "FertilityParams",
    forest_density_tex: np.ndarray | None = None,
    mesh: PolygonMesh | None = None,
) -> np.ndarray:
    """Compute per-pixel fertility in [0, 1].

    Args:
        temperature_tex: (N, H, W, K) float32 °C.
        precipitation_tex: (N, H, W, K) float32 mm/season.
        ocean_mask_tex: (N, H, W, 1) uint8; 1=ocean.
        face_biome: (N,) int biome index.
        face_elev_range_km: (N,) float32 — pixel elevation span per face.
        water_hops: (N,) int — face hops from any river/lake face.
        volcanism_hops: (N,) int — face hops from any volcanic face.
        mesh: if given, the per-face multiplier is rasterized smoothly; otherwise broadcast per face.

    Returns:
        (N, H, W, 1) float32 in [0, 1].
    """
    p = params
    face_climate = _climate_term_face(
        temperature_tex,
        precipitation_tex,
        optimal_temperature_c=p.optimal_temperature_c,
        temperature_sigma_c=p.temperature_sigma_c,
        optimal_precipitation_mm=p.optimal_precipitation_mm,
        precipitation_sigma_log=p.precipitation_sigma_log,
    )                                                                # (N,)

    face_terrain = (
        1.0 - np.clip(
            face_elev_range_km / max(p.elevation_range_threshold_km, 1e-3), 0.0, 1.0
        )
    ).astype(np.float32)                                             # (N,)

    face_water = _hop_bonus(
        water_hops, p.floodplain_bonus, p.floodplain_decay_per_hop, p.floodplain_max_hops
    )                                                                # (N,)
    face_geology = _hop_bonus(
        volcanism_hops, p.volcanic_bonus, p.volcanic_decay_per_hop, p.volcanic_max_hops
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

    if forest_density_tex is not None and p.forest_penalty_strength > 0.0:
        forest = forest_density_tex
        if forest.ndim == 4:
            forest = forest[..., 0]
        forest_factor = 1.0 - p.forest_penalty_strength * forest.astype(np.float32)
        fertility = fertility * forest_factor

    fertility = np.clip(fertility, 0.0, 1.0)

    ocean = ocean_mask_tex[..., 0] > 0                               # (N, H, W) bool
    fertility = np.where(ocean, 0.0, fertility).astype(np.float32)

    return fertility[..., None]


def apply_delta_fertility(
    fertility: np.ndarray,
    delta_tex: np.ndarray,
    temperature_tex: np.ndarray,
    params: "FertilityParams",
) -> np.ndarray:
    """Raise the river deltas' silt plains toward ``delta_fertility``.

    The river waters and re-silts a delta whatever the rain, so only the growing-season temperature bell
    limits it; marsh is docked by ``delta_marsh_penalty``.  Fertility is only ever raised, by
    ``plain × (target − fertility)``.

    Args:
        fertility: (N, H, W, 1) float32 in [0, 1].
        delta_tex: (N, H, W, 2) — plain membership, marsh (the delta stage's ``delta``).
        temperature_tex: (N, H, W, K) °C.

    Returns:
        (N, H, W, 1) float32 in [0, 1].
    """
    p = params
    plain = np.clip(delta_tex[..., 0].astype(np.float32), 0.0, 1.0)
    marsh = np.clip(delta_tex[..., 1].astype(np.float32), 0.0, 1.0)
    t_grow = _growing_season_temperature_pixel(temperature_tex)
    t_term = np.exp(-0.5 * ((t_grow - p.optimal_temperature_c) / max(p.temperature_sigma_c, 1e-3)) ** 2)
    target = p.delta_fertility * t_term * (1.0 - p.delta_marsh_penalty * marsh)
    fert = fertility[..., 0]
    out = fert + plain * np.maximum(target - fert, 0.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)[..., None]
