"""Resources stage orchestrator — ResourcesParams and ResourcesBuilder.

Runs after climate + hydrology (with lake feedback already applied). Reads the
`temperature`, `precipitation`, `elevation`, `ocean_mask`, and `lake_mask` face
textures plus the per-face `biome` and `volcanic_activity` metadata. Attaches
`fertility` and `forest_density` single-channel textures to the mesh, and writes
per-face `fertility_mean` / `forest_density_mean` aggregates.

See docs/resources_module_followups.md for deferred slices (crop suitability,
mineral deposits, rare resources).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from worldgen.mesh.mymesh import PolygonMesh
from worldgen.resources.crops import classify_crops
from worldgen.resources.fertility import compute_fertility
from worldgen.resources.forest import compute_forest_density
from worldgen.resources.terrain import (
    face_elevation_range,
    face_hops_from_seeds,
    lake_containing_faces,
    river_traversed_faces,
    volcanic_faces,
)


def _f(default, *, label: str, min: float, max: float, step: float | None = None,
       hidden: bool = False):
    meta = {"label": label, "min": min, "max": max}
    if step is not None:
        meta["step"] = step
    if hidden:
        meta["hidden"] = True
    return field(default=default, metadata=meta)


@dataclass
class FertilityParams:
    optimal_temperature_c: float = _f(18.0, label="Optimal T (°C)", min=0.0, max=35.0)
    temperature_sigma_c: float = _f(12.0, label="T σ (°C)", min=1.0, max=30.0)
    optimal_precipitation_mm: float = _f(900.0, label="Optimal P (mm/yr)", min=100.0, max=4000.0)
    precipitation_sigma_log: float = _f(0.7, label="P σ (log)", min=0.1, max=2.0)
    elevation_range_threshold_km: float = _f(2.5, label="Ruggedness threshold (km)", min=0.5, max=10.0)
    floodplain_bonus: float = _f(0.25, label="Floodplain bonus", min=0.0, max=2.0)
    floodplain_decay_per_hop: float = _f(0.5, label="Floodplain decay/hop", min=0.05, max=3.0)
    floodplain_max_hops: int = _f(4, label="Floodplain max hops", min=0, max=16)
    volcanic_bonus: float = _f(0.20, label="Volcanic bonus", min=0.0, max=2.0)
    volcanic_decay_per_hop: float = _f(0.3, label="Volcanic decay/hop", min=0.05, max=3.0)
    volcanic_max_hops: int = _f(6, label="Volcanic max hops", min=0, max=16)
    volcanism_threshold: float = _f(0.2, label="Volcanism threshold", min=0.0, max=1.0)
    forest_penalty_strength: float = _f(
        0.20, label="Forest penalty strength", min=0.0, max=1.0
    )


@dataclass
class ForestDensityParams:
    precipitation_modulation_strength: float = _f(
        0.3, label="P modulation strength", min=0.0, max=1.0
    )
    treeline_elevation_km: float = _f(3.0, label="Treeline elevation (km)", min=0.5, max=8.0)
    treeline_falloff_km: float = _f(0.5, label="Treeline falloff (km)", min=0.05, max=3.0)
    riparian_bonus: float = _f(0.05, label="Riparian bonus", min=0.0, max=0.5)
    riparian_decay_per_hop: float = _f(0.5, label="Riparian decay/hop", min=0.05, max=3.0)
    riparian_max_hops: int = _f(2, label="Riparian max hops", min=0, max=8)
    noise_amplitude: float = _f(0.03, label="Noise amplitude", min=0.0, max=0.3)
    noise_octaves: int = _f(3, label="Noise octaves", min=1, max=6)


@dataclass
class ResourcesParams:
    fertility: FertilityParams = field(
        default_factory=FertilityParams, metadata={"label": "Fertility"}
    )
    forest_density: ForestDensityParams = field(
        default_factory=ForestDensityParams, metadata={"label": "Forest density"}
    )
    seed: Optional[int] = field(
        default=None,
        metadata={"label": "Seed", "kind": "optional_int", "min": 0, "max": 2**31 - 1},
    )


def _face_modal_crop(crop_tex: np.ndarray, ocean_mask: np.ndarray) -> np.ndarray:
    """Return the most-common non-ocean crop index per face.

    Args:
        crop_tex: (N, H, W) uint8 crop indices.
        ocean_mask: (N, H, W) float32; 1 = ocean.

    Returns:
        (N,) int — modal crop index per face (0 = None if all ocean).
    """
    n = crop_tex.shape[0]
    result = np.zeros(n, dtype=np.int32)
    land = ocean_mask < 0.5                                    # (N, H, W) bool
    for i in range(n):
        land_pixels = crop_tex[i][land[i]]
        if land_pixels.size == 0:
            continue
        counts = np.bincount(land_pixels.astype(np.int64), minlength=1)
        result[i] = int(counts.argmax())
    return result


class ResourcesBuilder:
    """Runs the resources stage on a built (elevation + climate + hydrology) mesh.

    Call `build(mesh, rivers=...)` after `HydrologyBuilder.build()` and after
    `ClimateBuilder.apply_lake_humidity_feedback()` so biome metadata reflects
    the final precipitation field. Attaches `fertility` and `forest_density`
    single-channel textures and writes `fertility_mean` / `forest_density_mean`
    per-face metadata.
    """

    def __init__(self, params: ResourcesParams | None = None) -> None:
        self.params: ResourcesParams = params or ResourcesParams()
        self.fertility_textures: Optional[np.ndarray] = None       # (N, H, W, 1)
        self.forest_density_textures: Optional[np.ndarray] = None  # (N, H, W, 1)
        self.dominant_crop_textures: Optional[np.ndarray] = None   # (N, H, W, 1) uint8
        self.water_hops: Optional[np.ndarray] = None               # (N,) int
        self.volcanism_hops: Optional[np.ndarray] = None           # (N,) int

    def build(
        self,
        mesh: PolygonMesh,
        rivers=None,
        progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        def _report(msg: str) -> None:
            if progress is not None:
                progress(msg)

        p = self.params
        fp, gp = p.fertility, p.forest_density
        n = len(mesh.faces)

        elevation_tex = mesh.face_textures["elevation"]            # (N, H, W, 1) km
        temperature_tex = mesh.face_textures["temperature"]        # (N, H, W, K) °C
        precipitation_tex = mesh.face_textures["precipitation"]    # (N, H, W, K) mm/season
        ocean_mask_tex = mesh.face_textures["ocean_mask"]          # (N, H, W, 1)
        lake_mask_tex = mesh.face_textures.get("lake_mask")        # may be None

        face_biome = np.array(
            [int(mesh.face_metadata[i].get("biome", 0)) for i in range(n)],
            dtype=np.int32,
        )

        _report("Computing terrain ruggedness…")
        face_elev_range = face_elevation_range(elevation_tex)      # (N,) km

        _report("Computing water-proximity hops…")
        water_seed_mask = river_traversed_faces(rivers, n_faces=n)
        if lake_mask_tex is not None:
            water_seed_mask |= lake_containing_faces(lake_mask_tex)
        water_max_hops = max(fp.floodplain_max_hops, gp.riparian_max_hops)
        water_hops = face_hops_from_seeds(mesh, water_seed_mask, max_hops=water_max_hops)

        _report("Computing volcanism-proximity hops…")
        volcanism_seed_mask = volcanic_faces(mesh, fp.volcanism_threshold)
        volcanism_hops = face_hops_from_seeds(
            mesh, volcanism_seed_mask, max_hops=fp.volcanic_max_hops
        )

        _report("Computing forest density…")
        h_tex = elevation_tex.shape[1]
        w_tex = elevation_tex.shape[2]
        pixel_positions = mesh.face_texture_positions(h_tex, w_tex) if gp.noise_amplitude > 0 else None
        forest = compute_forest_density(
            precipitation_tex=precipitation_tex,
            elevation_tex=elevation_tex,
            ocean_mask_tex=ocean_mask_tex,
            face_biome=face_biome,
            riparian_hops=water_hops,
            pixel_positions=pixel_positions,
            precipitation_modulation_strength=gp.precipitation_modulation_strength,
            treeline_elevation_km=gp.treeline_elevation_km,
            treeline_falloff_km=gp.treeline_falloff_km,
            riparian_bonus=gp.riparian_bonus,
            riparian_decay_per_hop=gp.riparian_decay_per_hop,
            riparian_max_hops=gp.riparian_max_hops,
            noise_amplitude=gp.noise_amplitude,
            noise_octaves=gp.noise_octaves,
            seed=int(p.seed if p.seed is not None else 0),
            mesh=mesh,
        )                                                          # (N, H, W, 1)

        _report("Computing fertility…")
        fertility = compute_fertility(
            temperature_tex=temperature_tex,
            precipitation_tex=precipitation_tex,
            ocean_mask_tex=ocean_mask_tex,
            face_biome=face_biome,
            face_elev_range_km=face_elev_range,
            water_hops=water_hops,
            volcanism_hops=volcanism_hops,
            optimal_temperature_c=fp.optimal_temperature_c,
            temperature_sigma_c=fp.temperature_sigma_c,
            optimal_precipitation_mm=fp.optimal_precipitation_mm,
            precipitation_sigma_log=fp.precipitation_sigma_log,
            elevation_range_threshold_km=fp.elevation_range_threshold_km,
            floodplain_bonus=fp.floodplain_bonus,
            floodplain_decay_per_hop=fp.floodplain_decay_per_hop,
            floodplain_max_hops=fp.floodplain_max_hops,
            volcanic_bonus=fp.volcanic_bonus,
            volcanic_decay_per_hop=fp.volcanic_decay_per_hop,
            volcanic_max_hops=fp.volcanic_max_hops,
            forest_density_tex=forest,
            forest_penalty_strength=fp.forest_penalty_strength,
            mesh=mesh,
        )                                                          # (N, H, W, 1)

        _report("Computing crop suitability…")
        dominant_crop = classify_crops(
            temperature_tex=temperature_tex,
            precipitation_tex=precipitation_tex,
            elevation_tex=elevation_tex,
            fertility_tex=fertility,
            ocean_mask_tex=ocean_mask_tex,
        )                                                          # (N, H, W, 1) uint8

        mesh.set_face_textures("fertility", fertility, n_channels=1)
        mesh.set_face_textures("forest_density", forest, n_channels=1)
        mesh.set_face_textures("dominant_crop", dominant_crop, n_channels=1)

        fertility_mean = fertility[..., 0].mean(axis=(1, 2)).astype(np.float32)
        forest_mean = forest[..., 0].mean(axis=(1, 2)).astype(np.float32)
        # Per-face dominant crop: mode of land pixels.
        dominant_crop_face = _face_modal_crop(dominant_crop[..., 0], ocean_mask_tex[..., 0])
        mesh.apply_metadata_channel("fertility_mean", fertility_mean.tolist())
        mesh.apply_metadata_channel("forest_density_mean", forest_mean.tolist())
        mesh.apply_metadata_channel("dominant_crop", dominant_crop_face.tolist())

        self.fertility_textures = fertility
        self.forest_density_textures = forest
        self.dominant_crop_textures = dominant_crop
        self.water_hops = water_hops
        self.volcanism_hops = volcanism_hops

        _report("Resources done.")
