"""Resources stage: fertility, forest density, crops, mineral deposits and rare biological resources.

Runs after hydrology and the climate lake-feedback pass.  See docs/resources.md.
"""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
from scipy.spatial import KDTree

from core.display import DisplayRegistry
from core.pipeline import BuildContext, Stage
from core.world import World
from worldgen.climate.channels import BIOME, PRECIPITATION, TEMPERATURE
from worldgen.delta.channels import DELTA
from worldgen.geology.channels import ELEVATION, IS_OCEAN, OCEAN_MASK
from worldgen.hydrology.channels import LAKE_MASK, RIVER_OBJECTS
from worldgen.resources.channels import (
    FERTILITY, FERTILITY_MEAN, FOREST_DENSITY, MINERAL_SITES, MINERALS, RARE_BIO,
)
from worldgen.resources.crops import classify_crops
from worldgen.resources.display import (
    DOMINANT_CROP_RECIPE,
    FERTILITY_RECIPE,
    FOREST_DENSITY_RECIPE,
    crop_color,
    crop_legend_entries,
    fertility_color,
    fertility_legend_entries,
    forest_density_color,
    forest_density_legend_entries,
    mineral_legend_entries,
    rare_bio_legend_entries,
)
from worldgen.resources.fertility import (
    apply_delta_fertility, apply_wetland_fertility, compute_aquatic_fertility, compute_fertility,
    neighbour_water_fertility,
)
from worldgen.resources.forest import compute_forest_density
from worldgen.resources.minerals import place_mineral_deposits, place_mineral_sites
from worldgen.resources.rare_bio import place_rare_bio_resources
from worldgen.geology.ruggedness import face_relief
from worldgen.resources.terrain import (
    coastal_ocean_faces,
    face_hops_from_seeds,
    lake_containing_faces,
    river_traversed_faces,
    volcanic_faces,
)
from core.params import field, param_field as _f, paramclass


@paramclass
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
        0.30, label="Forest penalty strength", min=0.0, max=1.0
    )
    ocean_coast_decay_per_hop: float = _f(0.20, label="Ocean coast decay/hop", min=0.01, max=2.0)
    ocean_coast_max_hops: int = _f(8, label="Ocean coast max hops", min=1, max=20)
    ocean_optimal_temperature_c: float = _f(8.0, label="Ocean optimal T (°C)", min=-10.0, max=25.0)
    ocean_temperature_sigma_c: float = _f(12.0, label="Ocean T σ (°C)", min=1.0, max=30.0)
    ocean_upwelling_bonus: float = _f(0.6, label="Upwelling bonus", min=0.0, max=2.0)
    ocean_upwelling_scale_c: float = _f(4.0, label="Upwelling scale (°C)", min=1.0, max=20.0)
    lake_base_fertility: float = _f(0.40, label="Lake base fertility", min=0.0, max=1.0)
    lake_river_bonus: float = _f(0.40, label="Lake river bonus", min=0.0, max=2.0)
    lake_river_decay_per_hop: float = _f(0.40, label="Lake river decay/hop", min=0.05, max=3.0)
    lake_river_max_hops: int = _f(6, label="Lake river max hops", min=0, max=16)
    lake_optimal_temperature_c: float = _f(15.0, label="Lake optimal T (°C)", min=-5.0, max=30.0)
    lake_temperature_sigma_c: float = _f(10.0, label="Lake T σ (°C)", min=1.0, max=30.0)
    # Delta plains are raised toward this (× the temperature bell).
    delta_fertility: float = _f(0.9, label="Delta fertility", min=0.0, max=1.0)
    # Wetland-biome texels (delta marshes among them) are watered whatever the rain: raised toward
    # this (× the temperature bell), then lose ``wetland_penalty`` to waterlogging.
    wetland_fertility: float = _f(0.8, label="Wetland fertility (watered)", min=0.0, max=1.0)
    wetland_penalty: float = _f(0.4, label="Wetland fertility penalty", min=0.0, max=1.0)


@paramclass
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


@paramclass
class RareBioParams:
    noise_frequency: float = _f(4.0, label="Noise frequency", min=1.0, max=12.0)
    base_threshold: float = _f(0.70, label="Base threshold", min=0.3, max=0.95, step=0.05)
    max_spices: int = _f(8,  label="Max Spices",  min=0, max=30)
    max_tea:    int = _f(8,  label="Max Tea",     min=0, max=30)
    max_silk:   int = _f(5,  label="Max Silk",    min=0, max=20)
    max_ivory:  int = _f(6,  label="Max Ivory",   min=0, max=20)
    max_furs:   int = _f(10, label="Max Furs",    min=0, max=30)
    max_incense: int = _f(6, label="Max Incense", min=0, max=20)
    max_dyes:   int = _f(12, label="Max Dyes",    min=0, max=40)
    max_amber:  int = _f(5,  label="Max Amber",   min=0, max=20)
    max_rubber: int = _f(6,  label="Max Rubber",  min=0, max=20)
    max_pearls: int = _f(6,  label="Max Pearls",  min=0, max=20)


@paramclass
class MineralParams:
    noise_frequency: float = _f(3.0, label="Noise frequency", min=0.5, max=12.0)
    deposit_threshold: float = _f(0.55, label="Deposit threshold", min=0.1, max=0.95, step=0.05)
    iron_deposit_threshold: float = _f(0.3, label="Iron threshold", min=0.1, max=0.95, step=0.05)
    gold_deposit_threshold: float = _f(0.70, label="Gold threshold", min=0.1, max=0.95, step=0.05)
    gem_deposit_threshold: float = _f(0.78, label="Gems threshold", min=0.1, max=0.95, step=0.05)
    coal_deposit_threshold: float = _f(0.68, label="Coal threshold", min=0.1, max=0.95, step=0.05)
    salt_deposit_threshold: float = _f(0.58, label="Salt threshold", min=0.1, max=0.95, step=0.05)
    marble_deposit_threshold: float = _f(0.60, label="Marble threshold", min=0.1, max=0.95, step=0.05)
    rare_metals_deposit_threshold: float = _f(0.75, label="Rare metals threshold", min=0.1, max=0.95, step=0.05)
    volcanic_threshold_gold: float = _f(0.25, label="Gold/Gem volcanic min", min=0.0, max=1.0)
    volcanic_threshold_gems: float = _f(0.40, label="Gems volcanic min", min=0.0, max=1.0)
    volcanic_threshold_rare: float = _f(0.15, label="Rare metals volcanic min", min=0.0, max=1.0)
    coal_min_boundary_hops: int = _f(3, label="Coal min boundary hops", min=0, max=16)
    coal_max_elevation_km: float = _f(0.5, label="Coal max elevation (km)", min=-1.0, max=5.0)
    min_sites: int = _f(2, label="Min workings per deposit", min=1, max=12)
    max_sites: int = _f(5, label="Max workings per deposit", min=1, max=12)
    site_share_concentration: float = _f(
        1.0, label="Working share evenness (Dirichlet α)", min=0.1, max=10.0, step=0.1)


@paramclass
class ResourcesParams:
    fertility: FertilityParams = field(
        default_factory=FertilityParams, metadata={"label": "Fertility"}
    )
    forest_density: ForestDensityParams = field(
        default_factory=ForestDensityParams, metadata={"label": "Forest density"}
    )
    minerals: MineralParams = field(
        default_factory=MineralParams, metadata={"label": "Minerals"}
    )
    rare_bio: RareBioParams = field(
        default_factory=RareBioParams, metadata={"label": "Rare Resources"}
    )


def _face_modal_crop(crop_tex: np.ndarray, ocean_mask: np.ndarray) -> np.ndarray:
    """(N,) modal land crop index per face from (N, H, W) crops and ocean mask (0 = None if all ocean)."""
    n = crop_tex.shape[0]
    result = np.zeros(n, dtype=np.int32)
    land = ocean_mask == 0                                      # (N, H, W) bool
    for i in range(n):
        land_pixels = crop_tex[i][land[i]]
        if land_pixels.size == 0:
            continue
        counts = np.bincount(land_pixels.astype(np.int64), minlength=1)
        result[i] = int(counts.argmax())
    return result


class ResourcesBuilder(Stage):
    """Resources stage: writes ``fertility`` / ``forest_density`` / ``dominant_crop`` textures, their
    per-face means, and the ``"minerals"``, ``"mineral_sites"`` and ``"rare_bio"`` features."""

    name = "resources"

    def __init__(self, params: ResourcesParams | None = None) -> None:
        self.params: ResourcesParams = params or ResourcesParams()
        self.fertility_textures: Optional[np.ndarray] = None       # (N, H, W, 1)
        self.forest_density_textures: Optional[np.ndarray] = None  # (N, H, W, 1)
        self.dominant_crop_textures: Optional[np.ndarray] = None   # (N, H, W, 1) uint8
        self.water_hops: Optional[np.ndarray] = None               # (N,) int
        self.volcanism_hops: Optional[np.ndarray] = None           # (N,) int
        # Mineral deposits: 3-tuple (positions (M,3), kinds (M,) int8, abundances (M,) float32)
        self.minerals: Optional[tuple] = None
        # The workings each deposit is split into: (positions (K,3), deposit (K,) int32, share (K,) float32)
        self.mineral_sites: Optional[tuple] = None
        # Rare biological resources: same 3-tuple format as minerals.
        self.rare_bio: Optional[tuple] = None

    def build(self, ctx: BuildContext) -> None:
        mesh = ctx.mesh
        rivers = RIVER_OBJECTS.read(ctx.world)
        _report = ctx.report

        p = self.params
        # The world seed (0 when unseeded), salted per draw below.
        seed = int(ctx.world.seed or 0)
        fp, gp = p.fertility, p.forest_density
        n = len(mesh.faces)

        elevation_tex = ELEVATION.read(mesh)                       # (N, H, W, 1) km
        temperature_tex = TEMPERATURE.read(mesh)                   # (N, H, W, K) °C
        precipitation_tex = PRECIPITATION.read(mesh)               # (N, H, W, K) mm/season
        ocean_mask_tex = OCEAN_MASK.read(mesh)                     # (N, H, W, 1)
        lake_mask_tex = LAKE_MASK.read(mesh) if LAKE_MASK.present(mesh) else None

        face_biome = BIOME.read(mesh)
        is_ocean = IS_OCEAN.read(mesh)                             # (N,) bool
        is_lake_face = (
            lake_containing_faces(lake_mask_tex)
            if lake_mask_tex is not None
            else np.zeros(n, dtype=bool)
        )                                                          # (N,) bool

        _report("Computing terrain ruggedness…")
        face_relief_km = face_relief(mesh)                         # (N,) km, from ruggedness

        _report("Computing water-proximity hops…")
        river_seed_mask = river_traversed_faces(rivers, n_faces=n)
        river_hops = face_hops_from_seeds(
            mesh, river_seed_mask, max_hops=fp.lake_river_max_hops
        )
        water_seed_mask = river_seed_mask.copy()
        if lake_mask_tex is not None:
            water_seed_mask |= lake_containing_faces(lake_mask_tex)
        water_max_hops = max(fp.floodplain_max_hops, gp.riparian_max_hops)
        water_hops = face_hops_from_seeds(mesh, water_seed_mask, max_hops=water_max_hops)

        _report("Computing ocean depth hops…")
        ocean_depth_hops = face_hops_from_seeds(
            mesh, coastal_ocean_faces(mesh, is_ocean), max_hops=fp.ocean_coast_max_hops
        )

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
            params=gp,
            seed=seed,
            mesh=mesh,
        )                                                          # (N, H, W, 1)

        _report("Computing fertility…")
        fertility = compute_fertility(
            temperature_tex=temperature_tex,
            precipitation_tex=precipitation_tex,
            ocean_mask_tex=ocean_mask_tex,
            face_biome=face_biome,
            face_relief_km=face_relief_km,
            water_hops=water_hops,
            volcanism_hops=volcanism_hops,
            params=fp,
            forest_density_tex=forest,
            mesh=mesh,
        )                                                          # (N, H, W, 1)

        if DELTA.present(mesh):
            fertility = apply_delta_fertility(fertility, DELTA.read(mesh), temperature_tex, fp)
        if "biome" in mesh.face_textures:
            fertility = apply_wetland_fertility(fertility, mesh.face_textures["biome"], temperature_tex, fp)

        _report("Computing aquatic fertility…")
        current_temp_delta: np.ndarray | None = None
        if ctx.world.has_feature("currents"):
            currents_list = ctx.world.get_feature("currents")
            if currents_list:
                _, _, current_temp_delta = currents_list[0]
        aquatic_fert = compute_aquatic_fertility(
            temperature_tex=temperature_tex,
            is_ocean=is_ocean,
            is_lake=is_lake_face,
            ocean_depth_hops=ocean_depth_hops,
            river_hops=river_hops,
            current_temp_delta=current_temp_delta,
            params=fp,
        )                                                          # (N,) float32
        # A land face's own sea/lake pixels take the neighbouring water's aquatic fertility (fishing grounds).
        ocean_fert = neighbour_water_fertility(aquatic_fert, is_ocean, mesh.face_adjacency)
        lake_fert = neighbour_water_fertility(aquatic_fert, is_lake_face & ~is_ocean, mesh.face_adjacency)
        fert_arr = fertility[..., 0]                               # (N, H, W)
        ocean_px = ocean_mask_tex[..., 0] > 0
        fert_arr = np.where(ocean_px, ocean_fert[:, None, None], fert_arr)
        if lake_mask_tex is not None:
            lake_px = lake_mask_tex[..., 0] > 0
            fert_arr = np.where(lake_px, lake_fert[:, None, None], fert_arr)
        fertility = fert_arr[..., None].astype(np.float32)         # (N, H, W, 1)

        _report("Computing crop suitability…")
        dominant_crop = classify_crops(
            temperature_tex=temperature_tex,
            precipitation_tex=precipitation_tex,
            elevation_tex=elevation_tex,
            fertility_tex=fertility,
            ocean_mask_tex=ocean_mask_tex,
        )                                                          # (N, H, W, 1) uint8

        FERTILITY.write(mesh, fertility.astype(np.float16))
        FOREST_DENSITY.write(mesh, forest.astype(np.float16))
        mesh.set_face_textures("dominant_crop", dominant_crop, n_channels=1)

        fertility_mean = fertility[..., 0].mean(axis=(1, 2)).astype(np.float32)
        forest_mean = forest[..., 0].mean(axis=(1, 2)).astype(np.float32)
        dominant_crop_face = _face_modal_crop(dominant_crop[..., 0], ocean_mask_tex[..., 0])
        FERTILITY_MEAN.write(mesh, fertility_mean)
        mesh.apply_metadata_channel("forest_density_mean", forest_mean.tolist())
        mesh.apply_metadata_channel("dominant_crop", dominant_crop_face.tolist())

        _report("Placing mineral deposits…")
        mineral_seed = seed + 99991
        self.minerals = place_mineral_deposits(
            mesh,
            params=p.minerals,
            seed=mineral_seed,
        )
        self.mineral_sites = None
        if len(self.minerals[0]):
            centroids = np.asarray(mesh.face_centroids, dtype=np.float64)
            _, deposit_faces = KDTree(centroids).query(np.asarray(self.minerals[0], dtype=np.float64))
            land_px = ocean_mask_tex[..., 0] <= 0.5
            if lake_mask_tex is not None:
                land_px &= lake_mask_tex[..., 0] <= 0.5
            h, w = land_px.shape[1:]
            self.mineral_sites = place_mineral_sites(
                np.asarray(deposit_faces), mesh.face_texture_positions(h, w), land_px, centroids,
                min_sites=p.minerals.min_sites, max_sites=p.minerals.max_sites,
                share_concentration=p.minerals.site_share_concentration, seed=mineral_seed,
            )

        _report("Placing rare biological resources…")
        rare_bio_seed = seed + 55557
        self.rare_bio = place_rare_bio_resources(
            mesh,
            params=p.rare_bio,
            seed=rare_bio_seed,
        )

        self.fertility_textures = fertility
        self.forest_density_textures = forest
        self.dominant_crop_textures = dominant_crop
        self.water_hops = water_hops
        self.volcanism_hops = volcanism_hops

        if self.minerals is not None and len(self.minerals[0]) > 0:
            MINERALS.write(ctx.world, self.minerals)
        if self.mineral_sites is not None:
            MINERAL_SITES.write(ctx.world, self.mineral_sites)
        if self.rare_bio is not None and len(self.rare_bio[0]) > 0:
            RARE_BIO.write(ctx.world, self.rare_bio)

        _report("Resources done.")

    # ── Display ──────────────────────────────────────────────────────────────

    def register_display(
        self, world: World, reg: DisplayRegistry, *, debug: bool = False
    ) -> None:
        mesh = world.mesh

        for name, key, recipe, fn, legend in (
            ("Fertility", "fertility", FERTILITY_RECIPE, fertility_color,
             ("gradient", fertility_legend_entries())),
            ("Forest Density", "forest_density", FOREST_DENSITY_RECIPE, forest_density_color,
             ("gradient", forest_density_legend_entries())),
            ("Dominant Crop", "dominant_crop", DOMINANT_CROP_RECIPE, crop_color,
             ("categorical", crop_legend_entries())),
        ):
            if debug and key in mesh.face_textures:
                mesh.set_face_textures(f"{key}_rgba", recipe(mesh.face_textures))
                reg.add_scheme(name, fn, texture=f"{key}_rgba", legend=legend, recipe=recipe)

        if debug:
            # Resource → face lookups (built once, closed over by the hover fns).
            centroids = mesh.face_centroids.astype(np.float32)
            pos_to_face = {tuple(c): i for i, c in enumerate(centroids)}

            minerals_by_face: dict = {}
            if world.has_feature("minerals"):
                from worldgen.resources.minerals import MINERAL_NAMES
                m_pos, m_kinds, m_abund = world.get_feature("minerals")
                for p, k, a in zip(m_pos, m_kinds, m_abund):
                    fi = pos_to_face.get(tuple(p))
                    if fi is not None:
                        minerals_by_face.setdefault(fi, []).append((MINERAL_NAMES[int(k)], float(a)))
                reg.add_overlay(
                    "minerals", "Minerals", checked=False,
                    legend=("categorical", mineral_legend_entries()),
                )

            from worldgen.resources.crops import CROP_NAMES

            def _resources_meta_hover(face: int, _uv, _w) -> list:
                meta = mesh.face_metadata[face]
                parts: list = []
                if "fertility_mean" in meta:
                    parts.append(f"Fert: {meta['fertility_mean']:.2f}")
                if "forest_density_mean" in meta:
                    parts.append(f"Forest: {meta['forest_density_mean']:.2f}")
                if "dominant_crop" in meta:
                    idx = int(meta["dominant_crop"])
                    crop_name = CROP_NAMES[idx] if 0 <= idx < len(CROP_NAMES) else "?"
                    if crop_name != "None":
                        parts.append(f"Crop: {crop_name}")
                return parts

            reg.add_hover(_resources_meta_hover)

            def _minerals_hover(face: int, _uv, _w) -> list:
                if face in minerals_by_face:
                    best = max(minerals_by_face[face], key=lambda x: x[1])
                    return [f"{best[0]}: {best[1]:.2f}"]
                return []

            reg.add_hover(_minerals_hover)

            rare_bio_by_face: dict = {}
            if world.has_feature("rare_bio"):
                from worldgen.resources.rare_bio import RARE_BIO_NAMES
                rb_pos, rb_kinds, rb_abund = world.get_feature("rare_bio")
                for p, k, a in zip(rb_pos, rb_kinds, rb_abund):
                    fi = pos_to_face.get(tuple(p))
                    if fi is not None:
                        rare_bio_by_face.setdefault(fi, []).append((RARE_BIO_NAMES[int(k)], float(a)))
                reg.add_overlay(
                    "rare_bio", "Rare Resources", checked=False,
                    legend=("categorical", rare_bio_legend_entries()),
                )

            def _rare_bio_hover(face: int, _uv, _w) -> list:
                return [f"{n}: {a:.2f}" for n, a in rare_bio_by_face.get(face, [])]

            reg.add_hover(_rare_bio_hover)

    # ── Persistence ──────────────────────────────────────────────────────────

    def serialize(self) -> dict:
        arrays: dict = {}
        if self.minerals is not None and len(self.minerals[0]) > 0:
            m_pos, m_kinds, m_abund = self.minerals
            arrays["mineral_positions"] = m_pos.astype(np.float32)
            arrays["mineral_kinds"] = m_kinds.astype(np.int8)
            arrays["mineral_abundances"] = m_abund.astype(np.float32)
        if self.mineral_sites is not None:
            s_pos, s_dep, s_share = self.mineral_sites
            arrays["mineral_site_positions"] = s_pos.astype(np.float32)
            arrays["mineral_site_deposits"] = s_dep.astype(np.int32)
            arrays["mineral_site_shares"] = s_share.astype(np.float32)
        if self.rare_bio is not None and len(self.rare_bio[0]) > 0:
            rb_pos, rb_kinds, rb_abund = self.rare_bio
            arrays["rare_bio_positions"] = rb_pos.astype(np.float32)
            arrays["rare_bio_kinds"] = rb_kinds.astype(np.int8)
            arrays["rare_bio_abundances"] = rb_abund.astype(np.float32)
        return arrays

    def load(self, arrays: Mapping[str, np.ndarray], ctx: BuildContext) -> None:
        if "mineral_positions" in arrays:
            self.minerals = (
                arrays["mineral_positions"],
                arrays["mineral_kinds"],
                arrays["mineral_abundances"],
            )
            MINERALS.write(ctx.world, self.minerals)
        if "mineral_site_positions" in arrays:
            self.mineral_sites = (
                arrays["mineral_site_positions"],
                arrays["mineral_site_deposits"],
                arrays["mineral_site_shares"],
            )
            MINERAL_SITES.write(ctx.world, self.mineral_sites)
        if "rare_bio_positions" in arrays:
            self.rare_bio = (
                arrays["rare_bio_positions"],
                arrays["rare_bio_kinds"],
                arrays["rare_bio_abundances"],
            )
            RARE_BIO.write(ctx.world, self.rare_bio)
