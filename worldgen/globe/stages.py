"""Pipeline-stage wrappers for the worldgen physical stages.

Each wrapper adapts an existing ``*Builder`` to the core ``Stage`` protocol:

* ``build(ctx)`` runs the builder against ``ctx.world`` and registers that
  stage's vector/point features on the world.
* ``register_display(world, reg, debug)`` contributes that stage's color
  schemes, overlays, and hover fragments to a ``DisplayRegistry`` — this is the
  decomposition of the former ~300-line ``GlobeBuilder.apply_display`` god-method.

The builders are unchanged — they hold the real logic and remain accessible as
``stage.builder`` (and as ``GlobeBuilder`` attributes) so the serialization path
keeps working. The climate lake-feedback second pass is its own pipeline node
(``ClimateLakeFeedbackStage``) sharing the ``ClimateBuilder`` instance.

(These wrappers + their display slices currently live together for a low-churn
migration; they can move into their owning packages later without changing the
pipeline.)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from core.display import DisplayRegistry
from core.pipeline import BuildContext, Stage
from core.world import World

from worldgen.climate.builder import ClimateBuilder
from worldgen.cost.builder import CostBuilder
from worldgen.geology.elevation_builder import ElevationBuilder
from worldgen.hydrology.builder import HydrologyBuilder, LAKE_COLOR
from worldgen.resources.builder import ResourcesBuilder

from worldgen.geology.display import (
    BOUNDARY_COLORS,
    LAND_COLOR,
    OCEAN_COLOR,
    _elevation_color,
    _elevation_color_vec,
    _plate_palette,
    elevation_legend_entries,
)
from worldgen.climate.display import (
    BIOME_PALETTE,
    biome_color,
    biome_legend_entries,
    build_precipitation_display_texture,
    build_temperature_display_texture,
    humidity_color,
    precipitation_color,
    precipitation_legend_entries,
    temperature_color,
    temperature_legend_entries,
)
from worldgen.resources.display import (
    build_dominant_crop_display_texture,
    build_fertility_display_texture,
    build_forest_density_display_texture,
    crop_color,
    crop_legend_entries,
    fertility_color,
    fertility_legend_entries,
    forest_density_color,
    forest_density_legend_entries,
    mineral_legend_entries,
    rare_bio_legend_entries,
)


SEASON_LABELS = ["Spring", "Summer", "Autumn", "Winter"]


# ── Display-texture builders (RGBA from single-channel data layers) ──────────────


def build_elevation_display_texture(mesh) -> np.ndarray:
    elev_tex = mesh.face_textures["elevation"]          # (N, H, W, 1)
    ocean_mask = None
    if "ocean_mask" in mesh.face_textures:
        ocean_mask = mesh.face_textures["ocean_mask"][:, :, :, 0] > 0  # (N, H, W) bool
    rgba = _elevation_color_vec(elev_tex[:, :, :, 0], ocean_mask=ocean_mask)  # (N, H, W, 4)
    if "lake_mask" in mesh.face_textures:
        lake_tex = mesh.face_textures["lake_mask"]       # (N, H, W, 1)
        rgba[lake_tex[:, :, :, 0] > 0] = LAKE_COLOR
    return rgba


def build_biome_display_texture(mesh) -> np.ndarray:
    """Look up ``"biome"`` indices in BIOME_PALETTE → (N, H, W, 4) RGBA.

    Ocean pixels are overpainted with OCEAN_COLOR (using the ocean_mask texture
    if present, otherwise falling back to elevation < 0). Inland lake pixels are
    overpainted with LAKE_COLOR.
    """
    biome_tex = mesh.face_textures["biome"]              # (N, H, W, 1) uint8
    idx = biome_tex[..., 0].astype(np.int32)
    idx = np.clip(idx, 0, len(BIOME_PALETTE) - 1)
    rgba = BIOME_PALETTE[idx].copy()
    if "ocean_mask" in mesh.face_textures:
        ocean = mesh.face_textures["ocean_mask"][..., 0] > 0
        rgba[ocean] = np.asarray(OCEAN_COLOR, dtype=rgba.dtype)
    elif "elevation" in mesh.face_textures:
        elev_tex = mesh.face_textures["elevation"]       # (N, H, W, 1)
        ocean = elev_tex[..., 0] < 0.0
        rgba[ocean] = np.asarray(OCEAN_COLOR, dtype=rgba.dtype)
    if "lake_mask" in mesh.face_textures:
        lake_tex = mesh.face_textures["lake_mask"]       # (N, H, W, 1)
        rgba[lake_tex[:, :, :, 0] > 0] = LAKE_COLOR
    return rgba


# ── Stages ──────────────────────────────────────────────────────────────────────


class ElevationStage(Stage):
    """Builds the mesh and the elevation/tectonics layers; sets ``world.mesh``."""

    name = "elevation"

    def __init__(self, builder: ElevationBuilder) -> None:
        self.builder = builder

    def build(self, ctx: BuildContext) -> None:
        mesh = self.builder.build(progress=ctx.report)
        ctx.world.mesh = mesh
        eb = self.builder
        if eb.volcano_positions is not None and len(eb.volcano_positions) > 0:
            ctx.world.add_feature("volcanoes", eb.volcano_positions)

    def register_display(self, world: World, reg: DisplayRegistry, *, debug: bool = False) -> None:
        eb = self.builder
        mesh = world.mesh
        palette = _plate_palette(eb.params.plates.n_plates)
        mesh.set_face_colors(palette[np.array(eb.plate_ids, dtype=np.int32)])

        mesh.set_face_textures("elevation_rgba", build_elevation_display_texture(mesh))
        reg.add_scheme(
            "Elevation", _elevation_color,
            texture="elevation_rgba", legend=("gradient", elevation_legend_entries()),
        )
        if not debug:
            return

        max_speed = eb.max_speed

        def _speed_color(meta: dict) -> tuple:
            t = meta["surface_speed"] / max_speed
            return (t, 0.2 + 0.3 * (1.0 - t), 1.0 - t, 1.0)

        def _volcanic_color(meta: dict) -> tuple:
            t = meta["volcanic_activity"]
            if t <= 0.0:
                return (0.35, 0.35, 0.35, 1.0)
            elif t < 0.5:
                s = t / 0.5
                return (0.35 + 0.60 * s, 0.35 - 0.10 * s, 0.35 - 0.35 * s, 1.0)
            else:
                s = (t - 0.5) / 0.5
                return (0.95, 0.25 - 0.20 * s, 0.0, 1.0)

        reg.add_scheme("Plates", lambda meta: tuple(palette[meta["plate_id"]]))
        reg.add_scheme(
            "Plate Type",
            lambda meta: OCEAN_COLOR if meta["plate_type"] == "oceanic" else LAND_COLOR,
        )
        reg.add_scheme("Speed", _speed_color)
        reg.add_scheme("Boundary", lambda meta: BOUNDARY_COLORS[meta["boundary_type"]])
        reg.add_scheme("Volcanic Activity", _volcanic_color)

        if world.has_feature("volcanoes"):
            reg.add_overlay("volcanoes", "Volcanoes", checked=False)

        def _elev_px_hover(face: int, uv, _w) -> list:
            if uv is None or "elevation" not in mesh.face_textures:
                return []
            elev_tex = mesh.face_textures["elevation"]
            h_t, w_t = elev_tex.shape[1:3]
            tx = int(np.clip(uv[0] * w_t, 0, w_t - 1))
            ty = int(np.clip(uv[1] * h_t, 0, h_t - 1))
            return [f"[px: {float(elev_tex[face, ty, tx, 0]):.2f} km]"]

        reg.add_hover(_elev_px_hover)


class ClimateStage(Stage):
    """Runs the first climate pass; registers wind and current features + schemes."""

    name = "climate"

    def __init__(self, builder: ClimateBuilder) -> None:
        self.builder = builder

    def build(self, ctx: BuildContext) -> None:
        mesh = ctx.mesh
        self.builder.build(mesh, mesh.face_textures["elevation"], progress=ctx.report)
        cb = self.builder
        centroids = mesh.face_centroids.astype(np.float32)
        if cb.winds is not None:
            winds_feature = [
                (centroids.copy(), cb.winds[:, s, :].astype(np.float32))
                for s in range(cb.winds.shape[1])
            ]
            ctx.world.add_feature("winds", winds_feature)
        if cb.currents is not None:
            ctx.world.add_feature(
                "currents",
                [(
                    centroids.copy(),
                    cb.currents.currents.astype(np.float32),
                    cb.currents.temperature_delta.astype(np.float32),
                )],
            )

    def lake_feedback_stage(self) -> "ClimateLakeFeedbackStage":
        return ClimateLakeFeedbackStage(self.builder)

    def register_display(self, world: World, reg: DisplayRegistry, *, debug: bool = False) -> None:
        mesh = world.mesh
        if "temperature" not in mesh.face_textures:
            return
        cb = self.builder
        elev_tex = mesh.face_textures["elevation"]

        mesh.set_face_textures("biome_rgba", build_biome_display_texture(mesh))
        reg.add_scheme(
            "Biome", biome_color,
            texture="biome_rgba", legend=("categorical", biome_legend_entries()),
        )
        if not debug:
            return

        temp_tex = mesh.face_textures["temperature"]
        precip_tex = mesh.face_textures["precipitation"]
        ocean_mask_tex = mesh.face_textures.get("ocean_mask")
        mesh.set_face_textures(
            "temperature_annual_rgba",
            build_temperature_display_texture(temp_tex, elevation_tex=elev_tex, ocean_mask_tex=ocean_mask_tex),
        )
        mesh.set_face_textures(
            "temperature_summer_rgba",
            build_temperature_display_texture(
                temp_tex, season=1, elevation_tex=elev_tex, ocean_mask_tex=ocean_mask_tex
            ),
        )
        mesh.set_face_textures(
            "temperature_winter_rgba",
            build_temperature_display_texture(
                temp_tex, season=3, elevation_tex=elev_tex, ocean_mask_tex=ocean_mask_tex
            ),
        )
        mesh.set_face_textures(
            "precipitation_rgba",
            build_precipitation_display_texture(precip_tex, elevation_tex=elev_tex, ocean_mask_tex=ocean_mask_tex),
        )
        reg.add_scheme(
            "Annual Temperature", temperature_color,
            texture="temperature_annual_rgba", legend=("gradient", temperature_legend_entries()),
        )
        reg.add_scheme(
            "Summer Temperature", lambda meta: temperature_color(meta, key="temperature_warmest"),
            texture="temperature_summer_rgba", legend=("gradient", temperature_legend_entries()),
        )
        reg.add_scheme(
            "Winter Temperature", lambda meta: temperature_color(meta, key="temperature_coldest"),
            texture="temperature_winter_rgba", legend=("gradient", temperature_legend_entries()),
        )
        reg.add_scheme(
            "Annual Precipitation", precipitation_color,
            texture="precipitation_rgba", legend=("gradient", precipitation_legend_entries()),
        )
        reg.add_scheme("Humidity", humidity_color)

        if cb.winds is not None:
            for i, label in enumerate(SEASON_LABELS):
                reg.add_overlay(f"winds_season_{i}", f"Winds — {label}", checked=False)
        if cb.currents is not None:
            reg.add_overlay("currents", "Currents", checked=False)

        from worldgen.climate.classification import BIOME_NAMES

        def _climate_hover(face: int, _uv, _w) -> list:
            meta = mesh.face_metadata[face]
            parts: list = []
            if "biome" in meta:
                parts.append(BIOME_NAMES[int(meta["biome"])])
            if "temperature_annual" in meta:
                parts.append(f"T: {meta['temperature_annual']:.1f}°C")
            if "precipitation_annual" in meta:
                parts.append(f"P: {meta['precipitation_annual']:.0f} mm/yr")
            return parts

        reg.add_hover(_climate_hover)


class ClimateLakeFeedbackStage(Stage):
    """Second climate pass: lake evaporation feedback (runs after hydrology)."""

    name = "climate_lake_feedback"

    def __init__(self, builder: ClimateBuilder) -> None:
        self.builder = builder

    def build(self, ctx: BuildContext) -> None:
        mesh = ctx.mesh
        if "lake_mask" in mesh.face_textures:
            self.builder.apply_lake_humidity_feedback(
                mesh, mesh.face_textures["lake_mask"], progress=ctx.report
            )


class HydrologyStage(Stage):
    """Runs hydrology; registers river, lake, and delta features + overlays."""

    name = "hydrology"

    def __init__(self, builder: HydrologyBuilder) -> None:
        self.builder = builder

    def build(self, ctx: BuildContext) -> None:
        mesh = ctx.mesh
        self.builder.build(mesh, mesh.face_textures["elevation"], progress=ctx.report)
        hb = self.builder
        if hb.rivers:
            paths_with_flow = []
            for river in hb.rivers:
                polyline = river.as_polyline()
                if len(polyline) < 2:
                    continue
                positions = np.stack(polyline).astype(np.float32)
                flows = river.flow_rates()
                in_lake = np.array([s.in_lake for s in river.segments], dtype=bool)
                paths_with_flow.append((positions, flows, in_lake))
            if paths_with_flow:
                ctx.world.add_feature("rivers", paths_with_flow)
        if hb.lake_polygons:
            ctx.world.add_feature("lakes", hb.lake_polygons)
        if hb.delta_positions is not None and len(hb.delta_positions) > 0:
            ctx.world.add_feature("deltas", hb.delta_positions)

    def register_display(self, world: World, reg: DisplayRegistry, *, debug: bool = False) -> None:
        if world.has_feature("rivers"):
            reg.add_overlay("rivers", "Rivers")
        if debug and world.has_feature("deltas"):
            reg.add_overlay("deltas", "River Deltas", checked=False)


class ResourcesStage(Stage):
    """Runs resources (needs hydrology rivers); registers mineral/rare-bio features + schemes."""

    name = "resources"

    def __init__(
        self, builder: ResourcesBuilder, hydrology_builder: Optional[HydrologyBuilder] = None
    ) -> None:
        self.builder = builder
        self.hydrology_builder = hydrology_builder

    def build(self, ctx: BuildContext) -> None:
        rivers = self.hydrology_builder.rivers if self.hydrology_builder is not None else None
        self.builder.build(ctx.mesh, rivers=rivers, progress=ctx.report)
        rb = self.builder
        if rb.minerals is not None and len(rb.minerals[0]) > 0:
            ctx.world.add_feature("minerals", rb.minerals)
        if rb.rare_bio is not None and len(rb.rare_bio[0]) > 0:
            ctx.world.add_feature("rare_bio", rb.rare_bio)

    def register_display(self, world: World, reg: DisplayRegistry, *, debug: bool = False) -> None:
        mesh = world.mesh
        ocean_mask_tex = mesh.face_textures.get("ocean_mask")
        lake_mask_tex = mesh.face_textures.get("lake_mask")
        elev_tex = mesh.face_textures.get("elevation")

        if debug and "fertility" in mesh.face_textures:
            mesh.set_face_textures(
                "fertility_rgba",
                build_fertility_display_texture(
                    mesh.face_textures["fertility"],
                    elevation_tex=elev_tex, ocean_mask_tex=ocean_mask_tex, lake_mask_tex=lake_mask_tex,
                ),
            )
            reg.add_scheme(
                "Fertility", fertility_color,
                texture="fertility_rgba", legend=("gradient", fertility_legend_entries()),
            )
        if debug and "forest_density" in mesh.face_textures:
            mesh.set_face_textures(
                "forest_density_rgba",
                build_forest_density_display_texture(
                    mesh.face_textures["forest_density"],
                    elevation_tex=elev_tex, ocean_mask_tex=ocean_mask_tex, lake_mask_tex=lake_mask_tex,
                ),
            )
            reg.add_scheme(
                "Forest Density", forest_density_color,
                texture="forest_density_rgba", legend=("gradient", forest_density_legend_entries()),
            )
        if "dominant_crop" in mesh.face_textures:
            mesh.set_face_textures(
                "dominant_crop_rgba",
                build_dominant_crop_display_texture(
                    mesh.face_textures["dominant_crop"],
                    ocean_mask_tex=ocean_mask_tex, lake_mask_tex=lake_mask_tex,
                ),
            )
            reg.add_scheme(
                "Dominant Crop", crop_color,
                texture="dominant_crop_rgba", legend=("categorical", crop_legend_entries()),
            )

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

        # Debug crop/fertility/forest fragments come first (matching the original
        # status-bar order), then minerals (always), then rare-bio (debug).
        if debug:
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

        if debug:
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


class CostStage(Stage):
    """Runs the base traversal-cost surfaces; registers the cost_surfaces feature + schemes."""

    name = "cost"

    def __init__(
        self, builder: CostBuilder, hydrology_builder: Optional[HydrologyBuilder] = None
    ) -> None:
        self.builder = builder
        self.hydrology_builder = hydrology_builder

    def build(self, ctx: BuildContext) -> None:
        rivers = self.hydrology_builder.rivers if self.hydrology_builder is not None else None
        self.builder.build(ctx.mesh, rivers=rivers, progress=ctx.report)
        if self.builder.surfaces is not None:
            ctx.world.add_feature("cost_surfaces", self.builder.surfaces)

    def register_display(self, world: World, reg: DisplayRegistry, *, debug: bool = False) -> None:
        mesh = world.mesh
        if not (debug and mesh.face_metadata and "land_base_cost" in mesh.face_metadata[0]):
            return

        land_costs = np.array(
            [m.get("land_base_cost", 0.0) for m in mesh.face_metadata], dtype=np.float64
        )
        # Water = ocean ∪ lake (impassable on land); fall back to is_ocean for
        # worlds saved before the cost stage wrote is_water.
        is_water_arr = np.array(
            [bool(m.get("is_water", m.get("is_ocean", False))) for m in mesh.face_metadata],
            dtype=bool,
        )
        land_ref = (
            float(np.percentile(land_costs[~is_water_arr], 95)) if (~is_water_arr).any() else 1.0
        )
        land_ref = max(land_ref, 1e-6)

        def _land_cost_color(meta: dict, _ref=land_ref) -> tuple:
            if meta.get("is_water", meta.get("is_ocean", False)):
                return OCEAN_COLOR
            c = min(max(float(meta.get("land_base_cost", 0.0)) / _ref, 0.0), 1.0)
            return (0.15 + 0.80 * c, 0.75 - 0.55 * c, 0.20, 1.0)

        reg.add_scheme("Land Cost", _land_cost_color)

        # Sailable faces (water ∪ navigable rivers) carry the sea cost.
        is_sailable_arr = is_water_arr | np.array(
            [bool(m.get("is_navigable_river", False)) for m in mesh.face_metadata], dtype=bool
        )
        sea_costs = np.array(
            [m.get("sea_base_cost", 0.0) for m in mesh.face_metadata], dtype=np.float64
        )
        sea_ref = max(
            float(sea_costs[is_sailable_arr].max()) if is_sailable_arr.any() else 1.0, 1e-6
        )

        def _sea_cost_color(meta: dict, _ref=sea_ref) -> tuple:
            sailable = meta.get("is_water", meta.get("is_ocean", False)) or meta.get(
                "is_navigable_river", False
            )
            if not sailable:
                return LAND_COLOR
            if meta.get("is_strait", False):
                return (1.0, 0.85, 0.1, 1.0)  # highlight chokepoints
            if meta.get("is_navigable_river", False):
                return (0.30, 0.85, 0.95, 1.0)  # navigable rivers stand out
            c = min(max(float(meta.get("sea_base_cost", 0.0)) / _ref, 0.0), 1.0)
            return (0.1, 0.20 + 0.30 * (1.0 - c), 0.50 + 0.40 * c, 1.0)

        reg.add_scheme("Sea Cost", _sea_cost_color)
