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

from typing import Mapping, Optional

import numpy as np

from core.display import DisplayRegistry
from core.pipeline import BuildContext, Stage
from core.world import World

from worldgen.climate.builder import ClimateBuilder
from worldgen.climate.currents import CurrentsResult
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

    def serialize(self) -> dict:
        eb = self.builder
        arrays = {
            "eb_plate_ids": np.array(eb.plate_ids, dtype=np.int32),
            "eb_plate_omegas": eb.plate_omegas.astype(np.float64),
            "eb_face_velocities": eb.face_velocities.astype(np.float64),
            "eb_volcanism": eb.volcanism.astype(np.float64),
            "eb_elevations": eb.elevations.astype(np.float64),
            "eb_max_speed": np.array([eb.max_speed], dtype=np.float64),
        }
        if eb.volcano_positions is not None and len(eb.volcano_positions) > 0:
            arrays["eb_volcano_positions"] = eb.volcano_positions.astype(np.float32)
        return arrays

    def load(self, arrays: Mapping[str, np.ndarray], ctx: BuildContext) -> None:
        eb = self.builder
        eb.mesh = ctx.mesh
        eb.plate_ids = arrays["eb_plate_ids"].tolist()
        eb.plate_omegas = arrays["eb_plate_omegas"]
        eb.face_velocities = arrays["eb_face_velocities"]
        eb.volcanism = arrays["eb_volcanism"]
        eb.elevations = arrays["eb_elevations"]
        eb.max_speed = float(arrays["eb_max_speed"][0])
        if "eb_volcano_positions" in arrays:
            eb.volcano_positions = arrays["eb_volcano_positions"]
            ctx.world.add_feature("volcanoes", eb.volcano_positions)


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

    def serialize(self) -> dict:
        cb = self.builder
        arrays: dict = {}
        if cb.winds is not None:
            arrays["cb_winds"] = cb.winds.astype(np.float32)
        if cb.currents is not None:
            arrays["cb_currents_vectors"] = cb.currents.currents.astype(np.float32)
            arrays["cb_currents_streamfunction"] = cb.currents.streamfunction.astype(np.float32)
            arrays["cb_currents_temp_delta"] = cb.currents.temperature_delta.astype(np.float32)
            arrays["cb_currents_precip_delta"] = cb.currents.precipitation_delta.astype(np.float32)
            arrays["cb_currents_advected_insolation"] = cb.currents.advected_insolation.astype(np.float32)
        return arrays

    def load(self, arrays: Mapping[str, np.ndarray], ctx: BuildContext) -> None:
        cb = self.builder
        centroids = ctx.mesh.face_centroids.astype(np.float32)
        if "cb_winds" in arrays:
            cb.winds = arrays["cb_winds"]
            ctx.world.add_feature(
                "winds",
                [
                    (centroids.copy(), cb.winds[:, s, :].astype(np.float32))
                    for s in range(cb.winds.shape[1])
                ],
            )
        if "cb_currents_vectors" in arrays:
            cb.currents = CurrentsResult(
                currents=arrays["cb_currents_vectors"],
                streamfunction=arrays["cb_currents_streamfunction"],
                temperature_delta=arrays["cb_currents_temp_delta"],
                precipitation_delta=arrays["cb_currents_precip_delta"],
                advected_insolation=arrays["cb_currents_advected_insolation"],
            )
            ctx.world.add_feature(
                "currents",
                [(
                    centroids.copy(),
                    cb.currents.currents.astype(np.float32),
                    cb.currents.temperature_delta.astype(np.float32),
                )],
            )


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
        # The canonical ``(positions, flows, in_lake)`` polyline tuples registered
        # as the "rivers" feature. Cached here because ``serialize`` (no world
        # access) must reproduce it from either a fresh build (River objects) or a
        # load (already-flattened tuples).
        self._river_feature: Optional[list] = None

    def _flatten_rivers(self) -> list:
        """Flatten ``self.builder.rivers`` (List[River]) into ``(positions, flows,
        in_lake)`` polyline tuples — the form used by both the world feature and
        the serialized arrays."""
        paths = []
        for river in self.builder.rivers:
            polyline = river.as_polyline()
            if len(polyline) < 2:
                continue
            positions = np.stack(polyline).astype(np.float32)
            flows = river.flow_rates()
            in_lake = np.array([s.in_lake for s in river.segments], dtype=bool)
            paths.append((positions, flows, in_lake))
        return paths

    def build(self, ctx: BuildContext) -> None:
        mesh = ctx.mesh
        self.builder.build(mesh, mesh.face_textures["elevation"], progress=ctx.report)
        hb = self.builder
        if hb.rivers:
            paths_with_flow = self._flatten_rivers()
            if paths_with_flow:
                self._river_feature = paths_with_flow
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

    def serialize(self) -> dict:
        hb = self.builder
        arrays: dict = {}
        if self._river_feature:
            all_pos, all_flows, all_in_lake = [], [], []
            pos_offsets, flow_offsets = [0], [0]
            for positions, flows, in_lake in self._river_feature:
                all_pos.append(positions)
                all_flows.append(flows)
                all_in_lake.append(in_lake)
                pos_offsets.append(pos_offsets[-1] + len(positions))
                flow_offsets.append(flow_offsets[-1] + len(flows))
            arrays["river_positions_flat"] = np.concatenate(all_pos).astype(np.float32)
            arrays["river_flows_flat"] = np.concatenate(all_flows).astype(np.float32)
            arrays["river_in_lake_flat"] = np.concatenate(all_in_lake).astype(np.bool_)
            arrays["river_pos_offsets"] = np.array(pos_offsets, dtype=np.int64)
            arrays["river_flow_offsets"] = np.array(flow_offsets, dtype=np.int64)
        if hb.lake_polygons:
            all_verts, lake_offsets = [], [0]
            for poly in hb.lake_polygons:
                all_verts.append(poly)
                lake_offsets.append(lake_offsets[-1] + len(poly))
            arrays["lake_polys_flat"] = np.concatenate(all_verts).astype(np.float32)
            arrays["lake_poly_offsets"] = np.array(lake_offsets, dtype=np.int64)
        if hb.delta_positions is not None and len(hb.delta_positions) > 0:
            arrays["hb_delta_positions"] = hb.delta_positions.astype(np.float32)
        return arrays

    def load(self, arrays: Mapping[str, np.ndarray], ctx: BuildContext) -> None:
        hb = self.builder
        if "river_pos_offsets" in arrays:
            pos_flat = arrays["river_positions_flat"]
            flows_flat = arrays["river_flows_flat"]
            in_lake_flat = arrays["river_in_lake_flat"]
            pos_offs = arrays["river_pos_offsets"]
            flow_offs = arrays["river_flow_offsets"]
            # Rehydrated rivers are the flattened polyline tuples, not River
            # objects (no downstream stage re-runs after load).
            paths_with_flow = [
                (
                    pos_flat[pos_offs[i]:pos_offs[i + 1]],
                    flows_flat[flow_offs[i]:flow_offs[i + 1]],
                    in_lake_flat[flow_offs[i]:flow_offs[i + 1]],
                )
                for i in range(len(pos_offs) - 1)
            ]
            hb.rivers = paths_with_flow
            self._river_feature = paths_with_flow
            ctx.world.add_feature("rivers", paths_with_flow)
        if "lake_poly_offsets" in arrays:
            poly_flat = arrays["lake_polys_flat"]
            poly_offs = arrays["lake_poly_offsets"]
            hb.lake_polygons = [
                poly_flat[poly_offs[i]:poly_offs[i + 1]]
                for i in range(len(poly_offs) - 1)
            ]
            ctx.world.add_feature("lakes", hb.lake_polygons)
        if "hb_delta_positions" in arrays:
            hb.delta_positions = arrays["hb_delta_positions"]
            ctx.world.add_feature("deltas", hb.delta_positions)


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

    def serialize(self) -> dict:
        rb = self.builder
        arrays: dict = {}
        if rb.minerals is not None and len(rb.minerals[0]) > 0:
            m_pos, m_kinds, m_abund = rb.minerals
            arrays["mineral_positions"] = m_pos.astype(np.float32)
            arrays["mineral_kinds"] = m_kinds.astype(np.int8)
            arrays["mineral_abundances"] = m_abund.astype(np.float32)
        if rb.rare_bio is not None and len(rb.rare_bio[0]) > 0:
            rb_pos, rb_kinds, rb_abund = rb.rare_bio
            arrays["rare_bio_positions"] = rb_pos.astype(np.float32)
            arrays["rare_bio_kinds"] = rb_kinds.astype(np.int8)
            arrays["rare_bio_abundances"] = rb_abund.astype(np.float32)
        return arrays

    def load(self, arrays: Mapping[str, np.ndarray], ctx: BuildContext) -> None:
        rb = self.builder
        if "mineral_positions" in arrays:
            rb.minerals = (
                arrays["mineral_positions"],
                arrays["mineral_kinds"],
                arrays["mineral_abundances"],
            )
            ctx.world.add_feature("minerals", rb.minerals)
        if "rare_bio_positions" in arrays:
            rb.rare_bio = (
                arrays["rare_bio_positions"],
                arrays["rare_bio_kinds"],
                arrays["rare_bio_abundances"],
            )
            ctx.world.add_feature("rare_bio", rb.rare_bio)


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

    def serialize(self) -> dict:
        cs = self.builder.surfaces
        if cs is None:
            return {}
        return {
            "cost_land_base_cost": cs.land_base_cost.astype(np.float32),
            "cost_land_passable": cs.land_passable.astype(np.bool_),
            "cost_river_flow_dir": cs.river_flow_dir.astype(np.float32),
            "cost_river_flow_mag": cs.river_flow_mag.astype(np.float32),
            "cost_river_navigability": cs.river_navigability.astype(np.float32),
            "cost_sea_base_cost": cs.sea_base_cost.astype(np.float32),
            "cost_sea_passable": cs.sea_passable.astype(np.bool_),
            "cost_strait_mask": cs.strait_mask.astype(np.bool_),
        }

    def load(self, arrays: Mapping[str, np.ndarray], ctx: BuildContext) -> None:
        if "cost_land_base_cost" not in arrays:
            return
        from worldgen.cost.surfaces import CostSurfaces

        river_flow_mag = arrays["cost_river_flow_mag"]
        # Back-compat: navigability was added after the first saved worlds; default
        # to zeros when the key is absent.
        river_navigability = (
            arrays["cost_river_navigability"]
            if "cost_river_navigability" in arrays
            else np.zeros_like(river_flow_mag)
        )
        cs = CostSurfaces(
            land_base_cost=arrays["cost_land_base_cost"],
            land_passable=arrays["cost_land_passable"],
            river_flow_dir=arrays["cost_river_flow_dir"],
            river_flow_mag=river_flow_mag,
            river_navigability=river_navigability,
            sea_base_cost=arrays["cost_sea_base_cost"],
            sea_passable=arrays["cost_sea_passable"],
            strait_mask=arrays["cost_strait_mask"],
        )
        self.builder.surfaces = cs
        ctx.world.add_feature("cost_surfaces", cs)
