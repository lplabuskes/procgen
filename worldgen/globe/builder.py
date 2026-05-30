from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from typing import Callable, Optional

from worldgen.geology.display import (
    BOUNDARY_COLORS,
    LAND_COLOR,
    OCEAN_COLOR,
    _elevation_color,
    _elevation_color_vec,
    _plate_palette,
    elevation_legend_entries,
)

from worldgen.climate.builder import ClimateBuilder, ClimateParams
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
from worldgen.geology.elevation_builder import ElevationBuilder, ElevationParams
from worldgen.hydrology.builder import HydrologyBuilder, HydrologyParams, LAKE_COLOR
from worldgen.globe.globe import Globe


SEASON_LABELS = ["Spring", "Summer", "Autumn", "Winter"]


def build_elevation_display_texture(mesh) -> np.ndarray:
    elev_tex = mesh.face_textures["elevation"]          # (N, H, W, 1)
    ocean_mask = None
    if "ocean_mask" in mesh.face_textures:
        ocean_mask = mesh.face_textures["ocean_mask"][:, :, :, 0] > 0.5  # (N, H, W) bool
    rgba = _elevation_color_vec(elev_tex[:, :, :, 0], ocean_mask=ocean_mask)  # (N, H, W, 4)
    if "lake_mask" in mesh.face_textures:
        lake_tex = mesh.face_textures["lake_mask"]       # (N, H, W, 1)
        rgba[lake_tex[:, :, :, 0] > 0.5] = LAKE_COLOR
    return rgba


def build_biome_display_texture(mesh) -> np.ndarray:
    """Look up `"biome"` indices in BIOME_PALETTE → (N, H, W, 4) RGBA.

    Ocean pixels are overpainted with OCEAN_COLOR (using the ocean_mask texture
    if present, otherwise falling back to elevation < 0). Inland lake pixels are
    overpainted with LAKE_COLOR.
    """
    biome_tex = mesh.face_textures["biome"]              # (N, H, W, 1) uint8
    idx = biome_tex[..., 0].astype(np.int32)
    idx = np.clip(idx, 0, len(BIOME_PALETTE) - 1)
    rgba = BIOME_PALETTE[idx].copy()
    if "ocean_mask" in mesh.face_textures:
        ocean = mesh.face_textures["ocean_mask"][..., 0] > 0.5
        rgba[ocean] = np.asarray(OCEAN_COLOR, dtype=rgba.dtype)
    elif "elevation" in mesh.face_textures:
        elev_tex = mesh.face_textures["elevation"]       # (N, H, W, 1)
        ocean = elev_tex[..., 0] < 0.0
        rgba[ocean] = np.asarray(OCEAN_COLOR, dtype=rgba.dtype)
    if "lake_mask" in mesh.face_textures:
        lake_tex = mesh.face_textures["lake_mask"]       # (N, H, W, 1)
        rgba[lake_tex[:, :, :, 0] > 0.5] = LAKE_COLOR
    return rgba


@dataclass
class GlobeParams:
    """Top-level parameters for the full globe-generation pipeline.

    Stage-specific params are nested. New stages add their own params
    dataclass as a new field here.
    """
    elevation: ElevationParams = field(default_factory=ElevationParams)
    climate: ClimateParams = field(default_factory=ClimateParams, metadata={"label": "Climate"})
    hydrology: HydrologyParams = field(default_factory=HydrologyParams, metadata={"label": "Hydrology"})


class GlobeBuilder:
    """Orchestrates every stage of globe generation.

    Each stage's builder is stored as an attribute (`elevation_builder`,
    `climate_builder`, `hydrology_builder`) so callers can read intermediate
    results without re-running work. `build()` returns a `Globe` wrapping the
    resulting mesh.
    """

    def __init__(self, params: GlobeParams | None = None) -> None:
        self.params: GlobeParams = params or GlobeParams()
        self.elevation_builder: ElevationBuilder = ElevationBuilder(self.params.elevation)
        self.climate_builder: ClimateBuilder = ClimateBuilder(self.params.climate)
        self.hydrology_builder: HydrologyBuilder = HydrologyBuilder(self.params.hydrology)
        self.globe: Optional[Globe] = None

    def build(self, progress: Optional[Callable[[str], None]] = None) -> Globe:
        mesh = self.elevation_builder.build(progress=progress)

        elevation_textures = mesh.face_textures["elevation"]  # (N, H, W, 1)
        self.climate_builder.build(mesh, elevation_textures, progress=progress)
        self.hydrology_builder.build(mesh, elevation_textures, progress=progress)
        if "lake_mask" in mesh.face_textures:
            self.climate_builder.apply_lake_humidity_feedback(
                mesh, mesh.face_textures["lake_mask"], progress=progress
            )

        globe = Globe(mesh=mesh)
        hb = self.hydrology_builder
        cb = self.climate_builder

        if hb.rivers:
            paths_with_flow = []
            for river in hb.rivers:
                polyline = river.as_polyline()
                if len(polyline) < 2:
                    continue
                positions = np.stack(polyline).astype(np.float32)  # (M, 3)
                flows = river.flow_rates()                          # length M-1
                in_lake = np.array(
                    [s.in_lake for s in river.segments], dtype=bool
                )                                                    # length M-1
                paths_with_flow.append((positions, flows, in_lake))
            if paths_with_flow:
                globe.add_feature("rivers", paths_with_flow)

        if hb.lake_polygons:
            globe.add_feature("lakes", hb.lake_polygons)

        if hb.delta_positions is not None and len(hb.delta_positions) > 0:
            globe.add_feature("deltas", hb.delta_positions)

        if (self.elevation_builder.volcano_positions is not None
                and len(self.elevation_builder.volcano_positions) > 0):
            globe.add_feature("volcanoes", self.elevation_builder.volcano_positions)

        if cb.winds is not None:
            centroids = mesh.face_centroids.astype(np.float32)
            winds_feature = []
            for season_idx in range(cb.winds.shape[1]):
                winds_feature.append((centroids.copy(), cb.winds[:, season_idx, :].astype(np.float32)))
            globe.add_feature("winds", winds_feature)

        if cb.currents is not None:
            centroids = mesh.face_centroids.astype(np.float32)
            globe.add_feature(
                "currents",
                [(
                    centroids.copy(),
                    cb.currents.currents.astype(np.float32),
                    cb.currents.temperature_delta.astype(np.float32),
                )],
            )

        self.globe = globe
        return globe

    def apply_display(self, window, debug: bool = False) -> None:
        """Wire color schemes and overlays into a MapWindow after build()."""
        eb = self.elevation_builder
        hb = self.hydrology_builder
        cb = self.climate_builder
        assert eb.mesh is not None, "GlobeBuilder.build() must be called before apply_display"

        mesh = eb.mesh
        palette = _plate_palette(eb.params.plates.n_plates)
        max_speed = eb.max_speed

        mesh.set_face_colors(palette[np.array(eb.plate_ids, dtype=np.int32)])

        elevation_rgba = build_elevation_display_texture(mesh)
        mesh.set_face_textures("elevation_rgba", elevation_rgba)

        schemes: dict[str, Callable[[dict], tuple]] = {
            "Elevation": _elevation_color,
        }

        if debug:
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

            schemes.update({
                "Plates": lambda meta: tuple(palette[meta["plate_id"]]),
                "Plate Type": lambda meta: OCEAN_COLOR if meta["plate_type"] == "oceanic" else LAND_COLOR,
                "Speed": _speed_color,
                "Boundary": lambda meta: BOUNDARY_COLORS[meta["boundary_type"]],
                "Volcanic Activity": _volcanic_color,
            })

        if "temperature" in mesh.face_textures:
            elev_tex = mesh.face_textures["elevation"]
            mesh.set_face_textures(
                "biome_rgba",
                build_biome_display_texture(mesh),
            )
            schemes["Biome"] = biome_color

            if debug:
                temp_tex = mesh.face_textures["temperature"]
                precip_tex = mesh.face_textures["precipitation"]
                ocean_mask_tex = mesh.face_textures["ocean_mask"] if "ocean_mask" in mesh.face_textures else None
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
                    build_precipitation_display_texture(
                        precip_tex, elevation_tex=elev_tex, ocean_mask_tex=ocean_mask_tex
                    ),
                )
                schemes["Annual Temperature"] = temperature_color
                schemes["Summer Temperature"] = lambda meta: temperature_color(meta, key="temperature_warmest")
                schemes["Winter Temperature"] = lambda meta: temperature_color(meta, key="temperature_coldest")
                schemes["Annual Precipitation"] = precipitation_color
                schemes["Humidity"] = humidity_color

        window.set_color_schemes(schemes)
        window.associate_scheme_texture("Elevation", "elevation_rgba")
        window.register_legend("Elevation", "gradient", elevation_legend_entries())
        if "temperature" in mesh.face_textures:
            window.associate_scheme_texture("Biome", "biome_rgba")
            window.register_legend("Biome", "categorical", biome_legend_entries())
            if debug:
                window.associate_scheme_texture("Annual Temperature", "temperature_annual_rgba")
                window.associate_scheme_texture("Summer Temperature", "temperature_summer_rgba")
                window.associate_scheme_texture("Winter Temperature", "temperature_winter_rgba")
                window.associate_scheme_texture("Annual Precipitation", "precipitation_rgba")
                window.register_legend("Annual Temperature", "gradient", temperature_legend_entries())
                window.register_legend("Summer Temperature", "gradient", temperature_legend_entries())
                window.register_legend("Winter Temperature", "gradient", temperature_legend_entries())
                window.register_legend("Annual Precipitation", "gradient", precipitation_legend_entries())
        if hb.rivers:
            window.add_overlay("rivers", label="Rivers")
        if debug and cb.winds is not None:
            for i, label in enumerate(SEASON_LABELS):
                window.add_overlay(f"winds_season_{i}", label=f"Winds — {label}", checked=False)
        if debug and cb.currents is not None:
            window.add_overlay("currents", label="Currents", checked=False)
        if debug and self.globe is not None and self.globe.has_feature("deltas"):
            window.add_overlay("deltas", label="River Deltas", checked=False)
        if debug and self.globe is not None and self.globe.has_feature("volcanoes"):
            window.add_overlay("volcanoes", label="Volcanoes", checked=False)

        if debug:
            from worldgen.climate.classification import BIOME_NAMES

            def _hover_cb(face_src: Optional[int], uv: Optional[np.ndarray]) -> None:
                if face_src is None:
                    window.set_status("")
                    return
                meta = mesh.face_metadata[face_src]
                parts: list[str] = [f"Face {face_src}"]
                if "elevation" in meta:
                    parts.append(f"Elev: {meta['elevation']:.2f} km")
                if "plate_type" in meta:
                    parts.append(str(meta["plate_type"]))
                if "boundary_type" in meta:
                    bt = str(meta["boundary_type"])
                    if bt != "interior":
                        parts.append(bt)
                if "biome" in meta:
                    parts.append(BIOME_NAMES[int(meta["biome"])])
                if "temperature_annual" in meta:
                    parts.append(f"T: {meta['temperature_annual']:.1f}°C")
                if "precipitation_annual" in meta:
                    parts.append(f"P: {meta['precipitation_annual']:.0f} mm/yr")
                if uv is not None and "elevation" in mesh.face_textures:
                    elev_tex = mesh.face_textures["elevation"]
                    H_t, W_t = elev_tex.shape[1:3]
                    tx = int(np.clip(uv[0] * W_t, 0, W_t - 1))
                    ty = int(np.clip(uv[1] * H_t, 0, H_t - 1))
                    px_elev = float(elev_tex[face_src, ty, tx, 0])
                    parts.append(f"[px: {px_elev:.2f} km]")
                window.set_status("  |  ".join(parts))

            window.set_hover_callback(_hover_cb)

        _ = BIOME_PALETTE  # re-exported for downstream consumers
