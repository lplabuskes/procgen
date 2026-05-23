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
)

from worldgen.geology.elevation_builder import ElevationBuilder, ElevationParams
from worldgen.hydrology.builder import HydrologyBuilder, HydrologyParams, LAKE_COLOR
from worldgen.globe.globe import Globe


def build_elevation_display_texture(mesh) -> np.ndarray:
    elev_tex = mesh.face_textures["elevation"]          # (N, H, W, 1)
    rgba = _elevation_color_vec(elev_tex[:, :, :, 0])   # (N, H, W, 4)
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
    hydrology: HydrologyParams = field(default_factory=HydrologyParams, metadata={"label": "Hydrology"})


class GlobeBuilder:
    """Orchestrates every stage of globe generation.

    Each stage's builder is stored as an attribute (`elevation_builder`,
    `hydrology_builder`) so callers can read intermediate results without
    re-running work. `build()` returns a `Globe` wrapping the resulting mesh.
    """

    def __init__(self, params: GlobeParams | None = None) -> None:
        self.params: GlobeParams = params or GlobeParams()
        self.elevation_builder: ElevationBuilder = ElevationBuilder(self.params.elevation)
        self.hydrology_builder: HydrologyBuilder = HydrologyBuilder(self.params.hydrology)
        self.globe: Optional[Globe] = None

    def build(self, progress: Optional[Callable[[str], None]] = None) -> Globe:
        mesh = self.elevation_builder.build(progress=progress)

        elevation_textures = mesh.face_textures["elevation"]  # (N, H, W, 1)
        self.hydrology_builder.build(mesh, elevation_textures, progress=progress)

        globe = Globe(mesh=mesh)
        hb = self.hydrology_builder

        if hb.rivers:
            paths_with_flow = []
            for river in hb.rivers:
                polyline = river.as_polyline()
                if len(polyline) < 2:
                    continue
                positions = np.stack(polyline).astype(np.float32)  # (M, 3)
                flows = river.flow_rates()                          # length M-1
                paths_with_flow.append((positions, flows))
            if paths_with_flow:
                globe.add_feature("rivers", paths_with_flow)

        if hb.lake_polygons:
            globe.add_feature("lakes", hb.lake_polygons)

        self.globe = globe
        return globe

    def apply_display(self, window) -> None:
        """Wire color schemes and overlays into a MapWindow after build()."""
        eb = self.elevation_builder
        hb = self.hydrology_builder
        assert eb.mesh is not None, "GlobeBuilder.build() must be called before apply_display"

        mesh = eb.mesh
        palette = _plate_palette(eb.params.plates.n_plates)
        max_speed = eb.max_speed

        mesh.set_face_colors(palette[np.array(eb.plate_ids, dtype=np.int32)])

        elevation_rgba = build_elevation_display_texture(mesh)
        mesh.set_face_textures("elevation_rgba", elevation_rgba)

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

        schemes: dict[str, Callable[[dict], tuple]] = {
            "Elevation": _elevation_color,
            "Plates": lambda meta: tuple(palette[meta["plate_id"]]),
            "Plate Type": lambda meta: OCEAN_COLOR if meta["plate_type"] == "oceanic" else LAND_COLOR,
            "Speed": _speed_color,
            "Boundary": lambda meta: BOUNDARY_COLORS[meta["boundary_type"]],
            "Volcanic Activity": _volcanic_color,
        }

        window.set_color_schemes(schemes)
        window.associate_scheme_texture("Elevation", "elevation_rgba")
        if hb.rivers:
            window.add_overlay("rivers", label="Rivers")
