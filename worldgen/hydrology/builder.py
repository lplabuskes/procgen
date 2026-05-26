from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from worldgen.hydrology.rivers import River, RiverGenerator
from worldgen.mesh.mymesh import PolygonMesh


LAKE_COLOR = np.array([0.28, 0.65, 0.90, 1.0], dtype=np.float32)


def _f(default, *, label: str, min: float, max: float, step: float | None = None,
       hidden: bool = False):
    meta = {"label": label, "min": min, "max": max}
    if step is not None:
        meta["step"] = step
    if hidden:
        meta["hidden"] = True
    return field(default=default, metadata=meta)


@dataclass
class HydrologyParams:
    river_flow_threshold: float = _f(200.0, label="Flow threshold (faces)", min=1.0, max=1000.0)
    min_lake_pixels: int = _f(50, label="Min lake pixels", min=1, max=500)
    pet_baseline_mm_yr: float = _f(100.0, label="PET baseline (mm/yr)", min=0.0, max=500.0)
    pet_per_degree_mm_yr: float = _f(60.0, label="PET slope (mm/yr/°C)", min=0.0, max=200.0)
    pet_max_mm_yr: float = _f(2500.0, label="PET cap (mm/yr)", min=500.0, max=5000.0)


class HydrologyBuilder:
    """Runs the hydrology stage (river routing, lake detection) on a built mesh.

    Call `build(mesh, elevation_textures)` after `ElevationBuilder.build()`.
    Stores rivers, lake polygons, and attaches `filled_height` and `lake_mask`
    as single-channel face textures on the mesh.
    """

    def __init__(self, params: HydrologyParams | None = None) -> None:
        self.params: HydrologyParams = params or HydrologyParams()
        self.rivers: Optional[List[River]] = None
        self.lake_polygons: Optional[List[np.ndarray]] = None

    def build(
        self,
        mesh: PolygonMesh,
        elevation_textures: np.ndarray,
        progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Run the hydrology pipeline and attach results to the mesh."""
        def _report(msg: str) -> None:
            if progress is not None:
                progress(msg)

        _report("Computing rivers and lakes…")
        precip_tex = mesh.face_textures.get("precipitation")
        temperature_tex = mesh.face_textures.get("temperature")
        ocean_mask_tex = mesh.face_textures.get("ocean_mask")
        river_gen = RiverGenerator(
            mesh,
            elevation_textures,
            min_flow=self.params.river_flow_threshold,
            sea_level=0.0,
            min_lake_pixels=self.params.min_lake_pixels,
            precipitation_textures=precip_tex,
            temperature_textures=temperature_tex,
            pet_baseline_mm_yr=self.params.pet_baseline_mm_yr,
            pet_per_degree_mm_yr=self.params.pet_per_degree_mm_yr,
            pet_max_mm_yr=self.params.pet_max_mm_yr,
            ocean_mask_textures=ocean_mask_tex,
        )
        self.rivers = river_gen.generate()
        self.lake_polygons = river_gen.get_lake_polygons()

        filled_height = river_gen.get_filled_height_texture()  # (N, H, W, 1)
        lake_mask = river_gen.get_lake_mask_texture()          # (N, H, W, 1)
        mesh.set_face_textures("filled_height", filled_height, n_channels=1)
        mesh.set_face_textures("lake_mask", lake_mask, n_channels=1)
