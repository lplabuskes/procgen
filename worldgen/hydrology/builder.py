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
        """Run the hydrology pipeline and attach results to the mesh.

        Args:
            mesh: The dual polygon mesh from the elevation stage.
            elevation_textures: Single-channel (N, H, W, 1) float32 elevation
                texture from the elevation stage; channel 0 = elevation in km.
            progress: Optional callback for status messages.
        """
        def _report(msg: str) -> None:
            if progress is not None:
                progress(msg)

        _report("Computing rivers and lakes…")
        river_gen = RiverGenerator(
            mesh,
            elevation_textures,
            min_flow=self.params.river_flow_threshold,
            sea_level=0.0,
        )
        self.rivers = river_gen.generate()
        self.lake_polygons = river_gen.get_lake_polygons()

        filled_height = river_gen.get_filled_height_texture()  # (N, H, W, 1)
        lake_mask = river_gen.get_lake_mask_texture()          # (N, H, W, 1)
        mesh.set_face_textures("filled_height", filled_height, n_channels=1)
        mesh.set_face_textures("lake_mask", lake_mask, n_channels=1)
