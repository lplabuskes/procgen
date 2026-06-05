from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from worldgen.hydrology.rivers import River, RiverGenerator
from core.mesh import PolygonMesh
from utils.params import param_field as _f, paramclass


LAKE_COLOR = np.array([0.28, 0.65, 0.90, 1.0], dtype=np.float32)


@paramclass
class HydrologyParams:
    river_flow_threshold: float = _f(200.0, label="Flow threshold (faces)", min=1.0, max=1000.0)
    min_lake_pixels: int = _f(50, label="Min lake pixels", min=1, max=500)
    pet_baseline_mm_yr: float = _f(100.0, label="PET baseline (mm/yr)", min=0.0, max=500.0)
    pet_per_degree_mm_yr: float = _f(60.0, label="PET slope (mm/yr/°C)", min=0.0, max=200.0)
    pet_max_mm_yr: float = _f(2500.0, label="PET cap (mm/yr)", min=500.0, max=5000.0)
    delta_min_flow: float = _f(1000.0, label="Delta min flow", min=0.0, max=10000.0)
    delta_min_length_km: float = _f(500.0, label="Delta min river length (km)", min=0.0, max=5000.0)


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
        self.delta_positions: Optional[np.ndarray] = None  # (M, 3) mouth positions reaching the ocean

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
        mesh.set_face_textures("filled_height", filled_height.astype(np.float16), n_channels=1)
        mesh.set_face_textures("lake_mask", lake_mask, n_channels=1)

        delta_pts: list[np.ndarray] = []
        res = river_gen.resolution
        for river in self.rivers:
            if river.mouth is None or not river.segments:
                continue
            mouth_flow = river.segments[-1].flow_rate
            if mouth_flow < self.params.delta_min_flow:
                continue
            polyline = river.as_polyline()
            if len(polyline) >= 2:
                pts = np.stack(polyline)
                dots = np.clip(np.einsum("ij,ij->i", pts[:-1], pts[1:]), -1.0, 1.0)
                length_km = float(np.arccos(dots).sum()) * 6371.0
            else:
                length_km = 0.0
            if length_km < self.params.delta_min_length_km:
                continue
            face_id = river.mouth.face_id
            u, v = river.mouth.uv
            row = int(round(v * (res - 1)))
            col = int(round(u * (res - 1)))
            node_idx = river_gen._node_idx(face_id, row, col)
            if river_gen._ocean_mask[node_idx]:
                mesh.set_face_metadata(face_id, is_delta=True)
                delta_pts.append(river.mouth.position)
        self.delta_positions = (
            np.stack(delta_pts, axis=0).astype(np.float32) if delta_pts else None
        )
