from __future__ import annotations

from typing import List, Mapping, Optional

import numpy as np

from worldgen.climate.channels import PRECIPITATION, TEMPERATURE
from worldgen.geology.channels import ELEVATION, OCEAN_MASK
from worldgen.hydrology.channels import LAKE_MASK, RIVER_OBJECTS, RIVERS
from worldgen.hydrology.rivers import River, RiverGenerator
from core.display import DisplayRegistry
from core.params import param_field as _f, paramclass
from core.pipeline import BuildContext, Stage
from core.world import World


LAKE_COLOR = np.array([0.28, 0.65, 0.90, 1.0], dtype=np.float32)


@paramclass
class HydrologyParams:
    river_flow_threshold: float = _f(100.0, label="Flow threshold (faces)", min=1.0, max=1000.0)
    min_lake_pixels: int = _f(50, label="Min lake pixels", min=1, max=500)
    pet_baseline_mm_yr: float = _f(100.0, label="PET baseline (mm/yr)", min=0.0, max=500.0)
    pet_per_degree_mm_yr: float = _f(60.0, label="PET slope (mm/yr/°C)", min=0.0, max=200.0)
    pet_max_mm_yr: float = _f(2500.0, label="PET cap (mm/yr)", min=500.0, max=5000.0)
    delta_min_flow: float = _f(1000.0, label="Delta min flow", min=0.0, max=10000.0)
    delta_min_length_km: float = _f(500.0, label="Delta min river length (km)", min=0.0, max=5000.0)


class HydrologyBuilder(Stage):
    """Runs the hydrology stage (river routing, lake detection) — a pipeline ``Stage``.

    ``build(ctx)`` reads the mesh and its ``"elevation"`` texture from the world,
    stores rivers / lake polygons, attaches `filled_height` and `lake_mask`
    single-channel face textures, and registers the ``"rivers"`` (flattened
    polylines for rendering/serialization), ``"river_objects"`` (the rich
    ``River`` objects downstream stages consume), ``"lakes"``, and ``"deltas"``
    features.
    """

    name = "hydrology"

    def __init__(self, params: HydrologyParams | None = None) -> None:
        self.params: HydrologyParams = params or HydrologyParams()
        self.rivers: Optional[List[River]] = None
        self.lake_polygons: Optional[List[np.ndarray]] = None
        self.delta_positions: Optional[np.ndarray] = None  # (M, 3) mouth positions reaching the ocean
        # The flattened ``(positions, flows, in_lake)`` polyline tuples registered
        # as the "rivers" feature. Cached here because ``serialize`` (no world
        # access) must reproduce it from either a fresh build (River objects) or a
        # load (already-flattened tuples).
        self._river_feature: Optional[list] = None

    def build(self, ctx: BuildContext) -> None:
        """Run the hydrology pipeline, attach results to the mesh, register features."""
        mesh = ctx.mesh
        elevation_textures = ELEVATION.read(mesh)
        _report = ctx.report

        _report("Computing rivers and lakes…")
        precip_tex = PRECIPITATION.read(mesh) if PRECIPITATION.present(mesh) else None
        temperature_tex = TEMPERATURE.read(mesh) if TEMPERATURE.present(mesh) else None
        ocean_mask_tex = OCEAN_MASK.read(mesh) if OCEAN_MASK.present(mesh) else None
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
        LAKE_MASK.write(mesh, lake_mask)

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

        if self.rivers:
            # Rich River objects for downstream stages (resources, cost); the
            # flattened polylines for rendering + serialization.
            RIVER_OBJECTS.write(ctx.world, self.rivers)
            paths_with_flow = self._flatten_rivers()
            if paths_with_flow:
                self._river_feature = paths_with_flow
                RIVERS.write(ctx.world, paths_with_flow)
        if self.lake_polygons:
            ctx.world.add_feature("lakes", self.lake_polygons)
        if self.delta_positions is not None and len(self.delta_positions) > 0:
            ctx.world.add_feature("deltas", self.delta_positions)

    def _flatten_rivers(self) -> list:
        """Flatten ``self.rivers`` (List[River]) into ``(positions, flows,
        in_lake)`` polyline tuples — the form used by both the world feature and
        the serialized arrays."""
        paths = []
        for river in self.rivers:
            polyline = river.as_polyline()
            if len(polyline) < 2:
                continue
            positions = np.stack(polyline).astype(np.float32)
            flows = river.flow_rates()
            in_lake = np.array([s.in_lake for s in river.segments], dtype=bool)
            paths.append((positions, flows, in_lake))
        return paths

    # ── Display ──────────────────────────────────────────────────────────────

    def register_display(
        self, world: World, reg: DisplayRegistry, *, debug: bool = False
    ) -> None:
        if world.has_feature("rivers"):
            reg.add_overlay("rivers", "Rivers")
        if debug and world.has_feature("deltas"):
            reg.add_overlay("deltas", "River Deltas", checked=False)

    # ── Persistence ──────────────────────────────────────────────────────────

    def serialize(self) -> dict:
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
        if self.lake_polygons:
            all_verts, lake_offsets = [], [0]
            for poly in self.lake_polygons:
                all_verts.append(poly)
                lake_offsets.append(lake_offsets[-1] + len(poly))
            arrays["lake_polys_flat"] = np.concatenate(all_verts).astype(np.float32)
            arrays["lake_poly_offsets"] = np.array(lake_offsets, dtype=np.int64)
        if self.delta_positions is not None and len(self.delta_positions) > 0:
            arrays["hb_delta_positions"] = self.delta_positions.astype(np.float32)
        return arrays

    def load(self, arrays: Mapping[str, np.ndarray], ctx: BuildContext) -> None:
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
            self.rivers = paths_with_flow
            self._river_feature = paths_with_flow
            RIVERS.write(ctx.world, paths_with_flow)
        if "lake_poly_offsets" in arrays:
            poly_flat = arrays["lake_polys_flat"]
            poly_offs = arrays["lake_poly_offsets"]
            self.lake_polygons = [
                poly_flat[poly_offs[i]:poly_offs[i + 1]]
                for i in range(len(poly_offs) - 1)
            ]
            ctx.world.add_feature("lakes", self.lake_polygons)
        if "hb_delta_positions" in arrays:
            self.delta_positions = arrays["hb_delta_positions"]
            ctx.world.add_feature("deltas", self.delta_positions)
