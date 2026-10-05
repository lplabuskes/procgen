from __future__ import annotations

from typing import List, Mapping, Optional

import numpy as np

from worldgen.climate.channels import PRECIPITATION, TEMPERATURE
from worldgen.geology.channels import ELEVATION, OCEAN_MASK
from worldgen.hydrology.channels import LAKE_MASK, RIVER_OBJECTS, RIVERS, STREAMS
from worldgen.hydrology.rivers import River, RiverGenerator
from worldgen.hydrology.types import StreamNetwork
from core.display import DisplayRegistry
from core.params import param_field as _f, paramclass
from core.pipeline import BuildContext, Stage
from core.world import World


LAKE_COLOR = np.array([0.28, 0.65, 0.90, 1.0], dtype=np.float32)


@paramclass
class HydrologyParams:
    river_flow_threshold: float = _f(500.0, label="Flow threshold (faces)", min=1.0, max=1000.0)
    # The same accumulation tree cut lower: streams for town siting and region
    # tributaries. Units ≈ upstream texels at mean rainfall.
    stream_flow_threshold: float = _f(100.0, label="Stream threshold (faces)", min=1.0, max=1000.0)
    min_lake_pixels: int = _f(50, label="Min lake pixels", min=1, max=500)
    pet_baseline_mm_yr: float = _f(100.0, label="PET baseline (mm/yr)", min=0.0, max=500.0)
    pet_per_degree_mm_yr: float = _f(60.0, label="PET slope (mm/yr/°C)", min=0.0, max=200.0)
    pet_max_mm_yr: float = _f(2500.0, label="PET cap (mm/yr)", min=500.0, max=5000.0)


class HydrologyBuilder(Stage):
    """Hydrology stage: river routing and lake detection.

    Writes `filled_height` / `lake_mask` textures and the ``"rivers"``,
    ``"river_objects"``, ``"streams"`` and ``"lakes"`` features.  The delta stage
    later extends rivers and streams in place.
    """

    name = "hydrology"

    def __init__(self, params: HydrologyParams | None = None) -> None:
        self.params: HydrologyParams = params or HydrologyParams()
        self.rivers: Optional[List[River]] = None
        self.lake_polygons: Optional[List[np.ndarray]] = None
        self.streams: Optional[StreamNetwork] = None
        # The "rivers" feature tuples; cached since ``serialize`` has no world access.
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
        self.streams = river_gen.stream_network(self.params.stream_flow_threshold)
        STREAMS.write(ctx.world, self.streams)

        filled_height = river_gen.get_filled_height_texture()  # (N, H, W, 1)
        lake_mask = river_gen.get_lake_mask_texture()          # (N, H, W, 1)
        mesh.set_face_textures("filled_height", filled_height.astype(np.float16), n_channels=1)
        LAKE_MASK.write(mesh, lake_mask)

        if self.rivers:
            RIVER_OBJECTS.write(ctx.world, self.rivers)
            paths_with_flow = self._flatten_rivers()
            if paths_with_flow:
                self._river_feature = paths_with_flow
                RIVERS.write(ctx.world, paths_with_flow)
        if self.lake_polygons:
            ctx.world.add_feature("lakes", self.lake_polygons)

    def _flatten_rivers(self) -> list:
        """Flatten ``self.rivers`` into ``(positions, flows, in_lake)`` polyline tuples."""
        if self.rivers is None:
            return []
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

    # ── Persistence ──────────────────────────────────────────────────────────

    def serialize(self) -> dict:
        arrays: dict = {}
        # Re-flatten built rivers: the delta stage extends them after build.
        if self.rivers and isinstance(self.rivers[0], River):
            self._river_feature = self._flatten_rivers()
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
        if self.streams is not None:
            arrays.update(self.streams.pack())
        return arrays

    def load(self, arrays: Mapping[str, np.ndarray], ctx: BuildContext) -> None:
        if "river_pos_offsets" in arrays:
            pos_flat = arrays["river_positions_flat"]
            flows_flat = arrays["river_flows_flat"]
            in_lake_flat = arrays["river_in_lake_flat"]
            pos_offs = arrays["river_pos_offsets"]
            flow_offs = arrays["river_flow_offsets"]
            # Loaded rivers stay flattened tuples (no downstream stage re-runs).
            paths_with_flow = [
                (
                    pos_flat[pos_offs[i]:pos_offs[i + 1]],
                    flows_flat[flow_offs[i]:flow_offs[i + 1]],
                    in_lake_flat[flow_offs[i]:flow_offs[i + 1]],
                )
                for i in range(len(pos_offs) - 1)
            ]
            self.rivers = paths_with_flow  # type: ignore[assignment]
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
        if "stream_nodes" in arrays:
            self.streams = StreamNetwork.unpack(arrays)
            STREAMS.write(ctx.world, self.streams)
