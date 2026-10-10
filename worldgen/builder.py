from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from core.mesh_cache import MeshParams
from core.persistence import load_world, save_world
from core.pipeline import BuildContext, Pipeline
from core.world import World
from worldgen.climate.builder import ClimateBuilder, ClimateParams
from worldgen.coast.builder import CoastBuilder, CoastParams
from worldgen.cost.builder import CostBuilder, CostParams
from worldgen.danger.builder import DangerBuilder, DangerParams
from worldgen.delta.builder import DeltaBuilder, DeltaParams
from worldgen.geology.elevation import clear_raster_stencils
from worldgen.hydrology.rivers import clear_texel_graphs
from worldgen.geology.elevation_builder import ElevationBuilder, ElevationParams
from worldgen.hydrology.builder import HydrologyBuilder, HydrologyParams
from worldgen.resources.builder import ResourcesBuilder, ResourcesParams
from worldgen.wetland.builder import WetlandBuilder, WetlandParams

__all__ = [
    "GlobeParams",
    "GlobePipeline",
    "save_globe",
    "load_globe",
]


@dataclass
class GlobeParams:
    mesh: MeshParams = field(default_factory=MeshParams, metadata={"label": "Mesh Generation"})
    elevation: ElevationParams = field(default_factory=ElevationParams)
    climate: ClimateParams = field(default_factory=ClimateParams, metadata={"label": "Climate"})
    hydrology: HydrologyParams = field(default_factory=HydrologyParams, metadata={"label": "Hydrology"})
    delta: DeltaParams = field(default_factory=DeltaParams, metadata={"label": "River deltas"})
    coast: CoastParams = field(default_factory=CoastParams, metadata={"label": "Coast style"})
    wetland: WetlandParams = field(default_factory=WetlandParams, metadata={"label": "Wetlands"})
    resources: ResourcesParams = field(default_factory=ResourcesParams, metadata={"label": "Resources"})
    danger: DangerParams = field(default_factory=DangerParams, metadata={"label": "Danger"})
    cost: CostParams = field(default_factory=CostParams, metadata={"label": "Cost"})
    # World seed — drives mesh selection and every stage's draws. None → random each run.
    # The app sets it from ``AppParams.seed`` (top of the dialog), so it isn't shown here.
    seed: Optional[int] = field(default=None, metadata={"label": "Seed", "hidden": True})


class GlobePipeline(Pipeline):
    def __init__(self, params: GlobeParams | None = None) -> None:
        self.params: GlobeParams = params or GlobeParams()
        self.elevation_builder: ElevationBuilder = ElevationBuilder(
            self.params.elevation, seed=self.params.seed
        )
        self.climate_builder: ClimateBuilder = ClimateBuilder(self.params.climate)
        self.hydrology_builder: HydrologyBuilder = HydrologyBuilder(self.params.hydrology)
        self.delta_builder: DeltaBuilder = DeltaBuilder(self.params.delta)
        self.coast_builder: CoastBuilder = CoastBuilder(self.params.coast)
        # Shares hydrology's PET, so lakes and wetlands see one water balance.
        self.wetland_builder: WetlandBuilder = WetlandBuilder(self.params.wetland, pet=self.params.hydrology)
        self.resources_builder: ResourcesBuilder = ResourcesBuilder(self.params.resources)
        self.danger_builder: DangerBuilder = DangerBuilder(self.params.danger)
        self.cost_builder: CostBuilder = CostBuilder(
            self.params.cost, river_flow_threshold=self.params.hydrology.river_flow_threshold
        )

        # Lake feedback runs after the deltas so the land they build gets its own climate;
        # wetlands after it, as it reclassifies the biomes they are stamped onto.
        super().__init__([
            self.elevation_builder,
            self.climate_builder,
            self.hydrology_builder,
            self.delta_builder,
            self.climate_builder.lake_feedback_stage(),
            self.wetland_builder,
            self.coast_builder,
            self.resources_builder,
            self.danger_builder,
            self.cost_builder,
        ])

    def build(self, ctx: BuildContext) -> None:
        super().build(ctx)
        clear_raster_stencils(ctx.world.mesh)
        clear_texel_graphs(ctx.world.mesh)

    @property
    def globe(self) -> Optional[World]:
        return self.world


def save_globe(pipeline: GlobePipeline, path: Union[str, Path]) -> None:
    if pipeline.globe is None:
        raise RuntimeError("GlobePipeline.build() must be called before save_globe()")
    save_world(pipeline.globe, pipeline.params, pipeline, path)


def load_globe(path: Union[str, Path]) -> tuple[GlobePipeline, World]:
    params, world, data = load_world(path, GlobeParams)
    pipeline = GlobePipeline(params)
    pipeline.load(data, BuildContext(world=world))
    return pipeline, world
