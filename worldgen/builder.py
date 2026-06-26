from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from core.mesh_cache import MeshParams
from core.persistence import load_world, save_world
from core.pipeline import BuildContext, Pipeline
from core.world import World
from worldgen.climate.builder import ClimateBuilder, ClimateParams
from worldgen.cost.builder import CostBuilder, CostParams
from worldgen.danger.builder import DangerBuilder, DangerParams
from worldgen.geology.elevation_builder import ElevationBuilder, ElevationParams
from worldgen.hydrology.builder import HydrologyBuilder, HydrologyParams
from worldgen.resources.builder import ResourcesBuilder, ResourcesParams

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
    resources: ResourcesParams = field(default_factory=ResourcesParams, metadata={"label": "Resources"})
    danger: DangerParams = field(default_factory=DangerParams, metadata={"label": "Danger"})
    cost: CostParams = field(default_factory=CostParams, metadata={"label": "Cost"})
    # World seed — drives mesh selection and every stage's draws. None → random each run.
    seed: Optional[int] = field(
        default=None, metadata={"label": "Seed", "kind": "optional_int", "min": 0, "max": 2**31 - 1}
    )


class GlobePipeline(Pipeline):
    def __init__(self, params: GlobeParams | None = None) -> None:
        self.params: GlobeParams = params or GlobeParams()
        self.elevation_builder: ElevationBuilder = ElevationBuilder(
            self.params.elevation, seed=self.params.seed
        )
        self.climate_builder: ClimateBuilder = ClimateBuilder(self.params.climate)
        self.hydrology_builder: HydrologyBuilder = HydrologyBuilder(self.params.hydrology)
        self.resources_builder: ResourcesBuilder = ResourcesBuilder(self.params.resources)
        self.danger_builder: DangerBuilder = DangerBuilder(self.params.danger)
        self.cost_builder: CostBuilder = CostBuilder(self.params.cost)

        # Lake-feedback second pass is placed after hydrology, sharing the climate builder.
        super().__init__([
            self.elevation_builder,
            self.climate_builder,
            self.hydrology_builder,
            self.climate_builder.lake_feedback_stage(),
            self.resources_builder,
            self.danger_builder,
            self.cost_builder,
        ])

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
