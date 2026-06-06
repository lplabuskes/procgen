from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.display import DisplayRegistry
from core.pipeline import BuildContext, Pipeline
from worldgen.climate.builder import ClimateBuilder, ClimateParams
from worldgen.cost.builder import CostBuilder, CostParams
from worldgen.geology.elevation_builder import ElevationBuilder, ElevationParams
from worldgen.hydrology.builder import HydrologyBuilder, HydrologyParams
from worldgen.resources.builder import ResourcesBuilder, ResourcesParams
from core.world import World

__all__ = [
    "GlobeParams",
    "GlobePipeline",
]


@dataclass
class GlobeParams:
    """Top-level parameters for the full globe-generation pipeline.

    Stage-specific params are nested. New stages add their own params
    dataclass as a new field here.
    """
    elevation: ElevationParams = field(default_factory=ElevationParams)
    climate: ClimateParams = field(default_factory=ClimateParams, metadata={"label": "Climate"})
    hydrology: HydrologyParams = field(default_factory=HydrologyParams, metadata={"label": "Hydrology"})
    resources: ResourcesParams = field(default_factory=ResourcesParams, metadata={"label": "Resources"})
    cost: CostParams = field(default_factory=CostParams, metadata={"label": "Cost"})


class GlobePipeline(Pipeline):
    """Ordered pipeline of every worldgen stage.

    Each stage builder is stored as an attribute so callers can read
    intermediate results without re-running work. Call ``build(ctx)``
    with a shared ``BuildContext`` to run all stages in order.
    """

    def __init__(self, params: GlobeParams | None = None) -> None:
        self.params: GlobeParams = params or GlobeParams()
        self.elevation_builder: ElevationBuilder = ElevationBuilder(self.params.elevation)
        self.climate_builder: ClimateBuilder = ClimateBuilder(self.params.climate)
        self.hydrology_builder: HydrologyBuilder = HydrologyBuilder(self.params.hydrology)
        self.resources_builder: ResourcesBuilder = ResourcesBuilder(self.params.resources)
        self.cost_builder: CostBuilder = CostBuilder(self.params.cost)

        # The climate lake-feedback second pass is an explicit node placed after
        # hydrology, sharing the climate builder instance.
        super().__init__([
            self.elevation_builder,
            self.climate_builder,
            self.hydrology_builder,
            self.climate_builder.lake_feedback_stage(),
            self.resources_builder,
            self.cost_builder,
        ])
        self.globe: Optional[World] = None

    def build(self, ctx: BuildContext) -> None:
        super().build(ctx)
        self.globe = ctx.world

    def apply_display(self, window, debug: bool = False) -> None:
        """Wire color schemes, overlays, and hover into a MapWindow after build().

        Builds a ``DisplayRegistry`` by asking each pipeline stage to contribute
        its slice, then hands the registry to the viz adapter for Qt wiring. The
        adapter is imported lazily so this module stays Qt-free.
        """
        from viz.mapview.registry import apply_registry

        world = self.globe
        assert world is not None and world.mesh is not None, (
            "GlobePipeline.build() must be called before apply_display"
        )
        reg = DisplayRegistry()
        for stage in self:
            stage.register_display(world, reg, debug=debug)
        apply_registry(reg, window, world)
