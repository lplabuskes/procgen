from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from core.display import DisplayRegistry
from core.pipeline import BuildContext, Pipeline
from core.world import World
from worldgen.climate.builder import ClimateBuilder, ClimateParams
from worldgen.climate.display import build_biome_display_texture
from worldgen.cost.builder import CostBuilder, CostParams
from worldgen.geology.display import build_elevation_display_texture
from worldgen.geology.elevation_builder import ElevationBuilder, ElevationParams
from worldgen.hydrology.builder import HydrologyBuilder, HydrologyParams
from worldgen.resources.builder import ResourcesBuilder, ResourcesParams
from worldgen.globe.globe import Globe

# Re-exported for back-compat (these display-texture helpers live in the
# owning stages' display modules; re-exported here for existing importers).
__all__ = [
    "GlobeParams",
    "GlobeBuilder",
    "build_biome_display_texture",
    "build_elevation_display_texture",
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


class GlobeBuilder:
    """Orchestrates every stage of globe generation.

    Each stage's builder is stored as an attribute (`elevation_builder`,
    `climate_builder`, `hydrology_builder`) so callers can read intermediate
    results without re-running work. `build()` runs the pipeline and returns a
    `World` (aliased `Globe`).
    """

    def __init__(self, params: GlobeParams | None = None) -> None:
        self.params: GlobeParams = params or GlobeParams()
        self.elevation_builder: ElevationBuilder = ElevationBuilder(self.params.elevation)
        self.climate_builder: ClimateBuilder = ClimateBuilder(self.params.climate)
        self.hydrology_builder: HydrologyBuilder = HydrologyBuilder(self.params.hydrology)
        self.resources_builder: ResourcesBuilder = ResourcesBuilder(self.params.resources)
        self.cost_builder: CostBuilder = CostBuilder(self.params.cost)

        # Compose the generation pipeline. Each builder is itself a ``Stage`` that
        # reads its inputs from the ``BuildContext`` and registers its own
        # features; the climate lake-feedback second pass is an explicit node
        # placed after hydrology (sharing the climate builder instance).
        self.pipeline: Pipeline = Pipeline([
            self.elevation_builder,
            self.climate_builder,
            self.hydrology_builder,
            self.climate_builder.lake_feedback_stage(),
            self.resources_builder,
            self.cost_builder,
        ])
        self.globe: Optional[Globe] = None

    def build(self, progress: Optional[Callable[[str], None]] = None) -> Globe:
        """Run every stage in order and return the populated world.

        The elevation stage sets ``world.mesh``; each subsequent stage reads it
        from the context and registers its own features. The stage builders
        remain accessible as ``GlobeBuilder`` attributes for the display and
        serialization paths.
        """
        world = World()
        ctx = BuildContext(world=world)
        if progress is not None:
            ctx.progress = progress
        self.pipeline.build(ctx)
        self.globe = world
        return world

    def apply_display(self, window, debug: bool = False) -> None:
        """Wire color schemes, overlays, and hover into a MapWindow after build().

        Builds a ``DisplayRegistry`` by asking each pipeline stage to contribute
        its slice, then hands the registry to the viz adapter for Qt wiring. The
        adapter is imported lazily so this module stays Qt-free.
        """
        from utils.mapview.registry import apply_registry

        world = self.globe
        assert world is not None and world.mesh is not None, (
            "GlobeBuilder.build() must be called before apply_display"
        )
        reg = DisplayRegistry()
        for stage in self.pipeline:
            stage.register_display(world, reg, debug=debug)
        apply_registry(reg, window, world)
