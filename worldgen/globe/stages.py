"""Pipeline-stage wrappers for the worldgen physical stages.

Each wrapper adapts an existing ``*Builder`` to the core ``Stage`` protocol:
``build(ctx)`` runs the builder against ``ctx.world`` and registers that stage's
vector/point features on the world. The builders are unchanged — they hold the
real logic and remain accessible as ``stage.builder`` (and as ``GlobeBuilder``
attributes) so the current display/serialization paths keep working.

The climate lake-feedback second pass is modelled as its own pipeline node
(``ClimateLakeFeedbackStage``) sharing the ``ClimateBuilder`` instance, so the
ordering — *after* hydrology — is explicit in the pipeline list rather than an
``if "lake_mask" in textures`` branch in the orchestrator.

(These wrappers currently live together for a low-churn migration; they can move
into their owning packages later without changing the pipeline.)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from core.pipeline import BuildContext, Stage
from worldgen.climate.builder import ClimateBuilder
from worldgen.cost.builder import CostBuilder
from worldgen.geology.elevation_builder import ElevationBuilder
from worldgen.hydrology.builder import HydrologyBuilder
from worldgen.resources.builder import ResourcesBuilder


class ElevationStage(Stage):
    """Builds the mesh and the elevation/tectonics layers; sets ``world.mesh``."""

    name = "elevation"

    def __init__(self, builder: ElevationBuilder) -> None:
        self.builder = builder

    def build(self, ctx: BuildContext) -> None:
        mesh = self.builder.build(progress=ctx.report)
        ctx.world.mesh = mesh
        eb = self.builder
        if eb.volcano_positions is not None and len(eb.volcano_positions) > 0:
            ctx.world.add_feature("volcanoes", eb.volcano_positions)


class ClimateStage(Stage):
    """Runs the first climate pass; registers wind and current features."""

    name = "climate"

    def __init__(self, builder: ClimateBuilder) -> None:
        self.builder = builder

    def build(self, ctx: BuildContext) -> None:
        mesh = ctx.mesh
        self.builder.build(mesh, mesh.face_textures["elevation"], progress=ctx.report)
        cb = self.builder
        centroids = mesh.face_centroids.astype(np.float32)
        if cb.winds is not None:
            winds_feature = [
                (centroids.copy(), cb.winds[:, s, :].astype(np.float32))
                for s in range(cb.winds.shape[1])
            ]
            ctx.world.add_feature("winds", winds_feature)
        if cb.currents is not None:
            ctx.world.add_feature(
                "currents",
                [(
                    centroids.copy(),
                    cb.currents.currents.astype(np.float32),
                    cb.currents.temperature_delta.astype(np.float32),
                )],
            )

    def lake_feedback_stage(self) -> "ClimateLakeFeedbackStage":
        return ClimateLakeFeedbackStage(self.builder)


class ClimateLakeFeedbackStage(Stage):
    """Second climate pass: lake evaporation feedback (runs after hydrology)."""

    name = "climate_lake_feedback"

    def __init__(self, builder: ClimateBuilder) -> None:
        self.builder = builder

    def build(self, ctx: BuildContext) -> None:
        mesh = ctx.mesh
        if "lake_mask" in mesh.face_textures:
            self.builder.apply_lake_humidity_feedback(
                mesh, mesh.face_textures["lake_mask"], progress=ctx.report
            )


class HydrologyStage(Stage):
    """Runs hydrology; registers river, lake, and delta features."""

    name = "hydrology"

    def __init__(self, builder: HydrologyBuilder) -> None:
        self.builder = builder

    def build(self, ctx: BuildContext) -> None:
        mesh = ctx.mesh
        self.builder.build(mesh, mesh.face_textures["elevation"], progress=ctx.report)
        hb = self.builder
        if hb.rivers:
            paths_with_flow = []
            for river in hb.rivers:
                polyline = river.as_polyline()
                if len(polyline) < 2:
                    continue
                positions = np.stack(polyline).astype(np.float32)
                flows = river.flow_rates()
                in_lake = np.array([s.in_lake for s in river.segments], dtype=bool)
                paths_with_flow.append((positions, flows, in_lake))
            if paths_with_flow:
                ctx.world.add_feature("rivers", paths_with_flow)
        if hb.lake_polygons:
            ctx.world.add_feature("lakes", hb.lake_polygons)
        if hb.delta_positions is not None and len(hb.delta_positions) > 0:
            ctx.world.add_feature("deltas", hb.delta_positions)


class ResourcesStage(Stage):
    """Runs resources (needs hydrology rivers); registers mineral/rare-bio features."""

    name = "resources"

    def __init__(
        self, builder: ResourcesBuilder, hydrology_builder: Optional[HydrologyBuilder] = None
    ) -> None:
        self.builder = builder
        self.hydrology_builder = hydrology_builder

    def build(self, ctx: BuildContext) -> None:
        rivers = self.hydrology_builder.rivers if self.hydrology_builder is not None else None
        self.builder.build(ctx.mesh, rivers=rivers, progress=ctx.report)
        rb = self.builder
        if rb.minerals is not None and len(rb.minerals[0]) > 0:
            ctx.world.add_feature("minerals", rb.minerals)
        if rb.rare_bio is not None and len(rb.rare_bio[0]) > 0:
            ctx.world.add_feature("rare_bio", rb.rare_bio)


class CostStage(Stage):
    """Runs the base traversal-cost surfaces; registers the cost_surfaces feature."""

    name = "cost"

    def __init__(
        self, builder: CostBuilder, hydrology_builder: Optional[HydrologyBuilder] = None
    ) -> None:
        self.builder = builder
        self.hydrology_builder = hydrology_builder

    def build(self, ctx: BuildContext) -> None:
        rivers = self.hydrology_builder.rivers if self.hydrology_builder is not None else None
        self.builder.build(ctx.mesh, rivers=rivers, progress=ctx.report)
        if self.builder.surfaces is not None:
            ctx.world.add_feature("cost_surfaces", self.builder.surfaces)
