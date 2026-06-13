"""Placement stage — settlement seeding on the land cost graph.

Params and stage for the second phase of the realm pipeline.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from core.display import DisplayRegistry
from core.params import field, param_field as _f, paramclass
from core.pipeline import BuildContext, Stage
from realmgen.channels import SITUATIONAL_VALUE, SUITABILITY
from realmgen.placement.seeder import build_cost_graph, placement_score, seed_settlements
from realmgen.placement.types import Settlement
from worldgen.cost.channels import COST_SURFACES, IS_WATER


# ── Params ────────────────────────────────────────────────────────────────────


@paramclass
class TierParams:
    exclusion_radius: float = _f(20.0, label="Exclusion radius (cost-dist)", min=0.5, max=100.0)
    score_threshold: float = _f(0.30, label="Min score to place", min=0.0, max=1.0)
    initial_population: float = _f(0.5, label="Initial population seed", min=0.0, max=1.0)


@paramclass
class SeederParams:
    # Cities form at nodes — harbors, confluences, passes, straits — not merely
    # at fertile spots (fertile spots get villages, modeled separately).
    situational_weight: float = _f(0.55, label="Situational value weight", min=0.0, max=1.0)
    tier0: TierParams = field(
        default_factory=lambda: TierParams(exclusion_radius=25.0, score_threshold=0.45, initial_population=1.0),
        metadata={"label": "Capitals"},
    )
    tier1: TierParams = field(
        default_factory=lambda: TierParams(exclusion_radius=12.0, score_threshold=0.28, initial_population=0.5),
        metadata={"label": "Cities"},
    )
    tier2: TierParams = field(
        default_factory=lambda: TierParams(exclusion_radius=6.0, score_threshold=0.2, initial_population=0.2),
        metadata={"label": "Towns"},
    )
    tier3: TierParams = field(
        default_factory=lambda: TierParams(exclusion_radius=3.0, score_threshold=0.1, initial_population=0.08),
        metadata={"label": "Villages"},
    )


# ── PlacementStage ────────────────────────────────────────────────────────────


class PlacementStage(Stage):
    """Seeds settlements on the land cost graph from suitability + situational scores."""

    name = "realm_placement"
    requires = (SUITABILITY, SITUATIONAL_VALUE, IS_WATER, COST_SURFACES)
    provides = ()

    def __init__(self, params: SeederParams, danger_params=None) -> None:
        self.params = params
        self.danger_params = danger_params
        self.settlements: Optional[list[Settlement]] = None

    def build(self, ctx: BuildContext) -> None:
        world = ctx.world
        mesh = ctx.mesh
        n = len(mesh.faces)
        p = self.params

        is_water = IS_WATER.read(mesh)
        suitability = SUITABILITY.read(mesh)
        situational_value = SITUATIONAL_VALUE.read(mesh)
        centroids = np.asarray(mesh.face_centroids, dtype=np.float64)

        # Optional danger influence.
        danger = np.zeros(n, dtype=np.float32)
        if self.danger_params is not None:
            from worldgen.danger.channels import DANGER_ARRAY
            if DANGER_ARRAY.present(world):
                danger = DANGER_ARRAY.read(world).astype(np.float32)

        ctx.report("Building land cost graph…")
        if COST_SURFACES.present(world):
            from worldgen.cost.surfaces import build_land_edge_costs
            cs = COST_SURFACES.read(world)
            land_base_cost = cs.land_base_cost
            if self.danger_params is not None:
                scale = float(self.danger_params.movement_scale)
                land_base_cost = land_base_cost + scale * danger
            edge_costs = build_land_edge_costs(
                mesh,
                land_base_cost,
                cs.land_passable,
                cs.river_flow_dir,
                cs.river_flow_mag,
                cs.river_navigability,
            )
        else:
            base = np.where(is_water, np.inf, 1.5)
            if self.danger_params is not None:
                scale = float(self.danger_params.movement_scale)
                base = base + scale * danger
            edge_costs = [
                np.array([0.5 * (base[i] + base[j]) for j in adj], dtype=np.float64)
                for i, adj in enumerate(mesh.face_adjacency)
            ]

        combined = placement_score(
            suitability, situational_value, situational_weight=p.situational_weight,
        )
        if self.danger_params is not None:
            combined[danger >= float(self.danger_params.block_threshold)] = 0.0
        cost_graph = build_cost_graph(mesh.face_adjacency, edge_costs)
        tier_configs = [
            (p.tier0.exclusion_radius, p.tier0.score_threshold, p.tier0.initial_population),
            (p.tier1.exclusion_radius, p.tier1.score_threshold, p.tier1.initial_population),
            (p.tier2.exclusion_radius, p.tier2.score_threshold, p.tier2.initial_population),
            (p.tier3.exclusion_radius, p.tier3.score_threshold, p.tier3.initial_population),
        ]

        ctx.report("Seeding settlements…")
        self.settlements = seed_settlements(
            combined, is_water, centroids, cost_graph, tier_configs,
        )

        tier_arr = np.full(n, -1, dtype=np.int32)
        for s in self.settlements:
            tier_arr[s.face_id] = s.tier
        mesh.apply_metadata_channel("settlement_tier", tier_arr.tolist())

        world.add_feature("settlements", self.settlements)
        ctx.report(f"Seeder done: {len(self.settlements)} settlements.")

    def register_display(self, world, reg: DisplayRegistry, *, debug: bool = False) -> None:
        pass  # Final settlement display is owned by PopulationStage after culling.
