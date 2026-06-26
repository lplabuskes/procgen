"""Scoring stage — suitability and situational-value computation.

Params and stage for the first phase of the realm pipeline.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from core.display import DisplayRegistry
from core.params import field, param_field as _f, paramclass
from core.pipeline import BuildContext, Stage
from realmgen.channels import SITUATIONAL_VALUE, SUITABILITY
from realmgen.scoring.situational_value import (
    combine_situational_value,
    compute_harbor_quality,
    compute_mountain_passes,
    compute_navigable_heads,
    compute_river_confluences,
    compute_strait_adjacency,
)
from realmgen.scoring.suitability import (
    _positions_to_face_ids,
    combine_suitability,
    compute_climate_habitability,
    compute_resource_proximity,
    compute_ruggedness_penalty,
    compute_water_proximity,
)
from viz.color_stops import interp_stops_scalar
from worldgen.climate.channels import (
    DISTANCE_TO_COAST,
    PRECIPITATION_ANNUAL,
    TEMPERATURE_COLDEST,
    TEMPERATURE_WARMEST,
)
from worldgen.cost.channels import COST_SURFACES, IS_STRAIT, IS_WATER
from worldgen.geology.channels import ELEVATION, IS_OCEAN
from worldgen.geology.display import OCEAN_COLOR
from worldgen.hydrology.channels import LAKE_MASK, RIVERS
from worldgen.resources.channels import FERTILITY_MEAN, MINERALS, RARE_BIO
from worldgen.resources.terrain import face_elevation_range


# ── Params ────────────────────────────────────────────────────────────────────


@paramclass
class WaterProximityParams:
    freshwater_decay: float = _f(0.4, label="Freshwater hop decay", min=0.05, max=3.0)
    freshwater_max_hops: int = _f(12, label="Freshwater max hops", min=1, max=32)
    coast_scale_km: float = _f(150.0, label="Coastal scale (km)", min=10.0, max=2000.0)
    coast_weight: float = _f(0.4, label="Coastal weight", min=0.0, max=1.0)


@paramclass
class ClimateHabitabilityParams:
    t_grow_min_c: float = _f(0.0, label="Min growing T (°C)", min=-10.0, max=20.0)
    t_grow_opt_c: float = _f(18.0, label="Opt growing T (°C)", min=5.0, max=35.0)
    t_cold_min_c: float = _f(-30.0, label="Cold survival T (°C)", min=-60.0, max=0.0)
    p_dry_mm: float = _f(100.0, label="Dry threshold (mm/yr)", min=0.0, max=500.0)
    p_opt_mm: float = _f(800.0, label="Opt precip (mm/yr)", min=100.0, max=3000.0)
    p_wet_mm: float = _f(3000.0, label="Wet limit (mm/yr)", min=500.0, max=10000.0)


@paramclass
class RuggednessParams:
    penalty_scale_km: float = _f(1.0, label="Ruggedness penalty scale (km)", min=0.05, max=10.0)


@paramclass
class ResourceAccessParams:
    decay: float = _f(0.2, label="Resource hop decay", min=0.05, max=2.0)
    max_hops: int = _f(16, label="Resource max hops", min=1, max=32)


@paramclass
class SuitabilityWeightParams:
    water_weight: float = _f(0.30, label="Water weight", min=0.0, max=1.0)
    climate_weight: float = _f(0.20, label="Climate weight", min=0.0, max=1.0)
    fertility_weight: float = _f(0.4, label="Fertility weight", min=0.0, max=1.0)
    resource_weight: float = _f(0.1, label="Resource weight", min=0.0, max=1.0)


@paramclass
class HarborQualityParams:
    depth_scale_km: float = _f(0.5, label="Harbor depth scale (km)", min=0.05, max=5.0)


@paramclass
class MountainPassParams:
    min_neighbor_cost: float = _f(3.0, label="Min neighbor cost for pass", min=0.5, max=10.0)
    cost_scale: float = _f(1.5, label="Pass contrast scale", min=0.1, max=5.0)


@paramclass
class SituationalValueWeightParams:
    harbor_weight: float = _f(0.25, label="Harbor weight", min=0.0, max=1.0)
    confluence_weight: float = _f(0.20, label="Confluence weight", min=0.0, max=1.0)
    nav_head_weight: float = _f(0.20, label="Nav. river head weight", min=0.0, max=1.0)
    pass_weight: float = _f(0.20, label="Pass weight", min=0.0, max=1.0)
    strait_weight: float = _f(0.15, label="Strait weight", min=0.0, max=1.0)


@paramclass
class SuitabilityParams:
    water: WaterProximityParams = field(
        default_factory=WaterProximityParams, metadata={"label": "Water proximity"}
    )
    climate: ClimateHabitabilityParams = field(
        default_factory=ClimateHabitabilityParams, metadata={"label": "Climate"}
    )
    ruggedness: RuggednessParams = field(
        default_factory=RuggednessParams, metadata={"label": "Ruggedness"}
    )
    resource: ResourceAccessParams = field(
        default_factory=ResourceAccessParams, metadata={"label": "Resource access"}
    )
    weights: SuitabilityWeightParams = field(
        default_factory=SuitabilityWeightParams, metadata={"label": "Weights"}
    )


@paramclass
class SituationalValueParams:
    harbor: HarborQualityParams = field(
        default_factory=HarborQualityParams, metadata={"label": "Harbor quality"}
    )
    passes: MountainPassParams = field(
        default_factory=MountainPassParams, metadata={"label": "Mountain passes"}
    )
    weights: SituationalValueWeightParams = field(
        default_factory=SituationalValueWeightParams, metadata={"label": "Weights"}
    )


# ── Display ───────────────────────────────────────────────────────────────────

_SUITABILITY_STOPS = [
    (0.0, (0.45, 0.38, 0.28, 1.0)),
    (0.5, (0.65, 0.80, 0.20, 1.0)),
    (1.0, (0.10, 0.60, 0.20, 1.0)),
]

_SITUATIONAL_STOPS = [
    (0.0, (0.35, 0.33, 0.30, 1.0)),
    (0.5, (0.85, 0.75, 0.30, 1.0)),
    (1.0, (0.90, 0.55, 0.10, 1.0)),
]


def _suitability_color(meta: dict) -> tuple:
    if meta.get("is_water", meta.get("is_ocean", False)):
        return OCEAN_COLOR
    s = min(max(float(meta.get("suitability", 0.0)), 0.0), 1.0)
    return interp_stops_scalar(s, _SUITABILITY_STOPS)


def _situational_color(meta: dict) -> tuple:
    if meta.get("is_water", meta.get("is_ocean", False)):
        return OCEAN_COLOR
    s = min(max(float(meta.get("situational_value", 0.0)), 0.0), 1.0)
    return interp_stops_scalar(s, _SITUATIONAL_STOPS)


# ── ScoringStage ──────────────────────────────────────────────────────────────


class ScoringStage(Stage):
    """Computes SUITABILITY and SITUATIONAL_VALUE from the physical world."""

    name = "realm_scoring"
    requires = (
        IS_WATER, IS_OCEAN, IS_STRAIT, ELEVATION,
        DISTANCE_TO_COAST, FERTILITY_MEAN,
        TEMPERATURE_WARMEST, TEMPERATURE_COLDEST, PRECIPITATION_ANNUAL,
        LAKE_MASK, COST_SURFACES, RIVERS, MINERALS, RARE_BIO,
    )
    provides = (SUITABILITY, SITUATIONAL_VALUE)

    def __init__(
        self,
        suitability_params: SuitabilityParams,
        situational_params: SituationalValueParams,
        danger_params=None,
    ) -> None:
        self.suitability_params = suitability_params
        self.situational_params = situational_params
        self.danger_params = danger_params
        self.suitability: Optional[np.ndarray] = None
        self.situational_value: Optional[np.ndarray] = None

    def build(self, ctx: BuildContext) -> None:
        world = ctx.world
        mesh = ctx.mesh
        n = len(mesh.faces)
        sp = self.suitability_params
        svp = self.situational_params

        ctx.report("Reading physical layers…")
        is_water = IS_WATER.read(mesh)

        river_mask = np.zeros(n, dtype=bool)
        if COST_SURFACES.present(world):
            river_mask = COST_SURFACES.read(world).river_flow_mag > 0.0
        if LAKE_MASK.present(mesh):
            lake_mask = LAKE_MASK.read(mesh)[..., 0].mean(axis=(1, 2)) >= 0.5
        else:
            lake_mask = is_water & ~IS_OCEAN.read(mesh)
        freshwater_mask = river_mask | lake_mask

        dist_coast_km = DISTANCE_TO_COAST.read(mesh)
        fertility = FERTILITY_MEAN.read(mesh).clip(0.0, 1.0)
        t_warmest = TEMPERATURE_WARMEST.read(mesh)
        t_coldest = TEMPERATURE_COLDEST.read(mesh)
        p_annual = PRECIPITATION_ANNUAL.read(mesh)
        elev_range = face_elevation_range(ELEVATION.read(mesh))

        ctx.report("Mapping resource deposits to faces…")
        resource_mask = np.zeros(n, dtype=bool)
        centroids = np.asarray(mesh.face_centroids, dtype=np.float64)
        for channel in (MINERALS, RARE_BIO):
            if channel.present(world):
                positions, _, _ = channel.read(world)
                if len(positions) > 0:
                    resource_mask[_positions_to_face_ids(positions, centroids)] = True

        # ── Suitability ──────────────────────────────────────────────────────
        ctx.report("Computing water proximity…")
        water_score = compute_water_proximity(
            mesh, freshwater_mask, dist_coast_km, params=sp.water,
        )
        ctx.report("Computing climate habitability…")
        climate_score = compute_climate_habitability(
            t_warmest, t_coldest, p_annual, params=sp.climate,
        )
        ctx.report("Computing resource access…")
        resource_score = compute_resource_proximity(
            mesh, resource_mask,
            resource_decay=sp.resource.decay,
            resource_max_hops=sp.resource.max_hops,
        )
        ctx.report("Combining suitability…")
        suitability = combine_suitability(
            is_water, water_score, climate_score,
            fertility.astype(np.float32), resource_score,
            params=sp.weights,
        )
        ruggedness_mult = compute_ruggedness_penalty(
            elev_range, penalty_scale_km=sp.ruggedness.penalty_scale_km,
        )
        suitability = (suitability * ruggedness_mult).astype(np.float32)

        if self.danger_params is not None:
            from worldgen.danger.channels import DANGER_ARRAY
            if DANGER_ARRAY.present(world):
                danger = DANGER_ARRAY.read(world).astype(np.float32)
                w = float(self.danger_params.suitability_weight)
                suitability = np.clip(suitability * (1.0 - w * danger), 0.0, 1.0).astype(np.float32)

        self.suitability = suitability
        SUITABILITY.write(mesh, suitability)

        # ── Situational value ────────────────────────────────────────────────
        ctx.report("Computing situational value…")
        is_ocean = IS_OCEAN.read(mesh)
        face_mean_elev_km = ELEVATION.read(mesh)[..., 0].mean(axis=(1, 2))
        harbor_score = compute_harbor_quality(
            mesh, is_ocean, face_mean_elev_km,
            depth_scale_km=svp.harbor.depth_scale_km,
        )

        rivers = RIVERS.read(world) if RIVERS.present(world) else None
        confluence_score = compute_river_confluences(rivers, centroids, n)

        river_navigability = np.zeros(n, dtype=np.float32)
        if COST_SURFACES.present(world):
            river_navigability = COST_SURFACES.read(world).river_navigability
        nav_head_score = compute_navigable_heads(rivers, centroids, river_navigability, n)

        land_base_cost = np.zeros(n, dtype=np.float32)
        if COST_SURFACES.present(world):
            land_base_cost = COST_SURFACES.read(world).land_base_cost
        pass_score = compute_mountain_passes(
            mesh, land_base_cost, is_water, params=svp.passes,
        )

        is_strait = IS_STRAIT.read(mesh)
        strait_score = compute_strait_adjacency(mesh, is_strait, is_water)

        situational_value = combine_situational_value(
            is_water, harbor_score, confluence_score,
            nav_head_score, pass_score, strait_score,
            params=svp.weights,
        )
        self.situational_value = situational_value
        SITUATIONAL_VALUE.write(mesh, situational_value)

    def register_display(self, world, reg: DisplayRegistry, *, debug: bool = False) -> None:
        if not debug:
            return
        if self.suitability is not None:
            reg.add_scheme("Suitability", _suitability_color)
        if self.situational_value is not None:
            reg.add_scheme("Situational Value", _situational_color)

        def _suitability_hover(face: int, _uv, w) -> list:
            meta = w.mesh.face_metadata[face]
            if meta.get("is_water", meta.get("is_ocean", False)):
                return []
            suit = meta.get("suitability")
            return [f"Suit: {float(suit):.2f}"] if suit is not None else []

        reg.add_hover(_suitability_hover)
