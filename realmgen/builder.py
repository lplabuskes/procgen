"""Realm stage — human-geography layer over a fully-built world.

Downstream of the worldgen pipeline. ``RealmStage`` implements the core
``Stage`` protocol: ``build(ctx)`` reads upstream physical layers through typed
``Channel`` descriptors (no magic-string ``meta.get`` fallbacks), computes the
suitability and static situational-value scores, and writes them back as
channels; ``register_display`` contributes its color schemes to a
``DisplayRegistry``.

``RealmBuilder`` is a thin back-compat wrapper preserving the original
standalone API (``RealmBuilder(params).build(globe) -> suitability`` plus
``.suitability`` / ``.situational_value`` and ``apply_display(window)``).

Usage::

    from realmgen.builder import RealmStage
    from core.pipeline import BuildContext

    RealmStage(RealmParams()).build(BuildContext(world=globe))
    # globe.mesh.face_metadata[i]["suitability"] / ["situational_value"] are set
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from core.display import DisplayRegistry
from core.params import field, param_field as _f, paramclass
from core.pipeline import BuildContext, Pipeline, Stage
from core.world import World

from realmgen.channels import HINTERLAND_ID, POPULATION, SITUATIONAL_VALUE, SUITABILITY, TRADE_FLOW
from realmgen.population.calibration import (
    apply_population_to_settlements,
    compute_carrying_capacity,
    compute_hinterland_shares,
    compute_population,
    compute_trade_wealth,
    cull_settlements,
)
from realmgen.population.trade_spawn import spawn_trade_settlements
from realmgen.trade.network import compute_trade_network
from realmgen.placement.seeder import (
    build_cost_graph,
    placement_score,
    seed_settlements,
)
from realmgen.placement.types import Settlement, TIER_NAMES, population_display_tier
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
from worldgen.climate.channels import (
    DISTANCE_TO_COAST,
    PRECIPITATION_ANNUAL,
    TEMPERATURE_COLDEST,
    TEMPERATURE_WARMEST,
)
from viz.color_stops import interp_stops_scalar
from worldgen.cost.channels import COST_SURFACES, IS_STRAIT, IS_WATER
from worldgen.geology.channels import ELEVATION, IS_OCEAN
from worldgen.geology.display import OCEAN_COLOR
from worldgen.hydrology.channels import LAKE_MASK, RIVERS
from worldgen.resources.channels import FERTILITY_MEAN, MINERALS, RARE_BIO
from worldgen.resources.terrain import face_elevation_range

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

_TRADE_FLOW_STOPS = [
    (0.0, (0.30, 0.28, 0.20, 1.0)),
    (0.4, (0.80, 0.70, 0.20, 1.0)),
    (1.0, (0.95, 0.85, 0.10, 1.0)),
]

# Muted versions of TIER_COLORS used to fill hinterland territory.
_HINTERLAND_TIER_COLORS: list[tuple] = [
    (0.75, 0.65, 0.15, 1.0),  # muted gold   — capital hinterland
    (0.25, 0.45, 0.70, 1.0),  # muted blue   — city hinterland
    (0.30, 0.58, 0.30, 1.0),  # muted green  — town hinterland
    (0.60, 0.52, 0.38, 1.0),  # muted tan    — village hinterland
]
_HINTERLAND_UNASSIGNED = (0.30, 0.28, 0.25, 1.0)


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


@paramclass
class TierParams:
    exclusion_radius: float = _f(20.0, label="Exclusion radius (cost-dist)", min=0.5, max=100.0)
    score_threshold: float = _f(0.30, label="Min score to place", min=0.0, max=1.0)
    influence: float = _f(0.0, label="Territory influence (lower = wider)", min=0.0, max=30.0)
    demand: float = _f(1.0, label="Trade demand weight", min=0.0, max=2.0)


@paramclass
class TradeParams:
    port_cost: float = _f(5.0, label="Port cost", min=0.0, max=50.0)
    river_switch_cost: float = _f(0.0, label="River switch cost", min=0.0, max=10.0)
    demand_penalty: float = _f(5.0, label="Demand penalty", min=0.0, max=20.0)
    flow_decay: float = _f(0.05, label="Flow decay", min=0.001, max=1.0)
    hub_flow_quantile: float = _f(0.90, label="Hub quantile", min=0.5, max=0.99)


@paramclass
class TradeSpawnParams:
    hub_flow_quantile: float = _f(0.97, label="Hub flow quantile", min=0.80, max=1.00)
    hub_min_suitability: float = _f(0.15, label="Hub min suitability", min=0.0, max=0.5)
    hub_exclusion: float = _f(5.0, label="Hub exclusion radius", min=1.0, max=30.0)
    waypoint_min_cost: float = _f(30.0, label="Waypoint min cost dist", min=5.0, max=200.0)
    waypoint_min_suitability: float = _f(0.10, label="Waypoint min suitability", min=0.0, max=0.5)
    waypoint_exclusion: float = _f(3.0, label="Waypoint exclusion radius", min=1.0, max=20.0)


@paramclass
class PopulationParams:
    relaxation_passes: int = _f(4, label="Relaxation passes", min=1, max=5)
    trade_weight: float = _f(1.5, label="Trade wealth weight", min=0.0, max=5.0)
    min_pop_fraction: float = _f(0.012, label="Min pop fraction (cull)", min=0.0, max=0.1)
    hinterland_max_distance: float = _f(40.0, label="Hinterland max distance", min=5.0, max=200.0)
    hinterland_influence_scale: float = _f(8.0, label="Hinterland influence scale", min=0.5, max=30.0)
    hinterland_reach_scale: float = _f(45.0, label="Hinterland reach scale (∝ catchment radius)", min=1.0, max=1000.0)
    trade_spawn: TradeSpawnParams = field(
        default_factory=TradeSpawnParams, metadata={"label": "Trade-spawn settlements"}
    )


@paramclass
class SeederParams:
    situational_weight: float = _f(0.40, label="Situational value weight", min=0.0, max=1.0)
    tier0: TierParams = field(
        default_factory=lambda: TierParams(exclusion_radius=25.0, score_threshold=0.45, influence=0.0, demand=1.0),
        metadata={"label": "Capitals"},
    )
    tier1: TierParams = field(
        default_factory=lambda: TierParams(exclusion_radius=12.0, score_threshold=0.28, influence=4.0, demand=0.6),
        metadata={"label": "Cities"},
    )
    tier2: TierParams = field(
        default_factory=lambda: TierParams(exclusion_radius=6.0, score_threshold=0.2, influence=9.0, demand=0.3),
        metadata={"label": "Towns"},
    )
    tier3: TierParams = field(
        default_factory=lambda: TierParams(exclusion_radius=3.0, score_threshold=0.1, influence=15.0, demand=0.12),
        metadata={"label": "Villages"},
    )


@paramclass
class RealmParams:
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
    situational: SituationalValueParams = field(
        default_factory=SituationalValueParams, metadata={"label": "Situational value"}
    )
    seeder: SeederParams = field(
        default_factory=SeederParams, metadata={"label": "Settlement seeder"}
    )
    trade: TradeParams = field(
        default_factory=TradeParams, metadata={"label": "Trade network"}
    )
    population: PopulationParams = field(
        default_factory=PopulationParams, metadata={"label": "Population"}
    )


# ── Display color schemes ───────────────────────────────────────────────────────


def _make_hinterland_color(settlements: "list[Settlement]"):
    """Return a color-scheme callable that maps hinterland_id → display-tier color.

    Display tier is derived from each settlement's population relative to the
    world peak, not from the seeding tier label.
    """
    max_pop = max((s.population for s in settlements), default=0.0)
    display_tiers = [
        min(population_display_tier(s.population, max_pop), len(_HINTERLAND_TIER_COLORS) - 1)
        for s in settlements
    ]

    def fn(meta: dict) -> tuple:
        if meta.get("is_water", meta.get("is_ocean", False)):
            return OCEAN_COLOR
        hid = meta.get("hinterland_id", -1)
        if hid is None or int(hid) < 0 or int(hid) >= len(display_tiers):
            return _HINTERLAND_UNASSIGNED
        return _HINTERLAND_TIER_COLORS[display_tiers[int(hid)]]

    return fn


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


def _trade_flow_color(meta: dict, _max_flow: list) -> tuple:
    if meta.get("is_water", meta.get("is_ocean", False)):
        return OCEAN_COLOR
    f = float(meta.get("trade_flow", 0.0))
    mx = _max_flow[0]
    s = min(f / mx, 1.0) if mx > 0 else 0.0
    return interp_stops_scalar(s, _TRADE_FLOW_STOPS)


# ── Stage ───────────────────────────────────────────────────────────────────────


class RealmStage(Stage):
    """Computes human-geography scores from a fully-built world.

    Run after the worldgen pipeline so the channels listed in ``requires`` and
    the ``cost_surfaces`` / ``minerals`` / ``rare_bio`` / ``rivers`` features are
    present. Stores ``self.suitability`` and ``self.situational_value`` (``(N,)``
    float32 in [0, 1]) and writes them via the ``SUITABILITY`` /
    ``SITUATIONAL_VALUE`` channels.
    """

    name = "realm"
    requires = (
        IS_WATER, IS_OCEAN, IS_STRAIT, ELEVATION,
        DISTANCE_TO_COAST, FERTILITY_MEAN,
        TEMPERATURE_WARMEST, TEMPERATURE_COLDEST, PRECIPITATION_ANNUAL,
        LAKE_MASK, COST_SURFACES, RIVERS, MINERALS, RARE_BIO,
    )
    provides = (SUITABILITY, SITUATIONAL_VALUE, TRADE_FLOW)

    def __init__(self, params: RealmParams | None = None) -> None:
        self.params: RealmParams = params or RealmParams()
        self.suitability: Optional[np.ndarray] = None
        self.situational_value: Optional[np.ndarray] = None
        self.settlements: Optional[list[Settlement]] = None
        self.trade_result = None
        self.population: Optional[np.ndarray] = None
        self.hinterland: Optional[np.ndarray] = None

    def build(self, ctx: BuildContext) -> None:
        world = ctx.world
        mesh = ctx.mesh
        n = len(mesh.faces)
        p = self.params

        # ── Open-water mask (impassable for settlement) ──────────────────────
        ctx.report("Reading physical layers…")
        # IS_WATER (ocean ∪ lake) falls back to is_ocean for pre-cost worlds.
        is_water = IS_WATER.read(mesh)

        # ── Freshwater seed mask (rivers + lakes) ───────────────────────────
        # Rivers: cost_surfaces.river_flow_mag > 0 covers all non-in_lake river
        # segments (the most accurate signal available).
        river_mask = np.zeros(n, dtype=bool)
        if COST_SURFACES.present(world):
            river_mask = COST_SURFACES.read(world).river_flow_mag > 0.0

        # Lakes: use the lake_mask texture if present, else any non-ocean water.
        if LAKE_MASK.present(mesh):
            lake_tex = LAKE_MASK.read(mesh)
            lake_mask = lake_tex[..., 0].mean(axis=(1, 2)) >= 0.5
        else:
            lake_mask = is_water & ~IS_OCEAN.read(mesh)

        freshwater_mask = river_mask | lake_mask

        # ── Per-face climate and fertility ───────────────────────────────────
        dist_coast_km = DISTANCE_TO_COAST.read(mesh)
        fertility = FERTILITY_MEAN.read(mesh).clip(0.0, 1.0)
        t_warmest = TEMPERATURE_WARMEST.read(mesh)
        t_coldest = TEMPERATURE_COLDEST.read(mesh)
        p_annual = PRECIPITATION_ANNUAL.read(mesh)

        # ── Elevation range for ruggedness penalty ───────────────────────────
        elev_range = face_elevation_range(ELEVATION.read(mesh))

        # ── Resource deposit positions → face seeds ─────────────────────────
        ctx.report("Mapping resource deposits to faces…")
        resource_mask = np.zeros(n, dtype=bool)
        centroids = np.asarray(mesh.face_centroids, dtype=np.float64)
        for channel in (MINERALS, RARE_BIO):
            if channel.present(world):
                positions, _, _ = channel.read(world)
                if len(positions) > 0:
                    fids = _positions_to_face_ids(positions, centroids)
                    resource_mask[fids] = True

        # ── Factor computation ───────────────────────────────────────────────
        ctx.report("Computing water proximity…")
        water_score = compute_water_proximity(
            mesh,
            freshwater_mask,
            dist_coast_km,
            params=p.water,
        )

        ctx.report("Computing climate habitability…")
        climate_score = compute_climate_habitability(
            t_warmest,
            t_coldest,
            p_annual,
            params=p.climate,
        )

        ctx.report("Computing resource access…")
        resource_score = compute_resource_proximity(
            mesh,
            resource_mask,
            resource_decay=p.resource.decay,
            resource_max_hops=p.resource.max_hops,
        )

        ctx.report("Combining suitability…")
        suitability = combine_suitability(
            is_water,
            water_score,
            climate_score,
            fertility.astype(np.float32),
            resource_score,
            params=p.weights,
        )

        ruggedness_mult = compute_ruggedness_penalty(
            elev_range,
            penalty_scale_km=p.ruggedness.penalty_scale_km,
        )
        suitability = (suitability * ruggedness_mult).astype(np.float32)

        self.suitability = suitability
        SUITABILITY.write(mesh, suitability)

        # ── Situational value ────────────────────────────────────────────────
        ctx.report("Computing situational value…")
        sp = p.situational

        # Harbor quality: needs per-face mean elevation and ocean mask.
        is_ocean = IS_OCEAN.read(mesh)
        face_mean_elev_km = ELEVATION.read(mesh)[..., 0].mean(axis=(1, 2))
        harbor_score = compute_harbor_quality(
            mesh, is_ocean, face_mean_elev_km,
            depth_scale_km=sp.harbor.depth_scale_km,
        )

        # River confluences and navigable heads use the world's rivers feature
        # (positions/flows/in_lake tuples) with a centroid lookup.
        rivers = RIVERS.read(world) if RIVERS.present(world) else None
        confluence_score = compute_river_confluences(rivers, centroids, n)

        # Navigable river heads: fall-line faces per river.
        river_navigability = np.zeros(n, dtype=np.float32)
        if COST_SURFACES.present(world):
            river_navigability = COST_SURFACES.read(world).river_navigability
        nav_head_score = compute_navigable_heads(rivers, centroids, river_navigability, n)

        # Mountain passes: local land-cost minima in high-cost terrain.
        land_base_cost = np.zeros(n, dtype=np.float32)
        if COST_SURFACES.present(world):
            land_base_cost = COST_SURFACES.read(world).land_base_cost
        pass_score = compute_mountain_passes(
            mesh, land_base_cost, is_water,
            params=sp.passes,
        )

        # Strait adjacency: land faces bordering tagged strait ocean faces.
        is_strait = IS_STRAIT.read(mesh)
        strait_score = compute_strait_adjacency(mesh, is_strait, is_water)

        situational_value = combine_situational_value(
            is_water,
            harbor_score,
            confluence_score,
            nav_head_score,
            pass_score,
            strait_score,
            params=sp.weights,
        )

        self.situational_value = situational_value
        SITUATIONAL_VALUE.write(mesh, situational_value)

        # ── Settlement seeder ────────────────────────────────────────────────
        ctx.report("Building land cost graph…")
        sp = p.seeder
        if COST_SURFACES.present(world):
            from worldgen.cost.surfaces import build_land_edge_costs
            cs = COST_SURFACES.read(world)
            edge_costs = build_land_edge_costs(
                mesh,
                cs.land_base_cost,
                cs.land_passable,
                cs.river_flow_dir,
                cs.river_flow_mag,
                cs.river_navigability,
            )
        else:
            # Fallback: uniform cost 1.5 on land, impassable on water.
            base = np.where(is_water, np.inf, 1.5)
            edge_costs = [
                np.array([0.5 * (base[i] + base[j]) for j in adj], dtype=np.float64)
                for i, adj in enumerate(mesh.face_adjacency)
            ]

        combined = placement_score(
            suitability,
            situational_value,
            situational_weight=sp.situational_weight,
        )
        cost_graph = build_cost_graph(mesh.face_adjacency, edge_costs)
        tier_configs = [
            (sp.tier0.exclusion_radius, sp.tier0.score_threshold, sp.tier0.influence, sp.tier0.demand),
            (sp.tier1.exclusion_radius, sp.tier1.score_threshold, sp.tier1.influence, sp.tier1.demand),
            (sp.tier2.exclusion_radius, sp.tier2.score_threshold, sp.tier2.influence, sp.tier2.demand),
            (sp.tier3.exclusion_radius, sp.tier3.score_threshold, sp.tier3.influence, sp.tier3.demand),
        ]

        ctx.report("Seeding settlements…")
        self.settlements = seed_settlements(
            combined, is_water, centroids, cost_graph, tier_configs,
        )

        # Provisional per-face metadata written at seeding time; overwritten
        # with population-derived display tiers after calibration.
        tier_arr = np.full(n, -1, dtype=np.int32)
        for s in self.settlements:
            tier_arr[s.face_id] = s.tier
        mesh.apply_metadata_channel("settlement_tier", tier_arr)

        world.add_feature("settlements", self.settlements)
        ctx.report(f"Seeder done: {len(self.settlements)} settlements.")

        # ── Trade network ────────────────────────────────────────────────────
        ctx.report("Building trade network…")
        tp = p.trade

        # Sea edge costs (needed for the hybrid graph)
        if COST_SURFACES.present(world):
            from worldgen.cost.surfaces import build_sea_edge_costs
            cs = COST_SURFACES.read(world)
            sea_edge_costs = build_sea_edge_costs(mesh, cs.sea_base_cost, cs.sea_passable)
            sea_passable = cs.sea_passable
        else:
            sea_passable = np.zeros(n, dtype=bool)
            sea_edge_costs = [np.empty(0, dtype=np.float64) for _ in range(n)]

        # Collect good sources from minerals and rare_bio deposits
        good_sources: list[tuple] = []
        for channel in (MINERALS, RARE_BIO):
            if channel.present(world):
                positions, _, abundances = channel.read(world)
                if len(positions) > 0:
                    fids = _positions_to_face_ids(positions, centroids)
                    good_sources.extend(zip(fids.tolist(), abundances.tolist()))

        pp = p.population
        tsp = pp.trade_spawn
        face_areas = np.asarray(mesh.face_areas, dtype=np.float64)

        def _compute_trade(settlements, settlement_demands=None):
            return compute_trade_network(
                mesh,
                settlements,
                good_sources,
                edge_costs,
                ~is_water,
                sea_edge_costs,
                sea_passable,
                port_cost=tp.port_cost,
                river_switch_cost=tp.river_switch_cost,
                demand_penalty=tp.demand_penalty,
                flow_decay=tp.flow_decay,
                hub_flow_quantile=tp.hub_flow_quantile,
                settlement_demands=settlement_demands,
            )

        self.trade_result = _compute_trade(self.settlements)
        ctx.report(
            f"Initial trade done: {len(self.trade_result.routes)} route segments, "
            f"{self.trade_result.hub_mask.sum()} hub faces."
        )

        # ── Population & Calibration relaxation loop ──────────────────────────
        n_passes = max(1, int(pp.relaxation_passes))
        population = np.zeros(len(self.settlements), dtype=np.float64)

        for pass_idx in range(n_passes):
            ctx.report(f"Relaxation pass {pass_idx + 1}/{n_passes}…")

            # Spawn trade-driven settlements from current flow.
            new_settlements = spawn_trade_settlements(
                n,
                self.settlements,
                self.trade_result.face_flow,
                suitability,
                is_water,
                cost_graph,
                centroids,
                tier_configs,
                hub_flow_quantile=tsp.hub_flow_quantile,
                hub_min_suitability=tsp.hub_min_suitability,
                hub_exclusion=tsp.hub_exclusion,
                waypoint_min_cost=tsp.waypoint_min_cost,
                waypoint_min_suitability=tsp.waypoint_min_suitability,
                waypoint_exclusion=tsp.waypoint_exclusion,
            )
            if new_settlements:
                self.settlements = self.settlements + new_settlements
                ctx.report(f"  Trade-spawn: +{len(new_settlements)} ({len(self.settlements)} total).")

            # Assign hinterlands and carrying capacity.
            shares, self.hinterland = compute_hinterland_shares(
                n, self.settlements, cost_graph,
                max_distance=pp.hinterland_max_distance,
                influence_scale=pp.hinterland_influence_scale,
                reach_scale=pp.hinterland_reach_scale,
            )
            carrying_cap = compute_carrying_capacity(shares, fertility, face_areas)

            # Compute population.
            trade_wealth = compute_trade_wealth(
                self.settlements, self.trade_result.face_flow.astype(np.float64),
            )
            population = compute_population(
                carrying_cap, trade_wealth, trade_weight=pp.trade_weight,
            )

            # Cull weak settlements.
            self.settlements, population = cull_settlements(
                self.settlements, population, min_pop_fraction=pp.min_pop_fraction,
            )
            ctx.report(f"  {len(self.settlements)} settlements after cull.")

            # Recompute trade with population-weighted demands for next pass.
            max_p = float(population.max())
            norm_demands = population / max_p if max_p > 0 else np.ones_like(population)
            self.trade_result = _compute_trade(self.settlements, settlement_demands=norm_demands)

        self.settlements = apply_population_to_settlements(self.settlements, population)
        self.population = population

        # Final hinterland for the surviving settlement list.
        _shares, self.hinterland = compute_hinterland_shares(
            n, self.settlements, cost_graph,
            max_distance=pp.hinterland_max_distance,
            influence_scale=pp.hinterland_influence_scale,
            reach_scale=pp.hinterland_reach_scale,
        )

        # Final clean trade pass (no demand weighting) so routes reflect final geography.
        if self.settlements:
            self.trade_result = _compute_trade(self.settlements)

        # ── Write per-face metadata ───────────────────────────────────────────
        TRADE_FLOW.write(mesh, self.trade_result.face_flow)
        mesh.apply_metadata_channel("is_trade_hub", self.trade_result.hub_mask)

        # Per-face population = population of settlement (0 if no settlement).
        # Will add a second value for total population including rural areas
        pop_arr = np.zeros(n, dtype=np.float64)
        for k, s in enumerate(self.settlements):
            pop_arr[s.face_id] = population[k]
        POPULATION.write(mesh, pop_arr)
        HINTERLAND_ID.write(mesh, self.hinterland.tolist())

        # Write population-derived display tiers (overwriting seeder's provisional write).
        max_pop = float(population.max()) if population.size > 0 else 0.0
        tier_arr = np.full(n, -1, dtype=np.int32)
        for k, s in enumerate(self.settlements):
            tier_arr[s.face_id] = population_display_tier(population[k], max_pop)
        mesh.apply_metadata_channel("settlement_tier", tier_arr.tolist())

        world.add_feature("settlements", self.settlements)
        world.add_feature("trade_routes", self.trade_result.routes)
        ctx.report(
            f"Population & calibration done: {len(self.settlements)} settlements, "
            f"{len(self.trade_result.routes)} route segments."
        )

    def register_display(
        self, world: "World | None", reg: DisplayRegistry, *, debug: bool = False
    ) -> None:
        """Contribute realm color schemes and overlays (always shown — realm is an output).

        ``world`` is unused — results are read off the stage — so a façade caller
        may pass ``None``.
        """
        if self.suitability is not None:
            reg.add_scheme("Suitability", _suitability_color)
        if self.situational_value is not None:
            reg.add_scheme("Situational Value", _situational_color)
        if self.trade_result is not None:
            max_flow = float(self.trade_result.face_flow.max()) if self.trade_result.face_flow.size else 0.0
            _max_flow_ref = [max_flow]
            reg.add_scheme(
                "Trade Flow",
                lambda meta, _ref=_max_flow_ref: _trade_flow_color(meta, _ref),
            )
        if self.hinterland is not None and self.settlements is not None:
            reg.add_scheme("Hinterland", _make_hinterland_color(self.settlements))
        if self.settlements is not None:
            reg.add_overlay("settlements", "Settlements", checked=True)
        if self.trade_result is not None and self.trade_result.routes:
            reg.add_overlay("trade_routes", "Trade Routes", checked=True)


# ── Pipeline ────────────────────────────────────────────────────────────────────


class RealmPipeline(Pipeline):
    """Single-stage pipeline for the human-geography layer.

    Wraps :class:`RealmStage` in the ``Pipeline`` protocol so it can sit
    alongside ``GlobePipeline`` in a shared-context pipeline list.  After
    ``build(ctx)``, ``.suitability`` and ``.situational_value`` hold the
    computed ``(N,) float32`` arrays.
    """

    def __init__(self, params: RealmParams | None = None) -> None:
        self.realm_stage = RealmStage(params)
        super().__init__([self.realm_stage])
        self.world: Optional[World] = None

    @property
    def params(self) -> RealmParams:
        return self.realm_stage.params

    @property
    def suitability(self) -> Optional[np.ndarray]:
        return self.realm_stage.suitability

    @property
    def situational_value(self) -> Optional[np.ndarray]:
        return self.realm_stage.situational_value

    def build(self, ctx: BuildContext) -> None:
        super().build(ctx)
        self.world = ctx.world

    def apply_display(self, window, debug: bool = False) -> None:
        """Merge realm color schemes, overlays, and hover into a MapWindow after build().

        Call after the worldgen display is wired so the schemes are appended
        alongside the worldgen schemes.
        """
        reg = DisplayRegistry()
        self.realm_stage.register_display(None, reg, debug=debug)

        # Merge color schemes with any already registered by worldgen.
        if reg.schemes:
            merged = dict(window._map._color_schemes)
            for name, spec in reg.schemes.items():
                merged[name] = spec.fn
            window.set_color_schemes(merged)

        # Register feature overlays (settlements, etc.).
        for ov in reg.overlays:
            window.add_overlay(ov.feature, label=ov.label, checked=ov.checked)

        mesh = self.world.mesh if self.world else None
        if mesh is None:
            return

        prev_hover = window._map._hover_callback

        def _realm_hover(face_src, uv=None):
            if prev_hover is not None:
                prev_hover(face_src, uv)
            if face_src is None:
                return
            meta = mesh.face_metadata[face_src]
            if meta.get("is_water", meta.get("is_ocean", False)):
                return
            parts: list[str] = []
            suit = meta.get("suitability")
            if suit is not None:
                parts.append(f"Suit: {float(suit):.2f}")
            tier = meta.get("settlement_tier")
            if tier is not None and int(tier) >= 0:
                t = int(tier)
                parts.append(TIER_NAMES[min(t, len(TIER_NAMES) - 1)])
                pop = meta.get("population")
                if pop is not None and float(pop) > 0:
                    parts.append(f"Pop: {float(pop):.3g}")
            if parts:
                cur = window._status_label.text()
                fragment = "  ".join(parts)
                window.set_status(f"{cur}  |  {fragment}" if cur else fragment)

        window.set_hover_callback(_realm_hover)
