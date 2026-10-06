"""Population stage — trade network, relaxation loop, and population calibration."""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from core.display import DisplayRegistry
from core.params import field, param_field as _f, paramclass
from core.pipeline import BuildContext, Stage
from realmgen.channels import (
    FISHING_POPULATION, FORT_LEVY, FORT_POPULATION, HINTERLAND_ID, MINING_POPULATION, POPULATION, RURAL_POPULATION,
    SITUATIONAL_VALUE, SUITABILITY, TOWN_POPULATION, TRADE_FLOW,
)
from realmgen.placement.builder import SeederParams
from realmgen.placement.seeder import build_cost_graph
from realmgen.placement.types import (
    TIER_NAMES,
    Settlement,
    format_population,
    population_display_tier,
)
from realmgen.population.calibration import (
    apply_population_to_settlements,
    compute_carrying_capacity,
    compute_farm_and_fish_food,
    compute_food_base,
    compute_hinterland_shares,
    compute_miners,
    compute_rural_population,
    compute_trade_wealth,
    compute_urban_population,
    cull_settlements,
)
from realmgen.population.trade_spawn import spawn_trade_settlements
from realmgen.scoring.suitability import _positions_to_face_ids
from realmgen.persistence import pack_settlements, unpack_settlements
from realmgen.trade.network import GOOD_RARITY_WEIGHTS, RARE_BIO_TYPE_OFFSET, TradeResult, compute_trade_network
from viz.color_stops import interp_stops_scalar
from worldgen.cost.channels import COST_SURFACES, IS_WATER
from worldgen.geology.channels import OCEAN_MASK
from worldgen.geology.display import OCEAN_COLOR, is_water_meta
from worldgen.hydrology.channels import LAKE_MASK
from worldgen.resources.channels import FERTILITY_MEAN, MINERALS, RARE_BIO


# ── Params ────────────────────────────────────────────────────────────────────


@paramclass
class TradeParams:
    port_cost: float = _f(5.0, label="Port cost", min=0.0, max=50.0)
    river_switch_cost: float = _f(0.0, label="River switch cost", min=0.0, max=10.0)
    # Cost of a land↔water landing away from a settlement while relaxing; where trade
    # still pays it, a port town is founded.  The final network has no landings.
    landing_cost: float = _f(12.0, label="Fallback landing cost", min=0.0, max=200.0)
    flow_decay: float = _f(0.05, label="Flow decay", min=0.001, max=1.0)
    # Route cost is ~1 per 100 km of open land, ~2 on icecap.
    deposit_decay: float = _f(0.2, label="Deposit yield decay with distance", min=0.0, max=2.0)
    hub_display_quantile: float = _f(0.90, label="Hub display quantile", min=0.5, max=0.99)


@paramclass
class TradeSpawnParams:
    hub_spawn_quantile: float = _f(0.97, label="Hub spawn quantile", min=0.80, max=1.00)
    hub_min_suitability: float = _f(0.15, label="Hub min suitability", min=0.0, max=0.5)
    hub_exclusion: float = _f(5.0, label="Hub exclusion radius", min=1.0, max=30.0)
    waypoint_min_cost: float = _f(30.0, label="Waypoint min cost dist", min=5.0, max=200.0)
    waypoint_min_suitability: float = _f(0.10, label="Waypoint min suitability", min=0.0, max=0.5)
    waypoint_exclusion: float = _f(3.0, label="Waypoint exclusion radius", min=1.0, max=20.0)
    port_spawn_quantile: float = _f(0.50, label="Port spawn quantile", min=0.0, max=1.00)
    port_min_suitability: float = _f(0.05, label="Port min suitability", min=0.0, max=0.5)
    port_exclusion: float = _f(2.0, label="Port exclusion radius", min=0.0, max=20.0)


@paramclass
class PopulationParams:
    relaxation_passes: int = _f(2, label="Relaxation passes", min=1, max=5)
    rural_density_max: float = _f(14.0, label="Max rural density (people/km² at fertility 1)", min=1.0, max=100.0)
    # A land face's own shallows feed fishers on its shore; a water face's feed the
    # cities that reach it.
    fishing_density_max: float = _f(
        8.0, label="Max fishing density (people/km² of water at aquatic fertility 1)", min=0.0, max=100.0)
    fertility_exponent: float = _f(2.0, label="Fertility exponent γ (density contrast)", min=0.5, max=4.0)
    surplus_rate: float = _f(0.15, label="Agricultural surplus rate s", min=0.0, max=0.5)
    # ``s × cap`` is the whole region's urban headroom; the major city gets only this
    # share, the rest feeds towns (docs/plans/city_population_rework_plan.md §5 —
    # don't double-count).
    city_surplus_share: float = _f(0.04, label="Major-city share of regional surplus", min=0.0, max=1.0)
    trade_food_weight: float = _f(1500.0, label="Trade food weight β (people per flow unit)", min=0.0, max=10_000.0)
    city_min: float = _f(7_000.0, label="City existence floor (people)", min=0.0, max=100_000.0)
    # Share of the trade people fed by a city's own minerals who live at the deposits.
    mining_share: float = _f(0.5, label="Share of mineral-trade people who are miners", min=0.0, max=1.0)
    hinterland_max_distance: float = _f(40.0, label="Hinterland max distance", min=5.0, max=200.0)
    hinterland_influence_scale: float = _f(4.5, label="Hinterland influence scale", min=0.1, max=10.0)
    hinterland_reach_scale: float = _f(9.0, label="Hinterland reach scale", min=0.1, max=30.0)
    hinterland_water_cost: float = _f(
        20.0, label="Hinterland water step cost (coastal fishing reach)", min=0.0, max=200.0)
    trade_spawn: TradeSpawnParams = field(
        default_factory=TradeSpawnParams, metadata={"label": "Trade-spawn settlements"}
    )


# ── Display ───────────────────────────────────────────────────────────────────

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


def _make_hinterland_color(settlements: "list[Settlement]"):
    """Return a color-scheme callable that maps hinterland_id → display-tier color."""
    max_pop = max((s.population for s in settlements), default=0.0)
    display_tiers = [
        min(population_display_tier(s.population, max_pop), len(_HINTERLAND_TIER_COLORS) - 1)
        for s in settlements
    ]

    def fn(meta: dict) -> tuple:
        if is_water_meta(meta):
            return OCEAN_COLOR
        hid = meta.get("hinterland_id", -1)
        if hid is None or int(hid) < 0 or int(hid) >= len(display_tiers):
            return _HINTERLAND_UNASSIGNED
        return _HINTERLAND_TIER_COLORS[display_tiers[int(hid)]]

    return fn


def _trade_flow_color(meta: dict, _max_flow: list) -> tuple:
    if is_water_meta(meta):
        return OCEAN_COLOR
    f = float(meta.get("trade_flow", 0.0))
    mx = _max_flow[0]
    s = min(f / mx, 1.0) if mx > 0 else 0.0
    return interp_stops_scalar(s, _TRADE_FLOW_STOPS)


# ── PopulationStage ───────────────────────────────────────────────────────────


class PopulationStage(Stage):
    """Trade network, relaxation loop, and population calibration."""

    name = "realm_population"
    requires = (IS_WATER, COST_SURFACES, MINERALS, RARE_BIO, FERTILITY_MEAN, SUITABILITY)
    provides = (TRADE_FLOW, POPULATION, HINTERLAND_ID)

    def __init__(
        self,
        trade_params: TradeParams,
        population_params: PopulationParams,
        seeder_params: SeederParams,
        danger_params=None,
        defensibility_params=None,
    ) -> None:
        self.trade_params = trade_params
        self.population_params = population_params
        self.seeder_params = seeder_params
        self.danger_params = danger_params
        self.defensibility_params = defensibility_params
        self.settlements: Optional[list[Settlement]] = None
        self.trade_result = None
        self.population: Optional[np.ndarray] = None
        self.hinterland: Optional[np.ndarray] = None
        self.rural_population: Optional[np.ndarray] = None
        # Final-pass inputs to the urban populations, aligned with ``settlements``.
        self.carrying_cap: Optional[np.ndarray] = None
        self.trade_wealth: Optional[np.ndarray] = None
        # (M,) miners at each deposit of the ``minerals`` feature (index-aligned).
        self.mine_workers: Optional[np.ndarray] = None

    def build(self, ctx: BuildContext) -> None:
        world = ctx.world
        mesh = ctx.mesh
        n = len(mesh.faces)
        tp = self.trade_params
        pp = self.population_params
        sp = self.seeder_params

        is_water = IS_WATER.read(mesh)
        suitability = SUITABILITY.read(mesh)
        fertility = FERTILITY_MEAN.read(mesh).clip(0.0, 1.0).astype(np.float64)
        centroids = np.asarray(mesh.face_centroids, dtype=np.float64)
        face_areas = np.asarray(mesh.face_areas, dtype=np.float64)

        danger = np.zeros(n, dtype=np.float32)
        if self.danger_params is not None:
            from worldgen.danger.channels import DANGER_ARRAY
            if DANGER_ARRAY.present(world):
                danger = DANGER_ARRAY.read(world).astype(np.float32)

        # The countryside fills to its Malthusian ceiling regardless of cities: a land
        # face houses its farmers and its shallows' fishers; water faces house none.
        fert_tex = mesh.face_textures.get("fertility")
        if fert_tex is not None and OCEAN_MASK.present(mesh):
            water_tex = OCEAN_MASK.read(mesh)[..., 0] > 0.5
            if LAKE_MASK.present(mesh):
                water_tex |= LAKE_MASK.read(mesh)[..., 0] > 0.5
            farm, fish = compute_farm_and_fish_food(
                np.asarray(fert_tex[..., 0]), water_tex, face_areas,
                density_max=pp.rural_density_max,
                fishing_density_max=pp.fishing_density_max,
                fertility_exponent=pp.fertility_exponent,
            )
        else:
            farm = compute_food_base(
                fertility, face_areas,
                density_max=pp.rural_density_max,
                fertility_exponent=pp.fertility_exponent,
            )
            fish = np.zeros(n, dtype=np.float64)
        food_base = farm + fish
        self.rural_population = compute_rural_population(food_base, is_water)
        RURAL_POPULATION.write(mesh, self.rural_population)
        FISHING_POPULATION.write(mesh, compute_rural_population(fish, is_water))
        # All of it is countryside until the polity and town stages carve forts and towns out.
        TOWN_POPULATION.write(mesh, np.zeros(n, dtype=np.float64))
        FORT_POPULATION.write(mesh, np.zeros(n, dtype=np.float64))
        FORT_LEVY.write(mesh, np.zeros(n, dtype=np.float64))

        self.settlements = list(world.get_feature("settlements"))

        # ── Cost graphs (land + sea) ─────────────────────────────────────────
        # ``hinterland_edge_costs`` alone makes water passable, at a steep per-step
        # cost, so coastal settlements reach a limited band of fishing grounds.  It
        # must never leak into trade, placement, or polity formation.
        danger_scale = (
            float(self.danger_params.movement_scale) if self.danger_params is not None else 0.0
        )

        if COST_SURFACES.present(world):
            from worldgen.cost.surfaces import build_land_edge_costs, build_sea_edge_costs
            cs = COST_SURFACES.read(world)
            settlement_mask = np.zeros(n, dtype=bool)
            for s in self.settlements:
                settlement_mask[s.face_id] = True
            land_base_cost = cs.land_base_cost + danger_scale * danger
            edge_costs = build_land_edge_costs(
                mesh,
                land_base_cost,
                cs.land_passable,
                cs.river_flow_dir,
                cs.river_flow_mag,
                cs.river_navigability,
                settlement_mask=settlement_mask,
            )
            hinterland_land_cost = np.where(
                is_water, pp.hinterland_water_cost,
                cs.land_base_cost + danger_scale * danger,
            )
            hinterland_edge_costs = build_land_edge_costs(
                mesh,
                hinterland_land_cost,
                np.ones(n, dtype=bool),            # water passable for hinterlands only
                cs.river_flow_dir,
                cs.river_flow_mag,
                cs.river_navigability,
            )
            sea_edge_costs = build_sea_edge_costs(
                mesh, cs.sea_base_cost, cs.sea_passable,
                wind_dir=cs.wind_dir,
            )
            sea_passable = cs.sea_passable
        else:
            base = np.where(is_water, np.inf, 1.5) + danger_scale * danger
            edge_costs = [
                np.array([0.5 * (base[i] + base[j]) for j in adj], dtype=np.float64)
                for i, adj in enumerate(mesh.face_adjacency)
            ]
            hinterland_base = np.where(
                is_water, pp.hinterland_water_cost, 1.5 + danger_scale * danger
            )
            hinterland_edge_costs = [
                np.array(
                    [0.5 * (hinterland_base[i] + hinterland_base[j]) for j in adj],
                    dtype=np.float64,
                )
                for i, adj in enumerate(mesh.face_adjacency)
            ]
            sea_passable = np.zeros(n, dtype=bool)
            sea_edge_costs = [np.empty(0, dtype=np.float64) for _ in range(n)]

        cost_graph = build_cost_graph(mesh.face_adjacency, edge_costs)
        hinterland_cost_graph = build_cost_graph(mesh.face_adjacency, hinterland_edge_costs)

        # ── Good sources (minerals + rare bio) ──────────────────────────────
        # Minerals first: the first ``n_minerals`` sources are the ``minerals`` deposits in order.
        good_sources: list[tuple] = []
        n_minerals = 0
        for channel, type_offset in ((MINERALS, 0), (RARE_BIO, RARE_BIO_TYPE_OFFSET)):
            if channel.present(world):
                positions, kinds, abundances = channel.read(world)
                if len(positions) > 0:
                    fids = _positions_to_face_ids(positions, centroids)
                    for fid, kind, ab in zip(fids.tolist(), kinds.tolist(), abundances.tolist()):
                        good_sources.append((fid, ab, int(kind) + type_offset))
                    if channel is MINERALS:
                        n_minerals = len(positions)

        tier_configs = [
            (sp.tier0.exclusion_radius, sp.tier0.score_threshold, sp.tier0.initial_population),
            (sp.tier1.exclusion_radius, sp.tier1.score_threshold, sp.tier1.initial_population),
            (sp.tier2.exclusion_radius, sp.tier2.score_threshold, sp.tier2.initial_population),
            (sp.tier3.exclusion_radius, sp.tier3.score_threshold, sp.tier3.initial_population),
        ]

        def _compute_trade(settlements, settlement_demands=None, landings=True):
            return compute_trade_network(
                mesh, settlements, good_sources,
                edge_costs, ~is_water, sea_edge_costs, sea_passable,
                port_cost=tp.port_cost,
                river_switch_cost=tp.river_switch_cost,
                landing_cost=tp.landing_cost if landings else np.inf,
                flow_decay=tp.flow_decay,
                deposit_decay=tp.deposit_decay,
                hub_display_quantile=tp.hub_display_quantile,
                settlement_demands=settlement_demands,
            )

        population = np.array([s.population for s in self.settlements], dtype=np.float64)
        carrying_cap = np.zeros_like(population)
        trade_wealth = np.zeros_like(population)

        ctx.report("Building trade network…")
        self.trade_result = _compute_trade(self.settlements, settlement_demands=population)
        ctx.report(
            f"Initial trade done: {len(self.trade_result.routes)} route segments, "
            f"{self.trade_result.hub_mask.sum()} hub faces."
        )

        # ── Relaxation loop ──────────────────────────────────────────────────
        tsp = pp.trade_spawn
        n_passes = max(1, int(pp.relaxation_passes))

        for pass_idx in range(n_passes):
            ctx.report(f"Relaxation pass {pass_idx + 1}/{n_passes}…")

            # Sync s.population so hinterland sees this pass's values.
            self.settlements = [
                dataclasses.replace(s, population=float(p))
                for s, p in zip(self.settlements, population)
            ]

            new_settlements = spawn_trade_settlements(
                n, self.settlements, self.trade_result.face_flow,
                suitability, is_water, cost_graph, centroids, tier_configs,
                hub_spawn_quantile=tsp.hub_spawn_quantile,
                hub_min_suitability=tsp.hub_min_suitability,
                hub_exclusion=tsp.hub_exclusion,
                waypoint_min_cost=tsp.waypoint_min_cost,
                waypoint_min_suitability=tsp.waypoint_min_suitability,
                waypoint_exclusion=tsp.waypoint_exclusion,
                landing_flow=self.trade_result.landing_flow,
                port_spawn_quantile=tsp.port_spawn_quantile,
                port_min_suitability=tsp.port_min_suitability,
                port_exclusion=tsp.port_exclusion,
            )
            if new_settlements:
                self.settlements = self.settlements + new_settlements
                new_pops = np.array([s.population for s in new_settlements], dtype=np.float64)
                population = np.concatenate([population, new_pops])
                ctx.report(f"  Trade-spawn: +{len(new_settlements)} ({len(self.settlements)} total).")

            shares, self.hinterland = compute_hinterland_shares(
                n, self.settlements, hinterland_cost_graph,
                max_distance=pp.hinterland_max_distance,
                influence_scale=pp.hinterland_influence_scale,
                reach_scale=pp.hinterland_reach_scale,
            )
            carrying_cap = compute_carrying_capacity(shares, food_base)
            trade_wealth = compute_trade_wealth(
                self.settlements, self.trade_result.face_flow.astype(np.float64),
            )
            population = compute_urban_population(
                carrying_cap, trade_wealth,
                surplus_rate=pp.surplus_rate * pp.city_surplus_share,
                trade_food_weight=pp.trade_food_weight,
            )
            self.settlements, population, kept = cull_settlements(
                self.settlements, population, min_population=pp.city_min,
            )
            carrying_cap = carrying_cap[kept]
            trade_wealth = trade_wealth[kept]
            ctx.report(f"  {len(self.settlements)} cities after cull.")
            if not self.settlements:
                break

            if pass_idx == n_passes - 1:
                break    # the final trade network below replaces this one before anything reads it
            max_p = float(population.max())
            norm_demands = population / max_p if max_p > 0 else np.ones_like(population)
            self.trade_result = _compute_trade(self.settlements, settlement_demands=norm_demands)

        self.settlements = apply_population_to_settlements(self.settlements, population)
        self.population = population
        self.carrying_cap = carrying_cap
        self.trade_wealth = trade_wealth

        if self.defensibility_params is not None and self.settlements and COST_SURFACES.present(world):
            from realmgen.polity.seeding import compute_settlement_defensibility
            cs = COST_SURFACES.read(world)
            sv = (
                SITUATIONAL_VALUE.read(mesh).astype(np.float64)
                if SITUATIONAL_VALUE.present(mesh) else np.zeros(n, dtype=np.float64)
            )
            dp = self.defensibility_params
            defs, has_walls_arr, wall_str_arr = compute_settlement_defensibility(
                [s.face_id for s in self.settlements],
                cs.land_base_cost, sv, cs.river_flow_mag,
                trade_wealth, danger, ~is_water,
                int(world.seed) if world.seed is not None else 0,
                rugged_weight=dp.rugged_weight,
                situational_weight=dp.situational_weight,
                river_weight=dp.river_weight,
                wall_base_prob=dp.wall_base_prob,
                wall_danger_boost=dp.wall_danger_boost,
                wall_wealth_boost=dp.wall_wealth_boost,
                wall_min_strength=dp.wall_min_strength,
                wall_max_strength=dp.wall_max_strength,
            )
            self.settlements = [
                dataclasses.replace(
                    s,
                    defensibility=float(defs[k]),
                    has_walls=bool(has_walls_arr[k]),
                    wall_strength=float(wall_str_arr[k]),
                )
                for k, s in enumerate(self.settlements)
            ]

        _shares, self.hinterland = compute_hinterland_shares(
            n, self.settlements, hinterland_cost_graph,
            max_distance=pp.hinterland_max_distance,
            influence_scale=pp.hinterland_influence_scale,
            reach_scale=pp.hinterland_reach_scale,
        )
        if self.settlements:
            # No landings now: every land↔water hop is at a settlement.
            self.trade_result = _compute_trade(self.settlements, settlement_demands=population, landings=False)

        # ── Miners: out of the cities, onto their deposits ───────────────────
        mine_faces = np.array([f for f, _, _ in good_sources[:n_minerals]], dtype=np.int64)
        miners = np.zeros(n_minerals, dtype=np.float64)
        if self.settlements and n_minerals:
            tr = self.trade_result
            face_flow = tr.face_flow.astype(np.float64)
            s_faces = np.array([s.face_id for s in self.settlements], dtype=np.int64)
            own_share = np.where(
                face_flow[s_faces] > 0, tr.goods_flow / np.maximum(face_flow[s_faces], 1e-300), 0.0)
            rarity = np.array([GOOD_RARITY_WEIGHTS.get(int(g), 1.0) for _, _, g in good_sources])
            workable = np.zeros(len(good_sources), dtype=bool)
            workable[:n_minerals] = ~np.asarray(is_water, dtype=bool)[mine_faces]
            miners = compute_miners(
                population, pp.trade_food_weight * trade_wealth, own_share,
                tr.deposit_owner, tr.deposit_credit * rarity, workable,
                mining_share=pp.mining_share, city_min=pp.city_min,
            )[:n_minerals]
            owner = tr.deposit_owner[:n_minerals]
            population = population - np.bincount(
                owner[owner >= 0], weights=miners[owner >= 0], minlength=len(population))
            self.settlements = apply_population_to_settlements(self.settlements, population)
            self.population = population
        mining = np.bincount(mine_faces, weights=miners, minlength=n) if n_minerals else np.zeros(n)
        self.mine_workers = miners
        # The miners are countryside until the town stage gathers them into mining towns.
        self.rural_population = self.rural_population + mining
        RURAL_POPULATION.write(mesh, self.rural_population)
        MINING_POPULATION.write(mesh, mining)
        world.add_feature("mine_workers", miners)

        TRADE_FLOW.write(mesh, self.trade_result.face_flow)
        mesh.apply_metadata_channel("is_trade_hub", self.trade_result.hub_mask.tolist())

        pop_arr = np.zeros(n, dtype=np.float64)
        for k, s in enumerate(self.settlements):
            pop_arr[s.face_id] = population[k]
        POPULATION.write(mesh, pop_arr)
        HINTERLAND_ID.write(mesh, self.hinterland.tolist())

        max_pop = float(population.max()) if population.size > 0 else 0.0
        tier_arr = np.full(n, -1, dtype=np.int32)
        for k, s in enumerate(self.settlements):
            tier_arr[s.face_id] = population_display_tier(population[k], max_pop)
        mesh.apply_metadata_channel("settlement_tier", tier_arr.tolist())

        world.add_feature("settlements", self.settlements)
        world.add_feature("trade_routes", self.trade_result.routes)
        # (S, S), read by the polity stage for loyalty secession; guard a stale shape.
        attribution = self.trade_result.pair_flow
        if attribution.shape != (len(self.settlements), len(self.settlements)):
            attribution = np.zeros((len(self.settlements), len(self.settlements)), dtype=np.float32)
        world.add_feature("trade_attribution", attribution)
        rural_total = float(self.rural_population.sum())
        urban_total = float(population.sum())
        ctx.report(
            f"Population & calibration done: {len(self.settlements)} cities "
            f"(urban {urban_total / 1e6:.1f} M, rural {rural_total / 1e6:.0f} M people), "
            f"{len(self.trade_result.routes)} route segments."
        )

    def register_display(self, world, reg: DisplayRegistry, *, debug: bool = False) -> None:
        if debug and self.trade_result is not None:
            max_flow = float(self.trade_result.face_flow.max()) if self.trade_result.face_flow.size else 0.0
            _max_flow_ref = [max_flow]
            reg.add_scheme(
                "Trade Flow",
                lambda meta, _ref=_max_flow_ref: _trade_flow_color(meta, _ref),
            )
        if debug and self.hinterland is not None and self.settlements is not None:
            reg.add_scheme("Hinterland", _make_hinterland_color(self.settlements))
        if self.settlements is not None:
            reg.add_overlay("settlements", "Settlements", checked=True)
        has_routes = (
            (self.trade_result is not None and bool(self.trade_result.routes))
            or (world is not None and world.has_feature("trade_routes"))
        )
        if has_routes:
            reg.add_overlay("trade_routes", "Trade Routes", checked=True)

        def _settlement_hover(face: int, _uv, w) -> list:
            meta = w.mesh.face_metadata[face]
            tier = meta.get("settlement_tier")
            if tier is None or int(tier) < 0:
                return []
            parts = [TIER_NAMES[min(int(tier), len(TIER_NAMES) - 1)]]
            pop = meta.get("population")
            if pop is not None and float(pop) > 0:
                parts.append(f"Pop: {format_population(float(pop))}")
            return parts

        # Region and local views name the city under the cursor instead.
        reg.add_hover(_settlement_hover, world_only=True)

    # ── Persistence ───────────────────────────────────────────────────────────
    # Settlements (positions refined in place by RefineStage) are saved here; per-face
    # channels round-trip via metadata, so load() re-derives the attrs from the mesh.

    def serialize(self) -> dict:
        if self.settlements:
            out = pack_settlements(self.settlements)
            if self.mine_workers is not None:
                out["mine_workers"] = self.mine_workers
            return out
        return {}

    def load(self, arrays, ctx: BuildContext) -> None:
        world, mesh = ctx.world, ctx.mesh
        if "settle_face_id" in arrays:
            self.settlements = unpack_settlements(arrays)
            world.add_feature("settlements", self.settlements)
        if "mine_workers" in arrays:
            self.mine_workers = np.asarray(arrays["mine_workers"], dtype=np.float64)
            world.add_feature("mine_workers", self.mine_workers)
        if RURAL_POPULATION.present(mesh):
            self.rural_population = RURAL_POPULATION.read(mesh)
        if HINTERLAND_ID.present(mesh):
            self.hinterland = HINTERLAND_ID.read(mesh)
        if POPULATION.present(mesh):
            self.population = POPULATION.read(mesh)
        if TRADE_FLOW.present(mesh):
            face_flow = TRADE_FLOW.read(mesh).astype(np.float32)
            hub_mask = np.array(
                [bool(m.get("is_trade_hub", False)) for m in mesh.face_metadata],
                dtype=bool,
            )
            # Routes are restored by RefineStage.load; the debug scheme only reads face_flow.
            self.trade_result = TradeResult(face_flow=face_flow, hub_mask=hub_mask, routes=[])
