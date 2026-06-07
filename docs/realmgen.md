# Realmgen — Human Geography Layer

See also: `docs/plans/settlement_generation_design.md` for the original design.

## Overview

Standalone module downstream of worldgen. Takes a fully-built `World` (read-only physical data) and attaches human geography features. Integrated via `RealmPipeline(RealmParams()).build(BuildContext(world=globe))` — not wired into `GlobePipeline`; keeps realmgen optional.

## Files
- `realmgen/builder.py` — `RealmParams`, `RealmPipeline`
- `realmgen/channels.py` — MetaChannel definitions for realm outputs
- `realmgen/scoring/builder.py` — `ScoringStage`, `SuitabilityParams`, `SituationalValueParams` (+ sub-params)
- `realmgen/scoring/suitability.py` — pure suitability functions
- `realmgen/scoring/situational_value.py` — pure situational-value functions
- `realmgen/placement/builder.py` — `PlacementStage`, `SeederParams`, `TierParams`
- `realmgen/placement/seeder.py` — `placement_score`, `build_cost_graph`, `seed_settlements`
- `realmgen/placement/types.py` — `Settlement` dataclass, display constants
- `realmgen/population/builder.py` — `PopulationStage`, `TradeParams`, `TradeSpawnParams`, `PopulationParams`
- `realmgen/population/calibration.py` — pure population functions
- `realmgen/population/trade_spawn.py` — `spawn_trade_settlements` (adds hub/waypoint settlements per relaxation pass)
- `realmgen/trade/network.py` — `compute_trade_network`
- `realmgen/polity/builder.py` — `PolityStage`, `PolityParams`
- `realmgen/polity/formation.py` — pure polity/territory functions
- `realmgen/polity/types.py` — `Polity` dataclass + `POLITY_COLORS` palette

## `builder.py`

**`RealmParams`**: Nested sub-params — `suitability: SuitabilityParams`, `situational: SituationalValueParams`, `seeder: SeederParams`, `trade: TradeParams`, `population: PopulationParams`, `polity: PolityParams`.

**`RealmPipeline`** (extends `Pipeline`): Four-stage pipeline:
1. `ScoringStage` — suitability + situational value
2. `PlacementStage` — settlement seeding
3. `PopulationStage` — trade network, relaxation loop, population calibration
4. `PolityStage` — territory growth + border tension

After `build(ctx)`, each stage stores its results as attributes. `apply_display(window, debug=False)` merges realm color schemes (Suitability, Situational Value, Trade Flow, Hinterland, Polities, Border Tension), overlays (Settlements, Trade Routes), and chains hover with population/polity display.

Writes per-face metadata: `"suitability"`, `"situational_value"`, `"settlement_tier"`, `"trade_flow"`, `"is_trade_hub"`, `"population"`, `"hinterland_id"`, `"polity_id"`, `"border_tension"`.

Registers features: `"settlements"` (list of `Settlement` with `.population` set), `"trade_routes"`, `"polities"` (list of `Polity`), `"polity_tensions"` (list of `(a, b, score)`).

## Scoring

**`SuitabilityParams`**: Nested `water: WaterProximityParams`, `climate: ClimateHabitabilityParams`, `ruggedness: RuggednessParams`, `resource: ResourceAccessParams`, `weights: SuitabilityWeightParams`.

**`SituationalValueParams`**: Nested `harbor: HarborQualityParams`, `passes: MountainPassParams`, `weights: SituationalValueWeightParams`.

**`scoring/suitability.py`**: Pure functions for five suitability factors — water proximity (freshwater BFS + coastal distance), climate habitability (T/P bell curve), ruggedness penalty (elevation range), resource proximity (BFS from mineral/rare-bio deposit faces), weighted combination.

**`scoring/situational_value.py`**: Pure functions for five situational-value components — harbor quality (coastal faces, depth-scaled), river confluences, navigable river heads, mountain passes (local cost contrast), strait adjacency.

## Placement

**`SeederParams`**: `situational_weight: float`, `tier0`–`tier3: TierParams` (Capitals → Cities → Towns → Villages). Each `TierParams` has `exclusion_radius`, `score_threshold`, `influence`, and `demand`.

**`placement/seeder.py`**:
- `placement_score` — blend suitability + situational value
- `build_cost_graph` — convert per-face edge-cost list to scipy CSR sparse matrix
- `seed_settlements` — hierarchical greedy Dijkstra-exclusion seeder: tier 0 capitals first with widest radii, cumulative forbidden mask, scipy Dijkstra for fast cost-distance lookup

## Trade

**`trade/network.py`**: `compute_trade_network` — routes trade flows from resource deposits (minerals + rare bio) to settlements over the combined land + sea graph. Returns `face_flow` (per-face scalar), `routes` (polylines for rendering), and `hub_mask` (top-flow faces).

`TradeParams`: `port_cost`, `river_switch_cost`, `flow_decay`, `hub_display_quantile`.

## Population

**`PopulationParams`**: `relaxation_passes`, `trade_weight`, `min_pop_fraction`, `hinterland_max_distance`, `hinterland_influence_scale`, `hinterland_reach_scale`, `hinterland_water_cost`, `trade_spawn: TradeSpawnParams`.

**`PopulationStage`** runs a relaxation loop for `relaxation_passes` iterations: each pass spawns trade-hub and waypoint settlements (`trade_spawn.py`), recomputes hinterland shares, carrying capacity, trade wealth, population, culls weak settlements, then reruns the trade network with population-weighted demands.

**`population/calibration.py`**: Pure functions:
- `compute_hinterland_shares` — soft fractional multi-source undirected Dijkstra; passes a *water-permeable* graph so coastal settlements claim a limited coastal band for fishing (bounded by `hinterland_water_cost` × `hinterland_max_distance`)
- `compute_carrying_capacity` — Σ shares · fertility · face_area per settlement (`fertility_mean` already carries aquatic fishing productivity on coastal water faces)
- `compute_trade_wealth` — face_flow at settlement
- `compute_population` — carrying_cap × log-trade bonus
- `cull_settlements` — remove settlements below `min_pop_fraction` of peak
- `apply_population_to_settlements` — stamp population onto Settlement objects

**`population/trade_spawn.py`**: `spawn_trade_settlements` — identifies high-flow hub faces and remote waypoint faces and seeds additional settlements there; called once per relaxation pass.

## Polity

**`PolityParams`**: `capital_min_separation`, `capital_min_pop_fraction`, `power_bias`, `growth_max_reach`, `tension_barrier_scale`, `tension_flow_ref`, `tension_artery_weight`.

**`polity/formation.py`**: Pure functions:
- `seed_capitals` — pick polity capitals from settlements by population, spaced by min cost-distance
- `grow_territory` — power-weighted Voronoi on land cost graph: each face joins the capital minimising `cost_dist − power_bias·log(power/max_power)`; stronger capitals claim more, borders fall on barriers/coasts, unreachable islands stay unclaimed `-1`
- `compute_polity_power` — total member-settlement population per polity
- `compute_border_tension` — per-crossing `permeability · balance · artery`, where `artery = (1−w) + w·tanh(flow)` keeps a trade-independent baseline; reduced by `max` over each border's chokepoints

## Key design constraint

The land cost surface is land-only (`land_passable = ~is_water`), so polity formation uses a land-only graph — coastlines become natural borders, islands stay unclaimed. The **only** place that opts into limited water travel is the population stage's hinterland graph (coastal fishing grounds). That permeability never reaches polity formation.

Maritime reach (port-projected island/strait influence) is deferred — not yet implemented.
