# Realmgen — Human Geography Layer

See also: `docs/plans/settlement_generation_design.md` for the original design.

## Overview

Standalone module downstream of worldgen. Takes a fully-built `Globe` (read-only) and attaches human geography features. Integrated via `RealmBuilder.build(globe)` — not wired into `GlobeBuilder`; keep realmgen optional.

## Files
- `realmgen/builder.py` — `RealmParams`, `RealmStage`, `RealmPipeline`
- `realmgen/scoring/suitability.py`
- `realmgen/scoring/situational_value.py`
- `realmgen/placement/types.py`
- `realmgen/placement/seeder.py`
- `realmgen/population/calibration.py`
- `realmgen/polity/formation.py`
- `realmgen/polity/types.py`
- `realmgen/polity/builder.py`

## `builder.py`

**`RealmParams`**: Nested sub-params — `water: WaterProximityParams`, `climate: ClimateHabitabilityParams`, `ruggedness: RuggednessParams`, `resource: ResourceAccessParams`, `weights: SuitabilityWeightParams`, `situational: SituationalValueParams`, `seeder: SeederParams`, `trade: TradeParams`, `population: PopulationParams`, `polity: PolityParams`. All leaf fields carry `label`/`min`/`max` metadata.

**`RealmStage` / `RealmPipeline`**: Four-stage pipeline (scoring → placement → population → polity). `build(ctx)` reads the full physical Globe, computes suitability + situational value, seeds settlements, runs the trade network, runs the population & calibration relaxation loop, then grows polities.

Writes per-face metadata: `"suitability"`, `"situational_value"`, `"settlement_tier"`, `"trade_flow"`, `"population"`, `"hinterland_id"`, `"polity_id"`, `"border_tension"`.

Registers features: `"settlements"` (list of `Settlement` with `.population` set), `"trade_routes"`, `"polities"` (list of `Polity`), `"polity_tensions"` (list of `(a, b, score)`).

`RealmPipeline.apply_display(window)` merges realm color schemes (Suitability, Situational Value, Trade Flow, Hinterland, Polities, Border Tension), overlays (Settlements, Trade Routes), and chains hover with population/polity display.

## Scoring

**`scoring/suitability.py`**: Pure testable functions for five suitability factors (water proximity, climate habitability, ruggedness penalty, resource proximity, weighted combination).

**`scoring/situational_value.py`**: Pure testable functions for five situational-value components (harbor quality, river confluences, navigable heads, mountain passes, strait adjacency).

## Placement

**`placement/types.py`**: `Settlement` dataclass (`face_id`, `position (3,)`, `tier`, `score`, `population`) and display constants (`TIER_NAMES`, `TIER_COLORS`, `TIER_RADII`).

**`placement/seeder.py`**:
- `placement_score` — blend suitability + situational value
- `build_cost_graph` — convert per-face edge-cost list to scipy CSR sparse matrix
- `seed_settlements` — hierarchical greedy Dijkstra-exclusion seeder: tier 0 capitals first with widest radii, cumulative forbidden mask, scipy Dijkstra for fast cost-distance lookup

## Population

**`population/calibration.py`**: Stage-4 pure functions:
- `compute_hinterland_shares` — soft fractional multi-source undirected Dijkstra; each settlement bids for every face within `max_distance`, capacity split among bidders. The population stage passes a *water-permeable* hinterland graph so coastal settlements claim a **limited coastal band** of ocean/lake (water passable at a steep `hinterland_water_cost`, bounded by that cost × `max_distance`, ~1–2 rings off coast) and pull fishing productivity into carrying capacity.
- `compute_carrying_capacity` — Σ shares · fertility · face_area per settlement (`fertility_mean` already carries aquatic fishing productivity on coastal water faces)
- `compute_trade_wealth` — face_flow at settlement
- `compute_population` — cap × log-trade bonus
- `cull_settlements` — remove settlements below `min_pop_fraction` of peak; display tiers re-derived from population at render time
- `apply_population_to_settlements` — stamp population onto Settlement objects

## Polity

**`polity/formation.py`**: Stage-5 territory formation (land surface only). Pure functions:
- `seed_capitals` — pick polity capitals from settlements by population, spaced by min cost-distance
- `grow_territory` — power-weighted Voronoi on land cost graph: each face joins the capital minimising `cost_dist − power_bias·log(power/max_power)`; stronger capitals claim more, borders fall on barriers/coasts, unreachable islands stay unclaimed `-1`
- `compute_polity_power` — total member-settlement population per polity
- `compute_border_tension` — per-crossing `permeability · balance · artery`, where `artery = (1−w) + w·tanh(flow)` keeps a trade-independent baseline; reduced by `max` over each border's chokepoints

**`polity/types.py`**: `Polity` dataclass + `POLITY_COLORS` palette.

**`polity/builder.py`**: `PolityStage` — runs after population.

## Key design constraint

The land cost surface is land-only (`land_passable = ~is_water`), so polity formation uses a land-only graph — coastlines become borders, islands stay unclaimed. The **only** place that opts into limited water travel is the population stage's hinterland graph (coastal fishing grounds). That permeability never reaches polity formation.

Maritime reach (port-projected island/strait influence) is deferred — not yet implemented.
