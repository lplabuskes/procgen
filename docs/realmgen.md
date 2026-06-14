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
- `realmgen/polity/builder.py` — `PolityStage`, `PolityParams` (IO, cost graphs, tension, display)
- `realmgen/polity/forest.py` — `ForestBuilder` + `_Ctx` (the sovereign/vassal tree algorithm)
- `realmgen/polity/seeding.py` — defensibility, capital seeding, scalar rolls (Stage 0 pure functions)
- `realmgen/polity/authority.py` — authority field, claiming, classification (Stages 1-3 pure functions)
- `realmgen/polity/tension.py` — border-crossing tension
- `realmgen/polity/display.py` — color schemes
- `realmgen/polity/types.py` — `Polity` / `Scalars` / `ControlClass` + `POLITY_COLORS` palette

## `builder.py`

**`RealmParams`**: Nested sub-params — `suitability: SuitabilityParams`, `situational: SituationalValueParams`, `seeder: SeederParams`, `trade: TradeParams`, `population: PopulationParams`, `polity: PolityParams`.

**`RealmPipeline`** (extends `Pipeline`): Four-stage pipeline:
1. `ScoringStage` — suitability + situational value
2. `PlacementStage` — settlement seeding
3. `PopulationStage` — trade network, relaxation loop, population calibration
4. `PolityStage` — territory growth + border tension

After `build(ctx)`, each stage stores its results as attributes. `apply_display(window, debug=False)` merges realm color schemes (Suitability, Situational Value, Trade Flow, Hinterland, Polities, Border Tension; plus Sovereigns, Control Class, Authority Field under `--debug`), overlays (Settlements, Trade Routes), and chains hover with population/polity display.

Writes per-face metadata: `"suitability"`, `"situational_value"`, `"settlement_tier"`, `"trade_flow"`, `"is_trade_hub"`, `"population"` (urban headcount in people), `"rural_population"` (per-face people, land only), `"hinterland_id"`, `"polity_id"` (directly-governing), `"sovereign_id"` (root), `"control_class"`, `"border_tension"`.

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

All populations are **real people**, anchored to a medieval-tech Earth (~300–400 M world total). Design rationale: `docs/plans/city_population_rework_plan.md`. Three quantities: per-face **rural population** (`D_max · fertility^γ · area_km²` — the Malthusian food base, land only), per-settlement **carrying capacity** (food base summed over claimed hinterland shares), and per-settlement **urban population** (`s · share · cap + β · trade_flow` — additive trade lets a thin-hinterland chokepoint grow large on imported grain, the Constantinople/Venice case). A settlement exists as a city only if its urban population clears `city_min`; the modeled settlements are *major cities* — towns/villages are sub-face-scale and deliberately unmodeled (their slice of the regional surplus is reserved via `city_surplus_share`).

**`PopulationParams`**: `relaxation_passes`, `rural_density_max` (D_max, people/km² at fertility 1), `fertility_exponent` (γ ≈ 2 — restores density contrast), `surplus_rate` (s — regional agricultural surplus), `city_surplus_share` (major city's slice of `s·cap`; remainder reserved for the future town pipeline), `trade_food_weight` (β — people fed per unit trade flow), `city_min` (urban-formation floor in people), `hinterland_max_distance`, `hinterland_influence_scale`, `hinterland_reach_scale`, `hinterland_water_cost`, `trade_spawn: TradeSpawnParams`.

**`PopulationStage`** computes the per-face food base and rural population up front (written to `RURAL_POPULATION`), then runs a relaxation loop for `relaxation_passes` iterations: each pass spawns trade-hub and waypoint settlements (`trade_spawn.py`), recomputes hinterland shares, carrying capacity, trade wealth, urban population, culls cities below `city_min`, then reruns the trade network with population-weighted demands. Stores final per-settlement diagnostics (`carrying_cap`, `trade_wealth`) aligned with `settlements`/`population`. In debug mode registers a "Rural Population" color scheme.

**`population/calibration.py`**: Pure functions:
- `compute_food_base` — `D_max · fertility^γ · area_km²` per face (people the face can feed; aquatic fertility on coastal water counts)
- `compute_rural_population` — food base masked to land
- `compute_hinterland_shares` — soft fractional multi-source undirected Dijkstra; passes a *water-permeable* graph so coastal settlements claim a limited coastal band for fishing (bounded by `hinterland_water_cost` × `hinterland_max_distance`)
- `compute_carrying_capacity` — Σ shares · food_base per settlement
- `compute_trade_wealth` — face_flow at settlement
- `compute_urban_population` — `s_eff · cap + β · wealth` (additive trade)
- `cull_settlements` — drop settlements below the absolute `city_min` floor; also returns the keep mask for filtering aligned arrays
- `apply_population_to_settlements` — stamp population onto Settlement objects

Validation harnesses: `scripts/measure_population.py` (distribution shape — Zipf slope/R²/Gini) and `scripts/measure_population_absolute.py` (absolute grounding — world total, top-city band, trade-fed-city test).

**`population/trade_spawn.py`**: `spawn_trade_settlements` — identifies high-flow hub faces and remote waypoint faces and seeds additional settlements there; called once per relaxation pass.

## Polity

A polity is a **tree, not a flat partition** (`docs/plans/polity_generation_brief.md`): sovereigns are roots, vassals are children. The stage seeds sovereign capitals, projects an authority field from each over the land + sea cost surfaces, lets every face be claimed by the strongest projector, then classifies each settlement direct / vassal / independent by comparing projected authority against local power — recursing to carve vassal sub-territories. Per-polity scalars (`cohesion`, `infrastructure`, `legitimacy`, `maritime_reach`) — rolled from a per-capital RNG substream keyed on `world.seed` + capital face — modulate the field; their cross-product is the main generator of variety (Song ↔ Mongol ↔ HRE ↔ city-state belt).

**`PolityParams`** is composed of nested sub-paramclasses (one per concern). Those consumed only by the stage live in `polity/builder.py` — `seeding` (`CapitalSeedingParams`: `min_separation`, `min_pop_fraction`, `trade_weight`), `defensibility` (`DefensibilityParams`: `rugged_weight`, `situational_weight`), `tension` (`BorderTensionParams`: `barrier_scale`, `flow_ref`, `artery_weight`). Those consumed only by the forest algorithm live in `polity/forest.py` — `projection` (`ProjectionParams`: `base_range` — the master consolidation knob, `range_floor`, `sea_range_factor`, `legitimacy_spread`, `maritime_scale`, `cohesion_min`, `infrastructure_min`, `infrastructure_emergent`), `claiming` (`ClaimingParams`: `hysteresis`, `max_iters`, `outer_max_iters`, `max_capitals`), `hierarchy` (`HierarchyParams`: `direct_threshold`, `vassal_floor`, `cohesion_gain_floor`, `cohesion_gain_span`, `max_recursion_depth`).

**`Polity`** (`polity/types.py`): `capital_face/position`, the four scalars, `liege` (None ⇒ sovereign root), `autonomy`, `depth`, `power` (own-territory member population), `domain_power` (own + descendant vassals), `territory_faces`, `member_settlement_faces`, `vassals`. `ControlClass` (IntEnum) labels each settlement DIRECT / VASSAL / INDEPENDENT. `Scalars` (NamedTuple) carries the four per-polity scalar arrays with a `.take(idx)` sub-select.

**`polity/seeding.py`** (Stage 0 pure functions):
- `compute_defensibility` — terrain-derived hold-strength (ruggedness + situational value), a ≥1 multiplier on a settlement's population (no stored wall/garrison fields exist)
- `seed_capitals` — founder settlements weighted by population × trade × defensibility, spaced by cost-distance
- `roll_polity_scalars` / `cohesion_to_gain` — per-capital scalar rolls (deterministic, keyed on face → `Scalars`) and the cohesion→authority-gain map

**`polity/authority.py`** (Stages 1-3 pure functions):
- `compute_authority_fields` — `legitimacy · exp(−land_dist / range)` (range from infrastructure) max-combined with a gentler sea channel (`maritime_reach`, over the hybrid land/sea graph)
- `claim_faces` — argmax-authority face assignment with an incumbent hysteresis bonus, iterated to a bounded fixed point
- `classify_settlements` — direct/vassal/independent from `projecting_power[capital] · field · gain` vs `population · defensibility`

**`polity/forest.py`**: `ForestBuilder` threads the pure functions into the tree — `resolve_sovereigns` (claim + promote independents in an outer loop) then `_build_subtree` recursion (carve each sovereign into its directly-held core + vassal sub-territories). `_Ctx` holds the read-only shared inputs.

**`polity/tension.py`**: `compute_border_tension` — retained; scored on the final sovereign partition.

**Outputs**: per-face `POLITY_ID` (directly-governing), `SOVEREIGN_ID` (root), `CONTROL_CLASS`, `BORDER_TENSION`; features `polities` (the forest) and `polity_tensions`. Debug schemes (`--debug`): Authority Field (validates field saddles), Sovereigns (kingdom grouping), Control Class. Validation harness: `scripts/measure_polities.py`.

## Key design constraint

The land cost surface is land-only (`land_passable = ~is_water`), so land authority and claiming use a land-only graph — coastlines become natural borders. The sea authority channel reuses the hybrid land/sea graph (`realmgen/trade/network.py::build_base_hybrid_graph`): a high-`maritime_reach` coastal capital projects authority along coastlines and onto reachable islands; claiming stays restricted to land faces. Maritime reach scales with capital trade wealth, so a landlocked capital contributes none.
