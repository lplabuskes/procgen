# Realm Population Stage

Part of `realmgen/`. [Back to realmgen overview](realmgen.md)

## Files

- `realmgen/population/builder.py` — `PopulationStage`, `TradeParams`, `TradeSpawnParams`, `PopulationParams`
- `realmgen/population/calibration.py` — pure population functions
- `realmgen/population/trade_spawn.py` — `spawn_trade_settlements`
- `realmgen/trade/network.py` — `compute_trade_network`

## Design

All populations are **real people**, anchored to a medieval-tech Earth (~300–400 M world total). Design rationale: `docs/plans/city_population_rework_plan.md`. Three quantities: per-face **rural population** (`D_max · fertility^γ · area_km²` — the Malthusian food base, land only), per-settlement **carrying capacity** (food base summed over claimed hinterland shares), and per-settlement **urban population** (`s · share · cap + β · trade_flow` — additive trade lets a thin-hinterland chokepoint grow large on imported grain, the Constantinople/Venice case). A settlement exists as a city only if its urban population clears `city_min`; the modeled settlements are *major cities* — towns/villages are sub-face-scale and deliberately unmodeled (their slice of the regional surplus is reserved via `city_surplus_share`).

## Params

**`TradeParams`**: `port_cost`, `river_switch_cost`, `flow_decay`, `hub_display_quantile`.

**`TradeSpawnParams`**: `hub_spawn_quantile`, `hub_min_suitability`, `hub_exclusion`, `waypoint_min_cost`, `waypoint_min_suitability`, `waypoint_exclusion`.

**`PopulationParams`**: `relaxation_passes`, `rural_density_max` (D_max, people/km² at fertility 1), `fertility_exponent` (γ ≈ 2 — restores density contrast), `surplus_rate` (s — regional agricultural surplus), `city_surplus_share` (major city's slice of `s·cap`; remainder reserved for the future town pipeline), `trade_food_weight` (β — people fed per unit trade flow), `city_min` (urban-formation floor in people), `hinterland_max_distance`, `hinterland_influence_scale`, `hinterland_reach_scale`, `hinterland_water_cost`, `trade_spawn: TradeSpawnParams`.

## Trade network

**`trade/network.py`**: `compute_trade_network` — routes trade flows from resource deposits (minerals + rare bio) to settlements over the combined land + sea graph. Returns `face_flow` (per-face scalar), `routes` (polylines for rendering), `hub_mask` (top-flow faces), and `pair_flow` (the `(S, S)` directional inter-settlement trade matrix — `pair_flow[i, j]` = volume exporter *i* sends to importer *j*, index-aligned with `settlements`). Called at the start and after each relaxation pass with population-weighted demands.

The final-pass `pair_flow` is published by `PopulationStage` as the `trade_attribution` world feature (aligned with the `settlements` feature). It is per-settlement trade-route provenance — column *j* = settlement *j*'s imports by source, row *i* = its exports by destination — consumed by the polity stage's economic-grip / peripheral-wealth secession levers (a settlement whose trade is dominated by intra-polity partners has stronger economic incentive to stay).

## PopulationStage

Runs third in `RealmPipeline`. Reads `IS_WATER`, `COST_SURFACES`, `MINERALS`, `RARE_BIO`, `FERTILITY_MEAN`, `SUITABILITY`. Danger (`DangerInfluenceParams.movement_scale`) is added to land base cost in the cost graphs.

Computes the per-face food base and rural population up front (written to `RURAL_POPULATION`), then runs a relaxation loop for `relaxation_passes` iterations: each pass spawns trade-hub and waypoint settlements (`trade_spawn.py`), recomputes hinterland shares, carrying capacity, trade wealth, urban population, culls cities below `city_min`, then reruns the trade network with population-weighted demands.

After the relaxation loop, **defensibility is computed per settlement** and stamped onto the `Settlement` objects — see [Defensibility](#defensibility) below. Stores final per-settlement diagnostics (`carrying_cap`, `trade_wealth`) aligned with `settlements`/`population`. In debug mode registers a "Rural Population" color scheme.

## calibration.py

Pure functions:
- `compute_food_base` — `D_max · fertility^γ · area_km²` per face (people the face can feed; aquatic fertility on coastal water counts)
- `compute_rural_population` — food base masked to land
- `compute_hinterland_shares` — soft fractional multi-source undirected Dijkstra; passes a *water-permeable* graph so coastal settlements claim a limited coastal band for fishing (bounded by `hinterland_water_cost × hinterland_max_distance`)
- `compute_carrying_capacity` — Σ shares · food_base per settlement
- `compute_trade_wealth` — face_flow at settlement face
- `compute_urban_population` — `s_eff · cap + β · wealth`
- `cull_settlements` — drop settlements below `city_min`; returns keep mask
- `apply_population_to_settlements` — stamp population onto Settlement objects

## trade_spawn.py

`spawn_trade_settlements` — identifies high-flow hub faces and remote waypoint faces and seeds additional `Settlement` objects there; called once per relaxation pass.

## Defensibility

`polity/seeding.py::compute_settlement_defensibility` is called from `PopulationStage.build()` after the final population loop, once trade wealth and danger are finalized. Combines:
- **Terrain**: ruggedness (`land_base_cost` normalized over land p50..p90) + situational value
- **River barrier**: flow magnitude on the settlement's face (normalized to land p90)
- **Walls**: probabilistic per-settlement (`wall_base_prob + wall_danger_boost·danger + wall_wealth_boost·wealth`); wall quality biased by local danger and wealth; keyed on `(world.seed, face)` for reproducibility

Returns `(defensibility, has_walls, wall_strength)`, all stamped onto `Settlement` fields. Wall presence and strength are consumed by `PolityStage` via `Settlement.defensibility`. See `PolityParams.defensibility: DefensibilityParams` for the governing params — [realm_polity.md](realm_polity.md).

## Validation

`scripts/measure_population.py` (distribution shape — Zipf slope/R²/Gini) and `scripts/measure_population_absolute.py` (absolute grounding — world total, top-city band, trade-fed-city test).

## Outputs

Per-face: `TRADE_FLOW`, `POPULATION`, `HINTERLAND_ID`, `RURAL_POPULATION`, `is_trade_hub`, `settlement_tier`.
Features: `settlements` (updated with population, defensibility, has_walls, wall_strength), `trade_routes`, `trade_attribution` (`(S, S)` directional inter-settlement trade matrix, aligned with `settlements`).
Color schemes: Trade Flow, Hinterland, and (debug) Rural Population.
