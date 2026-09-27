# Realm Population Stage

Part of `realmgen/`. [Back to realmgen overview](realmgen.md)

## Files

- `realmgen/population/builder.py` — `PopulationStage`, `TradeParams`, `TradeSpawnParams`, `PopulationParams`
- `realmgen/population/calibration.py` — pure population functions
- `realmgen/population/trade_spawn.py` — `spawn_trade_settlements`
- `realmgen/trade/network.py` — `compute_trade_network`

## Design

All populations are **real people**, anchored to a medieval-tech Earth (~300–400 M world total). Design rationale: `docs/plans/city_population_rework_plan.md`. Three quantities: per-face **rural population** (`D_max · fertility^γ · area_km²` — the Malthusian food base, land only), per-settlement **carrying capacity** (food base summed over claimed hinterland shares), and per-settlement **urban population** (`s · share · cap + β · trade_flow` — additive trade lets a thin-hinterland chokepoint grow large on imported grain, the Constantinople/Venice case). A settlement exists as a city only if its urban population clears `city_min`; the modeled settlements are *major cities* — towns/villages are sub-face-scale and not modeled here (their slice of the regional surplus is reserved via `city_surplus_share`); display-only towns are carved out of the rural population later by the town stage, see [realm_towns.md](realm_towns.md).

## Params

**`TradeParams`**: `port_cost`, `river_switch_cost`, `landing_cost` (fallback land↔water transition away from a settlement, relaxation passes only — see [Ports](#ports)), `flow_decay`, `hub_display_quantile`.

**`TradeSpawnParams`**: `hub_spawn_quantile`, `hub_min_suitability`, `hub_exclusion`, `waypoint_min_cost`, `waypoint_min_suitability`, `waypoint_exclusion`, `port_spawn_quantile`, `port_min_suitability`, `port_exclusion`.

**`PopulationParams`**: `relaxation_passes`, `rural_density_max` (D_max, people/km² at fertility 1), `fertility_exponent` (γ ≈ 2 — restores density contrast), `surplus_rate` (s — regional agricultural surplus), `city_surplus_share` (major city's slice of `s·cap`; remainder reserved for the future town pipeline), `trade_food_weight` (β — people fed per unit trade flow), `city_min` (urban-formation floor in people), `hinterland_max_distance`, `hinterland_influence_scale`, `hinterland_reach_scale`, `hinterland_water_cost`, `trade_spawn: TradeSpawnParams`.

## Trade network

**`trade/network.py`**: `compute_trade_network` — routes trade flows from resource deposits (minerals + rare bio) to settlements over the combined land + sea graph. Returns `face_flow` (per-face scalar), `routes` (polylines for rendering), `hub_mask` (top-flow faces), and `pair_flow` (the `(S, S)` directional inter-settlement trade matrix — `pair_flow[i, j]` = volume exporter *i* sends to importer *j*, index-aligned with `settlements`). Called at the start and after each relaxation pass with population-weighted demands.

### Ports

Goods move between land and water **only at a settlement's face** (`settlement_ports=True`, the default): a coastal settlement is a port (port edges to its sea neighbours at `port_cost`), a settlement on a navigable river is a river port (land↔water switch at `river_switch_cost`). Navigable rivers are water all the way to their mouths, so an inland river city ships straight down the river and out to sea — the only other place a route crosses between land and water on the map is a river mouth (a water hop from a river face to the sea). The polity stage's sea-authority graph (`build_base_hybrid_graph`) still allows transitions everywhere.

A strict rule would strand coasts with no settlement, so during the relaxation passes every other coastal/river face gets a fallback *landing* at `landing_cost` (≈2.4× the port cost). `TradeResult.landing_flow` reports the flow each landing carries; where it reaches `port_spawn_quantile` of the positive face flows (the median — the map's drawing threshold), `spawn_trade_settlements` founds a port town there, and on the next pass it is an ordinary port. The **final** trade network is built without landings. Seed 4: land/sea hops away from a settlement went from 205 of 402 to 0 (the 14 left are river mouths); ~7 ports founded.

The final-pass `pair_flow` is published by `PopulationStage` as the `trade_attribution` world feature (aligned with the `settlements` feature). It is per-settlement trade-route provenance — column *j* = settlement *j*'s imports by source, row *i* = its exports by destination — consumed by the polity stage's economic-grip / peripheral-wealth secession levers (a settlement whose trade is dominated by intra-polity partners has stronger economic incentive to stay).

## PopulationStage

Runs third in `RealmPipeline`. Reads `IS_WATER`, `COST_SURFACES`, `MINERALS`, `RARE_BIO`, `FERTILITY_MEAN`, `SUITABILITY`. Danger (`DangerInfluenceParams.movement_scale`) is added to land base cost in the cost graphs.

Computes the per-face food base and rural population up front (written to `RURAL_POPULATION`, with `TOWN_POPULATION` zeroed — the town stage later moves its towns from one to the other), then runs a relaxation loop for `relaxation_passes` iterations: each pass spawns trade-hub and waypoint settlements (`trade_spawn.py`), recomputes hinterland shares, carrying capacity, trade wealth, urban population, culls cities below `city_min`, then reruns the trade network with population-weighted demands. The trade network after the loop is rebuilt without fallback landings.

After the relaxation loop, **defensibility is computed per settlement** and stamped onto the `Settlement` objects — see [Defensibility](#defensibility) below. Stores final per-settlement diagnostics (`carrying_cap`, `trade_wealth`) aligned with `settlements`/`population`. 

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

`spawn_trade_settlements` — identifies high-flow hub faces, remote waypoint faces, and busy fallback landings (port towns, see [Ports](#ports)) and seeds additional `Settlement` objects there; called once per relaxation pass.

## Defensibility

`polity/seeding.py::compute_settlement_defensibility` is called from `PopulationStage.build()` after the final population loop, once trade wealth and danger are finalized. Combines:
- **Terrain**: ruggedness (`land_base_cost` normalized over land p50..p90) + situational value
- **River barrier**: flow magnitude on the settlement's face (normalized to land p90)
- **Walls**: probabilistic per-settlement (`wall_base_prob + wall_danger_boost·danger + wall_wealth_boost·wealth`); wall quality biased by local danger and wealth; keyed on `(world.seed, face)` for reproducibility

Returns `(defensibility, has_walls, wall_strength)`, all stamped onto `Settlement` fields. Wall presence and strength are consumed by `PolityStage` via `Settlement.defensibility`. See `PolityParams.defensibility: DefensibilityParams` for the governing params — [realm_polity.md](realm_polity.md).

## Validation

`scripts/measure_population.py` (distribution shape — Zipf slope/R²/Gini) and `scripts/measure_population_absolute.py` (absolute grounding — world total, top-city band, trade-fed-city test).

## Outputs

Per-face: `TRADE_FLOW`, `POPULATION`, `HINTERLAND_ID`, `RURAL_POPULATION`, `TOWN_POPULATION` (zeroed), `is_trade_hub`, `settlement_tier`.
Features: `settlements` (updated with population, defensibility, has_walls, wall_strength), `trade_routes`, `trade_attribution` (`(S, S)` directional inter-settlement trade matrix, aligned with `settlements`).
Color schemes (debug): Trade Flow, Hinterland. (Rural density is drawn by the town stage.)
