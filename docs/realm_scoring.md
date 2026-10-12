# Realm Scoring Stage

Part of `realmgen/`. [Back to realmgen overview](realmgen.md)

## Files

- `realmgen/scoring/builder.py` — `ScoringStage`, `SuitabilityParams`, `SituationalValueParams` (+ sub-params)
- `realmgen/scoring/suitability.py`, `situational_value.py` — pure functions

## Params

- `SuitabilityParams`: `water`, `climate`, `ruggedness`, `resource`, `weights`.
- `SituationalValueParams`: `harbor`, `passes`, `weights`.
- Danger (`suitability_weight`, `block_threshold`) lives in `RealmParams.danger` ([realmgen.md](realmgen.md)).

## Pure functions

**Suitability** ∈ [0, 1] = weighted mean (`combine_suitability`, normalised) of
water proximity (`compute_water_proximity`: freshwater BFS + climate-stage coast
distance), climate habitability (`compute_climate_habitability`: T/P bells),
`fertility_mean`, resource access (`compute_resource_proximity`: BFS from
mineral/rare-bio faces); × ruggedness multiplier (`compute_ruggedness_penalty`:
`exp(−face relief / penalty_scale_km)`, the face relief from the world's
ruggedness, `worldgen.geology.ruggedness.face_relief`). Water faces 0.

**Situational value** (`combine_situational_value`): harbour quality (coastal,
depth-scaled), river confluences, navigable river heads, mountain passes (local
cost contrast), strait adjacency, delta apexes (`compute_delta_apexes`: 1 at each
`"delta_fans"` apex face, `delta_weight` 0.2; omitted from normalisation without
deltas).

## ScoringStage

First in `RealmPipeline`. Reads `IS_WATER`, `COST_SURFACES` (river faces,
navigability, land base cost), `lake_mask`, climate aggregates,
`fertility_mean`, `elevation`, features `rivers`, `minerals`, `rare_bio`,
`danger`. Suitability × `(1 − suitability_weight · danger)`. Writes `SUITABILITY`,
`SITUATIONAL_VALUE`.

Display (`--debug`): Suitability, Situational Value schemes; `Suit:` hover.
