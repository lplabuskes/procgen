# Realm Scoring Stage

Part of `realmgen/`. [Back to realmgen overview](realmgen.md)

## Files

- `realmgen/scoring/builder.py` — `ScoringStage`, `SuitabilityParams`, `SituationalValueParams` (+ sub-params)
- `realmgen/scoring/suitability.py` — pure suitability functions
- `realmgen/scoring/situational_value.py` — pure situational-value functions

## Params

**`SuitabilityParams`**: Nested `water: WaterProximityParams`, `climate: ClimateHabitabilityParams`, `ruggedness: RuggednessParams`, `resource: ResourceAccessParams`, `weights: SuitabilityWeightParams`.

**`SituationalValueParams`**: Nested `harbor: HarborQualityParams`, `passes: MountainPassParams`, `weights: SituationalValueWeightParams`.

Danger influence (`DangerInfluenceParams.suitability_weight`, `block_threshold`) is applied by the stage but lives in `RealmParams.danger` — see [realmgen.md](realmgen.md).

## Pure functions

**`scoring/suitability.py`**: suitability in [0, 1] = the weighted mean (`combine_suitability`, weights normalised) of water proximity (`compute_water_proximity`: freshwater BFS + climate-stage distance to coast), climate habitability (`compute_climate_habitability`: T/P bell curves), `fertility_mean` and resource access (`compute_resource_proximity`: BFS from mineral / rare-bio deposit faces), × a ruggedness multiplier (`compute_ruggedness_penalty`, from the face elevation span). Water faces are 0.

**`scoring/situational_value.py`**: Six situational-value components, weighted by `combine_situational_value` — harbor quality (coastal faces, depth-scaled), river confluences, navigable river heads, mountain passes (local cost contrast), strait adjacency, and delta apexes (`compute_delta_apexes`: 1 at the face holding each `"delta_fans"` apex, `delta_weight` 0.2; a world without deltas leaves the weight out of the normalization).

## ScoringStage

Runs first in `RealmPipeline`. Reads `IS_WATER`, `COST_SURFACES` (river faces, navigability, land base cost), `lake_mask`, climate aggregates, `fertility_mean`, `elevation`, the `rivers`, `minerals`, `rare_bio` and `danger` features. Danger lowers suitability by `× (1 − suitability_weight · danger)`. Writes per-face channels `SUITABILITY` and `SITUATIONAL_VALUE` to the mesh.

Display (`--debug` only): "Suitability" and "Situational Value" schemes and a `Suit:` hover fragment.
