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

**`scoring/suitability.py`**: Five suitability factors — water proximity (freshwater BFS + coastal distance), climate habitability (T/P bell curve), ruggedness penalty (elevation range), resource proximity (BFS from mineral/rare-bio deposit faces), weighted combination.

**`scoring/situational_value.py`**: Five situational-value components — harbor quality (coastal faces, depth-scaled), river confluences, navigable river heads, mountain passes (local cost contrast), strait adjacency.

## ScoringStage

Runs first in `RealmPipeline`. Writes per-face channels `SUITABILITY` and `SITUATIONAL_VALUE` to the mesh. Registers non-debug color schemes "Suitability" and "Situational Value".
