# Realm Placement Stage

Part of `realmgen/`. [Back to realmgen overview](realmgen.md)

## Files

- `realmgen/placement/builder.py` — `PlacementStage`, `SeederParams`, `TierParams`
- `realmgen/placement/seeder.py` — `placement_score`, `build_cost_graph`, `seed_settlements`
- `realmgen/placement/types.py` — `Settlement` dataclass, display constants

## Params

**`TierParams`**: `exclusion_radius` (minimum cost-distance between same-tier settlements), `score_threshold` (minimum combined score to place), `initial_population` (seed value fed to the PopulationStage relaxation loop).

**`SeederParams`**: `situational_weight: float` — blend weight for situational vs suitability score. Four `TierParams` tiers: `tier0` Capitals (`excl=25, thresh=0.45, init_pop=1.0`) → `tier1` Cities → `tier2` Towns → `tier3` Villages. Danger influence (`block_threshold`) can suppress placement on high-danger faces.

## Settlement

**`Settlement`** (`placement/types.py`): `face_id`, `position` (unit-sphere centroid), `tier` (seeder label: 0=capital…3=village), `score` (combined placement score at seeding time). Mutable fields set by later stages: `population` (initial seed from `TierParams.initial_population`; updated each relaxation pass by PopulationStage), `defensibility` (composite multiplier ≥1, set by PopulationStage after final populations are known), `has_walls: bool` (set by PopulationStage), `wall_strength: float` (wall contribution to defensibility, 0 if no walls).

Display tier (0-3) is derived from `population / max_pop` at render time via `population_display_tier` — unrelated to the seeding tier.

Display constants: `TIER_NAMES` (`["Great City", "Major City", "City", "Minor City"]`), `TIER_COLORS`, `TIER_RADII` (per display tier). `format_population` humanizes headcounts (`9400 → '9.4k'`).

## seeder.py

- `placement_score(suitability, situational_value, situational_weight)` — blends the two scores
- `build_cost_graph(face_adjacency, edge_costs)` — converts per-face edge-cost list to scipy CSR sparse matrix; shared with population and polity stages
- `seed_settlements` — hierarchical greedy Dijkstra-exclusion seeder: tier 0 capitals first with widest radii, cumulative forbidden mask, scipy Dijkstra for fast cost-distance lookup

## PlacementStage

Runs second in `RealmPipeline`. Reads `SUITABILITY`, `SITUATIONAL_VALUE`, `IS_WATER`, `COST_SURFACES`. Registers `settlements` feature (list of `Settlement` with initial populations from `TierParams.initial_population`). No per-face channel outputs; no display registration.
