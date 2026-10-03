# Realm Placement Stage

Part of `realmgen/`. [Back to realmgen overview](realmgen.md)

## Files

- `realmgen/placement/builder.py` — `PlacementStage`, `SeederParams`, `TierParams`
- `realmgen/placement/seeder.py` — `placement_score`, `build_cost_graph`, `seed_settlements`
- `realmgen/placement/types.py` — `Settlement` dataclass, display constants

## Params

**`TierParams`**: `exclusion_radius` (minimum cost-distance between same-tier settlements), `score_threshold` (minimum combined score to place), `initial_population` (seed value fed to the PopulationStage relaxation loop).

**`SeederParams`**: `situational_weight: float` (0.55) — blend weight for situational vs suitability score. Four `TierParams` tiers (exclusion radius / score threshold / initial population): `tier0` Capitals (25 / 0.45 / 1.0) → `tier1` Cities (12 / 0.28 / 0.5) → `tier2` Towns (6 / 0.2 / 0.2) → `tier3` Villages (3 / 0.1 / 0.08). The tier names are seeding labels only: every settlement that survives the population stage's cull is a city (display-only towns come from the town stage). Danger influence (`block_threshold`) zeroes the score of faces at or above it.

## Settlement

**`Settlement`** (`placement/types.py`): `face_id`, `position` (unit-sphere; the face centroid at seeding, then wherever the finest tier that refined it put it), `tier` (seeder label: 0=capital…3=village), `score` (combined placement score at seeding time). Mutable fields set by later stages: `population` (initial seed from `TierParams.initial_population`; updated each relaxation pass by PopulationStage), `defensibility` (composite multiplier ≥1, set by PopulationStage after final populations are known), `has_walls: bool` (set by PopulationStage), `wall_strength: float` (wall contribution to defensibility, 0 if no walls).

Siting fields (see [realm_refine.md](realm_refine.md) and [region_features.md](region_features.md#towns-and-cities)): `site_kind` (`SITE_KIND_OCEAN_COAST | _LAKE_COAST | _RIVER | _CONFLUENCE` bit flags of the refine stage's site), `site_position` (that site), `region_position` (the region tier's refinement), `refined` / `local_refined`. `set_site(position, kind)` puts the city on a new stage site; `reset_refinement()` returns it to the site, `reset_local_refinement()` to its region position.

Display tier (0-3) is derived from `population / max_pop` at render time via `population_display_tier` — unrelated to the seeding tier.

Display constants: `TIER_NAMES` (`["Great City", "Major City", "City", "Minor City"]`), `TIER_COLORS`, `TIER_RADII` (per display tier). `format_population` humanizes headcounts (`9400 → '9.4k'`).

## seeder.py

- `placement_score(suitability, situational_value, situational_weight)` — blends the two scores
- `build_cost_graph(face_adjacency, edge_costs)` — converts per-face edge-cost list to scipy CSR sparse matrix; shared with population and polity stages
- `seed_settlements(combined, is_water, centroids, cost_graph, tier_configs)` — hierarchical greedy Dijkstra-exclusion seeder: tier 0 capitals first with widest radii, cumulative forbidden mask, scipy Dijkstra for fast cost-distance lookup

## PlacementStage

Runs second in `RealmPipeline`. Reads `SUITABILITY`, `SITUATIONAL_VALUE`, `IS_WATER`, `COST_SURFACES` and `danger`. The cost graph is `build_land_edge_costs` over the land base cost + `movement_scale · danger`. Registers `settlements` feature (list of `Settlement` with initial populations from `TierParams.initial_population`) and writes per-face `settlement_tier` (-1 elsewhere; the population stage rewrites it after culling). No display registration — the Settlements overlay belongs to `PopulationStage`.
