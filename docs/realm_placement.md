# Realm Placement Stage

Part of `realmgen/`. [Back to realmgen overview](realmgen.md)

## Files

- `realmgen/placement/builder.py` — `PlacementStage`, `SeederParams`, `TierParams`
- `realmgen/placement/seeder.py` — `placement_score`, `build_cost_graph`, `seed_settlements`
- `realmgen/placement/types.py` — `Settlement`, display constants

## Params

- `TierParams`: `exclusion_radius` (min cost-distance between same-tier settlements), `score_threshold`, `initial_population` (seed for the population relaxation).
- `SeederParams`: `situational_weight` (0.55; situational vs suitability blend) and four tiers (radius / threshold / initial pop): `tier0` Capitals (25 / 0.45 / 1.0), `tier1` Cities (12 / 0.28 / 0.5), `tier2` Towns (6 / 0.2 / 0.2), `tier3` Villages (3 / 0.1 / 0.08). **Tier names are seeding labels only**: every survivor of the population cull is a city (display towns come from the town stage).
- Faces at/above `danger.block_threshold` score 0.

## Settlement

`Settlement` (`placement/types.py`):
- `face_id`, `position` (centroid at seeding, then wherever the finest refining tier put it), `tier` (seeding label 0–3), `score`.
- Set later: `population` (seeded from `initial_population`, updated per pass), `defensibility` (≥ 1), `has_walls`, `wall_strength` (0 without walls) — by `PopulationStage`.
- Siting ([realm_refine.md](realm_refine.md), [region_features.md](region_features.md#towns-and-cities)): `site_kind` (`SITE_KIND_OCEAN_COAST | _LAKE_COAST | _RIVER | _CONFLUENCE` flags), `site_position`, `region_position`, `refined` / `local_refined`. `set_site(position, kind)`; `reset_refinement()` → site; `reset_local_refinement()` → region position.

Display tier (0–3) comes from `population / max_pop` via
`population_display_tier`, unrelated to the seeding tier. Constants:
`TIER_NAMES` (`["Great City", "Major City", "City", "Minor City"]`),
`TIER_COLORS`, `TIER_RADII`; `format_population` (`9400 → '9.4k'`).

## seeder.py

- `placement_score(suitability, situational_value, situational_weight)` — blend
- `build_cost_graph(face_adjacency, edge_costs)` — scipy CSR; shared with population and polity
- `seed_settlements(combined, is_water, centroids, cost_graph, tier_configs)` — hierarchical greedy Dijkstra-exclusion: tier 0 first with widest radii, cumulative forbidden mask

## PlacementStage

Second in `RealmPipeline`. Reads `SUITABILITY`, `SITUATIONAL_VALUE`, `IS_WATER`,
`COST_SURFACES`, `danger`; cost graph = `build_land_edge_costs` over land base
cost + `movement_scale · danger`. Registers `settlements` and per-face
`settlement_tier` (-1 elsewhere; rewritten by population after culling). No
display (Settlements overlay belongs to `PopulationStage`).
