# Cost Surface Stage

## Files
- `worldgen/cost/builder.py` — stage orchestration + params
- `worldgen/cost/surfaces.py` — model + pure functions

## Concept

Final physical stage. Converts already-computed data into **base** land + sea traversal-cost layers consumed by `realmgen`, which adds its danger field on top.

**Sailable water = ocean ∪ lake**: lakes are impassable on land and navigable like the ocean. **Navigable rivers** (enough flow, flat enough — below the fall line) extend the sailable network inland; such faces are *amphibious* (passable on both land and sea surfaces — the only overlap).

## `builder.py`

**`CostParams`** (dataclass): Nested `land: LandCostParams`, `sea: SeaCostParams`, `strait: StraitParams`.
- Land knobs: biome-cost scale, ruggedness cost/km, river term (ford penalty, corridor discount, flow reference, min/max multiplier), `navigable_ruggedness_km` (local-relief limit where a river stops being navigable — the fall line)
- Sea knobs: coastal/open-ocean cost, open-ocean distance ramp, strait discount, `navigable_flow_min` (minimum river flow to float a boat), `ice_penalty_strength` / `ice_freeze_threshold` (sea-ice cost penalty above polar freeze)
- Strait knobs: `opposite_dot_threshold` + `min_water_neighbors`

**`CostBuilder`**: A pipeline `Stage`. Runs after elevation + climate + hydrology + resources. Reads: `biome` and `is_ocean` per-face metadata, `elevation` + `lake_mask` + `temperature` textures, hydrology rivers via `"river_objects"` world feature (temperature used for the sea-ice cost).

Derives `is_water = is_ocean | is_lake` (a face counts as lake when ≥50% of its pixels are lake).

Writes per-face metadata: `land_base_cost`, `sea_base_cost`, `is_strait`, `is_water`, `is_navigable_river`, `river_navigability`.

Stores `self.surfaces: CostSurfaces`; registers the `"cost_surfaces"` globe feature (includes the per-face river flow field).

## `surfaces.py` — Pure functions

**`CostSurfaces`** dataclass:
- `land_base_cost` + `land_passable` (= `~is_water`)
- `river_flow_dir (N,3)` + `river_flow_mag` + `river_navigability` (fall-line taper in [0,1])
- `sea_base_cost` + `sea_passable` (= water ∪ navigable river)
- `strait_mask`

Key functions:
- `compute_land_base_cost` — biome LUT `BIOME_TRAVERSAL_COST` × scale + ruggedness via pixel elevation span; water is impassable (`land_passable = ~is_water`). Realmgen's hinterland stage is the only consumer that opts into a limited coastal-water band via its own permeable graph.
- `compute_river_flow_field` — per-face downstream unit tangent + max flow; in-lake segments skipped; `exclude_mask` zeros river terms on water faces
- `compute_river_navigability` — linear taper from placid→1 to `navigable_ruggedness_km` relief→0 (the fall line)
- `navigable_river_mask` — flow ≥ `navigable_flow_min` ∧ navigability>0 ∧ not water
- `build_land_edge_costs` — folds the **directional** river term into per-edge cost: cheap *along* a navigable valley (flow ∥ step, discount tapered by navigability), expensive to *ford* (flow ⊥ step, not tapered); realmgen calls this same function after adding danger to the per-face base
- `detect_straits` — tags water faces pinched between opposite land neighbours
- `compute_sea_distance_to_coast` — Dijkstra over the sailable sub-graph from coastal water (open-ocean→coast)
- `compute_sea_cost` — ramps coastal→open with strait discount; lakes/navigable rivers read as coastal
- `build_sea_edge_costs` — isotropic ocean-graph edge averaging
