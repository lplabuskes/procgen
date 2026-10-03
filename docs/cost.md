# Cost Surface Stage

## Files
- `worldgen/cost/builder.py` — stage orchestration + params
- `worldgen/cost/surfaces.py` — model + pure functions
- `worldgen/cost/channels.py` — `IS_WATER`, `IS_STRAIT`, `COST_SURFACES`

## Concept

Final physical stage. Converts already-computed data into **base** land + sea traversal-cost layers consumed by `realmgen`, which adds its danger field on top.

**Sailable water = ocean ∪ lake**: lakes are impassable on land and navigable like the ocean. **Navigable rivers** (enough flow, flat enough — below the fall line) extend the sailable network inland; such faces are *amphibious* (passable on both land and sea surfaces — the only overlap).

## `builder.py`

**`CostParams`** (dataclass): Nested `land: LandCostParams`, `sea: SeaCostParams`, `strait: StraitParams`.
- Land knobs: `biome_cost_scale`, `ruggedness_cost_per_km`, `forest_cost_strength`, river term (`river_ford_strength`, `river_corridor_strength`, `river_flow_reference`, `river_min_mult` / `river_max_mult`), `navigable_ruggedness_km` (local-relief limit where a river stops being navigable — the fall line)
- Sea knobs: `coastal_cost` / `open_ocean_cost`, `open_ocean_distance_km` ramp, `strait_discount`, `navigable_flow_min` (minimum river flow to float a boat), `ice_penalty_strength` / `ice_freeze_threshold` (sea-ice penalty), wind (`wind_optimal_speed`, `wind_speed_width`, `wind_penalty_strength`, `wind_direction_strength`, `wind_coast_blend_km`)
- Strait knobs: `opposite_dot_threshold` + `min_water_neighbors`

**`CostBuilder`**: A pipeline `Stage`. Runs last in the globe pipeline, after resources and danger (it does not read danger). Reads: `biome` and `is_ocean` per-face metadata, `forest_density_mean` (when present), `elevation` + `lake_mask` + `temperature` textures, the hydrology rivers via the `"river_objects"` world feature, and the climate `"winds"` feature.

Derives `is_water = is_ocean | is_lake` (a face counts as lake when ≥50% of its pixels are lake).

Writes per-face metadata: `land_base_cost`, `sea_base_cost`, `is_strait`, `is_water`, `is_navigable_river`, `river_navigability`.

Stores `self.surfaces: CostSurfaces`; registers the `"cost_surfaces"` globe feature (includes the per-face river flow field).

Display (`--debug` only): **Land Cost** (normalised to the land 95th percentile) and **Sea Cost** (straits and navigable rivers highlighted).

## `surfaces.py` — Pure functions

**`CostSurfaces`** dataclass:
- `land_base_cost` + `land_passable` (= `~is_water`)
- `river_flow_dir (N,3)` + `river_flow_mag` + `river_navigability` (fall-line taper in [0,1])
- `sea_base_cost` + `sea_passable` (= water ∪ navigable river)
- `strait_mask`
- `wind_dir (N,3)` — coast-blended mean wind direction for the directional sea term, or `None` without winds

Key functions:
- `compute_land_base_cost` — `biome_cost_scale · BIOME_TRAVERSAL_COST[biome] + ruggedness_cost_per_km · (pixel elevation span) + forest_cost_strength · forest_density_mean`; water is impassable (`land_passable = ~is_water`). Realmgen's hinterland stage is the only consumer that opts into a limited coastal-water band via its own permeable graph.
- `compute_river_flow_field` — per-face downstream unit tangent + max flow; in-lake segments skipped; `exclude_mask` zeros river terms on water faces
- `compute_river_navigability` — linear taper from placid→1 to `navigable_ruggedness_km` relief→0 (the fall line)
- `navigable_river_mask` — flow ≥ `navigable_flow_min` ∧ navigability>0 ∧ not water
- `build_land_edge_costs` — mean of the two faces' base cost × a **directional** river multiplier per face, clipped to `[river_min_mult, river_max_mult]`: `1 + w·(ford_strength·(1 − align) − corridor_strength·navigability·align)`, `w = min(flow / river_flow_reference, 1)`, `align = |step · flow_dir|` — cheap *along* a navigable valley (discount tapered by navigability), expensive to *ford* (not tapered). `settlement_mask` suppresses the ford penalty on settlement faces (bridges). Realmgen calls this same function after adding danger to the per-face base
- `detect_straits` — tags water faces pinched between opposite land neighbours
- `compute_sea_distance_to_coast` — Dijkstra over the sailable sub-graph from coastal water (open-ocean→coast)
- `compute_sea_cost` — ramps coastal→open with strait discount, then × ice penalty (`compute_ice_penalty`: 1 + frozen-season fraction × strength) × wind-speed bell (`compute_wind_sea_modifier`: dearer when becalmed or stormy, blended to 1 within `wind_coast_blend_km` of the coast); lakes/navigable rivers read as coastal
- `compute_wind_mean_dir` — the season-mean wind direction, faded to 0 near the coast
- `build_sea_edge_costs` — mean of the two faces' sea cost; with `wind_dir`, × `1 − direction_strength · (step · mean wind dir)` (cheaper downwind, dearer upwind)
