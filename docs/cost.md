# Cost Surface Stage

## Files
- `worldgen/cost/builder.py` — stage + params
- `worldgen/cost/surfaces.py` — model + pure functions
- `worldgen/cost/channels.py` — `IS_WATER`, `IS_STRAIT`, `COST_SURFACES`

## Concept

Final physical stage: **base** land + sea traversal costs for `realmgen` (which
adds danger on top).

- **Sailable water = ocean ∪ lake**: lakes impassable on land, navigable like ocean.
- **Navigable rivers** (enough flow, below the fall line) extend the sailable network inland; such faces are *amphibious* — passable on both surfaces (the only overlap).

## `builder.py`

**`CostParams`**: `land`, `sea`, `strait`.
- Land: `biome_cost_scale`, `ruggedness_cost_per_km`, `forest_cost_strength`, `wetland_cost_strength`, river term (`river_ford_strength`, `river_corridor_strength`, `river_flow_reference`, `river_min_mult` / `river_max_mult`), `navigable_ruggedness_km` (relief at which a river stops being navigable — the fall line).
- Sea: `coastal_cost` / `open_ocean_cost`, `open_ocean_distance_km` ramp, `strait_discount`, `navigable_flow_factor` (× hydrology's `river_flow_threshold`, passed to `CostBuilder`), `ice_penalty_strength` / `ice_freeze_threshold`, wind (`wind_optimal_speed`, `wind_speed_width`, `wind_penalty_strength`, `wind_direction_strength`, `wind_coast_blend_km`).
- Strait: `opposite_dot_threshold`, `min_water_neighbors`.

**`CostBuilder`** — last in the globe pipeline (does not read danger). Reads
per-face `biome`, `is_ocean`, `forest_density_mean`, `wetland_fraction` (if
present); textures `elevation`, `lake_mask`, `temperature`; features
`"river_objects"`, `"winds"`. `is_water = is_ocean | is_lake` (lake when ≥ 50%
lake pixels).

Writes per-face `land_base_cost`, `sea_base_cost`, `is_strait`, `is_water`,
`is_navigable_river`, `river_navigability`; `self.surfaces` and feature
`"cost_surfaces"` (incl. per-face river flow field).

Display (`--debug`): **Land Cost** (normalised to land p95), **Sea Cost** (straits
and navigable rivers highlighted).

## `surfaces.py` — Pure functions

**`CostSurfaces`**: `land_base_cost` + `land_passable` (= `~is_water`);
`river_flow_dir (N,3)`, `river_flow_mag`, `river_navigability` (fall-line taper
[0,1]); `sea_base_cost` + `sea_passable` (= water ∪ navigable river);
`strait_mask`; `wind_dir (N,3)` (coast-blended mean wind direction, or `None`).

- `compute_land_base_cost` — `biome_cost_scale · BIOME_TRAVERSAL_COST[biome] + ruggedness_cost_per_km · (face relief, km) + forest_cost_strength · forest_density_mean`, then × `(1 + wetland_cost_strength · wetland_fraction)` in the builder ([wetland.md](wetland.md)). The face relief is the world's ruggedness in km (`worldgen.geology.ruggedness.face_relief`). Water impassable; only realmgen's hinterland stage opts into a limited coastal-water band via its own permeable graph.
- `compute_river_flow_field` — per-face downstream unit tangent + max flow; in-lake segments skipped; `exclude_mask` zeros water faces.
- `compute_river_navigability` — linear taper on the same face relief: placid → 1, `navigable_ruggedness_km` → 0.
- `navigable_river_mask` — flow ≥ `navigable_flow_min` (= `navigable_flow_factor` × river threshold) ∧ navigability > 0 ∧ not water.
- `build_land_edge_costs` — mean of the two faces' base cost × a **directional** river multiplier per face, clipped to `[river_min_mult, river_max_mult]`: `1 + w·(ford_strength·(1 − align) − corridor_strength·navigability·align)`, `w = min(flow / river_flow_reference, 1)`, `align = |step · flow_dir|`. Cheap *along* a navigable valley (tapered by navigability), dear to *ford* (not tapered). `settlement_mask` waives the ford penalty on settlement faces (bridges). Realmgen calls it after adding danger to the base.
- `detect_straits` — water faces pinched between opposite land neighbours.
- `compute_sea_distance_to_coast` — Dijkstra over the sailable graph from coastal water.
- `compute_sea_cost` — coastal→open ramp with strait discount, × ice penalty (`compute_ice_penalty`: 1 + frozen-season fraction × strength) × wind-speed bell (`compute_wind_sea_modifier`: dearer becalmed or stormy, blended to 1 within `wind_coast_blend_km` of coast). Lakes/navigable rivers read as coastal.
- `compute_wind_mean_dir` — season-mean wind direction, faded to 0 near the coast.
- `build_sea_edge_costs` — mean of the two faces' sea cost; with `wind_dir`, × `1 − direction_strength · (step · mean wind dir)` (cheaper downwind).
