# Realm Town Stage

Part of `realmgen/`. [Back to realmgen overview](realmgen.md)

## Overview

Last realm stage. Adds **towns** of `town_min` (1k) up to
`PopulationParams.city_min` (7k) people — the layer the population stage leaves
unmodeled (`city_surplus_share`, [realm_population.md](realm_population.md)).
**Display-only**: no stage reads them; drawn in region and local views.

- Town people come **out of** rural population: afterwards a face's
  `rural_population` is countryside *net of* towns, `town_population` the
  difference; rural + towns = the population stage's figure (global totals
  unchanged).
- Per global face, keyed `(world.seed, face)`, independent of any region viewed.
  Region refinement (below) is also view-order independent (each town has one
  owning region).

## Files

- `realmgen/towns/builder.py` — `TownStage`, `TownParams`, `town_site_context` (city exclusion discs, route/mineral points; shared with region/local re-siting), Rural Density display (`rural_density_recipe`)
- `realmgen/towns/placement.py` — `compute_rural_density`, `draw_town_sizes`, `site_log_weight_and_kind` (reused by finer tiers), `place_towns`, `texel_owners`
- `realmgen/towns/types.py` — `Towns` (`towns` feature: parallel arrays + refinement fields, `reset_refinement` / `reset_local_refinement`), kind codes, names, colours
- Reuses `realmgen/refine/sites.py` (`SiteField`; [realm_refine.md](realm_refine.md))

## Steps

1. **Rural density** (`compute_rural_density`) → `rural_density` face texture
   (people/km², float32; farmers + fishers + miners):
   - farmers (rural − `FISHING_POPULATION`) over the face's owned land texels
     (nearest centroid) ∝ `fertility_tex ** γ` (`PopulationParams.fertility_exponent`);
   - fishers evenly on owned **shore** texels (`shore_texels`: land with water in
     its 8-neighbourhood), else with the farmers; also written as
     `fishing_density` (so finer tiers keep them on the shore);
   - miners (`MINING_POPULATION`, [Miners](realm_population.md#miners)): each
     deposit's `mine_workers` split over its workings by share (`mineral_sites`,
     [resources.md](resources.md)), each working's on the nearest owned land
     texel (`compute_mining_density`) → `mining_density` texture.
   - Each owned texel = `face_area / n_owned`, so face totals are exact.
     Over-hang uses its owner face's scale (function of position, seamless).
     All-barren land → spread evenly.
2. **Town sizes** (`draw_town_sizes`). Budget `town_fraction ×` farmers+fishers,
   and separately `mining_town_fraction ×` miners for mining towns; truncated
   Pareto (`pdf ∝ x^-(α+1)` on `[town_min, city_min]`, α = `size_exponent`),
   largest first; the overspending draw is cut to what's left. A remainder <
   `town_min` becomes one minimum town with probability `remainder / town_min`
   (expected total exact; thin faces still get the odd town).
3. **Town sites** (`place_towns`). Globally largest first, each on an owned land
   texel of its face by Gumbel-max over the log weight:

   `w = (ρ / ρ̄_face + 0.05) × (1 + port·coast + confluence·conf + river·river + stream·stream + route·route) + mining·mine`

   - ρ = rural density. `stream` = channel of the hydrology `"streams"` network
     (cut at `stream_flow_threshold`) not already river/confluence; water bonuses
     don't stack. Cities get no network, so streams never draw cities. `route` =
     within `route_radius_km` of a trade route; `mine` = within
     `mining_radius_km` of a working. The 0.05 floor keeps barren sites possible.
   - **Spacing**: a pair must be ≥ the larger of their radii apart — town
     `town_spacing_km × (pop / town_min)^0.25` (20 km at 1k, ~33 at 7k), city
     `city_clearance_km × (pop / city_min)^0.25` (25 km at 7k, ~73 at 500k).
     Checked over the face and neighbours (exact while radii < a face, ~200 km).
   - Inland sites jittered ≤ `site_jitter` × texel spacing (off the lattice);
     coast sites not (could step into sea); jitter leaving the face dropped.
   - **Mining towns**: same pass and spacing, only among sites within `2 ×
     mining_radius_km` of the face's worked workings, weighted `Σ miners ·
     exp(−(d / mining_radius)²)`; always mining kind. Need no farmland (barren
     icecap/desert faces with worked deposits still get one).
4. **Accounting.** `town_population` = sited towns per face; `rural_population`
   and density reduced to match — mining towns out of miners (the odd minimum
   one beyond them out of farmers), others out of farmers+fishers. A town with no
   legal site is dropped; its people stay rural (a mining town's stay in the
   camps, drawn by local maps as mining villages).

**Idempotent**: the stage reads pre-town countryside as `rural_population +
town_population`, and `PopulationStage` zeroes `town_population` whenever it
writes fresh rural — re-running never carves towns twice.

**Kind** (precedence): port (coast), mining, river (river/confluence/stream),
waypoint (route), else market. **Parent** = the face's `hinterland_id` (city
whose hinterland holds it).

## `TownParams`

`town_fraction` (0.12), `mining_town_fraction` (0.6; the rest live in camps by
the mine), `town_min` (1000), `size_exponent` (1.0), `town_spacing_km` (20),
`city_clearance_km` (25); bonuses `port_bonus` (2), `confluence_bonus` (2),
`river_bonus` (1), `stream_bonus` (1), `route_bonus` (1), `mining_bonus` (1.5);
`route_radius_km` (15), `mining_radius_km` (25), `site_jitter` (hidden, 0.4).
Ceiling = `PopulationParams.city_min` (no size overlap with cities).

## Display

- **Region refinement.** Entering a region moves each town whose site is in it
  within ~one global texel onto finer terrain (river town → drawn river/stream,
  port → refined coast), writing into this feature: `positions` (actual),
  `site_positions` (stage's), `refined`, `region_positions` (all saved). Uses
  `site_log_weight_and_kind` and `town_site_context`
  ([region_features.md](region_features.md#towns-and-cities)). Local maps refine
  again from the region position (`local_refined`;
  [local_settlements.md](local_settlements.md#towns-and-cities)); cities are
  refined at both tiers too.
- **Towns overlay** (checked by default): dots by kind, sized by population
  (`_draw_towns`), categorical legend. **Region/local views only** (projector
  `regional` flag). Settlements on refined water are nudged to land
  ([viz_regionview.md](viz_regionview.md)).
- **Hover**: none on world map; region counts the hovered hex's towns (`Towns in
  hex: N (pop)` + kinds); local names the town under the cursor.
- **Rural Density** (`--debug`): `rural_density` sqrt-scaled to its 99th land
  percentile (`rural_density_recipe`). Region draws its refined
  `rural_density` ([region_fields.md](region_fields.md)); local maps draw
  countryside net of villages. Land inside globe-water faces shows empty (those
  faces have no rural people).

## Numbers (seeds 4 / 31 / 35)

| | 4 | 31 | 35 |
|---|---|---|---|
| towns | 14.9k | 18.6k | 18.5k |
| people in towns | 31 M | 39 M | 40 M |
| median / p90 size | 1.6k / 3.9k | 1.6k / 4.0k | 1.7k / 4.1k |
| nearest-town distance, median | 37 km | 37 km | 35 km |
| stage time | 1.3 s | 1.4 s | 1.3 s |

~70% market, 20–25% waypoints, rest ports, river and mining. The stream bonus
took seed 4 river towns from 586 (3.9%) to 956 (6.4%).

## Validation

`realmgen/towns/test/test_towns.py`: size bounds/budget/cap and tiny-budget
unbiasedness; density conservation and barren fallback; siting (spacing,
in-face, dropped stay rural, determinism); stream bonus (and no-network
`SiteField` ignoring it); `Towns` pack round trip; end-to-end (rural + towns
conserved, `town_population` matches, idempotent re-run).
