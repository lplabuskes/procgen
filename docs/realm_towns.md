# Realm Town Stage

Part of `realmgen/`. [Back to realmgen overview](realmgen.md)

## Overview

The last realm stage. It adds the settlements below the city floor: **towns** of
`town_min` (1k) up to `PopulationParams.city_min` (7k) people, the layer the
population stage deliberately leaves unmodeled (see `city_surplus_share` in
[realm_population.md](realm_population.md)). Towns are **display-only**: no
stage reads them. They are drawn in the region and local views.

Town people come **out of** the rural population. After this stage a face's
`rural_population` is its countryside *net of* its towns and `town_population`
is the difference, so rural + towns is exactly what the population stage
computed. Global totals (`measure_population_absolute.py`) are unchanged.

Everything is decided per global face, keyed on `(world.seed, face)`, and
independent of the region being viewed. A region view later *refines* each
town's position within its global texel (below); that too is independent of
which region is viewed, since each town has exactly one owning region.

## Files

- `realmgen/towns/builder.py` — `TownStage`, `TownParams`, `town_site_context` (the cities' exclusion discs and the route / mineral points, shared with the region and local re-siting), the Rural Density display (`rural_density_recipe`)
- `realmgen/towns/placement.py` — pure functions: `compute_rural_density`, `draw_town_sizes`, `site_log_weight_and_kind` (the site weight below, reused by the finer tiers), `place_towns`, `texel_owners`
- `realmgen/towns/types.py` — `Towns` (the `towns` feature: parallel arrays plus the refinement fields and `reset_refinement` / `reset_local_refinement`), kind codes, names and colours
- Reuses `realmgen/refine/sites.py` (`SiteField`) — see [realm_refine.md](realm_refine.md)

## Steps

1. **Rural density** (`compute_rural_density`). Each face's rural population is
   spread over the land texels it owns (nearest centroid) in proportion to the
   rural model's own food base, `fertility_tex ** γ` (γ =
   `PopulationParams.fertility_exponent`). Each owned texel stands for
   `face_area / n_owned`, so the face total is conserved exactly. An
   over-hanging texel uses its owner face's scale, which makes the field a
   function of position that agrees across face seams. A face whose owned land
   is all barren spreads its people evenly. Result: the `rural_density` face
   texture (people/km², float32), reduced by the towns in step 4.
2. **Town sizes** (`draw_town_sizes`). A face spends `town_fraction × rural`
   people on towns, drawing truncated-Pareto sizes (`pdf ∝ x^-(α+1)` on
   `[town_min, city_min]`, α = `size_exponent`), largest first. The draw that
   would overspend is cut to what is left. A remainder under `town_min` becomes
   one more minimum-size town with probability `remainder / town_min`, so the
   expected total is exact and thinly peopled faces still get the odd town.
3. **Town sites** (`place_towns`). All towns on the globe are sited largest
   first, each on one of its face's owned land texels, by weighted random choice
   (Gumbel-max over the log site weight):

   `w = (ρ / ρ̄_face + 0.05) × (1 + port·coast + confluence·conf + river·river + stream·stream + route·route) + mining·mine`

   ρ is the texel's rural density. `stream` means the texel is a channel of the
   hydrology stream network (`"streams"`, cut at `stream_flow_threshold`, far
   below the rivers) and not already a river or confluence site; the water
   bonuses do not stack. City siting shares `SiteField` but passes no network,
   so streams never draw cities. `route` means within `route_radius_km` of a
   trade route, and `mine` means within `mining_radius_km` of a mineral deposit.
   The 0.05 floor keeps barren sites possible for mining camps and waypoints. A
   site is rejected if it lies within the exclusion radius of a town or city
   already placed. A pair must be the larger of their two radii apart:
   `town_spacing_km × (pop / town_min)^0.25` for a town (20 km at 1k, ~33 km at
   7k) and `city_clearance_km × (pop / city_min)^0.25` for a city (25 km at 7k,
   ~73 km at 500k). The check only looks at the face and its neighbours, which
   is exact while every radius is shorter than a face (~200 km). Inland sites
   are jittered by up to `site_jitter` × texel spacing so towns do not sit on
   the texel lattice; coast sites are not jittered (it could step into the sea),
   and a jitter that would leave the face is dropped.
4. **Accounting.** `town_population` is the sum of sited towns per face, and
   `rural_population` and the density texture are reduced to match. A town with
   no legal site is dropped and its people stay rural. None were dropped on
   seeds 4/31/35.

The stage reads the pre-town countryside as `rural_population +
town_population`, and `PopulationStage` zeroes `town_population` whenever it
writes a fresh `rural_population`. Re-running the town stage alone therefore
gives the same result, and never carves towns out twice.

**Kind** is what the chosen site offered, in precedence order: port (coast),
mining, river (river, confluence or stream), waypoint (trade route), otherwise
market town. **Parent** is the `hinterland_id` of the town's face: the index of the
city whose hinterland holds it.

## `TownParams`

`town_fraction` (0.12), `town_min` (1000), `size_exponent` (1.0),
`town_spacing_km` (20), `city_clearance_km` (25), the site bonuses `port_bonus`
(2), `confluence_bonus` (2), `river_bonus` (1), `stream_bonus` (1), `route_bonus` (1),
`mining_bonus` (1.5), `route_radius_km` (15), `mining_radius_km` (25), and
`site_jitter` (hidden, 0.4). The town ceiling is `PopulationParams.city_min`,
so towns and cities do not overlap in size.

## Display

- **Region refinement.** Entering a region moves each town whose site is in it
  within ~one global texel of that site, onto the region's finer terrain — a
  river town onto the drawn river or stream, a port onto the refined coast —
  and writes the result into this feature: `positions` is the town's actual
  position, `site_positions` the stage's site, `refined` which towns have been
  refined, `region_positions` the region's result (all saved). It shares the
  stage's weights (`site_log_weight_and_kind`) and `town_site_context`. See
  [region_features.md](region_features.md#towns-and-cities). A local
  map then refines the town again from its region position
  (`local_refined`; [local_settlements.md](local_settlements.md#towns-and-cities)),
  and cities are refined at both tiers too.
- **Towns overlay** (checked by default). Small dots coloured by kind and sized
  by population (`_draw_towns` in `viz/mapview/overlays.py`), with a categorical
  legend. Drawn **only in the region and local views** (the overlay projector's
  `regional` flag), never on the world map. There, towns and cities that land
  on the refined water are nudged onto land (see [viz_regionview.md](viz_regionview.md)).
- **Hover** (every view): `Towns: N (pop)` for the face, plus a breakdown by kind.
- **Rural Density** (`--debug` layer). The `rural_density` texture,
  sqrt-scaled to its 99th land percentile (`rural_density_recipe`). A region
  redraws it from its refined `rural_density` (see
  [region_fields.md](region_fields.md)); a local map from the countryside net of
  its villages. Land inside faces the globe calls water shows as empty,
  because the population model gives those faces no rural people.

## Numbers (seeds 4 / 31 / 35)

| | 4 | 31 | 35 |
|---|---|---|---|
| towns | 14.9k | 18.6k | 18.5k |
| people in towns | 31 M | 39 M | 40 M |
| median / p90 size | 1.6k / 3.9k | 1.6k / 4.0k | 1.7k / 4.1k |
| nearest-town distance, median | 37 km | 37 km | 35 km |
| stage time | 1.3 s | 1.4 s | 1.3 s |

About 70% are market towns, 20–25% waypoints, and the rest ports, river towns
and mining towns. With the stream bonus, river towns on seed 4 went from 586
(3.9%) to 956 (6.4%), mostly out of market towns.

## Validation

`realmgen/towns/test/test_towns.py` covers:
- size bounds, budget and cap, and unbiasedness for tiny budgets
- per-face density conservation and the barren-face fallback
- siting: spacing, towns staying in their face, dropped towns staying rural, determinism
- the stream bonus making river towns, and a `SiteField` without a network ignoring it
- the `Towns` pack round trip
- the stage end to end: rural + towns conserved, `town_population` matches the towns, idempotent re-run
