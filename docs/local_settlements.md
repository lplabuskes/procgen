# Local Map Settlements

Part of the local tier — [back to local.md](local.md). The towns and cities a
local map re-sites, the villages it places, and the ground they build on and
farm. Drawing is in [viz_regionview.md](viz_regionview.md).

## Files
- `localmap/settlements.py` — `LocalSettlementParams`, `local_owned`, `refine_local_cities`, `refine_local_towns`
- `localmap/cities.py` — `CityParams`, `CityFootprints`, `city_footprints` (display-only)
- `localmap/farmland.py` — `FarmlandParams`, `Farmland`, `place_farmland`, `farmland_rgba` (display-only)
- `localmap/villages.py` — `VillageParams`, `Villages`, `place_villages`, `village_budget_report`

## Towns and cities

A local map re-sites towns and cities once more, from where the region put them,
with the region's own rules (`region.towns.refine_towns`,
`region.cities.refine_cities` at `LOCAL_TIER`; see
[region_features.md](region_features.md#towns-and-cities)) on its ~0.36 km
texels and its own coast, rivers and roads, within `refine_radius_parent_texels`
(1.5 regional texels, ~7 km). A harbour city or port town lands on the local
shore, a river city or river town on the local river, whenever one is in reach.
The result is **saved on the world** (`position`, flagged `local_refined`;
`region_position` keeps the region's), so the settlement stands in the same place
on every map.

- **Ownership.** Each settlement is refined by one local map, once: the one of
  the regional hex nearest its region position (`local_owned`), in the region
  that refined it. A map that shows a settlement another map owns draws it where
  it stands — at its region position, moved off the local water with
  `snap_to_land`, until its own hex is opened.
- **Order.** Cities first, so the local roads end on them (`refine_local_cities`
  returns `moved = (region positions, positions)` for `build_local_roads`); towns
  after the roads, since their route flag reads them. Town spacing is kept
  against the others' *region* positions, and city jitter and town noise are
  seeded per settlement and tier, so a map's result does not depend on which maps
  were opened before it.
- **Reset.** A local-params change puts every town and city back on its region
  position (`Towns.reset_local_refinement`, `Settlement.reset_local_refinement`);
  a region-params change back on its world site.

On seed 4, the harbour cities stood a median 2.1 km (up to 3.7 km) from the
local shore at their region position, and all 19 sampled land on it (moving a
median ~2 km). Villages keep clear of the settlements' final positions.

## City footprints (`localmap/cities.py`)

Display-only. A city of `P` people covers `P / density_per_ha` (200 people/ha —
a walled medieval city) of dry ground: that many local texels (~0.12 km² each,
at least one), claimed outward from its local position by distance × `1 +
edge_roughness · U[0,1)` (0.25, a ragged edge). Water is never claimed, so a
coastal city wraps its shore; rivers are vector lines, so a city spans its river.
Larger cities claim first; no texel is shared. `CityFootprints` holds the
per-texel `owner`, each claimed texel's `city` and four `corners` (for drawing)
and each city's `area_km2`. Seed 4's largest city (~540k) covers ~27 km² (~225
texels, ~6 km across); a 19k city ~1 km² (8 texels). Rural density is not carved
out.

## Farmland (`localmap/farmland.py`)

Display-only. How much ground each settlement farms follows medieval rules of
thumb:

| | Farmland | Default |
|---|---|---|
| Village | every villager × `ha_per_farmer` (cropland under three-field rotation, fallow, meadow and pasture) | 1.5 ha |
| Town | the share who farm (`town_farming_share`) × `ha_per_farmer`; the rest eat from the villages around | 0.5 |
| Mining village or town | only `mining_farming_share` of its people farm; the rest live off the mine and buy food | 0.1 |
| City | fed by its trade hinterland: a ring of market gardens and orchards `city_garden_ring_km` wide outside its footprint | 1.5 km |

Fields are claimed texel by texel, outward from the settlement like a
footprint, but a texel's distance is stretched by `1 + fertility_weight·(1 −
fertility) + slope_weight·slope/slope_saturation + forest_weight·forest` (1, 2,
1; 0.15 km/km) and jittered by `edge_roughness` (0.25), so fields fill valleys
and skip steep or wooded ground. The garden rings go first (measured against
the map's median stretch, so the ring is its nominal width on typical ground);
then towns and villages claim together, cheapest (settlement, texel) pair first
(a numba loop), each until its quota is met, so neighbours share the ground
between them. Water and built-up texels are never farmed; the hamlets and
farmsteads of the leftover rural density get no fields. `farmland_rgba(farmland,
show)` bakes it into translucent per-hex RGBA (wheat fields, dark-olive gardens,
±10% per-texel patchwork) for the kinds whose overlays are on.

Seed 4, the densest land: 940 villages farm ~5,100 km² and 59 towns ~700 km²
(every town inside the map gets its full quota; towns just outside it lose the
part that falls off the map), the one 67k city ~15 km² of gardens — ~11% of the
land, ~0.2 s. At the 90th-percentile density ~5%.

## Villages (`localmap/villages.py`)

`place_villages(local, rivers, roads, settlement_positions,
settlement_populations, seed, params) → Villages`.

- **Budgets** are the people in each *global texel ∩ map* (local rural density
  × texel area over the domain). `village_fraction` (0.7) of each is spent on
  villages; the rest stays as rural density (hamlets, farmsteads).
- **Sizes** are the town stage's truncated Pareto (`draw_town_sizes`) on
  `[village_min, village_max]` = [200, 1000], α = `size_exponent` (1), with its
  stochastic remainder.
- **Sites**, largest first, among the budget's dry domain texels by Gumbel-max
  over `log((ρ/ρ̄ + 0.05) · (1 + river·r + confluence·c + coast·k + road·d) ·
  (1 − slope_penalty · slope))` (bonuses 2 / +1 / 1 / 1; river — rivers and streams alike — and road within
  `water_reach_km` / `road_reach_km`, 0.6 km). A village claims `spacing_km`
  (2 km) × `(pop/200)^0.25`; towns and cities claim `settlement_clearance_km`
  (3 km) × `(pop/1000)^0.25`. Kinds, in precedence: mining (where miners are
  most of the site's rural density — `mining_density`, the camps around a
  worked deposit), fishing (on the coast, or where fishers are most of it —
  `fishing_density`; both in [region_fields.md](region_fields.md)), river,
  roadside, otherwise farming.
- **Accounting.** A village with no free site stays rural; each budget's rural
  density is scaled down by what its villages took, so rural + villages per
  budget is unchanged (`village_budget_report`: ~1e-8 relative on seed 4).
  `Villages.rural_density` is the countryside net of villages; the fields'
  `rural_density` stays the pre-village one (it is what the region conserves).
- Deterministic: one stream per budget, seeded by world, region, hex and global
  texel.

Seed 4, the densest land (~11 people/km²): ~940 villages per local map, mean
~370 people, median nearest neighbour 4.2 km (p10 2.7, p90 6.4). Sparser land
(~4/km²): ~350–400 villages, 6–7 km apart. Real high-medieval villages sat
2–3 km apart at about twice that density; the population calibration, not
this tier, sets the difference. Most are farming villages; with the region's
streams on the map (see [local.md](local.md#rivers-localmapriverspy)) 7–20%
are river villages (seeds 4/31/35, 2026-10-03), where the world's network alone
gave only 0–6 rivers per map and almost none.

`scripts/measure_villages.py`, seeds 4/31/35, five hexes each (2026-09-27):

| rural density | villages / 100 km² | mean size | nearest neighbour (median) | in villages | stranded in no-village budgets |
|---|---|---|---|---|---|
| ~10/km² | 1.9–2.2 | 350–370 | 4.0–4.5 km | 69–70% | 0.2–0.6% |
| ~5/km² | 1.0–1.1 | 320–335 | 5.8 km | 69% | 1% |
| ~2/km² | 0.4 | ~300 | 9 km | 65% | 7% |
| < 1/km² | 0–0.2 | ~200–235 | 11–12 km | 0–37% | 48–100% |

Budgets balance to ≤ 7e-9 everywhere. Village counts run ~10% above the budget's
prediction (`fraction · ρ / 402`) because each budget's last draw is cut to what
is left, pulling the mean size to ~330–370.

**Sparse land is nearly villageless.** Below ~1 person/km² a global texel
(~195 km²) holds fewer than 200 people and cannot afford a village, so its people
all stay dispersed — and on seed 4 the *median* land face is that sparse
(0.4/km²). This is the per-(global texel ∩ map) conservation rule meeting the
200-person floor. Kept deliberately: thin land shows its people dispersed rather
than pooling budgets or lowering the floor.

**Known artefact.** The Rural Density layer shows each budget (global texel ∩
map) as a faint block, since its countryside is scaled by its own realised
village share (0.27–0.35 on seed 4), and budgets that are small slivers of a
global texel (11% of budgets, 0.5% of the people) get no village and stay at full
density, showing as bright slivers along the hexagon edge and face seams.
