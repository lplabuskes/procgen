# Local Map Settlements

Part of the local tier — [back to local.md](local.md). Towns and cities a local
map re-sites, the villages it places, and the ground they build on and farm.
Drawing: [viz_regionview.md](viz_regionview.md).

## Files
- `localmap/settlements.py` — `LocalSettlementParams`, `local_owned`, `refine_local_cities`, `refine_local_towns`
- `localmap/cities.py` — `CityParams`, `CityFootprints`, `city_footprints` (display-only)
- `localmap/farmland.py` — `FarmlandParams`, `Farmland`, `place_farmland`, `farmland_rgba` (display-only)
- `localmap/villages.py` — `VillageParams`, `Villages`, `place_villages`, `village_budget_report`
- `localmap/banks.py` — `Channels`, `river_channels`, `covered_texels`, `onto_banks`, `crossing_pairs`

## River banks (`localmap/banks.py`)

Local maps draw rivers and streams at bankfull width (`channel_width_km`, 7 m ×
√Q; seed 4: world rivers ~180–380 m, up to km near big mouths; region streams
~20–170 m), so towns and cities sited *on* a river (within 1.5 texels of its centre
line) can sit in the channel.

- **Banks.** Towns, cities and villages are drawn `bank_clearance_km` (0.1)
  past the nearest bank, on the side they already stand (on the centre line:
  left of the flow); up to four rounds for confluences. Display only: the world
  keeps its positions. Road ends follow moved cities.
- **Covered texels** (centre under a channel) are never a village site, city
  footprint or field. Water masks and rural density are unchanged.
- **Field barrier.** A village farms only its own bank of a river at least
  `river_barrier_width_m` (50 m) wide: a (village, texel) pair whose straight
  line crosses such a channel is dropped. Towns and gardens may still cross.

Seed 4, six largest river cities: four stood in the channel (up to 1.6 km in);
all now 0.1 km off the bank. Before the barrier, 359 / 111 village-field texels
lay across a wide river on two of those maps.

## Towns and cities

Re-sited once more from the region's positions with the region's rules
(`refine_towns` / `refine_cities` at `LOCAL_TIER`;
[region_features.md](region_features.md#towns-and-cities)) on ~0.36 km texels and
local coast, rivers and roads, within `refine_radius_parent_texels` (1.5
regional texels, ~7 km). Harbours/ports → local shore, river cities/towns →
local river, when in reach; a city's water links are hard constraints, met past
the reach when they must be. **Saved on the world** (`position`, flagged
`local_refined`; `region_position` keeps the region's), so it stands in the same
place on every map.

- **Ownership.** Refined once, by the local map of the regional hex nearest its
  region position (`local_owned`) in the region that refined it. Other maps draw
  it at its region position, off local water via `snap_to_land`, until its own
  hex is opened.
- **Order.** Cities first (`refine_local_cities` returns `moved = (region
  positions, positions)` for `build_local_roads`); towns after roads (route flag
  reads them). Town spacing vs others' *region* positions, and city jitter/town
  noise seeded per settlement and tier → result independent of which maps were
  opened before.
- **Reset.** Local-params change → back to region positions
  (`Towns.reset_local_refinement`, `Settlement.reset_local_refinement`);
  region-params change → back to world sites.

Seed 4: harbour cities sat a median 2.1 km (≤ 3.7) from the local shore at
region positions; all 19 sampled land on it (median move ~2 km). Villages keep
clear of final positions.

## City footprints (`localmap/cities.py`)

Display-only. A city of `P` covers `P / density_per_ha` (200/ha — walled
medieval) of dry ground: that many local texels (~0.12 km², ≥ 1), claimed outward
by distance × `1 + edge_roughness · U[0,1)` (0.25). Never water (coastal cities
wrap the shore) or a river's covered texels; cities span their rivers. Larger cities claim
first; no sharing. `CityFootprints`: per-texel `owner`, each claimed texel's
`city` and four `corners`, each city's `area_km2`. Seed 4: largest (~540k) ~27
km² (~225 texels, ~6 km across); 19k city ~1 km² (8 texels). Rural density not
carved out.

## Farmland (`localmap/farmland.py`)

Display-only; medieval rules of thumb:

| | Farmland | Default |
|---|---|---|
| Village | every villager who farms (`Villages.farming_share`) × `ha_per_farmer` (three-field rotation, fallow, meadow, pasture) | 1.5 ha |
| Town | `town_farming_share` × `ha_per_farmer` (the rest eat from surrounding villages) | 0.5 |
| Mining town | only `mining_farming_share` farm | 0.1 |
| City | trade-fed; ring of market gardens/orchards `city_garden_ring_km` wide outside its footprint | 1.5 km |

- Claimed outward like a footprint, distance stretched by `1 +
  fertility_weight·(1 − fertility) + slope_weight·slope/slope_saturation +
  forest_weight·forest` (1, 2, 1; 0.15 km/km), jittered by `edge_roughness`
  (0.25) → fields fill valleys, skip steep/wooded ground.
- Garden rings first (vs the map's median stretch, so nominal width on typical
  ground); then towns and villages together, cheapest (settlement, texel) pair
  first (numba), each until its quota — neighbours share ground.
- Never water, built-up or under a river; village fields stop at rivers ≥
  `river_barrier_width_m` (see [River banks](#river-banks-localmapbankspy)).
  Leftover rural density (hamlets) gets no fields.
- `farmland_rgba(farmland, show)` bakes translucent per-hex RGBA (wheat fields,
  dark-olive gardens, ±10% patchwork) for kinds whose overlays are on.

Seed 4 densest land: 940 villages ~5,100 km², 59 towns ~700 km² (towns just
outside the map lose the off-map part), the one 67k city ~15 km² — ~11% of land,
~0.2 s. 90th-percentile density: ~5%.

## Villages (`localmap/villages.py`)

`place_villages(local, rivers, roads, settlement_positions,
settlement_populations, seed, params) → Villages`.

- **Budgets** = people per *global texel ∩ map* (local rural density × texel area
  over the domain); `village_fraction` (0.7) spent on villages, rest stays rural
  (hamlets, farmsteads).
- **Sizes**: town stage's truncated Pareto (`draw_town_sizes`) on
  `[village_min, village_max]` = [200, 1000], α = `size_exponent` (1), with
  stochastic remainder.
- **Sites**, largest first, among the budget's dry domain texels not under a river, Gumbel-max over
  `log((ρ/ρ̄ + 0.05) · (1 + river·r + confluence·c + coast·k + road·d) · (1 −
  slope_penalty · slope))` (bonuses 2 / +1 / 1 / 1; river = rivers and streams;
  reach `water_reach_km` / `road_reach_km` 0.6 km).
- **Spacing**: village claims `spacing_km` (2) × `(pop/200)^0.25`; towns/cities
  `settlement_clearance_km` (3) × `(pop/1000)^0.25`.
- **Kinds** (precedence): mining (miners are most of the site's rural density —
  `mining_density`), fishing (coast, or fishers most of it — `fishing_density`;
  both [region_fields.md](region_fields.md)), river, roadside, farming.
- **Farming share** (`Villages.farming_share`): fishers and miners are part of
  the rural density, so a village's people are its site's mix; the share that
  farms — `1 − (fishing_density + mining_density) / rural_density` at its texel
  — is all that needs fields (`place_farmland(village_farming=…)`; the
  settlement tier's hinterland reads it too). Seed 4: fishing villages ~0.08.
- **Accounting.** No free site → stays rural; each budget's rural density scaled
  down by what villages took, so rural + villages per budget unchanged
  (`village_budget_report`: ~1e-8 relative). `Villages.rural_density` = net of
  villages; fields' `rural_density` stays pre-village (what the region
  conserves).
- Deterministic: one stream per budget, seeded (world, region, hex, global
  texel).

Seed 4 densest land (~11/km²): ~940 villages per map, mean ~370 people, median
nearest neighbour 4.2 km (p10 2.7, p90 6.4). ~4/km²: ~350–400 villages, 6–7 km
apart. Real high-medieval villages sat 2–3 km apart at ~2× the density — set by
population calibration, not this tier. With the region's streams
([local.md](local.md#rivers-localmapriverspy)) 7–20% are river villages (seeds
4/31/35, 2026-10-03; ~none from the world network alone).

`scripts/measure_villages.py`, seeds 4/31/35, five hexes each (2026-09-27):

| rural density | villages / 100 km² | mean size | nearest neighbour (median) | in villages | stranded in no-village budgets |
|---|---|---|---|---|---|
| ~10/km² | 1.9–2.2 | 350–370 | 4.0–4.5 km | 69–70% | 0.2–0.6% |
| ~5/km² | 1.0–1.1 | 320–335 | 5.8 km | 69% | 1% |
| ~2/km² | 0.4 | ~300 | 9 km | 65% | 7% |
| < 1/km² | 0–0.2 | ~200–235 | 11–12 km | 0–37% | 48–100% |

Budgets balance to ≤ 7e-9. Counts run ~10% above prediction (`fraction · ρ /
402`) since each budget's last draw is cut, pulling mean size to ~330–370.

**Sparse land is nearly villageless (deliberate).** Below ~1/km² a global texel
(~195 km²) holds < 200 people and can't afford a village; on seed 4 the *median*
land face is that sparse (0.4/km²). Per-(global texel ∩ map) conservation meets
the 200-person floor; kept rather than pooling budgets or lowering the floor.

**Known artefact.** Rural Density shows each budget as a faint block (countryside
scaled by its own realised village share, 0.27–0.35 on seed 4); sliver budgets
(11% of budgets, 0.5% of people) get no village and show as bright slivers along
hex edges and face seams.
