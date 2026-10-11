# Realm Population Stage

Part of `realmgen/`. [Back to realmgen overview](realmgen.md)

## Files

- `realmgen/population/builder.py` — `PopulationStage`, `TradeParams`, `TradeSpawnParams`, `PopulationParams`
- `realmgen/population/calibration.py` — pure population functions
- `realmgen/population/trade_spawn.py` — `spawn_trade_settlements`
- `realmgen/trade/network.py` — `compute_trade_network`

## Design

All populations are **real people**, anchored to medieval-tech Earth (~300–400 M
world). Rationale: `docs/plans/city_population_rework_plan.md`.

- **Rural population** (per land face; Malthusian food base): farmers `D_max ·
  land_fertility^γ · land_km²` + fishers `D_fish · aquatic_fertility^γ · water_km²`
  from the face's own shallows ([Farmers and fishers](#farmers-and-fishers)).
- **Carrying capacity** (per settlement): food base summed over claimed
  hinterland shares.
- **Urban population**: `s · share · cap + β · trade_flow` — additive trade lets a
  thin-hinterland chokepoint grow on imported grain (Constantinople/Venice).
- A settlement is a city only if urban ≥ `city_min`; modeled settlements are
  *major cities*. Towns' slice of the surplus is reserved via
  `city_surplus_share`; display-only towns are carved out of rural later
  ([realm_towns.md](realm_towns.md)).

## Params

- **`TradeParams`**: `port_cost`, `river_switch_cost`, `landing_cost` (fallback
  land↔water away from settlements, relaxation passes only; [Ports](#ports)),
  `flow_decay`, `deposit_decay` (deposit yield to its worker ∝ `exp(−deposit_decay
  · cost)`; [Miners](#miners)), `hub_display_quantile`.
- **`TradeSpawnParams`**: `hub_` / `waypoint_` / `port_` spawn quantiles,
  min-suitability and exclusion knobs (`waypoint_min_cost` too).
- **`PopulationParams`**: `relaxation_passes`, `rural_density_max` (D_max,
  /km² at fertility 1), `fishing_density_max` (D_fish, /km² of water at aquatic
  fertility 1), `fertility_exponent` (γ ≈ 2, restores density contrast),
  `surplus_rate` (s), `city_surplus_share` (city's slice of `s·cap`),
  `trade_food_weight` (β, people per unit trade flow), `city_min`,
  `mining_share` ([Miners](#miners)), `hinterland_max_distance`,
  `hinterland_influence_scale`, `hinterland_reach_scale`,
  `hinterland_water_cost`, `trade_spawn`.

## Trade network

`compute_trade_network` — two-phase flow over the hybrid land + sea graph (land
node `i`, sea node `i + N`; one multi-source Dijkstra from all settlements):
1. Each deposit (minerals + rare bio, goods offset by `RARE_BIO_TYPE_OFFSET`)
   routes to its nearest settlement, credited `abundance · exp(−deposit_decay ·
   cost)`.
2. Every settlement ships to every other, weighted by `GOOD_RARITY_WEIGHTS`, the
   destination's lack of the good, its demand, and `exp(−flow_decay · cost)`.

`TradeResult`: `face_flow`, `routes` (`(start, end, flow)` centroid hops above the
display threshold), `hub_mask` (above `hub_display_quantile`), `landing_flow`,
`pair_flow` (`(S, S)`; `[i, j]` = exporter *i* → importer *j*, aligned with
`settlements`), per good source `deposit_owner` / `deposit_cost` /
`deposit_credit`, per settlement `goods_flow` (own goods at its face: hauls in +
exports). Called at start and after each pass with population-weighted demands.

### Ports

Goods cross land↔water **only at settlement faces** (`settlement_ports=True`):
coastal settlement = port (edges to sea neighbours at `port_cost`); on a navigable
river = river port (`river_switch_cost`). Navigable rivers are water to their
mouths, so the only other land/water crossing on the map is a river mouth. The
polity sea-authority graph (`build_base_hybrid_graph`) still allows transitions
everywhere.

**Water a face's texels reach** (`realmgen/trade/water_links.py`). A face is
~220 km across: bordering a water face, or a navigable river crossing it, does
not put water near a city in it. `build_water_access` reads per face the world
rivers whose course it owns and whether it owns coast (a land texel 4-adjacent
to ocean / lake; `SiteField` ownership). Then:

- a **port** (land-only face → water-only neighbour) needs coast of that
  water's kind (`port_links` → `compute_trade_network(port_ok=…)`);
- a **river face's water edge** needs a river shared with the neighbour (it
  flows on into it) or, onto a water-only face, coast of its kind
  (`mask_sea_edges` sets the rest to `inf`).

Before this ~a third of water-trading cities (seed 4: 49/152) had a port onto a
sea face their own face had no shore on — trade loaded ships tens of km inland.

**Water links.** `TradeResult.settlement_links` lists each settlement's
land↔water links (`(water face, flow)`: a port's water face, or each water edge
out of its face's river switch). `settlement_water_links` turns them into
`Settlement.water_links`: per link, strongest first, the tags of the water that
serves it — a world river (index into `rivers`), `TAG_OCEAN` or `TAG_LAKE` (by
kind) — links on the same water merged. Every siting tier keeps the city on it
([realm_refine.md](realm_refine.md#city-siting-settlementspy)). Saved with the
settlements.

Seed 4: 130 water-linked cities sit a median 4.3 km (max 6.2) from their
strongest link's water on the region map and 0.34 km (max 0.5) on the local map,
where before cities' water legs crossed up to 37 km of land. Population stays in
band; sovereigns rise (seed 4 53 → 64, seed 31 35 → 54), largest domains unchanged.

To avoid stranding settlement-less coasts, relaxation passes give every other
coastal/river face a fallback *landing* at `landing_cost` (≈2.4× port). Where
`landing_flow` reaches `port_spawn_quantile` of positive face flows (median = the
drawing threshold), `spawn_trade_settlements` founds a port town, an ordinary
port next pass. The **final** network has no landings. Seed 4: land/sea hops away
from settlements 205/402 → 0 (14 river mouths left); ~7 ports founded.

The final `pair_flow` is published as the `trade_attribution` feature (column *j*
= imports by source, row *i* = exports by destination), read by the polity
stage's loyalty levers.

## PopulationStage

Third in `RealmPipeline`. Reads `IS_WATER`, `COST_SURFACES`, `MINERALS`,
`RARE_BIO`, `FERTILITY_MEAN`, `SUITABILITY`, `SITUATIONAL_VALUE`, `settlements`,
`danger` (× `movement_scale` added to land base cost). The trade land graph waives
the ford penalty on settlement faces (`settlement_mask`); the hinterland graph is
a copy with water passable at `hinterland_water_cost`.

1. Food base and rural population up front (`RURAL_POPULATION`; `TOWN_POPULATION`
   zeroed — the town stage moves towns between them).
2. `relaxation_passes` × : spawn hub/waypoint/port settlements, recompute
   hinterland shares, carrying capacity, trade wealth, urban population; cull
   below `city_min`; rerun trade with population-weighted demands.
3. Rebuild final trade without landings; move miners out ([Miners](#miners)).
4. Defensibility per settlement ([Defensibility](#defensibility)); store
   `carrying_cap`, `trade_wealth` diagnostics aligned with `settlements`.

## Farmers and fishers

Food base = **land** pixels (mean fertility, at `rural_density_max`) + **water**
pixels (mean aquatic fertility, at `fishing_density_max`). A land face's water
pixels are its shallows, given neighbouring water faces' aquatic fertility
([resources.md](resources.md)) — so productive coasts house fishers even beside
tundra or desert. The fish part is also written to `FISHING_POPULATION` so the
town stage puts those people on the shore (→ port towns, local fishing villages;
[realm_towns.md](realm_towns.md), [region_fields.md](region_fields.md)). Water
faces house nobody; their food base feeds cities whose hinterland reaches them.

## Miners

After the final trade network, `compute_miners` moves some of each city's
trade-fed people to the deposits it works:

    miners_city = mining_share · β·trade_wealth · own_goods_share · mineral_share

`own_goods_share` = `goods_flow / face_flow` at the city (not imports/transit);
`mineral_share` = minerals' part of its owned value (`deposit_credit × rarity`;
rare bio employs no miners). Split over its minerals by value; no city cut below
`city_min`. Added to the deposit face's `RURAL_POPULATION` (and
`MINING_POPULATION`), removed from the city — world total unchanged. The town
stage gathers them into mining towns and camps.

`deposit_decay` (0.2; cost ~1 per 100 km open land, ~2 icecap) prevents polar
and desert cities growing on deposits 1,000 km away with nobody there: 250 km →
~55% yield, 700 km → ~14%. Seeds 4/31/35 (β 900 → 1500 to restore city sizes):
0.25–0.4 M miners.

## calibration.py

- `compute_food_base` — `D · fertility^γ · area_km²`
- `split_fertility` / `compute_farm_and_fish_food` → `(farm, fish)`
- `compute_rural_population` — food base masked to land
- `compute_hinterland_shares` — soft fractional multi-source Dijkstra on a *water-permeable* graph (coastal cities claim a fishing band bounded by `hinterland_water_cost × hinterland_max_distance`)
- `compute_carrying_capacity` — Σ shares · food_base
- `compute_trade_wealth` — face_flow at the settlement face
- `compute_urban_population` — `s_eff · cap + β · wealth`
- `compute_miners`, `cull_settlements` (keep mask), `apply_population_to_settlements`

## trade_spawn.py

`spawn_trade_settlements` — seeds `Settlement`s at high-flow hubs, remote
waypoints and busy fallback landings ([Ports](#ports)); once per pass.

## Defensibility

`polity/seeding.py::compute_settlement_defensibility`, called after the final
loop (trade wealth and danger final):
- **Terrain**: ruggedness (`land_base_cost` normalised over land p50..p90) + situational value
- **River barrier**: face flow magnitude (normalised to land p90)
- **Walls**: probability `wall_base_prob + wall_danger_boost·danger + wall_wealth_boost·wealth`, quality biased by danger and wealth; keyed `(world.seed, face)`

Rolled by `roll_walls` (shared with town walls, [realm_towns.md](realm_towns.md)).
Returns `(defensibility, has_walls, wall_strength)`, stamped on `Settlement`;
read by `PolityStage`. Params: `PolityParams.defensibility`
([realm_polity.md](realm_polity.md)).

## Validation

`scripts/measure_population.py` (Zipf slope/R²/Gini),
`scripts/measure_population_absolute.py` (world total, top-city band,
trade-fed-city test).

## Outputs

- Per face: `TRADE_FLOW`, `POPULATION`, `HINTERLAND_ID`, `RURAL_POPULATION`, `FISHING_POPULATION` (fishers within rural, pre-towns), `MINING_POPULATION` (miners within it, at deposit faces), `TOWN_POPULATION` (zeroed), `is_trade_hub`, `settlement_tier` (now the *display* tier from `population / max_pop`).
- Features: `settlements` (+ population, defensibility, walls), `trade_routes`, `trade_route_layers` (`(len(trade_routes),)` int8 aligned with it: `ROUTE_LAND` / `ROUTE_WATER` (sea layer — ocean, lake or navigable river) / `ROUTE_TRANSFER` (port or river switch); `TradeResult.route_layers`; the region and local tiers keep water hops on the water), `trade_attribution` (`(S, S)`), `mine_workers` (per deposit, aligned with `minerals`; saved with settlements).
- Overlays: **Settlements**, **Trade Routes** (checked). World-map hover: tier name + population (`world_only`; zoomed views name the city under the cursor). Debug schemes: Trade Flow, Hinterland. (Rural density is drawn by the town stage.)
