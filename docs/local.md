# Local Maps

Qt-free finest tier: **each regional hexagon becomes a local map**, refined from
the region exactly as a region is from the globe ([region.md](region.md)). Plan
and status: [plans/local_maps.md](plans/local_maps.md).

This doc: frame, fields, rivers, roads. Rest of the tier:
- [local_settlements.md](local_settlements.md) — towns and cities re-sited locally, river banks, city footprints, farmland, villages
- [viz_regionview.md](viz_regionview.md) — the viewer (`RegionView` shows local maps too)

One tier down: [settlement_maps.md](settlement_maps.md) — one town, city or village refined from its local map(s) at street scale.

## Files
- `localmap/geometry.py` — `LocalGeometry`: parent hex's texture square mapped affinely onto `[0, 1]²`
- `localmap/grid.py` — `LocalHexGrid`: `RegionHexGrid` kept to the parent hexagon; `texture_in_map`
- `localmap/upsample.py` — `LocalUpsampleParams` (incl. `conservation_halo_parent_texels`), `build_local_fields`, `local_conservation_report`
- `localmap/rivers.py` — `LocalRiverParams`, `build_local_rivers`
- `localmap/roads.py` — `LocalRoadParams`, `build_local_roads`
- `localmap/settlements.py`, `villages.py`, `cities.py`, `farmland.py`, `banks.py` — ([local_settlements.md](local_settlements.md))
- `localmap/params.py` — `LocalParams`: `upsample`, `streams` (`RegionStreamParams`), `rivers`, `roads`, `settlements`, `villages`, `cities`, `farmland` (`AppParams.local`, **Local map** dialog; view-side only)
- `localmap/builder.py` — `LocalMap`, `build_local_map`

## Pipeline (`localmap/builder.py`)

`build_local_map(globe, region_fields, hex_index, params, *, region_rivers,
region_roads, region_deltas, …world and realm tuning) → LocalMap`:

1. fields (`build_local_fields`);
2. rivers and the region's streams (`build_local_rivers`); region deltas snapped onto local mouths;
3. cities this map owns re-sited (`refine_local_cities`, saved on the world);
4. roads (`build_local_roads`), fording local rivers, ending on re-sited cities;
5. towns this map owns re-sited (`refine_local_towns`, saved); every other town/city in and around the map placed where it stands, moved off local water by `snap_to_land`;
6. villages (`place_villages`), clear of towns and cities;
7. city footprints (`city_footprints`);
8. farmland (`place_farmland`).

`LocalMap`: `fields`, `rivers`, `roads`, `villages`, `deltas`, `town_index` /
`town_positions`, `city_index` / `city_positions` (world indices + drawn
positions of settlements in and around the map), `city_footprints`,
`farmland`, `hex_index`. No region rivers/roads → none here. Deterministic given
which settlements earlier local maps re-sited; villages, footprints and farmland
share one seed per (world, region, hex).

## Scale (defaults)

| | Region | Local |
|---|---|---|
| Map covers | ~3,070 km cell | one regional hex, ~286 km (~53k km²) |
| Hex | ~286 km | ~23 km (`n_cols=16`) |
| Texel | ~4.5 km | ~365 m (`tex_size=64`) |
| Hexes | 216 | ~180 |
| Detail octaves | 2 | 4 |

Seed 4: ~1.5 s fields, ~2.1 s whole map.

## Geometry (`localmap/geometry.py`)

`LocalGeometry(region_geometry, region_grid, hex_index)` subclasses
`RegionGeometry`; `local_to_ecef` / `ecef_to_local` go through the region's
`(s, t)`: local `(a, b)` ↦ `origin + 2R·(a, b)`, `origin` = hex centre − `R`
(`to_region`). The region frame extends past the cell, so over-hanging hexes
work; handedness, `up_direction`, `local_to_lonlat`, `center_ecef` carry over.
`region_id` = parent's. Parent hexagon: flat-top, circumradius 0.5, centred
`(0.5, 0.5)` (`PARENT_CENTER`, `PARENT_RADIUS`).

## Grid (`localmap/grid.py`)

`LocalHexGrid` = `RegionHexGrid` filtered to hexes overlapping the parent hexagon
(~⅔ of the square). `texture_in_map(T)` marks texels inside it; the **domain**
(`RegionFields.domain` = own hex ∩ parent hexagon) is the aggregation weight, so
neighbouring local maps never double-count a texel.

## Refinement (`localmap/upsample.py`)

`build_local_fields(globe, region_fields, hex_index, params) → RegionFields` runs
`region.upsample.refine_fields` ([region_fields.md](region_fields.md)) with
`RegionSource(region_fields)`:

- **Locating**: point in region `(s, t)`, owned by the nearest-centre regional
  hex (KD-tree), read bilinearly from its texture through the over-hang.
- **Anchors** = regional texels (`hex·T·T + row·T + col`) in `parent_anchors`.
  `anchors` / `faces` / `face_uv` stay **global**, so display, hover and corridor
  code treat local maps like regions.
- **Slope** = region's refined slope; **lakes** from the region's
  `filled_height` (its small lakes re-flooded too) plus the map's own
  [small lakes](region_fields.md#small-lakes-regionlakespy), drained along the
  region's rivers and streams (`build_local_map` passes their `path_segments`
  as `drains`), with none within 1.25 regional texels of them; **danger** = region's continuous danger, bilinear, conserved
  per regional texel.
- **Noise** seeded `world.seed XOR 0x10CA`, shared by all local maps of a world
  (as the region tier's is, per world).
- **Conserved over a halo** `conservation_halo_parent_texels` (2 regional
  texels, ~9 km) past the parent hexagon (`RegionFields.conserved`;
  `LocalHexGrid(halo=…)` keeps the hexes that cover it, ~13% more). Conserving
  over the hexagon alone, neighbouring maps' smooth corrections faded out
  differently at their seam: seed 4 (region 16, hexes 96 | 110) disagreed by
  7.5 m mean / 24.5 m max at a point; with the halo, 0.4 m / 5.4 m, at no
  measurable build cost. The regional texels the hexagon edge cuts (~5%) are
  conserved over their whole area, so over a map's own part alone they are no
  longer exact (seed 4: p95 ~10 m, max ~30 m); every other texel is.

`local_conservation_report(local, region)` = `conservation_report` one tier down
(`anchor_residuals` per regional texel over the texels it is conserved with:
the parent hexagon and its halo). Seed 4: all fields at float32 noise
(elevation ~2e-8 km). Rural people match the parent hex to ~0.03% at `T=64`
(tiers count rim texels slightly differently, ~1/T).

### `LocalUpsampleParams`

`RegionUpsampleParams` with local defaults (only these are the local tier's own; the
other fields are the region's, copied by `with_region_detail` / `AppParams`): `n_cols=16`, `slope_reference=0.03`,
`slope_floor=0.03`, `max_relief_km=0.6`, `lake_smoothing_parent_texels=0.25` (~3 local
texels, ~1 km: rounds the staircase's parent-texel pits off; 0.5 leaves no local lakes),
`lake_max_parent_texels=1` (only lakes below the region's resolution, ~20 km²; uncapped,
a local map found region-sized lakes the region had left dry — 93% of its own lake area),
`lake_breach_parent_texels=1.5` (~7 km, near the region's ~15 km; with the cap it only sets
how many sub-texel lakes: seed 4, 40 maps in regions 48/51/16/45/25, 577 local-only lakes,
0.27% of land, median ~10 km², 50–80 per map in lake districts, 0–10 elsewhere),
`water_spread_texels=12` (~one regional
texel), `biome_smoothing=13` (~one regional texel, so biome edges follow the
region), `slope_saturation=0.1`. Relief numbers are a first cut, untuned (seed 4
region 34: ~50 m land relief std vs parent texel). Coast params keep region
defaults; band, blur and bays scale with the region's texels
([region_fields.md](region_fields.md#coast-style)).

## Rivers (`localmap/rivers.py`)

`build_local_rivers(local, region, region_rivers, globe, params)` re-traces each
region river within `corridor_parent_texels` (1) regional texels of its line over
local relief, with the region's loop (`region.rivers.route_stream_paths`) and
valley-seeking cost. A path leaving and re-entering the raster is routed once per
visit; the visit holding its last vertex ends on whatever it drains into (stem,
lake, local coast). Flow is the region's. Water costs ×50 (the distributaries'
`WATER_COST`) except within a corridor of the region path's own hidden stretches
(lake crossings, the run to its mouth), so a re-route keeps to land where its
parent does — on a flattened delta plain valley-seeking would otherwise hug the
refined shore and be hidden as a sea re-entry. Delta distributaries are ordinary
region paths ([region_features.md](region_features.md#delta-distributaries-regiondistributariespy)),
except that one starting on another path (`starts_on_path`) starts on the
nearest node of that path's local re-route, its first point snapped onto it.

**Region streams.** The world network stops at `stream_flow_threshold` (coarse
for villages); `build_local_map` re-traces the region's own streams
(`region_streams` from `build_region_streams`;
[region_features.md](region_features.md#streams-below-the-worlds-network-regionstreamspy))
after the rivers, so each ends on what it drains into. They go to
`RegionRivers.streams`, not `paths`: drawn (physical width, mostly hairlines)
and used by villages, but town/city refinement, confluences, mouths and deltas
keep to world channels, and roads don't ford them. No `region_streams` → none.

Seed 4: dense-land hex ~30 streams (~1,800 km) where it had 1 river; ~0.3 s per
map + ~0.35 s once per region. Local rivers ~1.1× the regional length (median
sinuosity 1.1–1.2), ~0.1 s per map.

### `LocalRiverParams`
`raster_texels` (1.0), `margin` (0.03), `corridor_parent_texels` (1.0),
`valley_seeking` (2.0), `smoothing_iterations` (8).

## Roads (`localmap/roads.py`)

`build_local_roads(local, region, region_roads, globe, rivers, params, land,
danger_scale, land_reach, moved)` re-traces each `RegionRoads.polylines` (with
`polyline_flows`) within `corridor_parent_texels` (2) regional texels, with the
world land-cost model and local river fords (`region.roads.route_polyline`).
Ends moved off local water and landed exactly on their points — a city's region
position first moved to its local re-site (`moved`;
[local_settlements.md](local_settlements.md#towns-and-cities)). River and lake
legs (`RegionRoads.water_polylines`) are re-traced the same way over the local
water and rivers within `water_corridor_parent_texels` (3) — the region's water
cost and straight sailing over open water
([region_features.md](region_features.md#roads-regionroadspy)); only their pinned
(port) ends move with cities and off the water. Sea legs and undrawn routes pass
straight. ~0.1 s per map.

### `LocalRoadParams`
`raster_texels` (2.0), `margin` (0.03), `corridor_parent_texels` (2.0),
`smoothing_iterations` (6), `water_corridor_parent_texels` (3.0),
`water_land_cost` (20, as the region's).

## Diagnostics

- `scripts/debug_local.py` — local map vs region upsampled-only, with features;
  prints conservation, shoreline movement, village budget. Defaults to the
  densest-land hex.
- `scripts/measure_villages.py` — village stats across the rural-density range →
  `scripts/results/village_results.json`
  ([local_settlements.md](local_settlements.md#villages-localmapvillagespy)).
