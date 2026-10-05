# Local Maps

Qt-free finest tier: **each regional hexagon becomes a local map**, refined from
the region exactly as a region is from the globe ([region.md](region.md)). Plan
and status: [plans/local_maps.md](plans/local_maps.md).

This doc: frame, fields, rivers, roads. Rest of the tier:
- [local_settlements.md](local_settlements.md) — towns and cities re-sited locally, city footprints, farmland, villages
- [viz_regionview.md](viz_regionview.md) — the viewer (`RegionView` shows local maps too)

## Files
- `localmap/geometry.py` — `LocalGeometry`: parent hex's texture square mapped affinely onto `[0, 1]²`
- `localmap/grid.py` — `LocalHexGrid`: `RegionHexGrid` kept to the parent hexagon; `texture_in_map`
- `localmap/upsample.py` — `LocalUpsampleParams`, `build_local_fields`, `local_conservation_report`
- `localmap/rivers.py` — `LocalRiverParams`, `build_local_rivers`
- `localmap/roads.py` — `LocalRoadParams`, `build_local_roads`
- `localmap/settlements.py`, `villages.py`, `cities.py`, `farmland.py` — ([local_settlements.md](local_settlements.md))
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
  `filled_height`; **danger** = region's continuous danger, bilinear, conserved
  per regional texel.
- **Noise** seeded `world.seed XOR 0x10CA`, shared by all local maps of a world,
  so neighbours agree where they meet.

`local_conservation_report(local, region)` = `conservation_report` one tier down
(`anchor_residuals` per regional texel inside the parent hexagon). Seed 4: all
fields at float32 noise (elevation ~2e-8 km). Rural people match the parent hex
to ~0.03% at `T=64` (tiers count rim texels slightly differently, ~1/T).

### `LocalUpsampleParams`

`RegionUpsampleParams` with local defaults: `n_cols=16`, `slope_reference=0.03`,
`slope_floor=0.03`, `max_relief_km=0.6`, `water_spread_texels=12` (~one regional
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
lake, local coast). Flow is the region's. Delta distributaries are ordinary
region paths ([region_features.md](region_features.md#delta-distributaries-regiondistributariespy)).

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
[local_settlements.md](local_settlements.md#towns-and-cities)). Sea legs and
undrawn routes pass straight. ~0.1 s per map.

### `LocalRoadParams`
`raster_texels` (2.0), `margin` (0.03), `corridor_parent_texels` (2.0),
`smoothing_iterations` (6).

## Diagnostics

- `scripts/debug_local.py` — local map vs region upsampled-only, with features;
  prints conservation, shoreline movement, village budget. Defaults to the
  densest-land hex.
- `scripts/measure_villages.py` — village stats across the rural-density range →
  `scripts/results/village_results.json`
  ([local_settlements.md](local_settlements.md#villages-localmapvillagespy)).
