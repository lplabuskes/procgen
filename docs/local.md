# Local Maps

Qt-free package for the finest tier: **each regional hexagon becomes a local
map**, refined from the region exactly the way a region is refined from the
globe ([region.md](region.md)). Plan and status:
[plans/local_maps.md](plans/local_maps.md).

This doc covers the frame, fields, rivers and roads. The rest of the tier:

- [local_settlements.md](local_settlements.md) — towns and cities re-sited on the local map, city footprints, farmland and villages
- [viz_regionview.md](viz_regionview.md) — the viewer (`RegionView` shows local maps too)

## Files
- `localmap/geometry.py` — `LocalGeometry`: the parent hex's texture square, mapped affinely onto `[0, 1]²`
- `localmap/grid.py` — `LocalHexGrid`: `RegionHexGrid` kept to the parent hexagon; `texture_in_map`
- `localmap/upsample.py` — `LocalUpsampleParams`, `build_local_fields`, `local_conservation_report`
- `localmap/rivers.py` — `LocalRiverParams`, `build_local_rivers`: the region's rivers and streams re-routed over the local relief
- `localmap/roads.py` — `LocalRoadParams`, `build_local_roads`: the region's roads re-routed over the local terrain
- `localmap/settlements.py` — `LocalSettlementParams`, `local_owned`, `refine_local_cities`, `refine_local_towns` ([local_settlements.md](local_settlements.md))
- `localmap/villages.py` — `VillageParams`, `Villages`, `place_villages`, `village_budget_report` ([local_settlements.md](local_settlements.md))
- `localmap/cities.py` — `CityParams`, `CityFootprints`, `city_footprints` ([local_settlements.md](local_settlements.md))
- `localmap/farmland.py` — `FarmlandParams`, `Farmland`, `place_farmland`, `farmland_rgba` ([local_settlements.md](local_settlements.md))
- `localmap/params.py` — `LocalParams`: `upsample`, `streams` (the region's `RegionStreamParams`), `rivers`, `roads`, `settlements`, `villages`, `cities`, `farmland` (`AppParams.local`, the **Local map** dialog section; view-side only, like `region`)
- `localmap/builder.py` — `LocalMap`, `build_local_map`: everything above for one regional hex

## Pipeline (`localmap/builder.py`)

`build_local_map(globe, region_fields, hex_index, params, *, region_rivers,
region_roads, region_deltas, …world and realm tuning) → LocalMap` runs, in order:

1. the fields (`build_local_fields`);
2. the rivers and the region's streams (`build_local_rivers`), and the region's deltas snapped onto the local river mouths;
3. the cities this map owns, re-sited (`refine_local_cities`, saved on the world);
4. the roads (`build_local_roads`), fording the local rivers and ending on the re-sited cities;
5. the towns this map owns, re-sited (`refine_local_towns`, saved on the world); every town and city in and around the map is then placed where it stands, moved off the local water with `snap_to_land` if it is not this map's to refine;
6. the villages (`place_villages`), clear of those towns and cities;
7. the cities' built-up footprints (`city_footprints`);
8. the farmland (`place_farmland`).

`LocalMap` holds `fields`, `rivers`, `roads`, `villages`, `deltas`,
`town_index` / `town_positions`, `city_index` / `city_positions` (world indices
and drawn positions of the settlements in and around the map),
`city_footprints`, `farmland`, and `hex_index`. Without the region's rivers or
roads the map has none. Deterministic, given which settlements earlier local
maps re-sited. Villages, footprints and farmland share one seed per (world,
region, hex).

## Scale (defaults)

| | Region | Local |
|---|---|---|
| Map covers | ~3,070 km cell | one regional hex, ~286 km across (~53k km²) |
| Hex | ~286 km | ~23 km (`n_cols=16`) |
| Texel | ~4.5 km | ~365 m (`tex_size=64`) |
| Hexes | 216 | ~180 |
| Detail octaves | 2 | 4 |

About 1.5 s for the fields and ~2.1 s for a whole local map (fields, rivers, roads,
villages) on seed 4.

## Geometry (`localmap/geometry.py`)

`LocalGeometry(region_geometry, region_grid, hex_index)` is a `RegionGeometry`
subclass whose `local_to_ecef` / `ecef_to_local` go through the parent region's
`(s, t)` frame: local `(a, b)` ↦ `origin + 2R·(a, b)`, `origin` being the hex
centre minus `R` (`to_region`). That frame extends smoothly past the region
cell, so a hex hanging over the cell edge works, and handedness, `up_direction`,
`local_to_lonlat` and `center_ecef` carry over unchanged. `region_id` is the
parent region's. The parent hexagon is flat-top, circumradius 0.5, centred on
`(0.5, 0.5)` (`PARENT_CENTER`, `PARENT_RADIUS`).

## Grid (`localmap/grid.py`)

`LocalHexGrid` is `RegionHexGrid` with the layout filtered to hexes overlapping
the parent hexagon (~2/3 of the square; the corners belong to neighbouring local
maps). `texture_in_map(T)` marks the texels inside the parent hexagon: the
map's **domain** (`RegionFields.domain` = own hex ∩ parent hexagon) is the
aggregation weight, so two neighbouring local maps never count a texel twice.

## Refinement (`localmap/upsample.py`)

`build_local_fields(globe, region_fields, hex_index, params) → RegionFields`
runs `region.upsample.refine_fields` ([region_fields.md](region_fields.md)) with
`RegionSource(region_fields)` as the parent:

- **Locating.** A point is placed in the region's `(s, t)` frame, owned by the
  regional hex with the nearest centre (a KD-tree over hex centres), and read
  from that hex's texture, bilinearly, through its over-hang.
- **Anchors** are regional texels (`hex·T·T + row·T + col`), stored as
  `parent_anchors`. `anchors` / `faces` / `face_uv` stay **global**, so the
  display, hover and corridor code read a local map as they read a region.
- **Slope** is the region's refined slope; **lakes** use the region's exported
  `filled_height`; **danger** is the region's (already continuous) danger read
  bilinearly and conserved per regional texel.
- **Noise** is seeded `world.seed XOR 0x10CA`, shared by every local map of a
  world, so neighbouring local maps of one region agree where they meet.

`local_conservation_report(local, region)` is `conservation_report` one tier
down (`anchor_residuals` against the region's textures): per regional texel
inside the parent hexagon, local mean vs regional value. On seed 4, every field
is at float32 noise (elevation ~2e-8 km).

Rural people in a local map match the parent hex's to ~0.03% at `T=64`; the
two tiers count the regional texels on the parent hexagon's rim slightly
differently (a ~1/T effect).

### `LocalUpsampleParams`

A `RegionUpsampleParams` subclass with local defaults: `n_cols=16`,
`slope_reference=0.03`, `slope_floor=0.03`, `max_relief_km=0.6`,
`water_spread_texels=12` (about one regional texel), `biome_smoothing=13`
(about one regional texel, so biome edges follow the region's pattern), and
`slope_saturation=0.1`. The relief numbers are a first cut and not yet tuned
(seed 4 region 34: ~50 m std of land relief against the parent texel).
The coast-style and shore-reshaping params keep the region's defaults; they
read the `coast_smoothness` the region copies down, and the band, blur and
bays scale with the region's texels ([region_fields.md](region_fields.md#coast-style)).

## Rivers (`localmap/rivers.py`)

`build_local_rivers(local, region, region_rivers, globe, params)` re-traces
each of the region's river paths within `corridor_parent_texels` (1) regional
texels of the region's line, over the local relief, with the region's own
routing loop (`region.rivers.route_stream_paths`) and valley-seeking cost. A
path leaving and re-entering the raster is routed once per visit; a visit that
holds the path's last vertex ends on whatever it drains into near there (its
stem, a lake, the local coast). Flow is the region's. A delta's distributaries
are ordinary region paths that start on their parent channel
([region_features.md](region_features.md#delta-distributaries-regiondistributariespy)),
so they are re-traced the same way.

**Then the region's streams.** The world's network stops at its
`stream_flow_threshold`, coarse for villages. `build_local_map` takes the
region's own streams below it (`region_streams`, from
`region.streams.build_region_streams`; see
[region_features.md](region_features.md#streams-below-the-worlds-network-regionstreamspy))
and re-traces them the same way, after the rivers, so each ends on the river or
stream it drains into. They go to `RegionRivers.streams`, not `paths`: the view
draws them (at their physical width, most as hairlines) and villages site on
them, but town and city refinement, confluences, mouths and deltas keep to the
world's channels, and roads do not ford them. Without `region_streams` a map
has none.

Seed 4, a dense-land hex: ~30 streams (~1,800 km) on a map that had 1 river;
~0.3 s more per map, plus ~0.35 s once per region for the network.

On seed 4 the local rivers run ~1.1× the length of the regional ones over the
same ground (median sinuosity 1.1–1.2), ~0.1 s per map.

### `LocalRiverParams`
`raster_texels` (1.0), `margin` (0.03), `corridor_parent_texels` (1.0),
`valley_seeking` (2.0), `smoothing_iterations` (8).

## Roads (`localmap/roads.py`)

`build_local_roads(local, region, region_roads, globe, rivers, params, land,
danger_scale, land_reach, moved)` re-traces each of the region's road polylines
(`RegionRoads.polylines`, with their `polyline_flows`) within
`corridor_parent_texels` (2) regional texels, with the world's land-cost model
per local raster node and the local rivers' fords (`region.roads.route_polyline`).
Road ends are moved off the local water and land exactly on their points — a
city's region position first moved to where the local map re-sited it (`moved`;
see [local_settlements.md](local_settlements.md#towns-and-cities)). Sea legs and
undrawn routes that cross the local raster pass through straight. ~0.1 s per map.

### `LocalRoadParams`
`raster_texels` (2.0), `margin` (0.03), `corridor_parent_texels` (2.0),
`smoothing_iterations` (6).

## Diagnostics

- **`scripts/debug_local.py`** — one local map side by side with the region's
  fields upsampled only, features and villages drawn over it; prints the
  conservation report against the region, shoreline movement and the village
  budget report. Defaults to the hex of the densest land.
- **`scripts/measure_villages.py`** — village statistics on hexes sampled
  across the rural-density range, appended to
  `scripts/results/village_results.json` (results in
  [local_settlements.md](local_settlements.md#villages-localmapvillagespy)).
