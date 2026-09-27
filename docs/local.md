# Local Maps

Qt-free package for the finest tier: **each regional hexagon becomes a local
map**, refined from the region exactly the way a region is refined from the
globe. Plan and status: [plans/local_maps.md](plans/local_maps.md). Built so far:
geometry, grid and field refinement, rivers, roads, settlements and villages
(phases 1–4), the viewer (phase 5) and the diagnostic scripts (phase 6).

## Files
- `localmap/geometry.py` — `LocalGeometry`: the parent hex's texture square, mapped affinely onto `[0, 1]²`
- `localmap/grid.py` — `LocalHexGrid`: `RegionHexGrid` kept to the parent hexagon; `texture_in_map`
- `localmap/upsample.py` — `LocalUpsampleParams`, `build_local_fields`, `local_conservation_report`
- `localmap/rivers.py` — `LocalRiverParams`, `build_local_rivers`: the region's rivers re-routed over the local relief
- `localmap/roads.py` — `LocalRoadParams`, `build_local_roads`: the region's roads re-routed over the local terrain
- `localmap/villages.py` — `VillageParams`, `Villages`, `place_villages`, `village_budget_report`
- `localmap/params.py` — `LocalParams`: `upsample`, `rivers`, `roads`, `villages`
- `localmap/builder.py` — `LocalMap`, `build_local_map`: everything above for one regional hex

## Scale (defaults)

| | Region | Local |
|---|---|---|
| Map covers | ~3,070 km cell | one regional hex, ~286 km across (~53k km²) |
| Hex | ~286 km | ~23 km (`n_cols=16`) |
| Texel | ~4.5 km | ~330–365 m (`tex_size=64`) |
| Hexes | ~250 | ~180 |
| Detail octaves | 2 | 4 |

About 1.5 s for the fields and ~2.1 s for a whole local map (fields, rivers, roads,
villages) on seed 4.

## Geometry (`localmap/geometry.py`)

`LocalGeometry(region_geometry, region_grid, hex_index)` is a `RegionGeometry`
subclass whose `local_to_ecef` / `ecef_to_local` go through the parent region's
`(s, t)` frame: local `(a, b)` ↦ `origin + 2R·(a, b)`, `origin` being the hex
centre minus `R`. That frame extends smoothly past the region cell, so a hex
hanging over the cell edge works, and handedness, `up_direction`,
`local_to_lonlat` and `center_ecef` carry over unchanged. `region_id` is the
parent region's. The parent hexagon is flat-top, circumradius 0.5, centred on
`(0.5, 0.5)`.

## Grid (`localmap/grid.py`)

`LocalHexGrid` is `RegionHexGrid` with the layout filtered to hexes overlapping
the parent hexagon (~2/3 of the square; the corners belong to neighbouring local
maps). `texture_in_map(T)` marks the texels inside the parent hexagon: the
map's **domain** (`RegionFields.domain` = own hex ∩ parent hexagon) is the
aggregation weight, so two neighbouring local maps never count a texel twice.

## Refinement (`localmap/upsample.py`)

`build_local_fields(globe, region_fields, hex_index, params) → RegionFields`
runs `region.upsample.refine_fields` with `RegionSource(region_fields)` as the
parent:

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
down: per regional texel inside the parent hexagon, local mean vs regional
value. On seed 4, every field is at float32 noise (elevation ~2e-8 km).

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

## Rivers (`localmap/rivers.py`)

`build_local_rivers(local, region, region_rivers, globe, params)` re-traces
each of the region's river paths within `corridor_parent_texels` (1) regional
texels of the region's line, over the local relief, with the region's own
routing loop (`region.rivers.route_stream_paths`) and valley-seeking cost. A
path leaving and re-entering the raster is routed once per visit; a visit that
holds the path's last vertex ends on whatever it drains into near there (its
stem, a lake, the local coast). Flow is the region's.

**Only the world's channels exist.** Every stream is one of the hydrology
stage's, down to its `stream_flow_threshold`, which is coarse for villages.
`_region_paths` carries the note on where a finer local network (flow
accumulation over the local relief) would add tributaries.

On seed 4 the local rivers run ~1.1× the length of the regional ones over the
same ground (median sinuosity 1.1–1.2), ~0.1 s per map.

## Roads (`localmap/roads.py`)

`build_local_roads(local, region, region_roads, globe, rivers, …)` re-traces
each of the region's road polylines (`RegionRoads.polylines`, with their
`polyline_flows`) within `corridor_parent_texels` (2) regional texels, with the
world's land-cost model per local raster node and the local rivers' fords
(`region.roads.route_polyline`). Road ends are moved off the local water and
land exactly on their points. Sea legs and undrawn routes that cross the local
raster pass through straight. ~0.1 s per map.

## Settlements (`localmap/builder.py`)

Towns and cities in and around the map get **local, display-only** positions:
moved off the local water; a river town onto its local river within 2.5
regional texels; a port onto the local coast within 2. The world's `towns`
feature keeps the region's refinement. Villages keep clear of these positions.

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
  (1 − slope_penalty · slope))`. A village claims `spacing_km` (2 km) ×
  `(pop/200)^0.25`; towns and cities claim `settlement_clearance_km` (3 km) ×
  `(pop/1000)^0.25`. Kinds: farming, river, fishing (coast), roadside.
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
this tier, sets the difference. Nearly all are farming villages: the world
network gives a map only 0–6 rivers.

## Viewer

`RegionView` shows either tier; `MapWindow` stacks a second instance for local
maps (see [viz.md](viz.md)).

- **Opening.** In a region, the **Local maps** toggle highlights the hex under
  the cursor; a click (a press and release under 4 px apart, so drags still
  pan) opens its local map. **← Region** goes back to the region as it was.
- **`RegionView.set_local(region_view, hex)`** builds the map with
  `build_local_map` from the region view's fields, rivers, roads and deltas and
  the world's tuning, or reuses it from an LRU cache (`LOCAL_CACHE_SIZE` = 6,
  ~0.02 s), valid while the region's fields are the same object and the local
  params are unchanged; a region-params change clears it.
- **Drawing.** Everything is clipped to the parent hexagon (`_outline`). Layers
  are the world map's recipes over the local fields, with `rural_density`
  swapped for `Villages.rural_density` (the countryside net of villages).
  Overlays get the local rivers, roads, deltas and a `Towns` at the local
  positions as `refined` features; villages are drawn under the towns by
  `overlays._draw_villages` (1.3–2.4 px dots coloured by kind: farming, river,
  fishing, roadside), behind the **Villages** overlay toggle.
- **Hover.** The status bar leads with the village under the cursor (within
  6 px: "Farming village · 340 people"), then the usual texel readout. Texels
  outside the parent hexagon report nothing.
- **Params.** `AppParams.local` (`LocalParams`, "Local map" in the dialog) is
  view-side only, like `region`.

Known artefact: the Rural Density layer shows each budget (global texel ∩ map)
as a faint block, since its countryside is scaled by its own realised village
share (0.27–0.35 on seed 4), and budgets that are small slivers of a global
texel (11% of budgets, 0.5% of the people) get no village and stay at full
density, showing as bright slivers along the hexagon edge and face seams.

## Diagnostics

- **`scripts/debug_local.py`** — one local map side by side with the region's
  fields upsampled only, features and villages drawn over it; prints the
  conservation report against the region, shoreline movement and the village
  budget report. Defaults to the hex of the densest land.
- **`scripts/measure_villages.py`** — village statistics on hexes sampled
  across the rural-density range, appended to
  `scripts/results/village_results.json`.

Seeds 4/31/35, five hexes each (2026-09-27):

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
200-person floor, the same effect as the edge slivers above. Kept deliberately: thin
land shows its people dispersed rather than pooling budgets or lowering the floor.
