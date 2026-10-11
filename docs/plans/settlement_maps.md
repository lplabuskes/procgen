# Settlement Maps — Plan

The fourth and finest tier: **a street-scale map of one town or city**, refined
from its local map exactly as a local map is from its region
([../local.md](../local.md)) — bilinear read of the parent, detail above the
parent's Nyquist, per-parent-texel conservation — through the same
`region.upsample.refine_fields` and a `RegionSource` over the local map's fields.
Built on demand: asking for a settlement's map builds its region and local map
first if they are missing.

[regional_local_maps_design.md](regional_local_maps_design.md) deferred
street-level maps as a separate project; this is that project.

Two halves, split by an inspectable contract:

- **Site** (phases 0–5): terrain, water, roads, footprint and the settlement's
  profile at street scale → `SettlementSite`. Derived from the tiers above;
  conserved where they conserve.
- **Layout** (phase 6): walls, streets, districts, lots → `SettlementLayout`.
  Reads only `SettlementSite` (so it can be tested on hand-built sites). Seeded,
  mostly generated rather than derived.

Both are **vector data** (Decision 8): the only raster is the terrain the site
is derived from.

## Scope

- **Cities** (`settlements` feature) **and towns** (`towns` feature) — both
  are saved on the world with a final, locally refined position.
- **Villages** (added 2026-10-10): they are local-view only, regenerated per
  (world, region, hex), so they are keyed by (region, hex, village index) on
  their local map — valid while the local params don't change.
- **Not the history layer.** Inputs it will eventually supply (founding age,
  past fires and sieges, ruins) are explicit stubs here (see Phase 5).

## Scale (defaults)

| | Local | Settlement |
|---|---|---|
| Map covers | one regional hex, ~286 km | one settlement: footprint + garden ring + margin, ~3–10 km square |
| Texel | ~365 m (`tex_size=64`) | `texel_m` = 15 m (T varies: ~200–700) |
| Refinement | ~12× | ~24× (~4–5 more octaves) |
| Footprint | 19k city ≈ 8 texels (~1 km²); 540k ≈ 225 (~27 km²) | the same areas at ~4,400 / ~120,000 texels |

Frame side = `2 · (footprint radius + city_garden_ring_km + margin_km)`, where
footprint radius = √(area / π) from `CityFootprints.area_km2` (towns: their
population at `CityParams.density_per_ha`, as cities — towns have no local
footprint, see Decision 4). `city_garden_ring_km` is `FarmlandParams`'
(1.5 km). Raster cost at 700² is fine for the corridor router (~0.5 M cells).

## Decisions

1. **One frame per settlement, not a tiling.** Settlement maps don't cover the
   world. Each is a square centred on the settlement's final `position`, in the
   owning local map's `(s, t)`. Conservation is per **local texel** inside the
   frame (the frame is the domain); a local texel cut by the frame edge is
   conserved over its in-frame part only.
   **A frame spans every local map it touches.** Near a regional hex edge it
   reaches past the owning local map's domain; there the neighbouring local
   map(s) are built (up to three at a hex corner, possibly in another region)
   and each fine texel's parent is the local texel of the map whose regional
   hexagon contains it (A4's ownership rule, applied per point). Fields,
   features and conservation all come from that map.
2. **The settlement doesn't move again.** Its position is final at the local tier
   (`local_refined`, saved). The settlement map places its anchor at that
   point; nothing here writes to the world. One-directional, like the display-only
   footprints and farmland. Note the local map *draws* it at the bank-shifted
   point (`onto_banks` in `build_local_map`; `LocalMap.city_positions` /
   `town_positions`), and the local footprint is claimed around that point —
   centre the frame on the drawn position, not `Settlement.position`.
3. **Noise per world, layout per settlement.** Terrain detail is seeded
   `world.seed XOR <tier constant>` (as the local tier's `_SEED_SALT_LOCAL =
   0x10CA`), so two overlapping frames (twin towns, a town in a city's garden
   ring) share their noise where they overlap. They won't match exactly: each
   frame conserves the local texels its edge cuts over its own in-frame part
   (Decision 1). Layout streams are seeded per (world, settlement kind, index).
4. **Built-up area is conserved (cities).** The local `city_footprints` is the
   coarse outline: the fine footprint claims, inside each local texel, the
   built-up share that texel had (1 or 0 now; fractional at the footprint edge
   after re-claiming). Totals match `area_km2` to the texel. Use the owning
   hex's local map: footprint jitter is seeded per (world, region, hex), so a
   city near a hex edge has a different footprint on its neighbour's map.
   **Towns have no local footprint** (`build_local_map` footprints cities
   only; towns get farmland), so a town's built-up area is claimed fresh at
   `density_per_ha` with nothing to conserve against.
5. **Nothing new is persisted.** Every tier is deterministic; a settlement map
   rebuilds from the world in a few seconds. The chain caches in memory (LRU,
   like `RegionView`'s local cache).
6. **Params are advanced-only.** `SettlementMapParams` hangs off `AppParams`
   like `LocalParams`; nothing is added to `BASIC_PARAMS` without asking.
7. **Visuals are debug-mode first.** Ask before adding to the main view.
8. **Vector, not raster (decided 2026-10-10).** The refined terrain (elevation,
   water masks) is the one raster: the field everything else is derived from
   and queried against (height, slope, wet/dry at a point or along a line).
   Everything `SettlementSite` and `SettlementLayout` hold is geometry in the
   frame's planar coordinates — metres from the frame centre along its axes
   (flat to well under 0.1% at ≤10 km), north given by
   `SettlementGeometry.up_direction()`:
   - points: gates, landings, crossings, anchor candidates, villages, wells;
   - polylines: shorelines (contours of the water masks), river and stream
     centrelines with width, roads, streets, wall lines;
   - polygons: water bodies, river channels (centrelines buffered to bankfull
     width), the footprint, the buildable area, fields and garden ring,
     districts, blocks, lots, building footprints.
   Where an algorithm is easier on a grid (claiming the footprint by cost
   distance, tracing a contour), it runs on the terrain raster internally and
   its result is traced and simplified to vector before it leaves the module.
   Needs a polygon library (clip, buffer, union, difference, area) — see open
   question in Status.
9. **No names.** A general-purpose name generator comes later; nothing here
   names streets, districts or landmarks.

## Interface assumptions (verify, flag if wrong)

Ordered by risk. **A1 is the gate for everything after Phase 0.** Checked
against the code 2026-10-10; notes in *italics* are the findings.

- **A1. `refine_fields` takes a single square tile.** It needs `positions
  (M, T, T, 3)`, a `weights` mask and spacing. `neighbours` (a
  `texel_neighbours` table) enables `connect_ocean`; `grid` enables small lakes.
  So `M = 1` with a square frame and neither should work, losing street-scale
  small lakes (fine: the local tier already has lakes down to ~10 km²) and
  ocean re-connection (check that a harbour frame doesn't strand sea pockets).
  *But the result has to be a `RegionFields`, which requires a `RegionHexGrid`
  (`grid`, `in_hex`, `n_hexes`), and every consumer downstream reads that grid:
  `region.corridor.raster_frame` / `sample_raster` (river and road routing),
  `snap_to_land`, `localmap.banks.covered_texels`, `city_footprints`
  (`grid.texture_local`). So build a one-cell grid —
  `SettlementGrid(RegionHexGrid)`, one centre at the frame centre, `radius =
  side / 2` (a hex grid's texture square has half-width `radius`), and
  `texture_in_hexagon` all True — rather than `grid=None`. Its
  `texel_neighbours` then gives `connect_ocean` for free.*
- **A2. `RegionSource(local.fields)` locates arbitrary points.** It finds the
  owning hex by nearest centre (KD-tree) over `LocalHexGrid`, and reads through
  the over-hang. Anchors become local texels (`hex·T·T + row·T + col`).
  *Two corrections. (a) It clamps silently: `_locate_nearest` clips row/col
  into the texture and ignores `ecef_to_local`'s `valid`, so the "fail loudly"
  check has to live in the settlement builder. (b) The local map's domain is
  the parent **regional hexagon** (`in_map`), not the local square:
  `LocalHexGrid` keeps only hexes overlapping the parent hex, and over-hang
  texels outside it carry fields but are unconserved, with no rivers, roads or
  footprints of their own — that ground belongs to the neighbouring local map,
  maybe in another region. A settlement may sit up to
  `refine_radius_parent_texels` (1.5 regional texels, ~7 km) from its region
  position, and the owning hex is the one nearest the region position, so a
  ~5 km half-width frame can cross the regional hex edge.* **Decided
  (2026-10-10): build the neighbouring local maps** and read each point from
  the map that owns it (Decision 1; Phase 2 builds them, Phase 3's source
  reads them). Costs up to 3× the local build at a hex corner.
- **A3. Relief params need a new scale.** Local defaults
  (`max_relief_km=0.6`, `slope_reference=0.03`) over 4–5 more octaves will put
  spurious cliffs in town. Start at `max_relief_km ≈ 0.03`, check slope
  distribution against the local slope, tune. *`LocalUpsampleParams`
  subclasses `RegionUpsampleParams` and inherits the fields it doesn't redeclare
  via `with_region_detail` (`SHARED_WITH_REGION`); follow the same pattern.*
- **A4. Ownership is reusable.** Region = `ecef_to_cell(site_position,
  antimeridian)`; owning hex = nearest `region.grid.centers` to
  `region_position` (an inline `argmin` inside `local_owned`). Factor that into
  one helper (`owning_hex`) used by both `local_owned` and the chain, rather
  than re-deriving it. *Confirmed.*
- **A5. Corridor routing scales down.** `route_stream_paths(fields, globe,
  paths, lo, step, n, radius_nodes, params)` (`region/rivers.py`) and
  `route_polyline(fields, raster, verts, flows, radius_nodes, edge_cost,
  smoothing_iterations)` (`region/roads.py`) re-trace at ~15 m with a corridor
  of ~one local texel. Check smoothing iterations (local defaults: rivers 8,
  roads 6) and that a river already at bankfull width (~200–400 m for city
  rivers) is routed as a channel area, not a hairline — the width is
  `localmap.banks.river_channels` (`channel_width_km` × discharge scale), and
  `covered_texels` the area. *Both routers go through `region.corridor`
  rasters, so they need A1's grid.*
- **A6. Winds** are the `winds` feature: per season, `(centroids (N, 3),
  vectors (N, 3) m/s)`, float32. Prevailing = annual mean at the settlement's
  face. *Confirmed.*
- **A7. Trade wealth.** *`TradeResult.goods_flow` is not saved on the world.
  What is: `trade_attribution` (the `(S, S)` `pair_flow` exporter → importer
  matrix, cities only) and the `trade_flow` metadata channel
  (`realmgen.channels.TRADE_FLOW`, per face — the polity stage's
  `trade_wealth`).* Cities: exports + imports from `trade_attribution` rows and
  columns, normalised by population. Towns: `trade_flow` at `face_id`,
  normalised by population. No world change needed.
- **A8. Polity relation.** *Saved: `control_class` channel
  (`CONTROL_CLASS`, `ControlClass.DIRECT / VASSAL / INDEPENDENT`) at city
  faces; `polity_id` / `sovereign_id` channels per face; `polities` feature
  (`Polity.capital_face`, `liege`, `autonomy`, `depth`); `border_tension`
  channel. Not saved: march status — `frontier_exposure` is computed inside
  `PolityStage` and only feeds the regrowth.* Use vassal `autonomy` +
  `border_tension` at the face as the march proxy; persisting the exposure
  would change the realm stage, so ask first. Towns have no control class:
  take the polity governing their face.
- **A9. Local refinement is order-dependent.** *`build_local_map` is
  "deterministic, given which settlements earlier local maps re-sited": local
  roads end on cities other maps have already moved, and towns are refined
  against those roads.* So a settlement's final position can depend on which
  neighbouring local maps were opened first. Phase 2 measures this rather than
  assuming it away. Building neighbour maps for a frame (A2) makes this more
  common: they re-site *their own* settlements as a side effect (as opening
  them in the viewer would). The subject itself is safe if its owning map is
  built first.

## Phases

### 0. Spike: validate A1–A3

`scripts/spike_settlement.py`, over Phase 3's field modules (built first in
`settlegen/`, since the spike needs them anyway: `geometry.py`, `grid.py`,
`source.py`, `upsample.py`, with tests). Build seed 4's six largest river
cities and three towns, plus settlements whose frame crosses a regional hex
edge: a square `refine_fields` at `texel_m = 15` from the owning local map,
and for the crossing case from the multi-map source; print `anchor_residuals` per local texel of every
contributing map (target: float noise, as local does), the elevation jump
across the map seam inside the frame (as `measure_seam_quality.py`), slope
histogram vs the local map's, and wall-clock. Also count how many frames
cross the parent regional hexagon (A2b) and whether opening neighbouring local
maps first moves any of them (A9).
**Stop and report** if conservation fails or the source needs more than a
small change.

### 1. Qt-free region map

Extract `RegionView._refine_features` and `_region_streams`
(`viz/mapview/regionview.py`) into `region/builder.py`:
`RegionMap(fields, rivers, roads, deltas, streams)` and
`build_region_map(globe, region_id, params, *, …world and realm tuning) →
RegionMap`, mirroring `build_local_map`. Same order: fields → rivers →
riparian wetland → refine cities → roads (`moved`) → deltas → refine towns;
streams built eagerly (or lazily behind a method, keyed on `RegionStreamParams`
as now). `RegionView` calls it. The extraction also takes
`RegionView._land_reach` (`(water_spread_texels + 2) × texel spacing`, from the
region's upsample params) and the delta snap reach; both read view state now.

Test: on seed 4, two regions, viewer-path and builder-path outputs identical
(rivers, roads, deltas, every settlement position) — a refactor, so
bit-identical.

### 2. The chain: settlement → local maps

`settlegen/chain.py`:

- `SettlementRef(kind: "city" | "town", index: int)`.
- `region_of(globe, ref)`, `owning_hex(region_fields, ref)` (A4).
- `ensure_local_maps(store, ref) → SettlementMaps` — through the shared
  `MapStore` (built only if not held), the region (Phase 1) and the owning local map first, so the settlement is
  `local_refined` at its final position; then sizes the frame from it
  (footprint + garden ring + margin), finds every regional hex the frame
  touches (A4's rule over points along the frame edge and inside), and builds
  those regions and local maps too. Owning map first in the list. Cache: LRU
  keyed `(region_id)` and `(region_id, hex_index)`, dropped when params change
  (as `RegionView`); sized for a corner frame (3 local maps, up to 3 regions).
- Mirror `RegionView`'s reset rules: realm/region/local params change →
  settlement positions reset as now; the chain never resets them itself.

Tests: a fresh world, ask for a city's map → its region and local map get
built once, it ends `local_refined`; asking again builds nothing; a city
near a regional hex edge gets the neighbour map(s), one across a region edge
gets the neighbour region; an interior one gets only its own map. Order of
requests: the local tier does **not** promise order independence (A9). Test
it; if positions move, the chain should fix the order (e.g. open the owning
local map before any neighbour that hasn't been opened yet) and document what
the viewer path can still differ by.

### 3. Frame and fields

`settlegen/geometry.py`: `SettlementGeometry(local_geometry, center, side)`
— a `RegionGeometry` subclass like `LocalGeometry`: frame `[0, 1]²` mapped
affinely onto a square of side `side` centred on `center` in the local map's
`(s, t)`; `local_to_ecef` / `ecef_to_local`, handedness, `up_direction`,
`local_to_lonlat` carry over.

`settlegen/grid.py`: `SettlementGrid(RegionHexGrid)`, the one-cell grid
of A1.

`settlegen/source.py`: `LocalMapsSource(ParentSource)` over the chain's
local maps: `locate` assigns each point to the map that owns it (Decision 1)
and delegates to that map's `RegionSource`; anchors are offset per map into
one concatenated anchor space (`n_anchors` = the sum), so `refine_fields`'
per-anchor conservation works unchanged. Over-hang past a map's parent
hexagon is re-owned to the nearest in-domain texel of the map owning that
ground, and with several maps every smooth read goes through the re-owned
stencil (Phase 0 findings in Status). With one map it is just that map's
`RegionSource`.

`settlegen/upsample.py`: `SettlementUpsampleParams` (A3 defaults,
`texel_m`), `build_settlement_fields(globe, local_maps, center, side_km,
params) → RegionFields` via `refine_terrain(LocalMapsSource(local_maps), …)` —
the relief half of `refine_fields` (split out 2026-10-10, region / local
output bit-identical), with `parent_anchors` = (map, local texel) in the
concatenated space, and `settlement_conservation_report` (as
`local_conservation_report`, over `anchor_residuals`, elevation only).
**Only the terrain is refined**: elevation, water masks and shoreline, coast
style, delta, and the small-lake flag following the refined lakes. Climate,
biome, wetland, resources, rural density and danger are one value for the
frame — the owning local map's at the settlement, broadcast so local
consumers read them unchanged (decided 2026-10-10: they don't vary at
settlement scale in any way the map uses).

### 4. Site features

`settlegen/site.py` → `SettlementSite`:

Every local feature below comes from all contributing maps, each clipped to
its own map's domain (`in_map`) and joined at the seams before re-tracing (the
neighbours re-trace the same regional rivers and roads, so their ends meet
within a corridor; the 15 m re-trace closes the gap).

- **Water**: shorelines traced from the refined water masks (polylines) and
  water bodies as polygons; rivers re-traced from the local maps' within ~one
  local texel (A5) as centrelines with `channel_width_km`, and their channels
  as polygons (centreline buffered by half the width — `localmap/banks.py`'s
  model, in vector); local streams too.
- **Roads**: each local road reaching the frame re-traced to the footprint
  edge → **gates** (position, bearing, flow); sea legs → **landings**.
- **Crossings**: road × river by segment intersection
  (`worldgen.hydrology.crossings.crossing_pairs`, great-circle segment pairs —
  not `localmap.banks.crossing_pairs`, which tests straight lines from origins
  to targets for the farmland) → bridge / ford candidates (bridge if channel
  width ≥ a param or the settlement is a city). The local roads already pay a
  ford cost to cross (`river_ford_strength`), so crossings are where they chose
  to.
- **Footprint**: a polygon (possibly several pieces, e.g. either side of a
  river). Claimed on the terrain raster under Decision 4 by the local
  `city_footprints` rule (distance × roughness) plus a slope penalty, never in
  a channel (cities span rivers), then traced and simplified. Conservation
  becomes: the polygon's area inside each local texel equals that texel's
  built-up share. Across a seam each local texel's share is that of its own
  map's footprint (every map claims the city around its drawn position, with
  its own jitter), so the total matches `area_km2` only to within the seam's
  ragged edge, not exactly.
- **Anchor candidates**, scored, not chosen: local high ground inside the
  footprint (citadel/keep), road convergence (market), waterfront on the
  harbour or river side (port/quay), sacred site (any — Phase 6 picks).
- **Hinterland**: villages (points), fields and garden ring (the local
  farmland texels outlined into polygons) from the local maps clipped to the
  frame (each to its own domain), for drawing and for the gate weights.
- **Buildable** polygon: the frame minus water bodies, river channels and
  ground steeper than a param (the slope contour of the terrain raster).

Test: on the Phase 0 cities, every gate lies on the footprint boundary and on
a re-traced road; every polygon valid (closed, no self-intersection);
footprint area within one fine texel of `area_km2` (frames
inside one hex; across a seam, within the seam tolerance); rivers and roads
continuous across the seam;
harbour/port settlements have ≥ 1 waterfront candidate on the shore they
were sited on.

### 5. Settlement profile

`settlegen/profile.py` → `SettlementProfile`, derived here from the world
and the local map, **not written back**:

| Field | From |
|---|---|
| `population`, `tier`, `kind` | record. Cities: `population_display_tier` (`Settlement.tier` is only a seeding label), `site_kind` flags (`SITE_KIND_*`). Towns: `Towns.kind` (`TOWN_*`) |
| `walls`, `wall_strength` | record (`has_walls`, `wall_strength` on both) |
| `polity_relation` | A8: `control_class` (direct / vassal / independent); capital flag (`Polity.capital_face == face_id`); march proxy |
| `wealth` | A7 |
| `industries` (ranked) | town kind; hinterland `mining_density` / `fishing_density`; `dominant_crop` (mills, breweries); resources in the frame |
| `wind` | A6 |
| `materials` | biome, forest density, ruggedness (stone / timber / brick / adobe) |
| `danger` | fields, at the settlement |
| `founding_age` | **stub**: seeded draw, skewed older for higher population and capitals. Marked for the history layer |
| `planned` | **stub**: seeded, weighted by kind (waypoint / march / colonial → grid-leaning) |

### 6. Layout (first cut — review before building)

`settlegen/layout.py` → `SettlementLayout`, reads `SettlementSite` +
`SettlementProfile` only. **Stop after Phase 5 and confirm the approach** —
the street generator is the biggest design choice here and has several
reasonable options (agent/growth from gates, tensor-field streets,
subdivision of blocks between arterials). A minimal first cut to test the
contract:

1. Anchors chosen from the candidates by profile (capital or march → citadel;
   port → quay; every city → market).
2. Old core = the footprint area implied by `founding_age` stub's earlier
   population; wall line around it from `wall_strength` (none / palisade /
   stone; towers by strength). Outer suburbs outside it.
3. Arterials: gate → market, shortest within the buildable polygon with slope
   cost (routed on the terrain raster, kept as polylines); bridges at the
   chosen crossings.
4. Districts: market, quay, citadel around their anchors; industry districts
   from `industries`, noxious trades (tanneries, dyers, butchers) downwind and
   downstream; elite near the citadel and upwind; poor outside the walls.
5. Minor streets and lots inside blocks (organic vs grid by `planned`) — blocks
   are the polygons between streets, lots subdivide them; wells, shrines,
   graveyards, small squares scattered by density.
6. No names (Decision 9).

### 7. Viewer

Debug mode only: in the local view, clicking a town or city opens its
settlement map (back button returns), built through the chain. The terrain
draws as a raster (hillshade, water); everything else is drawn from its
vectors: water bodies and channels, roads/gates, footprint, buildable area,
anchor candidates; layout once Phase 6 lands.
Hover: profile fields.

### 8. Tooling and docs

- `docs/settlement_maps.md` (frame, fields, site, profile, layout, scale), linked
  from `docs/local.md`.
- `scripts/debug_settlement.py [seed] [--city K | --town K | --largest]` —
  site layers + conservation report.
- `scripts/measure_settlements.py` — across seeds: build time, conservation,
  footprint error, gates per settlement, crossings, anchor counts by kind →
  `scripts/results/settlement_results.json`.

## Status

Plan checked against the code 2026-10-10 (A1–A9 findings above); A2 decided:
frames past the hex build the neighbouring local maps.

**Phase 0 done 2026-10-10** (`scripts/spike_settlement.py`, seed 4: 6 river
cities, 3 towns, 3 cities picked near a hex edge; 2 of the 12 frames cross a
local map's edge). Phase 3's field modules exist in `settlegen/` with tests.

- **A1 holds** with the one-cell `SettlementGrid`: conservation per local
  texel at float32 noise, 4 octaves. After the terrain-only change, elevation
  worst 2e-8 km and 0.1–0.3 s per frame at T = 312–604.
- **A2b, multi-map source works**, after two fixes in `LocalMapsSource`: a
  map's over-hang past its parent hexagon is re-owned to the *nearest in-domain*
  texel of the map that owns that ground (the texel under the point can be
  centred back across the seam, making a cycle of two unconserved over-hang
  texels), and `anchor_values` reads at the re-owned anchor. Seam step: 3.4 m
  at first, now 0.7–0.8 m against 0.2 m between other texels; what's left comes
  from the hexagon edge cutting local texels into slivers that each conserve
  their own value. Acceptable for now; revisit if it shows in hillshade.
- **A3 starting values look sane**: slopes 3–4× the local map's (p50
  0.011–0.028, p99 0.04–0.11 km/km), 17–66 m of land relief per frame, no
  cliffs. Not tuned.
- **A9**: opening the crossing frames' neighbour maps first moved neither
  settlement (0 m).
- Cost is the chain, not the frame: regions ~7 s, local maps ~4 s each,
  settlement fields ≤0.3 s.

**Phase 1 done 2026-10-10**: `region/builder.py` (`RegionMap`,
`build_region_map`, `refine_region_features`, `land_reach`); `RegionView`
builds through it. Seed 4, regions 28 and 16: rivers, roads, deltas, streams,
biome and every city and town position bit-identical to the viewer before the
change (25,620 arrays). Tests: `region/test/test_builder.py`, and
`test_the_view_builds_through_build_region_map` in the viewer tests.

**Phase 2 done 2026-10-10**: `localmap/store.py` `MapStore` — the region and
local maps built so far for one globe, their params and reset rules (moved out
of `RegionView`), LRU 3 regions / 6 local maps (~100 MB each). `MapWindow`'s two
views share one store, and the chain asks the same store, so **a region or
local map is built only if not already held**, whoever built it. Re-entering a
region in the viewer now reuses it instead of rebuilding.
`settlegen/chain.py`: `SettlementRef`, `ensure_local_maps(store, ref)` →
`SettlementMaps` (owning map first, frame centre / radius / side, what it had
built); `frame_keys` finds the maps a frame needs exactly with shapely (frame
square ∩ each regional hexagon clipped to its cell), from the hex grids alone,
so no region is built unless one of its hexes is needed.
`localmap.settlements.owning_hex` is the shared ownership rule (A4). Seed 4
spike through the chain: same maps and numbers as Phase 0, 23 maps built for
12 frames (region 28 once for its 4 cities), A9 still 0 m. Tests:
`localmap/test/test_store.py`, `settlegen/test/test_chain.py`.
`SettlementMapParams` on `AppParams` (Decision 6) waits for the viewer
(Phase 7); the chain takes `margin_km` for now.

**Phase 7, early cut 2026-10-10** (for evaluating the terrain), **in the main
view** (user decision, not debug): clicking a settlement wherever it is drawn
opens its settlement map — cities on the world, region and local maps, towns on
region and local maps, villages on local maps (and any of them on a settlement
map's own overlays) — built through the shared store (`settlegen/builder.py`
`build_settlement_map`); **Back** returns to the view it was opened from. A
click on a settlement wins over region / hex picking. **Villages** are now in
scope: `SettlementRef("village", i, map=(region, hex))`, the `i`-th village of
that local map (valid while its params don't change), framed like a town. `RegionView` has a settlement
mode (`set_settlement`): the refined terrain on the square frame in the active
layer, with the *owning local map's* rivers, roads, villages, towns and city
footprints over it (not yet re-traced — Phase 4). `AppParams.settlement`
(`SettlementMapParams.upsample`, advanced-only) rebuilds the map on screen.
Overlay markers and border lines are sized as on the owning local map at the
same view zoom (the true scale would draw them tens of pixels wide). Seed 4:
from the world map, a city 12.6 s (its region and local maps built); a town
from its region 6.4 s; a village from its local map 0.6 s.

**Phase 4a done 2026-10-10** (water): `settlegen/frame.py` (frame metres ↔
ECEF), `settlegen/contour.py` (marching squares on a smooth field, closed along
the frame and polygonized with shapely), `settlegen/water.py`: sea and lake
polygons traced at the water surface with the sign forced to the refined masks
(seed 4: area within 0.1% of the masks, all valid); the local maps' rivers and
streams clipped to each map's ground and chained across seams (by world river
id), re-traced with `build_local_rivers` one tier down (45 m raster, 12
smoothing passes — 15 m kept the router's 45° turns), channel polygons at
bankfull width (flat ends, rounded joints). ~0.1–0.2 s per site.
`settlegen/site.py` `SettlementSite` (frame, water), on `SettlementMap.site`;
the viewer draws the channels and shorelines on settlement maps (Rivers toggle).

**Seam slivers fixed upstream (Option C, 2026-10-10).** Neighbouring local maps
disagreed along their shared edge (seed 4: 7.5 m mean / 24.5 m max at a point),
which exact per-local-texel conservation forced into settlement terrain as a
line (max step 8–13 m). Cause: each local map conserved regional texels only
over its own hexagon, so the smooth corrections faded out differently at the
seam; regions likewise seeded their noise per region (cell edges: 7.3 m / 44 m).
Fix: local maps conserve over a halo 2 regional texels past the hexagon
(`LocalUpsampleParams.conservation_halo_parent_texels`, docs/local.md) and
regions seed their noise per world. Now local seams 0.4 / 5.4 m, region edges
0.2 / 10 m; settlement seam max steps 1.0 m (city 36, was 9.5) and 3.5 m (328,
was 12.8); one seam city (356) still 4.6 m max — residual per-map differences
(small lakes, likely). No measurable build-time change. Trade-off accepted:
regional texels cut by a local map's hexagon edge are exact over their whole
area, not the map's part.

**Phase 4b done 2026-10-10** (roads): `settlegen/roads.py` → `SiteRoads` on
`SettlementSite.roads` (lines, sea legs, landings, crossings, gates, and the
roads on the sphere for drawing). The viewer now draws the site's roads in
place of the owning local map's.
- **Roads**: the local maps' roads, each clipped to its own map's ground and
  chained across seams (undirected: either end may join), re-traced at 45 m within one
  local texel. At this scale a river is many raster nodes wide (seed 4 city 16:
  700 m), so the router treats it as a **channel**, not a line
  (`region.roads._road_cost` gained `bank_flow` / `channel_climb`, defaults
  unchanged): entering pays the world's ford cost, each step inside costs as
  much as climbing a 1 km/km slope (`channel_climb`; the street-scale relief's
  ruggedness otherwise sends roads down the flat bed), the along-river
  discount moves to a 60 m bank band, road ends inside a channel move to its
  nearest bank, and the corridor widens to clear the widest channel.
- **River and lake legs** (main's `RegionRoads.water_polylines`, from the
  world's `trade_route_layers`, merged 2026-10-10): chained across seams like
  roads, keeping their port pins, and re-routed over the site's still water and
  river channels (`region.roads._water_cost`).
- **Landings**: where a leg pinned at a port first reaches water — a quay on
  its sea, lake or river (`Landing.water`). A sea leg is still straight from a
  coast end on the region's ~9 km raster, so the road that reached it lands
  where it passes nearest the traced shore and is cut there. Two port hops
  sharing a coast point (chained city → coast → city) are split there into one road.
- **Crossings**: planar segment intersection in frame metres (shapely), not
  `crossing_pairs` — the frame is flat, so they're the same.
  A bridge if the channel is ≥ `bridge_width_m` (20 m) wide or the
  settlement is a city.
- **Gates**: where roads cross the footprint edge, facing out. Two routes
  re-traced onto one way share one gate and crossing, with their flows summed.
  **The footprint is provisional** until 4c: a disc of the local footprint's
  area, minus still water.
Seed 4, 15 frames: every gate on the footprint edge; quays at the river ports
(cities 16, 25, 3), the lake port (13) and sea ports (152, 27, 15); city 16's
road crosses its river once instead of running down the channel (5 false
bridges before). Water + roads ≤0.1 s per site. After the merge city 152 has
its roads. Seen upstream, not fixed: river/lake legs pinned at cities 7–37 km
from any regional river or coast (seed 4 cities 36, 220, 14 — refined off
their water, or the world's water face not drawn near) cross that land
overland; with no water in the frame they get no quay.

Next: Phase 4c (footprint, buildable area).

Polygon library for Decision 8: **shapely** (approved 2026-10-10; in
`requirements.txt`).
