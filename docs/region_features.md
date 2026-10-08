# Regional Rivers, Roads, Towns and Cities

Part of the region tier — [back to region.md](region.md). The world's line and
point features re-sited on a region's refined terrain
([region_fields.md](region_fields.md)). The local tier reuses all of it one level
down ([local.md](local.md), [local_settlements.md](local_settlements.md)).

## Files
- `region/corridor.py` — `Raster`, `raster_frame`, `sample_raster`, `corridor_reach`, `route_corridor`: shared raster + least-cost corridor router
- `region/rivers.py` — `RegionRiverParams`, `RegionRivers`, `build_region_rivers`, `route_stream_paths`, `snap_deltas`, `river_length_km`
- `region/distributaries.py` — `DistributaryParams`, `add_distributaries`, `mouth_count`
- `region/streams.py` — `RegionStreamParams`, `build_region_streams`: streams below the world's network, for local maps
- `region/roads.py` — `RegionRoadParams`, `RegionRoads`, `build_region_roads`, `route_polyline`, `move_points`
- `region/towns.py` — `RegionSettlementParams`, `refine_region_towns`, `refine_towns`, `tier_sites`, `REGION_TIER` / `LOCAL_TIER`
- `region/cities.py` — `refine_region_cities`, `refine_cities`

Order on entering a region ([viz_regionview.md](viz_regionview.md)): rivers →
cities → roads (ending on the re-sited cities) → deltas → towns (which read the
roads).

## Corridor routing (`region/corridor.py`)

A world line feature is fixed at global resolution; a region refines only *where
exactly* it runs.

- One uniform raster over the map's local `(s, t)` frame, at a multiple of texel
  spacing, `margin` past the cell (`raster_frame`).
- Only nodes within reach of some corridor are sampled (`corridor_reach`,
  `sample_raster`): refined elevation / masks / requested keys from the nearest
  hex texel through the over-hang, global fields beyond the textures.
- `route_corridor(raster, s, t, radius_nodes, edge_cost, *, sinks, mouth_radius)`:
  least-cost 8-connected path through nodes within `radius_nodes` of a world
  polyline. With `sinks`, it ends on the cheapest sink within `mouth_radius` of
  the line's end (first one touched). Caller supplies the edge cost.

## Rivers (`region/rivers.py`)

`build_region_rivers(fields, globe, params) → RegionRivers` (`paths`,
`confluences`, `mouths` + `mouth_flows`, `path_rivers`); each path a `(positions, flows,
in_lake)` polyline in the world `"rivers"` form. `path_rivers` holds, per path and
vertex, the index of the world river (`"rivers"`) it re-routes, -1 for a
sub-threshold headwater or a distributary: network nodes are tagged from
`"river_texels"` (`stream_river_ids`; a path's last node, its confluence, keeps
the path's id), each re-routed node takes its nearest source vertex's id
(`_run_ids`), and junction cuts realign them (`_meet_ids`). Each path is at most
one world river. Local maps carry the region's ids down the same way
(`localmap/rivers.py`). `RegionRivers.river_ids()` gives all -1 when unknown. Qt-free, deterministic; ~0.2 s
per region (0.05 s mostly-ocean).

**Source: the world's stream network.** Hydrology saves its single-receiver
drainage tree down to `HydrologyParams.stream_flow_threshold` (`"streams"`
feature, `StreamNetwork`; [hydrology.md](hydrology.md)); world rivers are its
part above `river_flow_threshold`. The region cuts the network nodes on its
raster into main stems with the global `main_stem_paths`, so rivers, streams and
confluences are the world's — a region never invents a channel, and the town
stage (which favours stream texels) agrees with it. Stems < `min_global_nodes`
are dropped. Network depth is a *world* param (towns are sited on it). No
`"streams"` → no regional rivers.

The routing loop `route_stream_paths(fields, globe, paths, …)` is also run by
local maps over the region's rivers ([local.md](local.md)).

**Re-routing.**
- Each stem: least-cost path through a corridor of `corridor_global_texels`
  either side of its world centreline (the global grid's positional
  uncertainty).
- Node cost `exp(valley_seeking · (anomaly − mean) / σ)`, anomaly = refined −
  coarse elevation, mean/σ over the corridor — finds the refined valley floor on
  plains and mountains alike, but can never change valley or basin.
- Stems routed in descending flow. A path whose world mouth is on the raster
  ends on the cheapest sink (its re-routed stem, a globe-sized lake, or the
  refined sea — small lakes are flowed through) in
  a disc of two corridors around the mouth (the stem moved up to a corridor). A
  path ending on another path is recorded in `confluences`.
- Flow = the network's tree accumulation (running max along the stem, so
  monotonic). Paths leaving the raster are cut at its edge.

**Keeping paths apart.** Corridors overlap, so two world stems a texel apart
would otherwise share one refined valley and braid.
- A later path pays `CROSS_COST` (1e4, `region/corridor.py`) on an earlier
  path's drawn nodes and on one corner of each of its diagonal steps
  (`diagonal_corners`, so nothing slips between), and × `1 + separation ·
  e^(−d/2)` within `d` nodes of one. Waived in its mouth disc for the nodes it
  may end on, and for a branch (`starts_on_path`) near its start.
- A source within two corridors of a path it is forced across starts past it
  (`_trim_crossed_head`); a path ending on a stem ends where it first comes
  beside that stem (`_join_at_first_touch`) — the stem's re-route can move
  across the tributary's course upstream of the world confluence.
- Smoothed independently, a junction can stop short of the line it meets or
  poke through it: `meet_line` snaps the end (or a branch's start) to the
  line's nearest vertex, then cuts at the crossing nearest it (planar test,
  `_segment_crossings`; the globe's `crossing_pairs` is ill-conditioned for
  sub-km segments).
- Left: a mid-course crossing where a stem's re-route cuts clean across another
  stem's corridor (a few per world).

**Deltas.** `mouths` = path ends on the refined sea. `snap_deltas` moves each
world delta to the largest-flow regional mouth within two river corridors + twice
the settlement snap reach, so the marker sits on its drawn river; none in reach
→ stays (as lake deltas do).

**Drawing.** Laplacian-smoothed (`smoothing_iterations`). A segment is `in_lake`
(hidden) when both ends are on refined water, with land runs ≤ 2 nodes between
water counted as water (no dashes over shoreline speckle). Neighbouring regions
route shared edges independently (shown alone, so no seam visible).

### `RegionRiverParams`
`raster_texels` (1.0), `margin` (0.08), `corridor_global_texels` (1.0),
`valley_seeking` (2.0), `min_global_nodes` (3), `smoothing_iterations` (8),
`separation` (3.0), `distributaries` (`DistributaryParams`).

## Delta distributaries (`region/distributaries.py`)

The world's delta stage ([delta.md](delta.md)) builds each delta as a plain with
one river across it. `build_region_rivers` ends with `add_distributaries`, which
fans that river out for every delta whose apex (`"delta_fans"`) is on the raster.
Needs the refined `delta` texture (copied down from the globe).

- **Stem.** Largest-flow re-routed path within two global texels of the apex,
  cut where it first meets open refined water (3 water nodes, or water to its
  end — a lake delta's river runs on through its lake). Its apex flow sets
  everything below.
- **Mouth count.** `mouth_count(flow)` = `base_count` (3) + `per_decade` (2) ×
  log10(flow / 1000), clipped to 1..`max_count` (8), stem included.
- **Mouth sites.** The delta front (refined water beside the refined plain,
  within 1.5 fan radii of the apex) is split by bearing within `max_angle_deg`
  (70°) of the axis into that many sectors. Stem's mouth holds one; each other
  sector takes the front node nearest its jittered middle bearing, unless within
  3 nodes of a taken mouth.
- **By shape** ([delta.md](delta.md#shape)). **Cuspate**: at most `wave_mouths`
  (2). **Bird's-foot**: no sectors — each world finger tip (`fan.fingers`) gets
  its nearest front node (same 3-node gap). **Arcuate**: sectors.
- **Junctions.** Per side, mouths taken outward; each branches off that side's
  outermost channel so far, at the node whose apex distance is nearest a random
  `junction_min`–`junction_max` (0.15–0.6) of the mouth's; never within 2.5
  nodes of another junction or on the trunk's last 3 nodes → nested fan.
  Bird's-foot aims at `birdfoot_junction_min`–`max` (0.75–1.0) of the apex's
  distance to the old shore (`fan.head`), splitting just behind the fingers.
- **Routing.** `route_corridor` junction → mouth, corridor `corridor` (0.3) ×
  length wide around a guide bent sideways up to `wander` (0.15) × length.
  Cost: valley-seeking (`valley_seeking` 1.0), ×50 on water before the mouth,
  × `1 + separation · exp(−d / 2)` within `d` nodes of an existing channel
  (`separation` 3.0, waived within 3 nodes of the junction), × `CROSS_COST` on
  any other channel's nodes. Routes not ending on water, or crossing another
  channel on land, are dropped. Each branch's start is put on its parent with
  `meet_line`, cut against the whole stem and the branches before it.
- **Flow.** Branch = apex flow × (its mouth's share + shares of branches leaving
  it downstream). Stem keeps `main_share` (0.5); other mouths split the rest
  evenly. The stem's path and flow are unchanged; branches are extra paths
  starting exactly on the parent's line, added to `mouths` / `mouth_flows` (not
  `confluences`).
- Local maps re-route branches like any regional river.

Deterministic (seed, region, delta index). ~0.1 s per delta.

## Streams below the world's network (`region/streams.py`)

`build_region_streams(fields, globe, rivers, params) → list[RiverPath]`: the
region's own drainage below the world's `stream_flow_threshold` (~100 global
texels, ~20,000 km² catchment) — the brooks villages sit on. **Not drawn by the
region view**; built once when the region's first local map opens, and every
local map re-routes the same network
([local.md](local.md#rivers-localmapriverspy)), so neighbours agree.

- **Routing.** Priority-flood over refined elevation on the rivers' raster (all
  nodes) from outlets: refined sea and the globe's lakes, nodes on re-routed
  rivers, raster edge. Small lakes ([region_fields.md](region_fields.md#small-lakes-regionlakespy))
  aren't outlets: streams run through them (`in_lake` there).
  Each node drains to the neighbour it was flooded from → depressions filled,
  every node reaches an outlet. numba with pure-Python fallback.
- **Flow** = catchment × annual precip / world mean, in world flow units (one
  global texel at mean rain), so drawn at physical width. Capped at the world's
  stream threshold (anything more is a world stream routed elsewhere).
- **Channels** = nodes with ≥ `min_catchment_km2`, cut into main stems with
  `main_stem_paths`. A stem ends on the river/lake/coast it drains into (only
  its last node may be on a river, so streams never cross one) or the raster
  edge.
- Not conserved (the world has no channels this small).

Seed 4 at 300 km²: ~0.35 s per region; drainage density ~0.05 km/km² vs ~0.006
for the world network; a dense-land local map gets ~30 streams (~1,800 km). Dry
land gets few.

### `RegionStreamParams`
Held in `LocalParams.streams` (**Local map** dialog). `min_catchment_km2` (300),
`min_nodes` (3), `raster_texels` (1.0), `margin` (0.08).

## Roads (`region/roads.py`)

`build_region_roads(fields, globe, rivers, params, land, danger_scale,
land_reach, moved) → RegionRoads` (`segments` in the world `"trade_routes"`
`(start, end, flow)` form, `polylines` + per-edge `polyline_flows`, `sea`,
`undrawn`). ~0.2 s per region. `route_polyline` re-routes each chain (also used
by local maps).

World routes are chains of straight face-centroid hops (~225 km); which places
trade, over which faces, and how much are world facts.

- **Which hops.** Only drawn ones (flow ≥ world median, the renderer's
  threshold) near the raster. Others → `undrawn`, straight, for town siting.
- **Chains.** A hop is land when both ends' faces are `land_passable`
  (`cost_surfaces`); land hops are chained at junctions (`_chains`; loops cut).
  A port hop (land → water) is split at the first refined water: land part joins
  the chain, rest is a straight `sea` leg. The world only puts land↔water hops at
  ports and river mouths ([realm_population.md](realm_population.md#ports)), so
  the land part is the short port-to-shore road. River faces count as land (river
  trade drawn as a road, kept in the valley by the along-river discount). Pure
  sea hops stay straight. Chains cut at the raster edge.
- **Cost** — the world land-cost model per node, same units (a face-width step at
  base 1 costs 1):
  - base = `biome_cost_scale · BIOME_TRAVERSAL_COST[biome] + forest_cost_strength
    · forest + danger_scale · danger` (realm `movement_scale`), per face-width;
  - climbing: `ruggedness_cost_per_km` per km of refined relief;
  - entering a regional river: ford penalty `min(ford_strength · w, max_mult − 1)`
    × base, `w` = flow / `river_flow_reference` ≤ 1; along one: × `max(1 −
    corridor_strength · w, min_mult)`. Rivers rasterised 4-connected so roads
    can't slip between diagonal nodes;
  - refined water impassable.
- **Routing** through a corridor `corridor_face_fraction` of a face spacing
  either side, raster `raster_texels` regional texels apart.
  - Chain ends at settlements land exactly on the marker: a city's world site is
    first moved to where the city now stands (`moved = (site positions,
    positions)`, matched within float32 round-off by `move_points`), then off
    refined water by `snap_to_land` (`land_reach`, the view's; default 6 texels).
  - Laplacian-smoothed; each hop's flow goes on the stretch between the path
    points nearest its ends. No dry way through → keeps the world line.

### `RegionRoadParams`
`raster_texels` (2.0), `margin` (0.08), `corridor_face_fraction` (0.35),
`smoothing_iterations` (6).

## Towns and cities

### Towns (`region/towns.py`)

`refine_region_towns(fields, rivers, globe, town_params, population_params,
river_threshold_factor, params, roads) → int` moves world towns **in place in the
`towns` feature** from the town stage's global texel to a spot on refined
terrain, flagging them `refined`. The refined position is authoritative from then
on (`positions`, `region_positions`) and saved; `Towns.site_positions` keeps the
stage's site.

- **Which.** Each town refined once, by the region containing its *site* —
  order-independent.
- **Where.** Dry in-hexagon regional texels of the town's own face within
  `refine_radius_global_texels` (~15 km) of its site, scored with the stage's
  `site_log_weight_and_kind` on: refined rural density; routes from regional
  roads (polylines, sea legs, undrawn hops, within `route_radius_km`; world
  straight routes if no roads); minerals (`town_site_context`); refined coast;
  river/confluence from regional paths (streams included) and
  `RegionRivers.confluences` within `river_threshold_factor` × regional texel
  spacing (~6 km). Gumbel noise seeded `(seed, town, 3)`.
- **Kind kept**: a river town lands on a drawn path, a port on the refined coast,
  whenever one is in reach.
- **Spacing** to every other town (current position) and city kept, by the
  stage's radii. No legal spot → stays on site.

`refine_towns(fields, rivers, globe, todo, tier, reach, …)` is the shared core;
the local tier calls it with `LOCAL_TIER` (from `region_positions`, spacing vs
others' region positions). `tier_sites` builds per-texel site features (dry
in-hex texels + owning face, coast, river, confluence, route, mine, rural
density) for both tiers and cities.

Seed 4 (regions 16/25/34/48): 73–100% of river towns on a drawn stream (vs
30–46% at sites); median move ~10 km; no spacing broken; ~0.15–0.4 s per region.

### Cities (`region/cities.py`)

`refine_region_cities(fields, rivers, globe, river_threshold_factor, params) →
int` — same idea, run first (before roads, so they end on cities; before towns,
so towns keep clear). The refine stage recorded each city's texel offer
(`Settlement.site_kind`: `SITE_KIND_OCEAN_COAST`, `_LAKE_COAST`, `_RIVER`,
`_CONFLUENCE`; [realm_refine.md](realm_refine.md#city-siting-settlementspy)).
Without this, harbour cities sat a median 2.3 km (up to 17 km) from the refined
coast on seed 4.

- **Which, where.** As towns: once, by the region containing the site; dry
  in-hex texels of its own face within `refine_radius_global_texels`.
- **Kind kept.** Harbour → refined coast (ocean, or a globe lake — not a small
  lake — for a lake port), then
  preferring a drawn river/confluence if it had one; river city → drawn
  river/confluence — whenever in reach.
- **Score.** Refine-stage weights (`SITE_OCEAN` 3, `SITE_LAKE` 2,
  `SITE_CONFLUENCE` 2, `SITE_RIVER` 1.5, `SITE_FERTILITY` 1 on refined fertility,
  `SITE_JITTER` 0.3 seeded `(seed, city, 11, tier)`), with the face-centre pull
  replaced by a pull back to the site (`SITE_CENTRE` 0.5 at the reach). Only
  features the site had earn bonuses (inland cities don't wander onto revealed
  streams/shores).
- Writes `Settlement.position` and `region_position`, flagged `refined`;
  `site_position` keeps the site. All saved.

Seed 4: 75 of 77 harbour cities in 20 regions land on the refined coast (median
move ~5 km); the other two have no coast in reach in their own face.

### `RegionSettlementParams`
`refine_radius_global_texels` (1.0; towns and cities).
