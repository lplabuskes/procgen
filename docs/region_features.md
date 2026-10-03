# Regional Rivers, Roads, Towns and Cities

Part of the region tier — [back to region.md](region.md). The world's line and
point features re-sited on a region's refined terrain
([region_fields.md](region_fields.md)). The local tier reuses all of it one level
down ([local.md](local.md), [local_settlements.md](local_settlements.md)).

## Files
- `region/corridor.py` — `Raster`, `raster_frame`, `sample_raster`, `corridor_reach`, `route_corridor`: the raster + least-cost corridor router the rivers and roads share
- `region/rivers.py` — `RegionRiverParams`, `RegionRivers`, `build_region_rivers`, `route_stream_paths`, `snap_deltas`, `river_length_km`
- `region/distributaries.py` — `DistributaryParams`, `add_distributaries`, `mouth_count`: the channels a delta river fans out into
- `region/streams.py` — `RegionStreamParams`, `build_region_streams`: the region's own streams below the world's network, for the local maps
- `region/roads.py` — `RegionRoadParams`, `RegionRoads`, `build_region_roads`, `route_polyline`, `move_points`
- `region/towns.py` — `RegionSettlementParams`, `refine_region_towns`, `refine_towns`, `tier_sites`, `REGION_TIER` / `LOCAL_TIER`
- `region/cities.py` — `refine_region_cities`, `refine_cities`

The view runs them in this order on entering a region (see
[viz_regionview.md](viz_regionview.md)): rivers → cities → roads (ending on the
re-sited cities) → deltas → towns (which read the roads).

## Corridor routing (`region/corridor.py`)

A world line feature is a fact at the global resolution; a region refines only
*where exactly* it runs. Both rivers and roads use one uniform raster over the
map's local `(s, t)` frame, at a multiple of the texel spacing and `margin` past
the cell (`raster_frame`), of which only the nodes within reach of some corridor
are ever sampled (`corridor_reach`, `sample_raster`: refined elevation / masks /
requested keys from the nearest hex texel through the over-hang, global fields
beyond the textures). `route_corridor(raster, s, t, radius_nodes, edge_cost, *,
sinks, mouth_radius)` is the least-cost 8-connected path through the nodes within
`radius_nodes` of a world polyline; given `sinks`, it ends on the cheapest sink
within `mouth_radius` of the line's end, stopping at the first sink it touches
there. The caller supplies the edge cost.

## Rivers (`region/rivers.py`)

`build_region_rivers(fields, globe, params) → RegionRivers` (`paths`,
`confluences`, `mouths` + `mouth_flows`), each path a `(positions, flows,
in_lake)` polyline in the world `"rivers"` feature's form. The rivers supply only
their valley-seeking edge cost. Qt-free and deterministic; ~0.2 s per region at
defaults (0.05 s on a mostly-ocean one).

**Source: the world's stream network.** The hydrology stage saves its
single-receiver drainage tree down to `HydrologyParams.stream_flow_threshold`
(the `"streams"` feature, a `StreamNetwork`; see
[hydrology.md](hydrology.md)). The world's rivers are its part above
`river_flow_threshold`; the rest are streams too small for the world map. The
region takes the network nodes on its raster and cuts them into main stems with
the global stage's own `main_stem_paths`, so rivers, streams and their
confluences are the world's — a region never invents a channel, and the town
stage (which favours stream texels) and the region agree on where streams are.
Stems shorter than `min_global_nodes` are dropped as stubs. How far down the
network goes is a *world* param, since towns are sited on it. A world without
`"streams"` gets no regional rivers.

The routing loop is `route_stream_paths(fields, globe, paths, …)`, which a
local map also runs over the region's own rivers (see [local.md](local.md)).

**Re-routing.** Each stem becomes the least-cost 8-connected path through a
*corridor* of `corridor_global_texels` (1 global texel) either side of its
world centreline — the positional uncertainty the global grid leaves. A node
costs `exp(valley_seeking · (anomaly − mean) / σ)`, the anomaly being refined
minus coarse elevation and mean / σ its mean and spread in the corridor, so the
path takes the refined valley floor on a plain as much as in the mountains but
can never change valley or basin. Stems are routed first (descending flow). A
path whose world mouth is on the raster ends on the cheapest sink near that
mouth — its re-routed stem, a lake, or the *refined* sea — found in a widened
disc of two corridors around the mouth, since the stem itself moved by up to a
corridor. A path that ends on another one (not on water) is recorded in
`confluences`. Flow is the network's tree accumulation (running max along the
stem, so monotonic).

Paths leaving the raster are cut at its edge.

**Deltas.** `mouths` are where paths end on the refined sea. `snap_deltas`
moves each world delta (a world river's mouth) to the largest-flow regional
mouth within reach — two river corridors plus twice the settlement snap reach,
since a re-routed stem can end that far from its world mouth — so a delta marker
sits on its drawn river; one with no mouth in reach stays.

Paths are Laplacian-smoothed (`smoothing_iterations`); a segment is flagged
`in_lake` (hidden) when both its ends are on refined water, with land runs of up
to 2 nodes between water treated as water so a lake crossing is not drawn as
dashes over the refined shoreline's speckle. Neighbouring regions route their
shared edge independently (each is shown alone, so no seam is visible).

### `RegionRiverParams`
`raster_texels` (1.0), `margin` (0.08), `corridor_global_texels` (1.0),
`valley_seeking` (2.0), `min_global_nodes` (3), `smoothing_iterations` (8),
`distributaries` (`DistributaryParams`, below).

## Delta distributaries (`region/distributaries.py`)

The world's delta stage ([delta.md](delta.md)) builds each delta as a plain of
global texels with one river carried across it. `build_region_rivers` ends with
`add_distributaries`, which fans that river out for every delta whose apex
(`"delta_fans"`) is on the region's raster. It needs the refined `delta`
texture, which the refinement copies down from the globe.

- **The stem.** The largest-flow re-routed path passing within two global
  texels of the apex and ending on the refined sea. Its flow at the apex sets
  everything below.
- **Mouths.** `mouth_count(flow)` = `base_count` (3) + `per_decade` (2) ×
  log10(flow / 1000), clipped to 1..`max_count` (8), the main stem's mouth
  included. The delta front (refined sea beside the refined plain, on a disc of
  1.5 fan radii around the apex) is split by bearing from the apex, within
  `max_angle_deg` (70°) of the axis, into that many sectors. The stem's mouth
  holds one sector, and each other sector gets the front node nearest its
  (jittered) middle bearing, unless that is within 3 raster nodes of a mouth
  already taken.
- **Junctions.** On each side of the stem, mouths are taken outward from it.
  Each branches off that side's outermost channel so far (the stem up to the
  last junction, then the last branch), at the node whose distance from the
  apex is nearest a random `junction_min`–`junction_max` (0.15–0.6) of the
  mouth's. It is never within 2.5 nodes of another junction or on the trunk's
  last 3 nodes. Outer branches thus leave further upstream and the fan nests.
- **Routing.** `route_corridor` from junction to mouth through a corridor
  `corridor` (0.3) × length wide around a guide bent sideways by up to `wander`
  (0.15) × length at its middle. Node cost is valley-seeking over the plain
  (`valley_seeking` 1.0), ×50 on water before the mouth, and × `1 + separation ·
  exp(−d / 2)` within `d` nodes of an existing channel (`separation` 3.0, waived
  within 3 nodes of the junction), so branches spread instead of doubling one
  another. A route not ending on the sea is dropped.
- **Flow.** Each branch carries `apex flow ×` (its mouth's share + the shares of
  the branches leaving it downstream). The stem keeps `main_share` (0.5) and the
  other mouths split the rest evenly. The stem's own path and flow are
  unchanged: branches are added as extra paths, each starting exactly on its
  parent's drawn line, and their mouths are added to `mouths` /
  `mouth_flows` (not to `confluences`).
- **Local maps** re-route each branch like any other regional river, so they
  need nothing of their own.

Deterministic: seeded by world seed, region and delta index. ~0.1 s per delta.

## Streams below the world's network (`region/streams.py`)

`build_region_streams(fields, globe, rivers, params) → list[RiverPath]` is the
region's own drainage network below the world's `stream_flow_threshold`
(~100 global texels, ~20,000 km² of catchment), the brooks a local map's
villages sit on. **The region view does not draw them**; the region builds
them once, when its first local map is opened, and every local map of the
region re-routes the same network ([local.md](local.md#rivers-localmapriverspy)),
so neighbouring local maps agree on which streams there are.

- **Routing.** On the rivers' raster (every node sampled), a priority-flood over
  the refined elevation from the outlets — refined sea and lakes, the nodes the
  region's re-routed rivers run through, the raster edge. Each node drains to the
  neighbour it was flooded from (its lowest), so depressions are filled over and
  every node reaches an outlet. numba-compiled, with a pure-Python fallback.
- **Flow** is each node's catchment at its annual precipitation over the world's
  mean, in the world's flow units (one global texel at mean rain), so a stream
  reads on the world's scale and is drawn at its physical width. It is capped at
  the world's stream threshold: more than that is a world stream the world
  routed elsewhere, and the region does not overrule it.
- **Channels** are nodes with at least `min_catchment_km2` of catchment, cut
  into main stems with `main_stem_paths`. A stem ends on the river, lake or
  coast it drains into (only its last node can be on a river, so streams never
  cross one) or is cut at the raster edge.
- Not conserved against anything: the world has no channels this small.

At 300 km² on seed 4: ~0.35 s per region, and a drainage density of ~0.05
km/km² against ~0.006 for the world's network (roughly 10× the channel
length); a dense-land local map gets ~30 streams (~1,800 km). Precipitation
weighting leaves dry land with few.

### `RegionStreamParams`
Held in `LocalParams.streams` (the **Local map** dialog), since only local maps
use them. `min_catchment_km2` (300), `min_nodes` (3), `raster_texels` (1.0),
`margin` (0.08).

## Roads (`region/roads.py`)

`build_region_roads(fields, globe, rivers, params, land, danger_scale,
land_reach, moved) → RegionRoads` (`segments` in the world `"trade_routes"`
feature's `(start, end, flow)` form, `polylines` with their per-edge
`polyline_flows`, `sea`, `undrawn`). ~0.2 s per region. Each chain is re-routed
by `route_polyline`, which a local map also runs over the region's own polylines
(see [local.md](local.md)).

The trade stage routes over the face graph, so a world route is a chain of
straight hops between face centroids (~225 km). Which places trade, over which
faces and how much are world facts; a region refines where the road runs.

- **Which hops.** Only those the map draws — flow ≥ the world median, the
  renderer's threshold — near the raster. The rest go to `undrawn`, straight,
  for town siting.
- **Chains.** A hop is land when both ends' faces are `land_passable` in the
  world's `cost_surfaces`. Land hops are chained at their junctions (`_chains`;
  loops are cut). A port hop (land → water face) is split at the first refined
  water along it: its land part joins the chain, the rest is a straight `sea`
  leg. The world only puts land↔water hops at settlements (ports) and river
  mouths (see [realm_population.md](realm_population.md#ports)), so the land part
  is the short road from the port town to its shore. A river face counts as land
  here, so trade carried on a river is drawn as a road, which the along-river
  discount keeps in the valley. Pure sea hops stay straight. Chains leaving the
  raster are cut at its edge.
- **Cost** — the world's land-cost model per raster node, in the same units (a
  face-width step of base cost 1 costs 1): base = `biome_cost_scale ·
  BIOME_TRAVERSAL_COST[biome] + forest_cost_strength · forest + danger_scale ·
  danger` (the realm's `movement_scale`), charged per face-width travelled;
  climbing costs `ruggedness_cost_per_km` per km of refined relief (the world's
  per-face relief term, made explicit); entering a regional river costs the
  world's ford penalty (`min(ford_strength · w, max_mult − 1)` × base, `w` =
  flow / `river_flow_reference` capped at 1) and travelling along one gets its
  corridor discount (× `max(1 − corridor_strength · w, min_mult)`). Rivers are
  rasterised 4-connected so a road cannot slip between diagonal river nodes.
  Refined water is impassable.
- **Routing** through a corridor `corridor_face_fraction` of a face spacing
  either side of the world chain (`route_corridor`), on a raster
  `raster_texels` regional texels apart. A chain end at a settlement lands
  exactly on its marker: a hop end at a city's world site is first moved to
  where the city now stands (`moved = (site positions, positions)`, matched
  within float32 round-off by `move_points`), then off refined water with
  `snap_to_land` (`land_reach` — the view's own reach; default six texels); the
  path is Laplacian-smoothed; each hop's flow is carried onto the stretch between
  the path points nearest its two ends. A stretch with no dry way through keeps
  the world line.

### `RegionRoadParams`
`raster_texels` (2.0), `margin` (0.08), `corridor_face_fraction` (0.35),
`smoothing_iterations` (6).

## Towns and cities

### Towns (`region/towns.py`)

`refine_region_towns(fields, rivers, globe, town_params, population_params,
river_threshold_factor, params, roads) → int` moves the world's towns, **in place
in the `towns` feature**, from the global texel the town stage chose to a spot on
the region's finer terrain, and flags them `refined`. The refined position is the
town's actual position from then on (`positions`, and `region_positions`) and is
saved with the world; `Towns.site_positions` keeps the stage's site.

- **Which towns.** Each town is refined by exactly one region — the one
  containing its *site* — and only once, so the result does not depend on which
  region is visited first or on neighbouring regions' overlap.
- **Where.** Among the dry in-hexagon regional texels of the town's own face
  within `refine_radius_global_texels` (1 global texel, ~15 km) of its site,
  scored with the stage's own `site_log_weight_and_kind`: the refined rural
  density; trade routes from the regional roads (re-routed polylines, sea legs
  and the undrawn world hops, within the stage's `route_radius_km`) — or the
  world's straight routes when no roads are given; minerals as the stage reads
  them (`town_site_context`); coast from the refined shoreline; river and
  confluence from the regional paths (streams included) and
  `RegionRivers.confluences`, within `river_threshold_factor` × the regional
  texel spacing (~6 km). Gumbel noise is seeded `(seed, town, 3)`.
- **Kind is kept**, and honoured: a river town (on a river or stream) lands on a
  drawn path, and a port on the refined coast, whenever one is in reach.
- **Spacing** to every other town (wherever it stands now) and city is kept, by
  the stage's radii. A town with no legal spot stays on its site.

`refine_towns(fields, rivers, globe, todo, tier, reach, …)` is the shared core;
the local tier calls it with `LOCAL_TIER`, refining from `region_positions` and
keeping spacing against the others' region positions. `tier_sites` builds the
per-texel site features (dry in-hex texels with their owning face, coast, river,
confluence, route, mine, rural density) both tiers and the cities read.

On seed 4 (regions 16/25/34/48), 73–100% of river towns end up on a drawn stream
against 30–46% at their sites; towns move a median ~10 km; no spacing is broken;
~0.15–0.4 s per region.

### Cities (`region/cities.py`)

`refine_region_cities(fields, rivers, globe, river_threshold_factor, params) →
int` refines cities the same way, first — before the roads are routed, so the
roads end on them, and before the towns, so the towns keep clear of where they
now stand. The refine stage placed each on a global texel and recorded what that
texel offered (`Settlement.site_kind`: `SITE_KIND_OCEAN_COAST`, `_LAKE_COAST`,
`_RIVER`, `_CONFLUENCE`; see
[realm_refine.md](realm_refine.md#city-siting-settlementspy)). The region
re-floods the shoreline and re-routes the rivers, so without this a harbour city
stands wherever its global texel's centre fell — on seed 4 a median 2.3 km and up
to 17 km from the refined coast, 70% more than 1 km off.

- **Which cities, where.** As for towns: the region containing the site, once;
  the dry in-hexagon texels of the city's own face within
  `refine_radius_global_texels` of its site.
- **Kind is kept.** A harbour city lands on the refined coast (ocean, or lake for
  a lake port), then among those on a drawn river or confluence if it had one;
  a river city on a drawn river or confluence — whenever one is in reach.
- **Score.** The refine stage's weights (`SITE_OCEAN` 3, `SITE_LAKE` 2,
  `SITE_CONFLUENCE` 2, `SITE_RIVER` 1.5, `SITE_FERTILITY` 1 on the refined
  fertility, `SITE_JITTER` 0.3 seeded `(seed, city, 11, tier)`), with the pull
  toward the face centre replaced by one back toward the site (`SITE_CENTRE` 0.5
  at the reach). Only features the site had earn their bonus, so an inland city
  does not wander onto a stream or a shore the region revealed.
- The result is `Settlement.position` and `region_position`, flagged `refined`;
  `site_position` keeps the stage's site. All are saved.

On seed 4, 75 of 77 harbour cities in 20 regions end up on the refined coast
(moving a median ~5 km). The other two have no coast in reach in their own face:
one's shoreline moved more than a global texel, the other's nearest shore is in
the neighbouring face.

### `RegionSettlementParams`
`refine_radius_global_texels` (1.0; towns and cities).
