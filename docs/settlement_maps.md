# Settlement Maps

Qt-free street-scale tier under the local maps ([local.md](local.md)): **one
city, town or village refined onto its own square frame** at ~15 m, from the
local map(s) the frame touches, as a local map is from its region. The site
(water, roads, footprint, anchors, fields) is held as vectors in frame metres;
a profile says what kind of place it is; a layout plans its walls, streets,
blocks and plots. Plan, decisions and dated status:
[plans/settlement_maps.md](plans/settlement_maps.md). Drawing:
[viz_regionview.md](viz_regionview.md) (`RegionView` settlement mode, main view).

Built on demand when a settlement is clicked, never part of the world build, and
never written back: the settlement map reads the world, region and local maps,
and changing its params only rebuilds the map on screen.

## Files
- `settlegen/builder.py` — `SettlementMap` (`chain`, `fields`, `site`, `profile`, `layout`), `build_settlement_map`
- `settlegen/chain.py` — `SettlementRef`, `SettlementMaps`, `ensure_local_maps`, `frame_keys`, `population`
- `settlegen/geometry.py`, `grid.py`, `frame.py` — `SettlementGeometry`, `SettlementGrid` (one cell), `Frame` (metres ↔ ECEF)
- `settlegen/source.py` — `LocalMapsSource`: several local maps as one parent
- `settlegen/upsample.py` — `SettlementUpsampleParams`, `build_settlement_fields`, `join_ocean_saddles`, `clear_thin_land`, `settlement_conservation_report`
- `settlegen/site.py` — `SettlementSite`, `build_site`
- `settlegen/contour.py` — marching squares on a smooth field → shapely polygons
- `settlegen/water.py` — `SettlementRiverParams`, `SiteWater`, `RiverLine`, `natural_channels`
- `settlegen/roads.py` — `SettlementRoadParams`, `SiteRoads`, `Gate`, `Landing`, `Crossing`
- `settlegen/footprint.py` — `SettlementFootprintParams`, `Footprint`, `base_density_per_ha`, `build_footprint`, `site_slope`
- `settlegen/anchors.py` — `SettlementAnchorParams`, `AnchorCandidates`, `Anchor`
- `settlegen/hinterland.py` — `Hinterland`, `Neighbour`
- `settlegen/profile.py` — `SettlementProfileParams`, `SettlementProfile`, `PolityRelation`, `Industry`, `settlement_key`
- `settlegen/layout.py` — `SettlementLayoutParams`, `SettlementLayout`, `Walls`, `Block`, `build_layout`, `describe_at`, `summary`
- `settlegen/village_layout.py` — villages' layouts (tofts along streets)
- `settlegen/docks.py` — quays and piers
- `settlegen/display.py` — the Elevation layer at street scale (`relief_rgba`)
- `settlegen/params.py` — `SettlementMapParams`: `upsample`, `rivers`, `roads`, `footprint`, `anchors`, `profile`, `layout` (`AppParams.settlement`, advanced only)

## Scale (defaults)

| | |
|---|---|
| Texel | 15 m (`texel_m`; local maps ~365 m: ~24×, 4–5 octaves of detail) |
| Frame | 2·(footprint radius + garden ring + 0.5 km margin): villages ~4–4.5 km, towns ~4.5–5 km, large cities ~6.5–9 km; 300²–600² texels |
| Density | 70/ha at 1,000 people × (pop/1,000)^0.2, ×1.3 walled, lognormal σ 0.3, ×(buildable share)^-0.5 ≤ 2, 10–450/ha |
| Time (`measure_settlements.py`, seeds 4/31/35) | after the chain: cities ~2.3 s (site 1.3, layout 0.7), towns and villages ~0.3 s; the chain adds ~3–7 s per region/local map not yet held |

## Pipeline (`settlegen/builder.py`)

`build_settlement_map(store, ref, params, timings=None) → SettlementMap`:

1. **chain** (`ensure_local_maps`, through the shared `localmap.store.MapStore`,
   which builds a region or local map only if it doesn't already hold it);
2. **terrain** (`build_settlement_fields`);
3. **site** (`build_site`): water → roads → footprint → gates → anchors → hinterland;
4. **profile** (`build_profile`);
5. **layout** (`build_layout`; villages `village_layout`).

`timings` (optional dict) gets each step's seconds. Seeded per (world, kind,
index[, map]) (`settlement_key`), so a settlement looks the same whichever maps
were opened first.

## The chain (`settlegen/chain.py`)

`SettlementRef(kind, index, map=None)`: a city or town by world index; a
village by its index on a local map (`map = (region, hex)`), valid while that
map's params don't change. `ensure_local_maps`:

1. the settlement's region (re-sites it there, if not yet);
2. its **owning local map**: the regional hex nearest its region position
   (`localmap.settlements.owning_hex`), which re-sites it locally; its position
   is final from here;
3. the **frame**: a square centred where that map draws the settlement, sized
   from the settlement tier's own footprint radius (its people at
   `base_density_per_ha`; the local map's footprint is only a preview);
4. every other local map whose ground the frame reaches (`frame_keys`: each
   regional hexagon, clipped to its region's cell, against the frame, with
   shapely), with their regions.

The chain never resets a settlement; the store's params do.

## Terrain (`settlegen/upsample.py`)

Only the relief is refined: `region.upsample.refine_terrain` one tier down, with
`LocalMapsSource` as parent (each point read from the local map whose regional
hexagon holds it; over-hang re-owned to the nearest in-domain texel of the
owning map). Elevation is **conserved per local texel** of every map the frame
reads (`settlement_conservation_report`: ~1e-10–1e-8 km); the water masks and
shoreline follow (`water_spread_texels` 24, about one local texel). Every other
field (climate, biome, soils, density, danger) is the owning map's value at the
settlement, broadcast over the frame: a town spans a few km.

Mask fixes (masks only; elevation stays conserved):
- **Diagonal sea saddles** (`join_ocean_saddles`): where two of a local map's
  sea texels touch only at a corner across two dry ones (a strait one texel wide
  running on diagonally), the sea wins: a `saddle_channel_m` (100 m) channel is
  cut between their centres. Interpolation would close it, and `connect_ocean`
  doesn't reopen water that runs off the frame's edge.
- **Thin land** (`clear_thin_land`): land narrower than `min_land_m` (75 m)
  between waters goes to the nearest water (on a delta flat the relief sits
  within centimetres of the sea and pokes through in one-texel ridges).

Seams: local maps conserve over a halo 2 regional texels past their hexagon
([local.md](local.md)), so neighbouring maps agree at the frame's seams to
~1 m. A ~1 m step between one local map's own hexes is accepted (faintly
visible in the exaggerated hillshade).

## Site (`settlegen/site.py`)

`SettlementSite`: `frame`, `water`, `footprint`, `density_per_ha`, `sparse`,
`buildable`, `roads`, `anchors`, `hinterland` — all shapely geometry or arrays
in frame metres.

### Water (`settlegen/water.py`)
- **Sea and lakes**: traced (`contour.py`) at the water surface, sign forced to
  the refined masks; within 1 m of the water the smoothed mask places the shore
  (no staircase on flats). Area within 0.1% of the masks.
- **Rivers and streams**: the local maps' lines, clipped to each map's ground and
  chained across seams (by world river id), re-traced with
  `localmap.rivers.build_local_rivers` on a 45 m raster.
  A line hooking back in at the frame's edge is cut; an end just inside heading
  out is carried past it (`_tidy_ends`). Then carved into the frame's terrain
  (`region.carve.carve_rivers`, conserved against the local maps;
  [region_features.md](region_features.md#valley-carving-regioncarvepy)): floor at
  least the channel's half-width (one texel), walls 4 texels, bed quantile 0.1.
  Everything after the water (roads, footprint, layout, slope) reads the carved terrain.
- **Channels** (`natural_channels`): bankfull width, the course smoothed over a
  width (`course_smoothing`), width wandering (lognormal σ 0.15) and banks
  shifting (0.1 × width) over 4 widths. Rivers ≥ 60 m get ~0.6 islands/km: long
  and narrow (0.3 × width broad, 7 × that long), blunt upstream, on the
  channel's middle, clear of the line's ends, sharp bends and **other lines**
  (a width plus half theirs: no island on a confluence); the channel widens by
  half the breadth round each. An end meeting still water, the frame's edge or
  another line runs on two widths, so the band is cut there, not stopped.
  `channel_polygons` (the plain outline) is the hinterland's crossing barrier.

### Roads (`settlegen/roads.py`)
- **Roads**: the local maps' roads, chained across seams, re-traced at 45 m. A
  river is a channel here, not a line: entering pays the world's ford cost, each
  step inside costs a 1 km/km climb (`channel_climb`), the along-river discount
  moves to a 60 m bank band.
- **Water legs** (river and lake trade): chained the same way, re-routed over the
  site's still water and channels, pinned at their ports.
- **Landings**: where a leg pinned at a port first reaches water (`Landing.water`:
  sea, lake, river). A sea leg's road lands where it passes nearest the traced
  shore and is cut there.
- **Crossings**: planar intersections with river centrelines; a bridge if the
  channel is ≥ `bridge_width_m` (20 m) or the settlement is a city, else a ford.
- **Gates**: where a road leaves the footprint for good. Routes re-traced onto one
  way share a gate and crossing, flows summed.

### Footprint (`settlegen/footprint.py`)
The settlement tier is the **authority for footprints** (the local map's coarse
one stays a preview there). `population / density` of dry ground claimed by
priority: nearness (the way there, each metre of water counting `crossing_cost`
3), a pull along the roads (ribbon growth) and to the waterfront, edge noise
(100 m, `edge_roughness` 0.5), a penalty on slope smoothed over 120 m. Never
water or a channel. Islands under `island_min_ha` (2 ha) are never built on;
larger ones are ground like any other. Detached pieces under 3% rejoin the body.
**Holes and notches** the claim encloses, and concavities narrower than
`compact` (0.15) × the radius, are built sparsely (`hole_density_mult` 0.3:
gardens, yards; `Footprint.sparse`, not drawn apart) and the dense claim gives up
their people. Traced by marching squares, the water cut out exactly.
**Buildable** = the frame less water, channels and slopes over 0.3 km/km.

Measured (18 cities, 9 towns, 9 villages): the dense area is 2–5% under
`population / density` (the sparse ground holds the rest; villages up to 8%:
tracing a few hundred texels rounds them off); cities 1–3 pieces (halves either
side of a river or inlet); sparse ~9% of a city.

### Anchors (`settlegen/anchors.py`)
Candidates, scored in [0, 1], `per_kind` (6) best per kind ≥ `separation_m`
(150 m) apart, not chosen: **citadel** (knolls by height above a 300 m blur and
steepness around; else the highest ground), **market** (along roads by the flow
within 75 m, nearness, flat ground), **waterfront** (footprint edge within 25 m
of water, by water kind and width, flatness, landings), **sacred** (central high
ground).

### Hinterland (`settlegen/hinterland.py`)
The settlement's own fields (a city's market-garden ring, a town's or village's
fields from its footprint's edge) and every neighbouring village's and town's in
the frame, by the local tier's farmland rules (`LocalParams.farmland`): gardens
first, then the cheapest (settlement, texel) pair; never water, a channel or the
footprint; villages farm their own bank of a river ≥ 50 m. Only farmers grow
fields (`Villages.farming_share`). On a ≤ 320² raster, traced to polygons. A
town's fields usually outgrow its frame: the hinterland is what lies inside it.

## Profile (`settlegen/profile.py`)

`SettlementProfile`, derived only (every realm input optional):
- **who**: `population`, `tier` (0–3 city display tiers, 4 town, 5 village),
  `kind` (towns/villages their own; cities `port` > `river` > `inland`), `site_kinds`;
- **walls** and `wall_strength` from the record;
- **polity** (`PolityRelation`): `control` (direct / vassal), `capital`
  (a polity's `capital_face`), `march` (`border_tension / 0.2`, clipped, ×
  `0.6 + 0.4 · autonomy`), `fort`;
- **wealth**: trade per head as a percentile among peers;
- **industries** (`Industry`, ranked, ≤ 8, ≥ 0.1): mining, fishing, shipping,
  hostelry, commerce, crop industries, timber, herding/tanning/wool, mineral
  workings and rare resources within 10 km;
- **wind** (annual mean, frame axes, with steadiness), **materials** (stone /
  timber / brick / adobe shares; stone from the world's `ruggedness` at the
  settlement, 0 at 0.25 to full at 0.65), **danger**;
- **history stubs**: `founding_age` (lognormal about 100 y × (pop/500)^0.35,
  ×1.5 for a capital) and `planned` (chance by kind), for a later history layer.

## Layout (`settlegen/layout.py`, `village_layout.py`, `docks.py`)

Block subdivision, planned **by tier** — a city, a town and a village are
different places, not one plan at three sizes. Reads only the site, profile and
slope, so it is tested on hand-built sites.

**Cities and towns**
1. **Anchors** from the candidates: a market always; a citadel for a capital,
   a march ≥ 0.3 or walls ≥ 0.6; a waterfront where it ships or fishes ≥ 0.5; a
   sacred site apart.
2. **Main streets**: least-cost on a 20 m grid (slope, water, off the footprint;
   the site's roads cheap — they hold the crossings — but no plan needs one). A
   town's runs through its market along the footprint's long axis; a city's run
   from the market to every gate and to the footprint's edge in the widest gaps
   (3–8 by size). Bridges where they cross channels.
3. **Market**: a town's main street widened (24 m); a city's plaza. Citadel and
   sacred precinct are open ground sized by people.
4. **Old core**: a wobbly blob about the market (and citadel) holding
   `core_share` of the footprint (more for walls, youth, towns). **Walls only
   when walled**: the core is first closed by `wall_closing` (0.35) × its radius
   (walls are dear: the line spans notches and goes round the citadel), dry
   ground it takes in beyond the footprint joining `SettlementLayout.footprint`
   as walled gardens; stone from strength 0.4 (towers every 70 m / strength),
   else palisade; open within 15 m of water; gates where main streets cross.
   Unwalled, blocks and plots just coarsen toward the edge.
5. **Docks** where a waterfront is chosen: a 12 m quay along the shore, piers
   every ~60 m (sea/lake 60 m × size; rivers ≥ 40 m wide, ≤ 0.2 × width), moles
   at landings.
6. **Blocks**: a town's main street lined with burgage plots (7 × 45 m, cross
   lanes every 140 m, a back lane); the rest cut recursively across its long
   axis by 5 m streets to ~0.35 ha (town ×1.5, outside the walls ×2, sparse ×4;
   lognormal σ 0.35); organic places jitter the cuts, `planned` ones cut on the
   longest main street's grid.
7. **Districts**: market, quay, elite (near the citadel, upwind), craft quarters
   from `industries` (noxious trades downwind, downstream, outside the walls),
   graveyards, squares; the rest houses, gardens on sparse ground. Towns keep to
   two trades, a churchyard and — only by a castle — an elite quarter.
8. **Plots**: strips along each block's long sides, 8 × 22 m (suburb × 1.5,
   gardens × 3); wells and shrines on the streets.

**Villages** (`village_layout.py`): households (`P / 5`) as tofts fronting a
street, sized from the footprint (300–3,000 m², 2.5:1). Forms: a fishing village
within 300 m of the sea or a lake is a **harbour** village — rows of streets
along the shore about its jetty; otherwise **street** (45%), **green** (30%) or
**cluster** (25%, 3–6 lanes). Arms stop at 300 m; short of room, back lanes a
plot-pair behind (arms meeting end to end offset as one line, stopping short of
other streets and the green, so lanes never cross). Church and churchyard at the
middle, a manor at ≥ 400 people (50%). Seed 536582844 map 50/115: 39 of 40
villages place ≥ 85% of households.

`describe_at` names what is under a point (hover); `summary` is the status line.
No names (Decision 9). Known: tofts crowd and overlap where village arms meet
(deferred); houses within plots aren't drawn.

## Tooling

- `scripts/debug_settlement.py [seed] [--city K | --town K | --village R:H:I | --largest] [--zoom Z] [--anchors] [--buildable]`
  — renders one settlement map with the viewer offscreen and prints the frame,
  timings, conservation, site, profile and layout. Writes
  `scripts/results/settlement_<seed>_<kind><index>.png`.
- `scripts/measure_settlements.py [seeds…|--random N] [--cities N --towns N --villages N] [--detail]`
  — the same numbers over each seed's largest cities, towns at population
  quantiles and villages of the largest city's local map, summarised per kind.
  Results appended to `scripts/results/settlement_results.json`.
- `scripts/spike_settlement.py` — the Phase 0 terrain spike (conservation,
  slopes, seam steps, `--order-check`).
- `scripts/_common.py`: `get_map_store(seed)` (a `MapStore` as the viewer sets
  it up) and `settlement_map_params(store)`.
