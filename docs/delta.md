# Delta Stage

River deltas: the low plains large rivers build where they meet the sea or a
lake. The stage reshapes the global relief, takes one of three shapes (bird's-foot, arcuate,
cuspate) from the balance of river and wave power, and four later stages read the result: fertility,
marsh (danger and land cost) and the apex as a city site. On region maps the
delta river fans out into distributaries (see
[region_features.md](region_features.md#delta-distributaries-regiondistributariespy)).

## Files
- `worldgen/delta/builder.py` — `DeltaParams`, `DeltaBuilder` (stage, display, persistence)
- `worldgen/delta/deposit.py` — `DeltaFan`, `TexelIndex`, `build_deltas`, `face_fractions`: the deposition itself
- `worldgen/delta/sediment.py` — `sediment_load`, `mouth_node`: the silt lakes trap upstream
- `worldgen/delta/shape.py` — `WaveField` (wave power), `wave_dominance`, `DeltaShape`, `make_shape`, `shape_metric`: which shape and how it bends the flood
- `worldgen/delta/channels.py` — `DELTA` (texture), `DELTA_FRACTION` / `MARSH_FRACTION` (per face), `DELTA_FANS` / `DELTA_MOUTHS` (features)
- `worldgen/delta/display.py` — `DELTA_RECIPE`, `delta_color`, `delta_legend_entries`

## Place in the pipeline

`… → hydrology → delta → climate lake feedback → coast → …`. It runs after
hydrology, which finds the rivers. It runs before the lake-feedback pass, which
re-derives precipitation, humidity and biome from the edited elevation, so new
land gets its own climate. On a world without lakes that pass does nothing, and
the new land keeps the precipitation and biome its sea texels had. Temperature is unchanged: sub-sea-level ground counts
as sea level for the lapse rate, and the plain is at most tens of metres high.

## Which rivers

A river gets a delta when its mouth flow ≥ `min_flow` (1000), its length ≥
`min_length_km` (500) and its mouth texel is ocean. It also gets one wherever
it **enters a lake** (its `in_lake` segments start) with that flow and length
so far. That covers rivers that end in the lake (endorheic sinks, tributaries
meeting their stem inside it) and rivers that run on through it. A river
through a chain of lakes can thus get several deltas, plus one at sea.

**Sediment.** A delta is built from silt, not water, and a lake on the way
keeps some of it (`sediment.py`):
- **Load.** The load is accumulated down the saved `StreamNetwork` like the flow,
  in the same units. Each node adds its own runoff (its flow less its inflow).
- **Lake outlets.** A lake traps `lake_trap_efficiency` (0.2) of the load
  reaching it. Across its outlet (a lake node draining to one off the lake),
  the rest passes.
- **Full lakes spill.** A lake can only hold so much. When one of its deltas
  is cut short by `max_lake_share`, the share of its area budget left unbuilt
  is silt the lake could not hold. That fraction of the delta's arriving load
  **spills**: it passes the outlet untrapped, on top of the usual
  `1 − lake_trap_efficiency` of the rest. A delta that finds its lake already
  full spills its whole load. Spills from several deltas add up, but never
  more passes than arrives.
- **Supply.** A mouth's **sediment share** is the load ÷ the flow at its
  stream node: the biggest node on the mouth texel (the first lake texel for a
  lake mouth, so a lake delta gets the whole load arriving there). Its
  **supply** is the flow × that share.
- **What the supply sets.** It replaces flow in the `min_flow` threshold, the
  area budget, the finger count and the river side of the wave dominance. A
  river draining a big lake builds a small delta, or none (the St Lawrence),
  until fresh tributaries load it up again.
- **Order.** A delta below a lake needs that lake's spill settled first, so
  **lake deltas are built before sea deltas**. Lakes are taken upstream first,
  ranked by the drainage-order position of their last stream node, which
  places every lake after all the lakes above it. Within a lake, and among sea
  deltas, the largest river goes first. The load is re-derived after each
  spill.
- **Bookkeeping.** A lake mouth is tracked by its entry segment rather than a
  vertex index, because a delta further up the same river may have rebuilt the
  polyline before it. A mouth whose entry now lies under a delta upstream is
  skipped.

Without a stream network every share is 1.

A mouth that already lies on a bigger delta gets none. At defaults
(`lake_trap_efficiency` 0.2), seeds 4/31/35 get 29/15/23 sea and 41/42/29 lake
deltas. Lakes spill 4/10/2 times.

**Lakes.** Lakes are the connected pieces of the `lake_mask` (labelled over
in-polygon texels; over-hang texels join the nearest). A lake delta works like a
sea one, with these differences:
- Heights and depths are measured from the lake's surface (the median
  `filled_height` over its texels) instead of sea level.
- The flood may take only open water of *its own* lake.
- All the deltas in one lake together may fill at most `max_lake_share` (0.25)
  of its area. The flood stops at the first lake texel that would pass that,
  as it does at the area budget, so the plain keeps its usual mix of built
  land and lowland. At defaults lakes lose 0.4–1.4% of their area.

## Deposition (`deposit.py`)

The footprint is a set of global texels grown by a **priority flood** from the
river's lower course:

- **Apex.** The head of the delta lies on the river `apex_distance · √area`
  upstream of the mouth, but never further back than the river's last exit
  from a lake. The fan's **axis** runs from the apex to the mouth.
- **Area budget.** `area_km2 · (supply / 1000) ** area_exponent` (1500 km², 0.7).
  That is ~1.5k km² at the threshold, ~7.5k at flow 10k and ~37k at 100k, about
  Ganges–Mekong scale for the biggest rivers. The flood stops when the budget is
  spent.
- **Priority.** Distance from the apex × `1 + lateral_cost · (1 − cos θ)` (θ off
  the axis) × `land_cost` on land, or × `1 + depth / depth_scale_km` at sea. The
  fan therefore narrows over land and bulges out over a shallow shelf: at
  defaults about 70% of a footprint is built land, the rest flattened lowland. A
  steep coast spreads the delta along its narrow shelf instead.
- **Shape.** The distance in the priority is the blend of three shape metrics
  described [below](#shape). The arcuate one is the distance above.
- **Never taken:** land above `max_land_elevation_km` (0.12), water deeper than
  `max_depth_km` (0.4), other water (lakes for a sea delta; the sea and other
  lakes for a lake delta), texels behind the apex, and texels already in
  another delta.
- **Surface.** Every footprint texel becomes a plain falling linearly from
  `apex_height_km` (20 m) at the apex to `front_height_km` (2 m) at the farthest
  texel, above the water's surface. Water texels turn to land (`ocean_mask` /
  `lake_mask`, `filled_height`). The open water within `skirt_rings` (2) texels
  of the plain is shoaled to at most `skirt_depth_km · ring / (rings + 1)` below
  the surface (the prodelta). It never rises above the surface.
- **Seams.** The flood runs over texels *inside* their face polygons (texel area
  = face area ÷ in-polygon count, shared-edge duplicates counted once). Every
  texel near the delta, over-hang included, then takes the role of the
  in-polygon texel nearest it, so neighbouring faces agree.

**Rivers across the plain.** A river that ended on the old shore now ends
inland, so it is extended in place:

- the delta river along the plain to the open-water texel at the fan's tip
  (furthest along the axis, least off it). A river that ran on through a lake
  is cut at its old mouth, carried across, and then rejoins its old course at
  the first vertex past the plain. The rejoining reach runs through the lake
  and stays hidden (`in_lake`). Its old vertices on the plain are dropped;
- any other river ending on the plain to the nearest open water, or onto the
  delta river's new reach if that is nearer;
- in the saved `StreamNetwork`, the delta river's own node at its old mouth (the
  nearest big node on the plain) follows the river's new reach. That reach then
  drains into the first node of its old course past the plain, or stops at the
  front. Its old course over the plain is **dropped**: left in place it would
  keep its large flow and out-rank the new reach as the main stem on region
  maps. Streams that ended on the plain, or drained into that old course, route
  like the other rivers. New nodes are appended and the old node re-linked, so
  nothing upstream changes.

Extensions are Laplacian-smoothed (4 iterations, ends pinned) and carry the
mouth flow.

## Shape

`shape.py` follows Galloway's triangle without tides. A delta's plan is set by
the river's sediment against the waves that rework it.

| Shape | When | Plan | Earth |
|---|---|---|---|
| bird's-foot | river-dominated: calm or sheltered water, big river | a few long fingers out to sea, bays between them | Mississippi |
| arcuate | balanced | a rounded fan | Nile, Niger |
| cuspate | wave-dominated: exposed coast, onshore winds | a blunt cusp at the mouth, beaches running along shore | São Francisco, Tiber, Ebro |

**Wave power.** `WaveField.power(mouth, axis)` takes 13 bearings within 90° of
the river's seaward heading at its mouth. Each bearing marches a ray over
ocean-majority faces (½-face steps, starting ¾ face out) until it meets land:
the **fetch**, capped at `fetch_cap_km` (1500). In a lake the ray instead
marches over that lake's own texels (one-texel steps, starting a texel out),
and no swell counts. Lakes are small and sheltered, so most lake deltas
come out bird's-foot. The bearing contributes
`cos φ · (mean onshore wind³ + swell_speed³) · fetch / cap`, where the onshore
wind is each season's wind at the ray's sea faces, resolved toward the mouth.
The result is the cos-weighted mean over `(7 m/s)³`. Winds come from the
climate's `"winds"` feature; without them only swell (`swell_speed` 3 m/s) counts.

**Dominance.** `wave_dominance = W / (W + river_weight · (supply / 1000) ** river_exponent)`
(0.08, 0.5): 0 is all river, 1 all wave. Below `birdfoot_below` (0.3) the
delta is bird's-foot, above `cuspate_above` (0.7) cuspate, and in between
arcuate, with the weights smoothstepped over `shape_blend` (0.1). At defaults,
seeds 4/31/35 give 11 bird's-foot, 33 arcuate and 25 cuspate sea deltas
(16/48/36%), and 88 / 21 / 9 lake deltas (75/18/8%).

**Metrics** (km, then × the land/depth multiplier as before). They are blended
by the weights and measured from the **head**, the river's last vertex on land
(the old shore):

- **arcuate:** distance from the apex × `1 + lateral_cost · (1 − cos θ)`;
- **bird's-foot**, seaward of the head: `|apex→head| + along + finger_cost ·
  (max(perp − half, 0) + max(along − length, 0))` to the nearest **finger**.
  The fingers are rays from the head:
  - `finger_base` (2) + `finger_per_decade` (1) per tenfold flow of them, at
    most `finger_max` (5);
  - spread evenly over ±`finger_spread_deg` (45°) with seeded jitter;
  - the one nearest the axis `finger_length` (1.4) × √area long, the rest up to
    `finger_jitter` (0.5) shorter;
  - `half` = `finger_half_width` (0.4) texels, `finger_cost` 8.

  Landward of the head the arcuate metric applies;
- **cuspate**, seaward of the head: `|apex→head| + forward + cot(cusp_angle / 2) · lateral`
  (`cusp_angle_deg` 140). Its level sets are two straight beaches meeting at a
  blunt point, and the sea along the shore is cheap, so the plain spreads along
  the coast.

The footprint's area budget is the same for every shape; only where the area
goes changes. Small deltas are only a few texels across, so their shapes barely
differ on the globe. The shape shows from about flow 10k up.

A cuspate front is sandy beach ridges, so its marsh is × `cuspate_marsh` (0.4,
blended by the cuspate weight). A bird's-foot's built **finger tips** (the
farthest plain texel along each finger, ≥ 1.5 texels out) are kept on the fan
for the region tier, which puts a distributary mouth at each. A cuspate delta
gets at most two mouths there (see
[region_features.md](region_features.md#delta-distributaries-regiondistributariespy)).

## Outputs

| Output | Kind | What |
|---|---|---|
| `elevation`, `ocean_mask`, `lake_mask`, `filled_height` | textures (edited) | the plain, the new land, the prodelta |
| `is_ocean`, `elevation` | per-face meta (recomputed) | aggregates of the edited textures |
| `delta` | texture `(N, H, W, 2)` | ch 0: 1 on the plain; ch 1: marsh, `smoothstep(marsh_start, 1, s)` with `s` = apex→front fraction |
| `delta_fraction`, `marsh_fraction` | per-face meta | share of the face's in-polygon texels on the plain / mean marsh |
| `"delta_fans"` | feature | `List[DeltaFan]`: `apex`, `mouth` (new), `axis`, `radius_km`, `flow` (water), `sediment` (share carrying silt), `area_km2`, `head` (old shore), `wave_dominance`, `shape` (`"birdfoot"` / `"arcuate"` / `"cuspate"`), `fingers` (`(F, 3)` built finger tips), `lake` (built into a lake) |
| `"deltas"` | feature | `(M, 3)` the new mouths (overlay; region/local snap them onto re-routed mouths) |
| `"river_objects"`, `"rivers"`, `"streams"` | features (extended in place) | see above |

Consumers:
- **Fertility** (`resources`): `apply_delta_fertility` raises the plain toward
  `delta_fertility` (0.9). The river waters and re-silts it, so only the
  growing-season temperature bell limits it, and the marsh is docked by
  `delta_marsh_penalty` (0.4). It only ever raises fertility, so it feeds
  `fertility_mean` and rural population.
- **Land cost** (`cost`): `× (1 + marsh_cost_strength · marsh_fraction)` (3.0).
- **Danger** (`danger`): the **Lizardfolk Fen** source, seeded on faces with
  `marsh_fraction ≥ 0.05`.
- **City sites** (realm scoring): the **delta apex** situational factor (1 at
  each apex face, `delta_weight` 0.2).
- **Region / local tiers**: the `delta` texture is copied down (bilinear, not
  conserved) and the region's distributaries are sited on it, ending in the
  sea or the lake.

## Display (`--debug` only)

The **Delta** scheme (plain green, marsh teal, off-delta stone; a texture recipe,
so it also renders refined on region and local maps), the **River Deltas**
overlay (the new mouths, unchecked), and a hover line
`Delta plain: 40% (marsh 10%) — bird's-foot, wave dominance 0.21` (shape from
the fan whose apex is nearest; `bird's-foot lake` for a lake delta; `, sediment 12% of flow` when lakes upstream trapped some).

## Persistence

Textures and per-face values go with the mesh. The fans are saved as
`delta_apex`, `delta_mouth`, `delta_axis`, `delta_head`, `delta_scalars`
(radius, flow, area, wave dominance, sediment share), `delta_shape` (code), `delta_lake` and the ragged
`delta_fingers` / `delta_finger_count` and re-published as `"delta_fans"` / `"deltas"` on load. The extended
rivers and streams are saved by the hydrology stage, which re-flattens its
`River` objects at save time.
