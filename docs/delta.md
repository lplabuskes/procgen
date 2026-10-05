# Delta Stage

River deltas: low plains large rivers build where they meet the sea or a lake.
The stage reshapes global relief into one of three shapes (bird's-foot, arcuate,
cuspate) by river vs wave power; fertility and the apex (city site) are read
downstream. Delta marshes are the wetland stage's ([wetland.md](wetland.md)): a low
flat plain beside a big river scores wet. On region maps the delta river fans into
distributaries
([region_features.md](region_features.md#delta-distributaries-regiondistributariespy)).

## Files
- `worldgen/delta/builder.py` — `DeltaParams`, `DeltaBuilder` (stage, display, persistence)
- `worldgen/delta/deposit.py` — `DeltaFan`, `TexelIndex`, `build_deltas`, `face_fractions`: deposition
- `worldgen/delta/sediment.py` — `sediment_load`, `mouth_node`: silt trapped by lakes upstream
- `worldgen/delta/shape.py` — `WaveField`, `wave_dominance`, `DeltaShape`, `make_shape`, `shape_metric`
- `worldgen/delta/channels.py` — `DELTA` (texture), `DELTA_FRACTION` (per face), `DELTA_FANS` / `DELTA_MOUTHS` (features)
- `worldgen/delta/display.py` — `DELTA_RECIPE`, `delta_color`, `delta_legend_entries`

## Place in the pipeline

`… → hydrology → delta → climate lake feedback → coast → …`. After hydrology
(finds rivers); before lake feedback, which re-derives precipitation, humidity
and biome from the edited elevation so new land gets its own climate (on a
lakeless world that pass is a no-op and new land keeps its former sea texels'
precipitation/biome). Temperature is unchanged (sub-sea-level counts as sea
level; the plain is ≤ tens of metres).

## Which rivers

A delta forms where mouth flow ≥ `min_flow` (1000), length ≥ `min_length_km`
(500) and the mouth texel is ocean — and also wherever a river **enters a lake**
(start of an `in_lake` run) with that flow/length so far (endorheic sinks,
tributaries meeting their stem in the lake, rivers running through). A river
through a lake chain can get several, plus one at sea. A mouth already on a
bigger delta gets none.

**Sediment** (`sediment.py`) — deltas are built from silt, and lakes keep some:
- **Load** accumulates down the saved `StreamNetwork` like flow, same units; each
  node adds its own runoff (flow − inflow).
- **Trapping.** A lake traps `lake_trap_efficiency` (0.2) of arriving load; the
  rest passes its outlet (a lake node draining off the lake).
- **Spill.** When a lake's deltas are cut short by `max_lake_share`, the unbuilt
  share of the area budget is silt the lake couldn't hold: that fraction of the
  delta's arriving load passes untrapped (plus the usual `1 − trap` of the rest).
  A delta finding its lake full spills everything. Spills add, capped at what
  arrives.
- **Supply.** Sediment share = load ÷ flow at the mouth's stream node (biggest
  node on the mouth texel; first lake texel for a lake mouth). Supply = flow ×
  share. Supply replaces flow in the `min_flow` threshold, area budget, finger
  count and river side of wave dominance — a river draining a big lake builds a
  small delta or none (St Lawrence).
- **Order.** Lake deltas before sea deltas (spill must be settled); lakes
  upstream-first by the drainage-order position of their last stream node;
  within a lake, and among sea deltas, largest river first. Load re-derived after
  each spill.
- Lake mouths are tracked by entry segment, not vertex index (an upstream delta
  may have rebuilt the polyline); a mouth whose entry is now under an upstream
  delta is skipped.
- No stream network → all shares 1.

Defaults, seeds 4/31/35: 29/15/23 sea and 41/42/29 lake deltas; lakes spill
4/10/2 times.

**Lakes** = connected pieces of `lake_mask` (labelled over in-polygon texels;
over-hang joins the nearest). Differences from sea deltas:
- heights/depths relative to the lake surface (median `filled_height`);
- flood takes only open water of *its own* lake;
- all deltas in a lake fill ≤ `max_lake_share` (0.25) of its area (flood stops
  there as at the area budget). Lakes lose 0.4–1.4% of area at defaults.

## Deposition (`deposit.py`)

Footprint = global texels grown by **priority flood** from the lower course:

- **Apex**: `apex_distance · √area` upstream of the mouth, never past the river's
  last lake exit. **Axis**: apex → mouth.
- **Area budget**: `area_km2 · (supply / 1000) ** area_exponent` (1500 km², 0.7)
  → ~1.5k km² at threshold, ~7.5k at 10k, ~37k at 100k (Ganges–Mekong).
- **Priority**: shape-blended distance ([below](#shape)) × `land_cost` on land,
  or × `1 + depth / depth_scale_km` at sea → narrows over land, bulges over a
  shallow shelf (~70% built land at defaults); a steep coast spreads it
  alongshore.
- **Never taken**: land above `max_land_elevation_km` (0.12), water deeper than
  `max_depth_km` (0.4), other water (lakes for sea deltas; sea and other lakes
  for lake deltas), texels behind the apex, texels in another delta.
- **Surface**: plain falling linearly from `apex_height_km` (20 m) to
  `front_height_km` (2 m) above the water surface; water texels become land
  (`ocean_mask` / `lake_mask`, `filled_height`). Open water within `skirt_rings`
  (2) is shoaled to ≤ `skirt_depth_km · ring / (rings + 1)` below the surface
  (prodelta), never above it.
- **Seams**: flood runs over in-polygon texels (texel area = face area ÷
  in-polygon count, shared-edge duplicates once); every nearby texel, over-hang
  included, then copies the nearest in-polygon texel so faces agree.

**Rivers across the plain** (extended in place):
- Delta river → along the plain to the fan-tip open-water texel (furthest along,
  least off the axis). A river running through a lake is cut at its old mouth,
  carried across, and rejoins its course at the first vertex past the plain
  (hidden, `in_lake`); old vertices on the plain dropped.
- Other rivers ending on the plain → nearest open water, or the delta river's
  new reach if nearer.
- `StreamNetwork`: the delta river's node at its old mouth follows the new reach,
  which drains into the first old-course node past the plain or stops at the
  front. The old course over the plain is **dropped** — left, its large flow
  would out-rank the new reach as main stem on region maps. New nodes appended,
  old node re-linked; nothing upstream changes.
- Extensions Laplacian-smoothed (4 iterations, ends pinned), carry mouth flow.

## Shape

Galloway's triangle without tides: river sediment vs wave reworking.

| Shape | When | Plan | Earth |
|---|---|---|---|
| bird's-foot | river-dominated: calm/sheltered, big river | few long fingers, bays between | Mississippi |
| arcuate | balanced | rounded fan | Nile, Niger |
| cuspate | wave-dominated: exposed, onshore winds | blunt cusp, beaches alongshore | São Francisco, Tiber, Ebro |

**Wave power** — `WaveField.power(mouth, axis)` over 13 bearings within 90° of
the seaward heading:
- Fetch: ray over ocean-majority faces (½-face steps from ¾ face out) until land,
  capped `fetch_cap_km` (1500). In a lake: over that lake's texels (one-texel
  steps), no swell — so most lake deltas are bird's-foot.
- Bearing contributes `cos φ · (mean onshore wind³ + swell_speed³) · fetch / cap`
  (onshore = each season's wind at the ray's sea faces resolved toward the
  mouth); result = cos-weighted mean over `(7 m/s)³`. Winds from climate's
  `"winds"`; absent → swell only (`swell_speed` 3 m/s).

**Dominance** `= W / (W + river_weight · (supply / 1000) ** river_exponent)`
(0.08, 0.5), 0 river … 1 wave. < `birdfoot_below` (0.3) bird's-foot, >
`cuspate_above` (0.7) cuspate, else arcuate; weights smoothstepped over
`shape_blend` (0.1). Seeds 4/31/35: sea 11/33/25 (16/48/36%), lake 88/21/9
(75/18/8%).

**Metrics** (km, then × land/depth multiplier), blended by weight, measured from
the **head** (river's last land vertex = old shore):
- **arcuate**: distance from apex × `1 + lateral_cost · (1 − cos θ)`;
- **bird's-foot** (seaward of head; arcuate landward): `|apex→head| + along +
  finger_cost · (max(perp − half, 0) + max(along − length, 0))` to the nearest
  finger. Fingers are rays from the head: `finger_base` (2) + `finger_per_decade`
  (1) per tenfold flow, ≤ `finger_max` (5); spread over ±`finger_spread_deg` (45°)
  with seeded jitter; nearest-axis one `finger_length` (1.4) × √area long, others
  up to `finger_jitter` (0.5) shorter; `half` = `finger_half_width` (0.4) texels,
  `finger_cost` 8;
- **cuspate** (seaward of head): `|apex→head| + forward + cot(cusp_angle / 2) ·
  lateral` (`cusp_angle_deg` 140) — two straight beaches meeting at a blunt
  point; cheap alongshore so the plain spreads along the coast.

Area budget is shape-independent. Shapes only show from ~flow 10k (small deltas
are a few texels). Bird's-foot **finger tips** (farthest plain texel per
finger, ≥ 1.5 texels out) are kept for the region tier's distributary mouths;
cuspate gets ≤ 2 mouths there.

## Outputs

| Output | Kind | What |
|---|---|---|
| `elevation`, `ocean_mask`, `lake_mask`, `filled_height` | textures (edited) | plain, new land, prodelta |
| `is_ocean`, `elevation` | per-face meta (recomputed) | aggregates of edited textures |
| `delta` | texture `(N, H, W, 1)` | 1 on plain |
| `delta_fraction` | per-face meta | in-polygon share on plain |
| `"delta_fans"` | feature | `List[DeltaFan]`: `apex`, `mouth` (new), `axis`, `radius_km`, `flow` (water), `sediment` (silt share), `area_km2`, `head` (old shore), `wave_dominance`, `shape` (`"birdfoot"`/`"arcuate"`/`"cuspate"`), `fingers` (`(F, 3)` tips), `lake` |
| `"deltas"` | feature | `(M, 3)` new mouths (overlay; region/local snap to re-routed mouths) |
| `"river_objects"`, `"rivers"`, `"streams"` | features (extended in place) | see above |

Consumers:
- **Fertility** (`resources`): `apply_delta_fertility` raises the plain toward
  `delta_fertility` (0.9), limited only by the growing-season temperature bell.
  Only raises → feeds `fertility_mean` and rural population (Wetland texels on it
  are then docked, [wetland.md](wetland.md)).
- **City sites** (realm scoring): **delta apex** factor (1 at apex faces,
  `delta_weight` 0.2).
- **Region/local**: `delta` texture copied down (bilinear, not conserved);
  distributaries sited on it.

## Display (`--debug` only)

**Delta** scheme (plain green, else stone; texture recipe, so refined
on region/local maps), **River Deltas** overlay (new mouths, unchecked), hover
`Delta plain: 40% — bird's-foot, wave dominance 0.21` (nearest apex's
fan; `bird's-foot lake` for lakes; `, sediment 12% of flow` when trapped
upstream).

## Persistence

Textures and per-face values go with the mesh. Fans saved as `delta_apex`,
`delta_mouth`, `delta_axis`, `delta_head`, `delta_scalars` (radius, flow, area,
wave dominance, sediment share), `delta_shape` (code), `delta_lake`, ragged
`delta_fingers` / `delta_finger_count`; re-published as `"delta_fans"` /
`"deltas"` on load. Extended rivers/streams are saved by hydrology (re-flattens
`River` objects at save time).
