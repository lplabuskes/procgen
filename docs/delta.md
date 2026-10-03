# Delta Stage

River deltas: the low plains large rivers build at their mouths. The stage
reshapes the global relief, and four later stages read the result: fertility,
marsh (danger and land cost) and the apex as a city site. On region maps the
delta river fans out into distributaries (see
[region_features.md](region_features.md#delta-distributaries-regiondistributariespy)).

## Files
- `worldgen/delta/builder.py` — `DeltaParams`, `DeltaBuilder` (stage, display, persistence)
- `worldgen/delta/deposit.py` — `DeltaFan`, `TexelIndex`, `build_deltas`, `face_fractions`: the deposition itself
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
`min_length_km` (500) and its mouth texel is ocean. Deltas are built largest
first, and a river whose mouth already lies on a bigger delta gets none. That
gives about 15–35 per world at defaults.

## Deposition (`deposit.py`)

The footprint is a set of global texels grown by a **priority flood** from the
river's lower course:

- **Apex.** The head of the delta lies on the river `apex_distance · √area`
  upstream of the mouth. The fan's **axis** runs from the apex to the mouth.
- **Area budget.** `area_km2 · (flow / 1000) ** area_exponent` (1500 km², 0.7).
  That is ~1.5k km² at the threshold, ~7.5k at flow 10k and ~37k at 100k, about
  Ganges–Mekong scale for the biggest rivers. The flood stops when the budget is
  spent.
- **Priority.** Distance from the apex × `1 + lateral_cost · (1 − cos θ)` (θ off
  the axis) × `land_cost` on land, or × `1 + depth / depth_scale_km` at sea. The
  fan therefore narrows over land and bulges out over a shallow shelf: at
  defaults about 70% of a footprint is built land, the rest flattened lowland. A
  steep coast spreads the delta along its narrow shelf instead.
- **Never taken:** land above `max_land_elevation_km` (0.12), sea deeper than
  `max_depth_km` (0.4), lakes, texels behind the apex, and texels already in
  another delta.
- **Surface.** Every footprint texel becomes a plain falling linearly from
  `apex_height_km` (20 m) at the apex to `front_height_km` (2 m) at the farthest
  texel. Sea texels turn to land (`ocean_mask`, `filled_height`). The open sea
  within `skirt_rings` (2) texels of the plain is shoaled to at most
  `skirt_depth_km · ring / (rings + 1)` deep (the prodelta). It never rises
  above sea level.
- **Seams.** The flood runs over texels *inside* their face polygons (texel area
  = face area ÷ in-polygon count, shared-edge duplicates counted once). Every
  texel near the delta, over-hang included, then takes the role of the
  in-polygon texel nearest it, so neighbouring faces agree.

**Rivers across the plain.** A river that ended on the old shore now ends
inland, so it is extended in place:

- the delta river along the plain to the open-sea texel at the fan's tip
  (furthest along the axis, least off it);
- any other river ending on the plain to the nearest open sea, or onto the delta
  river's new reach if that is nearer;
- in the saved `StreamNetwork`, the delta river's outlet node follows the
  river's new reach, and other outlets on the plain route like the other rivers.
  New nodes are appended and the old outlet re-linked, so nothing upstream
  changes.

Extensions are Laplacian-smoothed (4 iterations, ends pinned) and carry the
mouth flow.

## Outputs

| Output | Kind | What |
|---|---|---|
| `elevation`, `ocean_mask`, `filled_height` | textures (edited) | the plain, the new land, the prodelta |
| `is_ocean`, `elevation` | per-face meta (recomputed) | aggregates of the edited textures |
| `delta` | texture `(N, H, W, 2)` | ch 0: 1 on the plain; ch 1: marsh, `smoothstep(marsh_start, 1, s)` with `s` = apex→front fraction |
| `delta_fraction`, `marsh_fraction` | per-face meta | share of the face's in-polygon texels on the plain / mean marsh |
| `"delta_fans"` | feature | `List[DeltaFan]`: `apex`, `mouth` (new), `axis`, `radius_km`, `flow`, `area_km2` |
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
  conserved) and the region's distributaries are sited on it.

## Display (`--debug` only)

The **Delta** scheme (plain green, marsh teal, off-delta stone; a texture recipe,
so it also renders refined on region and local maps), the **River Deltas**
overlay (the new mouths, unchecked), and a hover line `Delta plain: 40% (marsh 10%)`.

## Persistence

Textures and per-face values go with the mesh. The fans are saved as
`delta_apex`, `delta_mouth`, `delta_axis` and `delta_scalars` (radius, flow,
area) and re-published as `"delta_fans"` / `"deltas"` on load. The extended
rivers and streams are saved by the hydrology stage, which re-flattens its
`River` objects at save time.
