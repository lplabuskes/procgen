# Regional Field Refinement

Part of the region tier — [back to region.md](region.md). How a region's
textures are made from the globe's (and, through the same code, a local map's
from its region's — see [local.md](local.md)).

## Files
- `region/upsample.py` — `RegionUpsampleParams`, `RegionFields`, `refine_fields`, `build_region_fields`, conservation steps, water flood, coast style (`CoastStyle`, `coast_style`) and shore reshaping (`reshape_to_target`), `conservation_report` / `anchor_residuals`
- `region/resources.py` — `RefineContext`, `refine_resources`, `refined_slope`
- `region/lakes.py` — `find_small_lakes`, `basin_lake_levels` (the tier's own small lakes)
- `region/drainage.py` — `priority_flood`, `accumulate` (shared with `region/streams.py`)
- `region/smooth.py` — `FaceFanField`
- `region/source.py` — the `ParentSource` the refinement reads ([region.md](region.md#parent-sources-regionsourcepy))

## Refinement (`region/upsample.py`)

`build_region_fields(globe, region_id, params, *, resources, fertility_exponent)
→ RegionFields`, seeded `world.seed XOR region_id` (regenerates identically). It
wraps the tier-agnostic `refine_fields(source, positions, weights, spacing_rad,
texel_area_km2, params, seed, …)` with a `GlobalSource`; the local tier runs the
same function over a region. Water masks also write back `filled_height` (lake
surface on lake texels, refined ground elsewhere) so the tier below can refine
the region's lakes.

Per field: **bilinear coarse read → detail → per-anchor correction.**

**Smooth conservation** (`conserve_smooth`; elevation, temperature,
precipitation). A per-anchor constant would leave a step at every parent-texel
edge (a bilinear read doesn't average back to its texels where the parent is
curved — obvious in hillshade one tier down). Instead:
- Per-anchor residuals are read back through the parent's bilinear stencil
  (`ParentSource.interpolate`) and added, over-relaxed (`_RELAXATION = 1.4`, ≤ 16
  passes) until the well-covered anchors' worst residual is 1% of the start
  (usually 7–8 passes) or a pass stops helping.
- Then the exact constant step (`conserve_additive` / `conserve_multiplicative`)
  closes the rest, plus parent-texel slivers cut off by the map rim.
- Stencil corners go through an *owner* map (anchor owning each parent texel's
  position), so over-hang reads its neighbour's residual and the correction is
  continuous across hex and face seams.
- **Invariant: every anchor owns its own position** (`ParentSource.owners`).
  Otherwise an anchor no stencil reads has no say over its mean; with ~10% such
  anchors (seam duplicates / over-hang) the passes diverged, leaving ~10 m rms
  (hundreds of m on seams) to the exact step, which drew every parent texel as a
  block. Owner-anchored: ~2× shrink per pass, exact step ~0.1 m.
- Precipitation runs in log space (stays positive).

**Elevation is the only field with a noise stream.** Everything else is a
physical consequence of the refined relief, or copied. Independent noise in a
second field scatters threshold-classified fields (`biome`) across cuts; one
shared relief signal puts moisture/temperature bands on the contours.

| field | detail | conserved |
|---|---|---|
| `elevation` | slope-proportional fBm — **the only noise source**; restyled/reshaped near water by coast smoothness | additively, smooth; shore edit keeps every parent mean |
| `temperature` | lapse rate on refined relief (sub-sea-level ground counts as sea level) | additively, smooth |
| `precipitation` | orographic: × `1 + orographic_gain · anomaly` (per km, clipped ±0.9) | multiplicatively, smooth |
| `biome` | reclassified from **area-averaged** refined climate | derived |
| `ocean_mask`, `lake_mask`, `filled_height` | re-flooded to the parent's water surface over refined relief, plus the tier's [small lakes](#small-lakes-regionlakespy) | surface exact; shoreline refined |
| `small_lake_mask` | the part of `lake_mask` below the globe's resolution (`SMALL_LAKE_KEY`), the parent's re-flooded + this tier's | no |
| `coast_smoothness` | bilinear copy of parent's (if any) | no — a style for the tier below |
| `delta` | bilinear copy of parent's delta plain ([delta.md](delta.md)), clipped [0, 1] | no — guides distributaries |
| `wetland` | parent's score ± sub-grid relief, stamped into `biome` at the tier's lower threshold; riparian marsh added once rivers are routed ([wetland.md](wetland.md)) | no — derived |
| `forest_density` | treeline on refined elevation | multiplicatively, over land |
| `fertility` | valleys richer, steep slopes poorer, refined forest penalty | multiplicatively, over land |
| `dominant_crop` | reclassified from refined fields (climate area-averaged) | derived |
| `rural_density` | each global texel's farmers over its refined land by `fertility ** γ`, fishers over its refined shore | **total exact** |
| `fishing_density` | fishers' part of `rural_density` | **total exact** |
| `mining_density` | miners' part, in camps by their deposit's workings | **total exact** |
| `danger` | per face: made continuous (`FaceFanField`), nudged by terrain | per face (smooth base); nudge per global texel |

The last five need the world's resource / town / danger stages
(`region/resources.py`, below).

**Why biome averages.** Refined temperature swings ~1 °C between adjacent ~4 km
texels (mean |Laplacian| 0.63 °C vs 0.068 coarse), fraying biome edges into
dither. `classify_textures` reads a `biome_smoothing`² box average (5 at region
defaults; 0/1 disables); the temperature/precipitation layers stay sharp. Seed 4
region 34: texels adjacent to a biome change 9.2% → 2.7% (2.1% no-refinement
reference).

**Water masks.** A water body's *surface* is a global fact (ocean at sea level,
lake at its Priority-Flood `filled_height`); only the shoreline moves.
- A texel is water when its **refined** elevation is below the surface of a body
  it connects to (`flood_to_level`: the level-constrained flood
  `compute_ocean_mask_texture` uses, seeded from the parent's mask, per hex
  texture).
- Water may cross low ground into ≤ `water_spread_texels` texels the parent
  called land (≈ one global texel) but never over a rise above its surface.
- Ocean and lake flood separately and can't take each other's texels (a walled
  coastal lake below sea level isn't swallowed).
- **Cut-off sea is channelled back** (`connect_ocean`, on with `connect_ocean`).
  Sub-grid rises can wall off part of the seeded sea — on coastal local maps
  most often (seeds 4/31/35: about half of coastal maps, pockets of 1 to ~1,350
  texels). Across the whole map (drawn texels linked over hexagon edges by
  `RegionHexGrid.texel_neighbours`), the open sea is the ocean touching the
  map's edge (else the largest body). A Dijkstra out from it gives every
  pocket its cheapest route — 1 per texel plus the cut to `channel_depth_km`
  below sea level per `channel_cut_km` — and a one-texel channel is cut along
  it, avoiding lakes. The cut is absorbed by raising the rest of the parent
  texel's dry ground (`reshape_to_target`, ≤ `channel_max_scale` ×), so
  conservation is exact; texels whose parent texel can't absorb it are routed
  around on the next round. Pockets with no route under `channel_max_cost` go
  dry (seeds 4/31/35: none). Over-hang texels keep their own flood. ~0.15 s
  per map.
- No `filled_height` (no hydrology) → lakes copied from the anchor.
  `refine_water_masks=False` → verbatim anchor copy.
- Seed 4 region 49: coast gains 0.14% / loses 0.17% of the region.

- The parent's small lakes and its other lakes flood apart too (and keep
  `small_lake_mask`), so a refined tier's lakes carry down a tier.

**Detail fades under water.** Lakes are shallow (median ~40 m) vs ~15–30 m of
sub-grid relief; zero-mean noise would poke lake beds into the air and dither the
mask into confetti. Amplitude is scaled by water depth over
`water_relief_depth_km` (0.1 km), keeping bumps smaller than the water above.
Deep ocean (median 3.4 km) keeps full detail (painted over anyway).
`debug_region.py` prints flooded/dried percentages.

**Detail band.** ≤ `octaves` (4) from the parent grid's Nyquist (`0.5 / parent
texel spacing`) up to `nyquist_safety / texel spacing`. A tier no finer than its
parent gets `detail_octaves == 0`. At defaults: **2 octaves** over ~4.5 km texels
(globe ~14 km).

**Detail amplitude.** `relief_factor × (detail wavelength, km) × effective
slope`, capped at `max_relief_km`; effective slope saturates at
`slope_reference` and floors at `slope_floor` of it (near water see
[Coast style](#coast-style)). The global field is smooth at ~15 km (mountains
only ~0.007 km/km), hence the small `slope_reference` (0.005). Typical: tens of
metres on plains, a few hundred in rough country.

**The critical invariant.** `conservation_report(fields, globe)` averages each
global texel's in-hexagon regional samples vs the global value — elevation,
temperature, precipitation, rural density over the whole domain; fertility and
forest over texels dry on both maps (`anchor_residuals`, reused by the local
tier). Residuals are float32 rounding (~1e-7 km) by construction (each field's
last step is that correction). Pinned by `region/test/test_upsample.py`.

## Small lakes (`region/lakes.py`)

The globe keeps lakes of `min_lake_pixels` (50) global texels, ~10,000 km², and
up. Each refined tier finds its own in its refined relief, after the water masks
and climate and before resources (`_add_small_lakes`; `small_lakes=False` →
none). They join `lake_mask` / `filled_height` and are marked in
`small_lake_mask`; resources, rural density and fishers then treat them as water
(conservation unchanged — all totals still exact), and the tier below re-shores
them with its own (below). No noise of its own.

- **Smoothed relief.** Everything below runs on the raster's relief
  Gaussian-smoothed over `lake_smoothing_parent_texels` (σ; water left out of
  the average, `smooth_land`). On raw texels every few-metre detail bump is a
  sill or an island (lacy shores, speckle islands), and a tier down the
  conservation staircase (metre steps at parent-texel edges) makes pits that
  fill whole parent texels (square lakes).

- **Basins.** Priority-flood on a raster at the tier's texel spacing (one hex
  radius past the map; nearest-centre texel, uncovered nodes at the parent's
  height) from the outlets: water already there, uncovered nodes, the raster
  edge and the parent's channels (`drains`: a region drains along the world's
  stream network, a local map along the region's rivers and streams) — a river
  that big would have cut any sill on its course — widened to
  `lake_drain_clearance_parent_texels` (1.25) each side. The tier re-routes
  each river anywhere within its corridor (1 parent texel each side of the
  parent's line), so a lake any closer sat tangent to the drawn river or was
  clipped by its edge (seed 4 regions 48/51/16/45/25: 63 of 363 lakes before,
  4 after, at 18% of the lake area; local maps: 172 of 535 before, 8 after).
  So rivers never pass through a tier's own small lakes, only through
  inherited ones.
- **Water balance.** Inflow = catchment runoff (Fu's Budyko curve, w = 2.6,
  from precipitation and the globe's PET formula, `HydrologyParams` defaults);
  a basin fills to its spill level when that covers PET off the full lake, else
  to the level where it does (as the globe's endorheic lakes). None where
  precipitation / PET < `lake_min_aridity_index` (0.2, UNEP "arid": playas)
  or on ice cap (warmest season < 0 °C); both judged per basin.
- **Breaching.** A lake must be deeper than the parent surface drops over
  `lake_breach_parent_texels` parent texels (parent slope × length) — erosion
  cuts sills fastest on steep ground, and slope-proportional detail digs its
  deepest pits there, so without this lakes crowd mountain flanks — and than
  `lake_min_depth_km`, and at least `lake_min_texels` big. In the cold the
  length shrinks by `lake_glacial_factor` (annual mean ≤ 0 °C, fading out by
  10 °C): glacial lake country.
- **Size cap.** With `lake_max_parent_texels` > 0, a lake bigger than that many
  parent texels is dropped (its basin stays dry): the parent could have drawn
  it and didn't. Off (0) on regions, whose lakes run up to the globe's
  `min_lake_pixels`; 1 on local maps, so they only add lakes below the region's
  resolution (~20 km²) rather than region-sized ones the region left out.
- **Texels.** Each node's surface is carried a node onto its shore (not onto
  outlets); a texel is lake where its smoothed height (bilinear) is below its
  nearest node's surface.
- **Relief edit.** The raw elevation is then dragged to agree — under the
  surface in the lake, above it on a band ~one parent texel onto the shore —
  by `reshape_to_target` with `two_sided=True`: an anchor the lake covers
  (nothing dry to raise) instead makes its lake bed shallower, never crossing
  the surface. Each anchor's mean is kept, so conservation stays exact. A
  texel no edit can move stays dry (seed 4 region 48: 7 of 1,436). The fields
  after (biome, wetland, resources) read the edited relief.
- **Inherited lakes.** The parent's small lakes (re-flooded at its surface by
  `_refine_water_masks`) are re-flooded again on the smoothed relief from the
  same seed, at the same surface and spread, and edited alike. They are water
  to this tier's basin search (no lake grows into one), smoothed over as land,
  and not drained by `drains` (the parent's river runs under them).

Region defaults (`lake_smoothing_parent_texels` 0.5, `lake_breach_parent_texels` 1,
`lake_glacial_factor` 0.3, `lake_min_depth_km` 0.001, `lake_min_texels` 2,
`lake_drain_clearance_parent_texels` 1.25). With the clearance the five regions
below hold 0.41% of land; the figures below predate it.
Smoothed basins are shallower, so the breach length and depth floor dropped
from 6 / 0.002 on raw relief. Seed 4's five lake-richest regions (48/51/16/45/25):
0.51% of land in 438 lakes with 1 island, vs 1.20% in 2,207 lakes with 33
islands on raw relief. Most of the old share was texel-scale pits; restoring
it (L 0.5) brings back ~800 speckle lakes. (Before smoothing, seed 4's six
most land-rich regions held 0.7% of land — taiga 11%, tundra 2.4% — not
re-measured since. Earth: lakes 40–10,000 km² ~0.5–1% of land, concentrated in
formerly glaciated country.) ~0.2 s per region. `debug_region.py` prints the
share.

Rivers and streams flow through small lakes rather than end in them
(`Raster.sinks` = sea + the globe's lakes): stream outlets, river mouth sinks
and confluences use it, and segments in small lakes are hidden (`in_lake`).
Harbours (`tier_sites.coast_lake`) stay on the globe's lakes.

## Coast style

The globe's `coast_smoothness` ([coast.md](coast.md)) gives each shore a style
from 1 (long sweeping) to 0 (crenellated). Without it, neither step applies. Both
act only in a **shore band**: a tent on the parent's bilinear water fraction, 1
where parent texels are mixed, 0 within `coast_band` (0.25) of all-land/all-water
(~one parent texel either side).

**Why styling noise isn't enough.** Parent-scale coasts are nearly flat (seed 4:
±40 m within three regional texels of shore), so exact per-parent-texel
conservation puts the refined coastline on parent texel boundaries — a staircase
of ~14 km steps in a region (4.5 km locally). The correction causes it (blurring
the coarse read beforehand does nothing); zero-mean-per-texel noise can only
fringe it by 1–3 texels. Hence two steps.

**1. Detail style (`coast_style`).** In the band, fBm persistence moves from
`persistence` toward `coast_persistence_smooth` (0.25) / `_rugged` (0.8) per
texel (`_varying_persistence_fbm`). Amplitude becomes `wander × detail wavelength
× parent slope`, `wander` from `coast_wander_smooth` (0.03) to `_rugged` (0.5), so
the shore moves ~that many wavelengths regardless of gradient (the
slope-saturated amplitude would swamp a flat coast). Over sea, the underwater
fade is lifted in the band so rugged coasts raise skerries; lakes keep it.

**2. Shore reshaping (`reshape_coast`, default on).** After elevation is
conserved, a *target* shore is drawn and elevation edited to match:

- **Target**: conserved elevation box-blurred twice over `coast_blur_texels`
  (1.5) parent texels (curves through the staircase), plus fine detail (elevation
  − blur) × (1 − smoothness). Rugged shores add "bays": fBm one octave below the
  detail band and the next (scales conservation removes), amplitude `target
  wander × 2 wavelengths × parent slope`, wander from
  `coast_target_wander_smooth` (0) to `_rugged` (0.5).
- **Edit** (`reshape_to_target`): each texel on the wrong side of the surface is
  set to the target clipped to `coast_edit_min_km`–`coast_edit_max_km` (1–20 m)
  past it. The parent-mean change is taken back from that parent texel's texels
  that can absorb it without crossing the surface (dry raised / wet lowered, in
  proportion to height above / depth below), so every parent mean is unchanged
  and no other texel changes side. Needing > `coast_max_scale` (1×) of that
  height, or nothing to absorb → left as is (keeps its step). Seed 4 region 49:
  ~⅔ of wanted edits applied.
- Two passes share the target: sea vs sea level (lakes excluded), then lakes vs
  their own surfaces (ocean excluded). A lake surface is carried ~two parent
  texels onto shore via the lowest neighbouring surface (as the later flood
  spreads it).
- A final `conserve_additive` clears float crumbs; water masks are then
  re-flooded as usual (`water_spread_texels` still bounds shore movement).

Effect: a ~1-parent-texel band around the parent coastline — smooth shores lose
the staircase and fringe, rugged ones gain coves/headlands/islands ~a parent
texel across. Invisible at full-region zoom, clear zoomed in and on local maps
(applied again there against the region). `reshape_coast=False` → styled detail
only. Pinned by `region/test/test_coast_style.py`, `localmap/test/test_coast_style.py`
(conservation, water invariants, smooth shores shorter than rugged).

## Resources, population and danger (`region/resources.py`)

No noise of their own: all sub-grid variation comes from refined relief (slope,
relief anomaly, treeline, refined shoreline), so layers line up with terrain. The
world's tuning is reused: `build_region_fields(..., resources=ResourcesParams,
fertility_exponent=γ)` (viewer passes `GlobeParams.resources` and
`PopulationParams.fertility_exponent`). `RefineContext` carries texels, anchors,
refined/coarse elevation and slope, land masks.

- **Refined slope** (`refined_slope`): gradient of refined elevation per hex
  texture / `slope_saturation` (sub-grid slopes ~0.003–0.035 km/km on seed 4).
- **Forest** = coarse × `(treeline(refined) + 0.05) / (treeline(coarse) + 0.05)`.
- **Fertility** = coarse × `clip(1 − fertility_valley_gain · anomaly, 0.5, 1.5)`
  × `(1 − fertility_slope_penalty · slope)` × the forest penalty's change.
- Forest and fertility are *land* densities, conserved over texels dry on both
  maps: conservation and the `[0, 1]` clip alternated (`conserve_unit`), then
  closed exactly by a headroom step (a short group moves each texel toward 1 ∝
  `1 − v`, an over one scales down) — needed where targets sit near 1. Residuals
  at float noise. Water keeps the parent's aquatic fertility; newly flooded
  texels take it from neighbouring water.
- **Dominant crop**: `classify_crops` over box-averaged climate, refined
  elevation and fertility.
- **Rural density** (all three parts conserved exactly — the design's hard
  constraint):
  - farmers (`rural_density − fishing_density`) over refined land by
    `fertility ** γ`;
  - fishers over refined land weighted by water share within `fishing_shore_km`
    (2 km; box of `round(km / texel_km)` texels) — following the refined shore;
    no shore left → with the farmers;
  - miners (`mining_density`) on refined land by a Gaussian of `mining_camp_km`
    (3 km) about the nearest worked deposit working (`worked_mine_points`;
    measured from each global texel's closest land so it can't underflow); no
    land → with the farmers.
  - Nobody on refined water unless the global texel has no dry ground; all-barren
    land spreads evenly. Read by town refinement
    ([region_features.md](region_features.md#towns-and-cities)).
- **Danger** (one value per face on the globe):
  - `FaceFanField`: piecewise linear over each face's fan triangles (centroid,
    vertex i, i+1), vertex values shared, each face's area mean an exact
    constraint. Non-negative with exact means ⇒ 0 on every 0-danger face incl.
    edges, so danger fades out *inside* carrying faces. Node values are the
    smoothest (least-squares over fan edges) solution: one sparse KKT solve
    (~0.08 s), repeated with negative nodes pinned to 0; cached for the last
    world. Uniform blocks → domes with raised rims; an isolated face → cone up to
    3× its value (layer clamps at "lethal").
  - Then × `exp(danger_terrain_gain · score)`, score = `biome_smoothing`-box
    (slope + forest) on land, 0 on water, renormalised per global texel (rugged
    forest more dangerous; each face's mean unchanged).
  - Local maps read the region's continuous danger and conserve per regional
    texel.

## `RegionUpsampleParams`

| param | default | |
|---|---|---|
| `n_cols`, `tex_size` | 14, 64 | grid size |
| `relief_factor`, `slope_reference`, `slope_floor`, `max_relief_km` | 1.0, 0.005, 0.10, 1.0 | detail amplitude |
| `octaves`, `lacunarity`, `persistence`, `nyquist_safety` | 4, 2.0, 0.5, 0.5 | detail band |
| `orographic_gain` | 0.5 /km | precipitation |
| `lapse_rate_c_per_km` | 6.5 | temperature |
| `water_relief_depth_km`, `refine_water_masks`, `water_spread_texels` | 0.1, True, 4 | water |
| `biome_smoothing` | 5 | biome / crop / danger-score box average |
| `coast_band` | 0.25 | coast band (parent water fraction) |
| `coast_persistence_smooth`, `coast_persistence_rugged`, `coast_wander_smooth`, `coast_wander_rugged` | 0.25, 0.8, 0.03, 0.5 | coast detail style |
| `reshape_coast`, `coast_blur_texels`, `coast_target_wander_smooth`, `coast_target_wander_rugged` | True, 1.5, 0.0, 0.5 | shore target |
| `coast_max_scale`, `coast_edit_min_km`, `coast_edit_max_km` | 1.0, 0.001, 0.02 | shore edit |
| `slope_saturation`, `fertility_valley_gain`, `fertility_slope_penalty`, `danger_terrain_gain` | 0.02, 2.0, 0.5, 1.0 | resources and danger |

`LocalUpsampleParams` subclasses it with local defaults
([local.md](local.md#localupsampleparams)).
