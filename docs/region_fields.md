# Regional Field Refinement

Part of the region tier — [back to region.md](region.md). How a region's
textures are made from the globe's (and, through the same code, a local map's
from its region's — see [local.md](local.md)).

## Files
- `region/upsample.py` — `RegionUpsampleParams`, `RegionFields`, `refine_fields`, `build_region_fields`, the conservation steps, the water flood, the coast style (`CoastStyle`, `coast_style`) and shore reshaping (`reshape_to_target`), `conservation_report` / `anchor_residuals`
- `region/resources.py` — `RefineContext`, `refine_resources`, `refined_slope`
- `region/smooth.py` — `FaceFanField`
- `region/source.py` — the `ParentSource` the refinement reads (see [region.md](region.md#parent-sources-regionsourcepy))

## Refinement (`region/upsample.py`)

`build_region_fields(globe, region_id, params, *, resources, fertility_exponent)
→ RegionFields`. Seeded with `world.seed XOR region_id`, so a region always
regenerates identically.

The work is done by the tier-agnostic `refine_fields(source, positions, weights,
spacing_rad, texel_area_km2, params, seed, …)`, which reads the tier above only
through a `ParentSource`; `build_region_fields` wraps it with a `GlobalSource`.
The local tier runs the same function over a region. The water masks also write
back `filled_height` (the lake surface on lake texels, the refined ground
elsewhere, as on the globe) so the tier below can refine the region's lakes.

Per field: **bilinear coarse read → detail → per-anchor correction.**

The correction for elevation, temperature and precipitation is **smooth**
(`conserve_smooth`). A per-anchor constant would leave a step at every
parent-texel edge, because a bilinear read never averages back to its own
texels where the parent is curved. That step was obvious in hillshade one tier
down, since the region is far more curved than the globe. Instead the
per-anchor residuals are read back through the same bilinear stencil as the
parent (`ParentSource.interpolate`) and added. This runs for several
over-relaxed passes (`_RELAXATION = 1.4`, at most 16), until the well-covered
anchors' worst residual is 1% of where it started (usually 7–8 passes) or a pass
stops helping. Then the exact constant step (`conserve_additive` /
`conserve_multiplicative`) closes the last bit, plus the slivers of parent texels
a map's rim cuts off. The stencil's corners go through an *owner* map (the
anchor owning each parent texel's position), so a texture's over-hang reads its
neighbour's residual and the correction stays continuous across hex and face
seams. Precipitation runs in log space, so it stays positive.

**Elevation is the only field with a noise stream.** Everything else is a
physical consequence of the refined relief, or is copied verbatim. Independent
noise in a second field is what breaks the derived categorical ones: `biome` is a
threshold classifier, so uncorrelated jitter in its inputs scatters neighbouring
texels across a cut. Sharing one relief signal puts the moisture and temperature
bands on the contours instead.

| field | detail | conserved |
|---|---|---|
| `elevation` | slope-proportional fBm — **the only noise source**; restyled and reshaped near water by the globe's coast smoothness | additively, smooth; the shore edit keeps every parent mean |
| `temperature` | lapse rate on the refined relief (sub-sea-level ground counts as sea level, as on the globe) | additively, smooth |
| `precipitation` | orographic response: × `1 + orographic_gain · anomaly` (per km, clipped to ±0.9) | multiplicatively, smooth (stays positive) |
| `biome` | reclassified from the **area-averaged** refined climate | derived |
| `ocean_mask`, `lake_mask`, `filled_height` | re-flooded to the parent's water surface over the refined relief | surface exact; shoreline refined |
| `coast_smoothness` | bilinear copy of the parent's (when it has one) | not conserved — a style, read by the tier below |
| `forest_density` | treeline on the refined elevation | multiplicatively, over land |
| `fertility` | valley floors richer, steep slopes poorer, forest penalty on the refined forest | multiplicatively, over land |
| `dominant_crop` | reclassified from the refined fields (climate area-averaged, as for `biome`) | derived |
| `rural_density` | each global texel's people over its refined land by `fertility ** γ` | **total exact** |
| `danger` | per face on the globe: made continuous (`FaceFanField`), then nudged by terrain | per face (smooth base); nudge per global texel |

The last five need the world's resource / town / danger stages; a world without
them gets none (`region/resources.py`, see **Resources, population and danger**).

**Why biome averages.** The refined temperature swings ~1 °C between adjacent
~4 km texels (mean \|Laplacian\| 0.63 °C vs the coarse field's 0.068 °C), which is
enough to fray every biome edge into a dithered band. A biome is a property of a
neighbourhood rather than of a single texel, so `classify_textures` reads a
`biome_smoothing`×`biome_smoothing` box average (5 at the region's defaults) —
the `temperature` and `precipitation` layers themselves stay sharp. On seed 4
region 34 this takes texels adjacent to a biome change from 9.2% back to 2.7%
against a 2.1% no-refinement reference, while 1.2% of texels still change biome
because of the terrain. `biome_smoothing` 0 or 1 disables it.

**Water masks.** A water body's *surface* is a global fact — the ocean sits at
sea level, a lake at the fill height Priority-Flood gave it (`filled_height`) —
so the level is copied from the parent and only the shoreline moves. Each texel is
water when its **refined** elevation lies below the surface of a body it is
connected to (`flood_to_level`): the same level-constrained flood
`compute_ocean_mask_texture` runs one tier up, seeded from the parent's mask
instead of from deep-ocean faces, within each hex texture. Water may cross low
ground into up to `water_spread_texels` texels the parent called land (≈ one
global texel at the defaults) but never over a rise above its surface; ocean and
lake are flooded separately and neither may take the other's texels, so a coastal
lake below sea level is not swallowed by the sea it is walled off from. Without
`filled_height` (no hydrology) the lakes are copied from the anchor.
`refine_water_masks=False` restores the verbatim anchor copy. On seed 4 region
49 the coast gains 0.14% of the region and loses 0.17% — bays and inlets against
islands and spits.

**Why the detail fades out under water.** Lakes are *shallow* (median ~40 m of
water over the bed on a real world) against ~15–30 m of sub-grid relief, so
zero-mean noise pokes half a lake bed back into the air and the refined mask —
a threshold on the refined elevation — dithers the whole water body into
confetti rather than only rearranging its shore. Detail amplitude is therefore
scaled by the water depth over `water_relief_depth_km` (0.1 km), which keeps the
bump strictly smaller than the water above it. The deep ocean (median 3.4 km) is
far past the fade and keeps its full detail, and submerged relief is painted over
by the water anyway. `debug_region.py` prints the flooded/dried percentages.

**Detail band.** Octaves (at most `octaves`, 4) run from the parent grid's
Nyquist (`0.5 / parent texel spacing`) up to this grid's
(`nyquist_safety / texel spacing`). A tier no finer than its parent gets
`detail_octaves == 0` — no invented structure. At defaults (10,242 faces,
`tex_size=16` global; `n_cols=14`, `tex_size=64` regional) this is **2 octaves**
over ~4.5 km texels vs the globe's ~14 km.

**Detail amplitude.** `relief_factor × (detail wavelength, km) × effective slope`
(near water it is set differently; see [Coast style](#coast-style)),
capped at `max_relief_km`, where the effective slope saturates at
`slope_reference` and never falls below `slope_floor` of it. The global field is
smooth at ~15 km, where even mountains only reach ~0.007 km/km — hence the small
`slope_reference` (0.005) and the `relief_factor` (2) above 1. Typical result:
tens of metres on plains, a few hundred in rough country.

**The critical invariant.** `conservation_report(fields, globe)` averages every
global texel's in-hexagon regional samples and compares against the global value
— elevation, temperature, precipitation and rural density over the whole domain,
fertility and forest density over the texels dry on both maps
(`anchor_residuals`, which the local tier reuses). Residuals are at float32
rounding (~1e-7 km) by construction, because the last step of each field is
exactly that correction. `region/test/test_upsample.py` pins it.

## Coast style

The globe's `coast_smoothness` ([coast.md](coast.md)) gives each shore a
style, from 1 (long sweeping shores) to 0 (crenellated, rugged ones). A world
without it gets neither step below, and its detail is exactly as above. Both
steps apply only in a **shore band**: a tent on the parent's bilinear water
fraction, 1 where the parent's texels are mixed and fading to 0 within
`coast_band` (0.25) of all-land or all-water, about one parent texel either
side of the coast.

**Why styling the noise is not enough.** At the parent's scale coasts are
nearly flat: on seed 4, texels within three regional texels of the shore lie
only ~±40 m from sea level. Exact per-parent-texel conservation then puts the
refined coastline on the parent's texel boundaries, a staircase of ~14 km steps
in a region (4.5 km in a local map). The correction is what puts it there,
which is why blurring the coarse read before conservation does nothing. Noise
that is zero-mean within each parent texel can only fringe that staircase, by
1–3 texels. Hence two steps.

**1. Detail style (`coast_style`).** Within the band the fBm's persistence
moves from `persistence` toward `coast_persistence_smooth` (0.25) or
`coast_persistence_rugged` (0.8), per texel (`_varying_persistence_fbm`). The
amplitude is replaced by `wander × detail wavelength × parent slope`, with
`wander` from `coast_wander_smooth` (0.03) to `coast_wander_rugged` (0.5), so
the shore moves about that many wavelengths whatever the gradient. The
slope-saturated amplitude used elsewhere would swamp a flat coast and the shore
would just take the noise's sign out to the spread cap. Over the sea the
underwater fade is lifted in the band, so rugged coasts can raise skerries;
lakes keep it.

**2. Shore reshaping (`reshape_coast`, on by default).** After elevation is
conserved, a *target* shore is drawn and the elevation is edited to match it:

- The target starts from the conserved elevation box-blurred twice over
  `coast_blur_texels` (1.5) parent texels, which curves through the staircase.
  The fine detail (elevation − blur) is added back × (1 − smoothness), so
  smooth shores lose the fringe and rugged ones keep it. Rugged shores also get
  "bays": fBm one octave below the detail band and the next (the scales
  conservation removes from the elevation itself), of amplitude
  `target wander × 2 wavelengths × parent slope`, the wander running from
  `coast_target_wander_smooth` (0) to `coast_target_wander_rugged` (0.5).
- `reshape_to_target` sets each texel on the wrong side of the water surface
  to the target clipped to `coast_edit_min_km`–`coast_edit_max_km` (1–20 m)
  past it. The change to its parent texel's mean is taken back from that parent
  texel's texels on the side that can absorb it without crossing the surface
  (the dry ones raised, or the wet ones lowered, in proportion to their height
  above or depth below it), so every parent mean is unchanged and no other
  texel changes side. A parent texel that would need more than
  `coast_max_scale` (1×) of that height, or has nothing to absorb it, is left
  as it was, keeping its staircase step. On seed 4 region 49 about two thirds
  of the wanted edits are applied.
- Two passes share the target: the sea against sea level (lake texels
  excluded), then the lakes against their own surfaces (ocean texels
  excluded). A lake's surface is known on its texels only, so it is carried
  about two parent texels onto the shore by taking the lowest neighbouring
  surface, as the flood that later re-draws the lake spreads it.
- A final `conserve_additive` clears float crumbs. The water masks are then
  re-flooded as usual, so they agree with the edited relief, and
  `water_spread_texels` still bounds how far a shore moves.

The effect is a ~1 parent texel band around the parent's coastline: smooth
shores lose the staircase and fringe; rugged ones gain coves, headlands and
islands about a parent texel across. That is invisible at full-region zoom
(under 1% of a region's width) and clear zoomed in and on local maps, where it
is applied again against the region. `reshape_coast=False` restores the styled
detail alone. `region/test/test_coast_style.py` and
`localmap/test/test_coast_style.py` pin conservation, the water invariants, and
that smooth shores are shorter than rugged ones.

## Resources, population and danger (`region/resources.py`)

Same rule as the physical fields — no noise of their own; all sub-grid variation
comes from the refined relief (slope, the relief anomaly, the treeline, the
refined shoreline), so the layers line up with each other and the terrain. The
world's own tuning is reused: `build_region_fields(..., resources=ResourcesParams,
fertility_exponent=γ)` (the viewer passes the world's `GlobeParams.resources`
and `PopulationParams.fertility_exponent`). `RefineContext` carries the texels,
their anchors, the refined and coarse elevation and slope, and the land masks.

- **Refined slope** (`refined_slope`) is the gradient of the refined elevation
  inside each hex texture, normalised by `slope_saturation` (sub-grid slopes run
  ~0.003–0.035 km/km on seed 4).
- **Forest** is the coarse forest × `(treeline(refined) + 0.05) /
  (treeline(coarse) + 0.05)`, the world's treeline taper at the two elevations.
  **Fertility** is the coarse fertility × `clip(1 − fertility_valley_gain ·
  anomaly, 0.5, 1.5)` × `(1 − fertility_slope_penalty · slope)` × the forest
  penalty's change. Both are densities of *land*: conserved over the texels dry
  on both maps, with conservation and the `[0, 1]` clip alternated
  (`conserve_unit`), then closed exactly by one headroom step (a group still
  short moves each texel toward 1 in proportion to `1 − v`, one still over
  scales down) — needed where a target sits at or near 1, which the alternation
  only approaches geometrically. Residuals: both at float noise. Water keeps the
  parent's aquatic fertility; water the refined shoreline added takes it from the
  neighbouring water.
- **Dominant crop** is `classify_crops` over the box-averaged climate, the
  refined elevation and the refined fertility.
- **Rural density** redistributes each global texel's people over its refined
  land by the rural model's own weight `fertility ** γ` — the design doc's hard
  constraint: the regional sum equals the global one to float noise, and nobody
  stands on refined water unless the global texel has no dry ground left (a
  texel whose land is all barren spreads its people evenly). Town refinement
  reads it ([region_features.md](region_features.md#towns-and-cities)).
- **Danger** is one value per face on the globe. `FaceFanField` makes it a
  continuous surface: piecewise linear over each face's fan triangles (centroid,
  vertex i, vertex i+1), with vertex values shared between faces and each face's
  area mean an exact equality constraint. A non-negative field with exact means
  must be 0 on every 0-danger face, edges included, so danger fades out *inside*
  the faces carrying it; within that, the node values are the smoothest
  (least-squares over the fan edges) solution — one sparse KKT solve (~0.08 s),
  repeated with negative nodes pinned to 0, and cached for the last world.
  Uniform blocks become domes with slightly raised rims; an isolated face is a
  cone up to 3× its value (the layer clamps at "lethal"). The smooth base is
  then multiplied by `exp(danger_terrain_gain · score)`, score =
  `biome_smoothing`-box-averaged (slope + forest) on land, 0 on water, and
  renormalised per global texel — rugged, forested ground more dangerous, open
  flats less, each global texel's (and so each face's) mean unchanged. A local
  map reads the region's continuous danger instead and conserves it per regional
  texel.

## `RegionUpsampleParams`

| param | default | |
|---|---|---|
| `n_cols`, `tex_size` | 14, 64 | grid size |
| `relief_factor`, `slope_reference`, `slope_floor`, `max_relief_km` | 2.0, 0.005, 0.10, 1.0 | detail amplitude |
| `octaves`, `lacunarity`, `persistence`, `nyquist_safety` | 4, 2.0, 0.5, 0.5 | detail band |
| `orographic_gain` | 0.5 /km | precipitation |
| `lapse_rate_c_per_km` | 6.5 | temperature |
| `water_relief_depth_km`, `refine_water_masks`, `water_spread_texels` | 0.1, True, 4 | water |
| `biome_smoothing` | 5 | biome / crop / danger-score box average |
| `coast_band` | 0.25 | coast style / reshaping band (parent water fraction) |
| `coast_persistence_smooth`, `coast_persistence_rugged`, `coast_wander_smooth`, `coast_wander_rugged` | 0.25, 0.8, 0.03, 0.5 | coast detail style |
| `reshape_coast`, `coast_blur_texels`, `coast_target_wander_smooth`, `coast_target_wander_rugged` | True, 1.5, 0.0, 0.5 | shore target |
| `coast_max_scale`, `coast_edit_min_km`, `coast_edit_max_km` | 1.0, 0.001, 0.02 | shore edit |
| `slope_saturation`, `fertility_valley_gain`, `fertility_slope_penalty`, `danger_terrain_gain` | 0.02, 2.0, 0.5, 1.0 | resources and danger |

`LocalUpsampleParams` subclasses it with the local tier's defaults
([local.md](local.md#localupsampleparams)).
