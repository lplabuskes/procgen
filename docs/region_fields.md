# Regional Field Refinement

Part of the region tier — [back to region.md](region.md). How a region's
textures are made from the globe's (and, through the same code, a local map's
from its region's — see [local.md](local.md)).

## Files
- `region/upsample.py` — `RegionUpsampleParams`, `RegionFields`, `refine_fields`, `build_region_fields`, the conservation steps, the water flood, `conservation_report` / `anchor_residuals`
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
| `elevation` | slope-proportional fBm — **the only noise source** | additively, smooth |
| `temperature` | lapse rate on the refined relief (sub-sea-level ground counts as sea level, as on the globe) | additively, smooth |
| `precipitation` | orographic response: × `1 + orographic_gain · anomaly` (per km, clipped to ±0.9) | multiplicatively, smooth (stays positive) |
| `biome` | reclassified from the **area-averaged** refined climate | derived |
| `ocean_mask`, `lake_mask`, `filled_height` | re-flooded to the parent's water surface over the refined relief | surface exact; shoreline refined |
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

**Detail amplitude.** `relief_factor × (detail wavelength, km) × effective slope`,
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
| `slope_saturation`, `fertility_valley_gain`, `fertility_slope_penalty`, `danger_terrain_gain` | 0.02, 2.0, 0.5, 1.0 | resources and danger |

`LocalUpsampleParams` subclasses it with the local tier's defaults
([local.md](local.md#localupsampleparams)).
