# Regional Maps

Qt-free package implementing the region tier of
[docs/plans/regional_local_maps_design.md](plans/regional_local_maps_design.md):
the sphere is cut into equal-area cells, each cell is tiled with hexagons, and
each hexagon carries textures holding the global fields **refined** — never
replaced.

## Files
- `region/geometry.py` — equal-area cubed-sphere cells; sphere ↔ local `(s, t)` maps
- `region/grid.py` — `RegionHexGrid`: hexagons + their over-hanging textures inside one cell
- `region/sampling.py` — `GlobalSampler`: read global face textures at arbitrary sphere points; anchor assignment
- `region/upsample.py` — `build_region_fields`: the refinement itself, plus `conservation_report`
- `region/display.py` — RGBA layer textures for the viewer (reuses the world map's palettes)
- `region/rivers.py` — `build_region_rivers`: the world's rivers re-routed over the refined relief, plus sub-threshold tributaries
- `region/params.py` — `RegionParams`: `upsample` (`RegionUpsampleParams`) + `rivers` (`RegionRiverParams`), the **Region** section of the params dialog
- `viz/mapview/regionview.py` — `RegionView`, the zoomed-in widget

---

## Geometry (`region/geometry.py`)

The sphere is wrapped in a cube; each cube face carries gnomonic coordinates
`(u, v) ∈ [-1, 1]²`, stretched by a separable equal-area CDF (`ea_forward` /
`ea_inverse`) so uniform steps carve equal solid angle. Each face is cut
`N_SUBDIV × N_SUBDIV` → **54 regions** of ~9.4 M km² each.

- `ecef_to_cell(positions, antimeridian)` — sphere → region id
- `cell_border_polylines(antimeridian)` — overlay curves for the world map
- `RegionGeometry(region_id, antimeridian)` — that cell's local `[0, 1]²` frame:
  `local_to_ecef`, `local_to_lonlat`, `sample_grid`, `center_ecef`, `up_direction`,
  and `ecef_to_local` (the inverse: `(s, t, valid)`, projected through the cell's
  own cube face so the frame extends smoothly past the cell; `valid` is False
  only on the face's far hemisphere)

Every entry point takes the viewer's `antimeridian` so region assignment is
stable however the mesh is oriented.

**Orientation.** The local `(s, t)` frame is *not* uniformly north-up — on the
4 equatorial cube faces +t happens to be exactly true north (moving along t at
fixed s is exactly a meridian), but on the 2 polar faces the grid is a flat
gnomonic cap where +t is an arbitrary direction that varies cell to cell.
`RegionGeometry.up_direction()` returns the local `(ds, dt)` direction, at the
cell centre, that a renderer should treat as up: true north ordinarily (the
tangent of increasing latitude), or — for the one cell per pole actually
centred on the pole, where north has no direction — the tangent toward the
antimeridian meridian instead, found the same way (via the local Jacobian) so
it degrades continuously into the equatorial case (a no-op there). It is 0 on
every equatorial region.

**Handedness.** `_cube_face_gnomonic` negates `u_g` on faces 1, 2 and 5 (and
`_face_gnomonic_to_ecef` undoes it on the way back) so every face is
right-handed the same way: `ds x dt` points outward everywhere, i.e. east is
always to the right of north (`test_local_frame_is_right_handed`). Without
that negation, sharing one raw `yr/m`-or-`xr/m` formula between a face and
its opposite mirrors one of the two, since they're reflections of each other
through the origin — invisible while `up_direction()` wasn't applied yet
(any orientation looked equally arbitrary), but a hard left-right mirror once
it was, since a rotation can't undo a reflection.

## Hex grid (`region/grid.py`)

`RegionHexGrid(geometry, n_cols)` tiles the local square with flat-top hexagons
(`n_cols` columns across the s-axis). A hexagon is kept when it *overlaps* the
cell, not merely when its centre is inside, so the cell is fully covered;
neighbouring regions overlap slightly at their shared edge.

Each hexagon carries a square texture of side `2·radius` — the same
square-over-hangs-the-polygon arrangement mesh faces use one level up. Two
deliberate differences from the mesh convention:

| | mesh face textures | region hex textures |
|---|---|---|
| texel positions | grid **endpoints** (`linspace(0, 1, T)`) | **cell centres** (`(i+0.5)/T`) — so a plain mean is an area mean |
| row order | v ascending | row 0 = largest t, so textures blit without a flip (see **Orientation** above for what "up" — +t — actually points at) |

- `texture_local(T)` / `texture_positions(T)` — `(M, T, T, 2 | 3)`
- `texture_in_hexagon(T)` — `(M, T, T)` bool; the over-hang is excluded from every aggregate
- `texel_spacing_rad(T)` — measured angular spacing, used to band-limit the detail

## Sampling the globe (`region/sampling.py`)

`GlobalSampler(mesh)` answers two questions per regional texel:

- `locate(points) → (faces, u, v)` — owning face by nearest centroid (a KD-tree
  over `mesh.face_centroids`), then that face's tangent-frame UV.
- `sample(texture, faces, u, v)` — **bilinear** read of any `(N, H, W, C)` face
  texture → `(P, C)`. This is the smooth coarse field.
- `anchors(faces, u, v, H, W)` — the flat index `face*H*W + row*W + col` of the
  **nearest** global texel. The anchor is the conservation target.
- `anchor_values(texture, anchors)` — that texel's exact global value.

`anchor_means(values, anchors, weights, n_anchors)` reduces per-anchor with the
over-hang weighted out.

## Refinement (`region/upsample.py`)

`build_region_fields(globe, region_id, params) → RegionFields`. Seeded with
`world.seed XOR region_id`, so a region always regenerates identically.

Per field: **bilinear coarse read → detail → per-anchor correction.**

**Elevation is the only field with a noise stream.** Everything else is a
physical consequence of the refined relief, or is copied verbatim. Independent
noise in a second field is what breaks the derived categorical ones: `biome` is a
threshold classifier, so uncorrelated jitter in its inputs scatters neighbouring
texels across a cut. Sharing one relief signal puts the moisture and temperature
bands on the contours instead.

| field | detail | conserved |
|---|---|---|
| `elevation` | slope-proportional fBm — **the only noise source** | additively |
| `temperature` | lapse rate on the refined relief | additively |
| `precipitation` | orographic response to the refined relief (`orographic_gain`, per km) | multiplicatively (stays positive) |
| `biome` | reclassified from the **area-averaged** refined climate | derived |
| `ocean_mask`, `lake_mask` | re-flooded to the global water surface over the refined relief | surface exact; shoreline refined |

**Why biome averages.** The refined temperature swings ~1 °C between adjacent
~4 km texels (mean \|Laplacian\| 0.63 °C vs the coarse field's 0.068 °C), which is
enough to fray every biome edge into a dithered band. A biome is a property of a
neighbourhood rather than of a single texel, so `classify_textures` reads a
`biome_smoothing`×`biome_smoothing` box average — the `temperature` and
`precipitation` layers themselves stay sharp. On seed 4 region 34 this takes
texels adjacent to a biome change from 9.2% back to 2.7% against a 2.1%
no-refinement reference, while 1.2% of texels still change biome because of the
terrain. `biome_smoothing=0` disables it.

**Water masks.** A water body's *surface* is a global fact — the ocean sits at
sea level, a lake at the fill height Priority-Flood gave it (`filled_height`) —
so the level is copied from the globe and only the shoreline moves. Each texel is
water when its **refined** elevation lies below the surface of a body it is
connected to: the same level-constrained flood
`compute_ocean_mask_texture` runs one tier up, seeded from the global mask
instead of from deep-ocean faces. Water may cross low ground into up to
`water_spread_texels` texels the globe called land (≈ one global texel at the
defaults) but never over a rise above its surface; ocean and lake are flooded
separately and neither may take the other's texels, so a coastal lake below sea
level is not swallowed by the sea it is walled off from. `refine_water_masks=0`
restores the verbatim anchor copy. On seed 4 region 49 the coast gains 0.14% of
the region and loses 0.17% — bays and inlets against islands and spits.

**Why the detail fades out under water.** Lakes are *shallow* (median ~40 m of
water over the bed on a real world) against ~15–30 m of sub-grid relief, so
zero-mean noise pokes half a lake bed back into the air and the refined mask —
a threshold on the refined elevation — dithers the whole water body into
confetti rather than only rearranging its shore. Detail amplitude is therefore
scaled by the water depth over `water_relief_depth_km` (0.1 km), which keeps the
bump strictly smaller than the water above it. The deep ocean (median 3.4 km) is
far past the fade and keeps its full detail, and submerged relief is painted over
by the water anyway. `debug_region.py` prints the flooded/dried percentages.

**Detail band.** Octaves run from the global grid's Nyquist
(`0.5 / global texel spacing`) up to the regional grid's
(`nyquist_safety / regional spacing`). A region no finer than the globe gets
`detail_octaves == 0` — no invented structure. At defaults (10,242 faces,
`tex_size=16` global; `n_cols=14`, `tex_size=64` regional) this is **2 octaves**
over ~16 km² texels vs the globe's ~195 km².

**Detail amplitude.** `relief_factor × (detail wavelength, km) × effective slope`,
where the effective slope saturates at `slope_reference` and never falls below
`slope_floor` of it. The global field is smooth at ~15 km, where even mountains
only reach ~0.007 km/km — hence the small `slope_reference` (0.005) and the
`relief_factor` above 1. Typical result: tens of metres on plains, a few hundred
in rough country.

**The critical invariant.** `conservation_report(fields, globe)` averages every
global texel's in-hexagon regional samples and compares against the global value.
Residuals are at float32 rounding (~1e-7 km) by construction, because the last
step of each field is exactly that correction. `region/test/test_upsample.py`
pins it.

### `RegionUpsampleParams`
`n_cols`, `tex_size`, `relief_factor`, `slope_reference`, `slope_floor`,
`max_relief_km`, `orographic_gain`, `lapse_rate_c_per_km`,
`water_relief_depth_km`, `refine_water_masks`, `water_spread_texels`,
`biome_smoothing`, `octaves`, `lacunarity`, `persistence`, `nyquist_safety`.

### `RegionParams` (`region/params.py`)
Groups both tiers of region tuning — `upsample` ("Detail refinement") and
`rivers` ("Rivers") — as `AppParams.region`, the **Region** section of the params
dialog. It is view-side only: `main.py` hands it to
`MapWindow.set_region_params` → `RegionView.set_params`, and a dialog edit that
changes nothing but `region` never rebuilds the world. On screen, a new
`upsample` re-refines the region (keeping zoom and pan); a new `rivers` only
drops the cached rivers, which rebuild on the next paint.

### `RegionFields`
`region_id`, `geometry`, `grid`, `tex_size`, `positions (M,T,T,3)`,
`in_hex (M,T,T)`, `anchors (M,T,T)`, `faces (M,T,T)` (owning global face), `face_uv (M,T,T,2)` (texture coords on it), `textures {name: (M,T,T,C)}`,
`detail_octaves`, `texel_area_km2`.

## Rivers (`region/rivers.py`)

`build_region_rivers(fields, globe, params) → RegionRivers` (`rivers`,
`tributaries`, `paths` = both), each a `(positions, flows, in_lake)` polyline in
the world `"rivers"` feature's form. Qt-free and deterministic; ~0.6–0.9 s per
region at defaults.

Everything runs on one uniform raster over the local frame, at the regional
texel spacing (`raster_texels`) and `margin` past the cell. It reads the refined
elevation / masks / precipitation from the nearest hex texel (reaching through
the over-hang), and the global fields bilinearly beyond the textures.

**World rivers are re-routed, not replaced.** Each stretch of a world river on
the raster becomes the least-cost 8-connected path through a *corridor* of
`corridor_global_texels` (1 global texel) either side of the world polyline —
the positional uncertainty the global grid leaves. A node costs
`exp(valley_seeking · anomaly / σ)`, the anomaly being refined minus coarse
elevation and σ its spread in the corridor, so the river takes the refined
valley floor on a plain as much as in the mountains, but can never change
valley or basin. Stems are routed first (descending flow). A river whose world
mouth is on the raster ends on the cheapest sink near that mouth — its
re-routed stem, a lake, or the *refined* sea — found in a widened disc of two
corridors around the mouth, since the stem itself moved by up to a corridor.
Flow is copied from the nearest world vertex (running max, so monotonic).

**Tributaries.** The refined relief is drained by a *carving* Priority-Flood
(`worldgen.hydrology.routing.priority_flood_tree`) whose outlets are the sea,
lakes, the re-routed rivers and the raster rim: each node drains to the node
that flooded it. The ordinary filled-DEM + steepest-descent routing drew
dead-straight lines across every filled depression in the sub-grid noise (the
epsilon tilt picks the route, not the terrain); carving drains a depression
down its lowest route instead. Flow is accumulated precipitation-weighted **in
the world's flow units** (a node weighs `precip / global mean precip ×` its
share of a global texel's area), so `tributary_min_flow` (default 40) reads on
the scale of the world's `river_flow_threshold` (default 500). Channels are cut
into main stems with the global stage's own
`worldgen.hydrology.rivers.main_stem_paths`, each ending on the river, lake or
coast it drains into; stubs under `tributary_min_nodes` are dropped.

Both kinds are Laplacian-smoothed (`smoothing_iterations`); a segment is flagged
`in_lake` (hidden) when both its ends are on refined water, with land runs of up
to 2 nodes between water treated as water so a lake crossing is not drawn as
dashes over the refined shoreline's speckle.

Known limits: a tributary flowing in across the raster rim only has the
catchment inside the raster behind it; neighbouring regions route their shared
edge independently (each is shown alone, so no seam is visible); and a
tributary's accumulated flow can occasionally exceed the world threshold where
the world's D-infinity split sent that water elsewhere.

### `RegionRiverParams`
`raster_texels`, `margin`, `corridor_global_texels`, `valley_seeking`,
`tributary_min_flow`, `tributary_min_nodes`, `smoothing_iterations`.

## Display (`region/display.py`)

The region view offers exactly the world map's layers (its registered color
schemes), under the same names and in the same order.
`build_layer_texture(fields, spec, mesh)` → `(M, T, T, 4)` RGBA, where `spec`
is the world map's `SchemeSpec`. There is no region-side list of layers or
palettes: a scheme with a `TextureRecipe` whose `sources` the region has refined
(`has_refined_layer`) is drawn by running that recipe — the stage's own
function — on `fields.textures`. Today that is Elevation, Biome,
Annual/Summer/Winter Temperature, Annual Precipitation and Polities; a new stage
layer with a recipe over refined keys appears refined automatically. Besides the
refined textures, a recipe may read the region's texel positions under
`core.display.POSITIONS_KEY` (`recipe_inputs`). Polities uses that: the polity is a
function of position (`realmgen.refine.fill.PolityLabelField`), so its fill is
evaluated at the region's own texels, with the refined shoreline, and lines up
with the border overlay instead of stepping at the global texel size. Recipes with
`relief=True` (Elevation, Biome) are then `hillshade(...)`d on land from the
refined elevation — without it a 100 m bump is invisible at ~16 km²/texel;
water stays flat. Every other layer goes through
`sample_global_layer`: the scheme's per-face RGBA texture read at each texel's
anchor, or else the scheme's color function applied to each texel's owning face
— the globe's own two rendering paths, with no detail invented.

## Viewer (`viz/mapview/regionview.py`)

`RegionView.set_region(geometry, globe)` refines synchronously (~0.8 s for a
region at defaults) and caches a `QImage` per hexagon. Local `(s, t)` → screen is
a flip-and-scale, so each hex texture is an axis-aligned blit clipped to its
hexagon (drawn 0.75 px proud so antialiased neighbours leave no seam), wrapped
in a whole-cell rotation from `geometry.up_direction()` so the rendered map is
actually up-is-north (up-is-antimeridian on the 2 pole regions); this is the
one place that direction gets turned into a screen rotation, since `region/`
itself stays Qt-free. Without a globe it falls back to the bare scaffolding.

`MapWindow` wires it up: **Regions** toggles the world-map overlay, clicking a
cell enters the region view, the shared **Coloration** combo picks the layer in
both views (so it carries across switching), and
`RegionView.status_text()` reports hex/texel counts and the detail-octave count
in the status bar. The active layer's legend is drawn outside the cell, to the
left of its (rotated) footprint; the cell is sized so the rotated footprint fits
beside it. Hover maps the cursor back through the rotation to a texel
(`RegionHexGrid.texel_at`) and shows the world map's full hover text for the
texel's owning face and UV, except that contributors registered with
`add_hover(..., refined=keys)` are replaced in place by the refined values when
the region has all of those keys (`describe_texel`: elevation → `[px: …]`,
biome/temperature/precipitation → water-or-biome, T, P). Everything not refined
regionally (fertility, danger, polity, settlements, …) is the world value.
Leaving the cell restores the summary.

Scroll zooms (1×–`RegionView.MAX_ZOOM` = 20×, anchored to the cursor) and drag
pans, like the world map; the toolbar's −/+/Reset act on whichever view is
showing. The zoom is a screen-space magnification of the laid-out cell, composed
after the north-up rotation, so hover goes through the same inverse transform.
Pan is stored in footprint widths (resize-safe) and clamped so the view never
leaves the cell; the map is clipped short of the legend. Entering a region
resets the view. Outlines are cosmetic pens, so they stay 1 px at any zoom.

The world map's feature overlays (rivers, borders, settlements, trade routes,
resources, winds, …) draw on the region too, from the same Overlays menu toggles
and the same renderers (`viz/mapview/overlays.py`). `_RegionProjector` maps
sphere points through `ecef_to_local` and the cell's screen transform, culling
anything beyond 0.15 of the cell or on the far side; overlays are drawn in
screen space (upright markers, pixel widths) and clipped to the rotated cell.
They size themselves by the *equivalent world zoom* — the world-map zoom that
would show this ground at the same pixel scale — so a region looks like the
world map zoomed to the same place. Overlays are world-resolution data — borders
follow the global paths rather than the refined terrain — except **rivers**:
the first time the Rivers overlay is shown in a region, `build_region_rivers`
runs and its paths are handed to `draw_overlays(..., refined={"rivers": …})` in
place of the world feature (cached until the region changes). Widths still
scale against the *world's* largest flow, so a re-routed river is drawn as wide
as on the world map and tributaries at the thin end. The **Towns** overlay draws only here (`regional` on the
projector; see [realm_towns.md](realm_towns.md)), and cities are drawn to stand
apart from the towns: 1.5× their world-map size (and at least twice the
largest town dot), opaque, and ringed dark-then-white.

**Settlements onto land.** Towns and cities are sited on global texels, so the
refined shoreline can flood one (~1% of towns, a few coastal cities). The
projector's `on_land` runs `region.display.snap_to_land`: a marker on refined
water moves to the nearest in-hexagon land texel within
`water_spread_texels + 2` regional texels (the farthest the refinement lets water
advance), typically 3–6 km. Trade-route endpoints at a moved city follow it.
`RegionView` caches the result per feature. It is display-only, and the world
features keep their global positions.

## Not yet implemented

Lakes as a regional hydrology product (the lake overlay is still the world's
polygons; the refined lake mask is drawn by the layers), resources, cost
surfaces and realmgen inside a region; region
serialization; locals. See the design doc's phase list.
