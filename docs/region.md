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
  `local_to_ecef`, `local_to_lonlat`, `sample_grid`, `center_ecef`

Every entry point takes the viewer's `antimeridian` so region assignment is
stable however the mesh is oriented.

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
| row order | v ascending | row 0 = **north**, so textures blit without a flip |

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
`biome_smoothing`, `octaves`, `lacunarity`, `persistence`, `nyquist_safety` — a
`paramclass`, so it is dialog-ready.

### `RegionFields`
`region_id`, `geometry`, `grid`, `tex_size`, `positions (M,T,T,3)`,
`in_hex (M,T,T)`, `anchors (M,T,T)`, `textures {name: (M,T,T,C)}`,
`detail_octaves`, `texel_area_km2`.

## Display (`region/display.py`)

`build_layer_texture(fields, layer)` → `(M, T, T, 4)` RGBA for
`REGION_LAYERS = ["Terrain", "Elevation", "Temperature", "Precipitation"]`, all
routed through the world map's palettes so a region reads as a continuation of
the globe. Terrain and Elevation get `hillshade(...)` from the refined
elevation — without it a 100 m bump is invisible at ~16 km²/texel.

## Viewer (`viz/mapview/regionview.py`)

`RegionView.set_region(geometry, globe)` refines synchronously (~0.8 s for a
region at defaults) and caches a `QImage` per hexagon. Local `(s, t)` → screen is
a flip-and-scale, so each hex texture is an axis-aligned blit clipped to its
hexagon (drawn 0.75 px proud so antialiased neighbours leave no seam). Without a
globe it falls back to the bare scaffolding.

`MapWindow` wires it up: **Regions** toggles the world-map overlay, clicking a
cell enters the region view, a **Layer** combo picks the layer, and
`RegionView.status_text()` reports hex/texel counts and the detail-octave count
in the status bar.

## Not yet implemented

Hydrology, resources, cost surfaces and realmgen inside a region; region
serialization; locals. See the design doc's phase list.
