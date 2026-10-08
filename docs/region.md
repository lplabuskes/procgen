# Regional Maps

Qt-free region tier of
[docs/plans/regional_local_maps_design.md](plans/regional_local_maps_design.md):
the sphere is cut into equal-area cells, each tiled with hexagons whose textures
hold the global fields **refined** — never replaced. The local tier
([local.md](local.md)) runs the same machinery over one regional hexagon.

This doc: scaffolding (geometry, grid, sampling, params, display). Rest of the
tier:
- [region_fields.md](region_fields.md) — field refinement (elevation, climate, water, resources, rural density, danger) and the conservation invariant
- [region_features.md](region_features.md) — rivers, distributaries, streams, roads, towns and cities re-sited on refined terrain
- [viz_regionview.md](viz_regionview.md) — `RegionView`, the Qt widget for a region or local map

## Files
- `region/geometry.py` — equal-area cubed-sphere cells; sphere ↔ local `(s, t)`
- `region/grid.py` — `RegionHexGrid`: hexagons + over-hanging textures in one cell
- `region/sampling.py` — `GlobalSampler`: read global face textures at sphere points; anchor assignment
- `region/source.py` — `ParentSource`: `GlobalSource` (globe → region), `RegionSource` (region → local map)
- `region/upsample.py`, `region/resources.py`, `region/smooth.py` — field refinement ([region_fields.md](region_fields.md))
- `region/corridor.py`, `rivers.py`, `distributaries.py`, `streams.py`, `roads.py`, `towns.py`, `cities.py` — features ([region_features.md](region_features.md))
- `region/display.py` — RGBA layer textures (runs the world map's recipes), hover text, hillshade, `snap_to_land`
- `region/borders.py` — `view_polity_borders`: polity border lines contoured from a region / local view's own fill ([viz_regionview.md](viz_regionview.md))
- `region/params.py` — `RegionParams`, the **Region** params-dialog section
- `viz/mapview/regionview.py` — `RegionView` ([viz_regionview.md](viz_regionview.md))

---

## Geometry (`region/geometry.py`)

Cube-wrapped sphere; each cube face has gnomonic `(u, v) ∈ [-1, 1]²` stretched by
a separable equal-area CDF (`ea_forward` / `ea_inverse`). Each face is cut
`N_SUBDIV × N_SUBDIV` (3 × 3) → **54 regions** (`N_REGIONS`) of ~9.4 M km². Id =
`face · 9 + ci · 3 + cj`.

- `ecef_to_cell(positions, antimeridian)` — sphere → region id
- `cell_border_polylines(antimeridian)` — world-map overlay curves
- `RegionGeometry(region_id, antimeridian)` — the cell's local `[0, 1]²` frame:
  `local_to_ecef`, `local_to_lonlat`, `sample_grid`, `center_ecef`,
  `up_direction`, `ecef_to_local` (→ `(s, t, valid)`, projected through the
  cell's own cube face so the frame extends past the cell; `valid` False only on
  the far hemisphere)

All entry points take the world's `antimeridian` (a rotation about Z puts a cube
edge on it), so assignment is stable however the mesh is oriented.

**Orientation.** `(s, t)` is *not* uniformly north-up: on the 4 equatorial faces
+t is exactly north; on the 2 polar faces +t varies per cell.
`RegionGeometry.up_direction()` gives the local `(ds, dt)` a renderer should
treat as up at the cell centre: true north (via the local Jacobian), or, for the
one cell per pole centred on the pole, toward the antimeridian. Equatorial
regions → `+t` (rotation 0).

**Handedness.** `_cube_face_gnomonic` negates `u_g` on faces 1, 2, 5 (undone by
`_face_gnomonic_to_ecef` / `ecef_to_local`) so `ds × dt` points outward
everywhere — east right of north (`test_local_frame_is_right_handed`). Opposite
faces are reflections through the origin, so without it one of each pair is
mirrored, which no `up_direction()` rotation can undo.

## Hex grid (`region/grid.py`)

`RegionHexGrid(geometry, n_cols)` tiles the local square with flat-top hexagons
(`n_cols` across s; circumradius `1 / (1.5·n_cols + 0.5)`). A hexagon is kept if
it *overlaps* the cell, so the cell is fully covered and neighbours overlap
slightly. Each carries a square texture of side `2·radius` (over-hanging the
polygon, as mesh faces do), with two deliberate differences:

| | mesh face textures | region hex textures |
|---|---|---|
| texel positions | grid **endpoints** (`linspace(0, 1, T)`) | **cell centres** (`(i+0.5)/T`) — a plain mean is an area mean |
| row order | v ascending | row 0 = largest t (blit without flip; see **Orientation** for where +t points) |

- `texture_local(T)` / `texture_positions(T)` — `(M, T, T, 2 | 3)`
- `texture_in_hexagon(T)` — `(M, T, T)` bool; over-hang excluded from every aggregate
- `texel_at(s, t, T)` / `texels_at(...)` — texel drawn at a local point (owning hex = nearest centre)
- `texel_spacing_rad(T)` — measured angular spacing (band-limits detail)

Defaults (`n_cols=14`, `tex_size=64`): 216 hexes ~286 km across, ~4.5 km texels.

## Sampling the globe (`region/sampling.py`)

`GlobalSampler(mesh)`:
- `locate(points) → (faces, u, v)` — owning face by nearest centroid (KD-tree),
  then its tangent-frame UV.
- `sample(texture, faces, u, v)` — **bilinear** read of `(N, H, W, C)` → `(P, C)`,
  the smooth coarse field. Points just past the face's UV square (Voronoi cell
  poking out near corners) are extrapolated, not clamped.
- `anchors(faces, u, v, H, W)` — flat index `face*H*W + row*W + col` of the
  global texel **owning** the point (nearest texel, through `owners`): the
  conservation target.
- `owners(H, W)` — per global texel, the texel owning its *position*: itself if
  its centre is in its own face's territory, else the neighbour's texel there
  (seam duplicates, over-hang). Made idempotent by `settle_owners` so every
  anchor owns itself — required for `conserve_smooth` to converge
  ([region_fields.md](region_fields.md#refinement-regionupsamplepy)).
- `anchor_values(texture, anchors)` — exact global value.

`anchor_means(values, anchors, weights, n_anchors)` reduces per anchor with
over-hang weighted out. The refine stage's border tracer reuses the sampler
([realm_refine.md](realm_refine.md)).

## Parent sources (`region/source.py`)

`refine_fields` reads the tier above only through a `ParentSource`: `locate` (→
`Located` with anchors), `coarse` (bilinear), `anchor_values`, `stencil` /
`interpolate` (smooth read of a per-anchor field, corners through `owners` so
over-hang reads its neighbour), `slope`, `danger`, `has`, `n_anchors`,
`spacing_rad`.

- `GlobalSource(globe)` — mesh face textures; slope from `global_slope`; danger via `FaceFanField`.
- `RegionSource(fields)` — a region's hex textures for a local map: points placed in the region's `(s, t)` on the nearest hex (read through its over-hang), anchored on the texel owning their nearest one; slope = refined slope; danger = the region's continuous texture.

## `RegionParams` (`region/params.py`)

`AppParams.region` (**Region** dialog section): `upsample`
(`RegionUpsampleParams`; [region_fields.md](region_fields.md)), `rivers`,
`roads`, `settlements` ([region_features.md](region_features.md)).
- Never rebuilds the world: `main.py` → `MapWindow.set_region_params` →
  `RegionView.set_params`; an edit touching only `region` (or `local`) stops
  there.
- Any change resets every refined town and city to its stage site
  (`Towns.reset_refinement`, `Settlement.reset_refinement`; local tier too, which
  refines from region positions); they re-refine as regions are visited.
- On screen: new `upsample` re-refines the region (keeps zoom/pan); new
  `rivers`/`roads`/`settlements` rebuilds rivers and roads and re-refines cities
  and towns. Cached local maps dropped either way.

## `RegionFields` (`region/upsample.py`)

`region_id`, `geometry`, `grid`, `tex_size`, `positions (M,T,T,3)` float32
(~0.4 m), `in_hex (M,T,T)`, `anchors (M,T,T)` int32 (global texel), `faces
(M,T,T)` int32 (owning global face), `face_uv (M,T,T,2)` float32, `textures
{name: (M,T,T,C)}` (globe keys + `danger`; water masks uint8, conserved fields
float32 — float16 would lift the conservation residuals off float32 noise), `detail_octaves`, `texel_area_km2` (cell area × texel's share of the
equal-area local square — not cell area ÷ in-hex count, since hexes spill past
the cell). Local maps only: `parent_anchors` (regional texel conserved) and
`in_map` (inside the parent hexagon). Aggregation weight `domain` = `in_hex`
(∩ `in_map` locally).

## Display (`region/display.py`)

The region view offers exactly the world map's registered schemes, same names
and order. `build_layer_texture(fields, spec, mesh)` → `(M, T, T, 4)` RGBA for a
`SchemeSpec`; no region-side layer/palette list.

- A scheme whose `TextureRecipe` `sources` the region refined
  (`has_refined_layer`) runs that recipe on `fields.textures`: Elevation, Biome,
  Polities; with `--debug` also Annual/Summer/Winter Temperature, Annual
  Precipitation, Fertility, Forest Density, Dominant Crop, Rural Density, Danger.
  New stage layers with recipes over refined keys appear refined automatically.
  Danger's recipe is region-only (globe has no danger texture).
- Recipes may read texel positions under `core.display.POSITIONS_KEY`
  (`recipe_inputs`). Polities does: `realmgen.refine.fill.PolityLabelField` is a
  function of position, so fill matches the border overlay and refined shoreline
  instead of stepping per global texel.
- `build_layer_texture(..., river_segments=)` adds the view's drawn rivers under
  `core.display.RIVER_SEGMENTS_KEY` (`(M, 8)` `[a, b, flow, world river id]`). Polities
  moves its river-following borders onto them, so borders follow the re-routed rivers on
  screen ([realm_refine.md](realm_refine.md#river-following-river_followpy)).
- `relief=True` recipes (Elevation, Biome, Fertility, Forest Density, Dominant
  Crop) are `hillshade(...)`d on land from refined elevation (light NW); water
  flat. Needed — a 100 m bump is invisible on ~4.5 km texels.
- Everything else → `sample_global_layer`: the scheme's per-face RGBA texture at
  each texel's anchor, else its colour function on the owning face (no detail
  invented).

`describe_texel(fields, hex, row, col, keys)` — refined hover text (elevation →
`[px: …]`, biome → water or name, T, P) for hover contributors registered with
`refined=keys`. `snap_to_land(fields, points, max_dist)` — move points on refined
water to the nearest in-hex land texel within `max_dist` (roads, viewer).

## Not yet implemented

A lake-polygon overlay (refined lake masks, small lakes included, are drawn as
texels), minerals / rare resources and cost *layers* in a region
(roads evaluate the cost model themselves), the rest of realmgen in a region,
region serialization (fields/rivers/roads rebuilt per visit; only refined
town/city positions saved). Non-refined layers are sampled per global texel, so
their coast and lakes are the globe's. See the design doc's phase list.
