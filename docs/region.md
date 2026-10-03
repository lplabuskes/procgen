# Regional Maps

Qt-free package implementing the region tier of
[docs/plans/regional_local_maps_design.md](plans/regional_local_maps_design.md):
the sphere is cut into equal-area cells, each cell is tiled with hexagons, and
each hexagon carries textures holding the global fields **refined** — never
replaced. The local tier ([local.md](local.md)) runs the same machinery one level
down, over one regional hexagon.

This doc covers the scaffolding (geometry, grid, sampling, params, display).
The rest of the tier:

- [region_fields.md](region_fields.md) — the field refinement (`region/upsample.py`, `region/resources.py`, `region/smooth.py`): elevation, climate, water, resources, rural density, danger, and the conservation invariant
- [region_features.md](region_features.md) — the line and point features (`region/corridor.py`, `region/rivers.py`, `region/roads.py`, `region/towns.py`, `region/cities.py`): rivers, roads, towns and cities re-sited on the refined terrain
- [viz_regionview.md](viz_regionview.md) — `RegionView`, the Qt widget that shows a region or a local map

## Files
- `region/geometry.py` — equal-area cubed-sphere cells; sphere ↔ local `(s, t)` maps
- `region/grid.py` — `RegionHexGrid`: hexagons + their over-hanging textures inside one cell
- `region/sampling.py` — `GlobalSampler`: read global face textures at arbitrary sphere points; anchor assignment
- `region/source.py` — `ParentSource`: what a refinement reads from the tier above — `GlobalSource` (the globe, for a region) and `RegionSource` (a region, for a local map)
- `region/upsample.py` — `RegionUpsampleParams`, `RegionFields`, `build_region_fields`, `refine_fields`, `conservation_report` ([region_fields.md](region_fields.md))
- `region/resources.py` — `refine_resources`: forest, fertility, crops, rural density and danger ([region_fields.md](region_fields.md))
- `region/smooth.py` — `FaceFanField`: a continuous, non-negative surface with exact per-face means through per-face values (danger)
- `region/corridor.py` — the raster + least-cost corridor router the rivers and roads share ([region_features.md](region_features.md))
- `region/rivers.py` — `RegionRiverParams`, `build_region_rivers`, `route_stream_paths`, `snap_deltas`
- `region/distributaries.py` — `DistributaryParams`, `add_distributaries`: delta distributaries
- `region/roads.py` — `RegionRoadParams`, `build_region_roads`, `route_polyline`
- `region/towns.py` — `RegionSettlementParams`, `refine_region_towns` / `refine_towns`, `tier_sites` (the site features both tiers and the cities read)
- `region/cities.py` — `refine_region_cities` / `refine_cities`
- `region/display.py` — RGBA layer textures for the viewer (runs the world map's own recipes), hover text, hillshade, `snap_to_land`
- `region/params.py` — `RegionParams`, the **Region** section of the params dialog
- `viz/mapview/regionview.py` — `RegionView`, the zoomed-in widget ([viz_regionview.md](viz_regionview.md))

---

## Geometry (`region/geometry.py`)

The sphere is wrapped in a cube; each cube face carries gnomonic coordinates
`(u, v) ∈ [-1, 1]²`, stretched by a separable equal-area CDF (`ea_forward` /
`ea_inverse`) so uniform steps carve equal solid angle. Each face is cut
`N_SUBDIV × N_SUBDIV` (3 × 3) → **54 regions** (`N_REGIONS`) of ~9.4 M km² each.
Region ids are `face · 9 + ci · 3 + cj`.

- `ecef_to_cell(positions, antimeridian)` — sphere → region id
- `cell_border_polylines(antimeridian)` — overlay curves for the world map
- `RegionGeometry(region_id, antimeridian)` — that cell's local `[0, 1]²` frame:
  `local_to_ecef`, `local_to_lonlat`, `sample_grid`, `center_ecef`, `up_direction`,
  and `ecef_to_local` (the inverse: `(s, t, valid)`, projected through the cell's
  own cube face so the frame extends smoothly past the cell; `valid` is False
  only on the face's far hemisphere)

Every entry point takes the world's `antimeridian` (a whole-sphere rotation about
Z puts a cube edge on it) so region assignment is stable however the mesh is
oriented.

**Orientation.** The local `(s, t)` frame is *not* uniformly north-up — on the
4 equatorial cube faces +t happens to be exactly true north (moving along t at
fixed s is exactly a meridian), but on the 2 polar faces the grid is a flat
gnomonic cap where +t is an arbitrary direction that varies cell to cell.
`RegionGeometry.up_direction()` returns the local `(ds, dt)` direction, at the
cell centre, that a renderer should treat as up: true north ordinarily (the
tangent of increasing latitude, solved through the local Jacobian), or — for the
one cell per pole actually centred on the pole, where north has no direction —
the tangent toward the antimeridian meridian instead. On every equatorial region
it is `+t`, so the view's rotation is 0.

**Handedness.** `_cube_face_gnomonic` negates `u_g` on faces 1, 2 and 5 (and
`_face_gnomonic_to_ecef` / `ecef_to_local` undo it on the way back) so every face
is right-handed the same way: `ds x dt` points outward everywhere, i.e. east is
always to the right of north (`test_local_frame_is_right_handed`). Without
that negation, sharing one raw `yr/m`-or-`xr/m` formula between a face and
its opposite mirrors one of the two, since they're reflections of each other
through the origin — a hard left-right mirror once `up_direction()` is applied,
since a rotation can't undo a reflection.

## Hex grid (`region/grid.py`)

`RegionHexGrid(geometry, n_cols)` tiles the local square with flat-top hexagons
(`n_cols` columns across the s-axis; circumradius `1 / (1.5·n_cols + 0.5)`). A
hexagon is kept when it *overlaps* the cell, not merely when its centre is
inside, so the cell is fully covered; neighbouring regions overlap slightly at
their shared edge.

Each hexagon carries a square texture of side `2·radius` — the same
square-over-hangs-the-polygon arrangement mesh faces use one level up. Two
deliberate differences from the mesh convention:

| | mesh face textures | region hex textures |
|---|---|---|
| texel positions | grid **endpoints** (`linspace(0, 1, T)`) | **cell centres** (`(i+0.5)/T`) — so a plain mean is an area mean |
| row order | v ascending | row 0 = largest t, so textures blit without a flip (see **Orientation** above for what "up" — +t — actually points at) |

- `texture_local(T)` / `texture_positions(T)` — `(M, T, T, 2 | 3)`
- `texture_in_hexagon(T)` — `(M, T, T)` bool; the over-hang is excluded from every aggregate
- `texel_at(s, t, T)` / `texels_at(...)` — the texel drawn at a local point (owning hex = nearest centre)
- `texel_spacing_rad(T)` — measured angular spacing, used to band-limit the detail

At defaults (`n_cols=14`, `tex_size=64`) a region has 216 hexes ~286 km across,
with ~4.5 km texels.

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
over-hang weighted out. The refine stage's border tracer reuses the sampler on
the globe ([realm_refine.md](realm_refine.md)).

## Parent sources (`region/source.py`)

`refine_fields` reads the tier above only through a `ParentSource`: `locate`
(→ a `Located` with each texel's anchor), `coarse` (bilinear read), `anchor_values`,
`stencil` / `interpolate` (a smooth read of a per-anchor field, its corners mapped
through an *owner* map so a texture's over-hang reads its neighbour's values),
`slope`, `danger`, `has`, `n_anchors` and `spacing_rad`.

- `GlobalSource(globe)` — the mesh's face textures; slope from `global_slope`; danger made continuous by `FaceFanField`.
- `RegionSource(fields)` — a region's hex textures, for a local map: points are placed in the region's `(s, t)` frame on the nearest hex and read through its over-hang; slope is the refined slope; danger is the region's (already continuous) texture.

## `RegionParams` (`region/params.py`)

Groups the region tuning — `upsample` (`RegionUpsampleParams`, "Detail
refinement"; [region_fields.md](region_fields.md)), `rivers`
(`RegionRiverParams`), `roads` (`RegionRoadParams`) and `settlements`
(`RegionSettlementParams`, "Towns and cities"; all three in
[region_features.md](region_features.md)) — as `AppParams.region`, the **Region**
section of the params dialog. It never rebuilds the world: `main.py` hands it to
`MapWindow.set_region_params` → `RegionView.set_params`, and a dialog edit that
changes nothing but `region` (or `local`) stops there. Any change makes every
refined town and city stale — at the local tier too, which refines from the
region's positions — so all of them go back to their stage sites
(`Towns.reset_refinement`, `Settlement.reset_refinement`) and are refined again
as their regions are visited. On screen, a new `upsample` re-refines the region
(keeping zoom and pan); a new `rivers`, `roads` or `settlements` only rebuilds
the rivers and roads and re-refines the region's cities and towns. Cached local
maps are dropped either way.

## `RegionFields` (`region/upsample.py`)

`region_id`, `geometry`, `grid`, `tex_size`, `positions (M,T,T,3)`,
`in_hex (M,T,T)`, `anchors (M,T,T)` (global texel), `faces (M,T,T)` (owning
global face), `face_uv (M,T,T,2)` (texture coords on it), `textures {name:
(M,T,T,C)}` (the globe's keys plus `danger`), `detail_octaves`,
`texel_area_km2` (cell area × the texel's share of the equal-area local square; not cell area ÷ in-hex count, since the hexes spill past the cell); local maps only: `parent_anchors` (the regional texel each
texel conserves) and `in_map` (inside the parent hexagon). `domain` = `in_hex`
(∩ `in_map` on a local map) is the aggregation weight.

## Display (`region/display.py`)

The region view offers exactly the world map's layers (its registered color
schemes), under the same names and in the same order.
`build_layer_texture(fields, spec, mesh)` → `(M, T, T, 4)` RGBA, where `spec`
is the world map's `SchemeSpec`. There is no region-side list of layers or
palettes: a scheme with a `TextureRecipe` whose `sources` the region has refined
(`has_refined_layer`) is drawn by running that recipe — the stage's own
function — on `fields.textures`. Today that is Elevation, Biome and Polities,
and with `--debug` Annual/Summer/Winter Temperature, Annual Precipitation,
Fertility, Forest Density, Dominant Crop, Rural Density and Danger; a new stage
layer with a recipe over refined keys appears refined automatically. Danger's
recipe exists only for the region: the globe has no danger texture, so the world
map draws the per-face colour function.

Besides the refined textures, a recipe may read the region's texel positions
under `core.display.POSITIONS_KEY` (`recipe_inputs`). Polities uses that: the
polity is a function of position (`realmgen.refine.fill.PolityLabelField`), so
its fill is evaluated at the region's own texels, with the refined shoreline, and
lines up with the border overlay instead of stepping at the global texel size.

Recipes with `relief=True` (Elevation, Biome, Fertility, Forest Density, Dominant
Crop) are then `hillshade(...)`d on land from the refined elevation (light from
the north-west) — without it a 100 m bump is invisible on ~4.5 km texels; water
stays flat. Every other layer goes through `sample_global_layer`: the scheme's
per-face RGBA texture read at each texel's anchor, or else the scheme's color
function applied to each texel's owning face — the globe's own two rendering
paths, with no detail invented.

`describe_texel(fields, hex, row, col, keys)` is the refined hover text
(elevation → `[px: …]`, biome → water or biome name, T, P) for hover
contributors registered with `refined=keys`. `snap_to_land(fields, points,
max_dist)` moves points on refined water to the nearest in-hexagon land texel
within `max_dist`; the roads and the viewer use it.

## Not yet implemented

Lakes as a regional hydrology product (the refined lake mask is drawn by the
layers; there is no lake-polygon overlay in the menus), minerals / rare
resources and the cost *layers* inside a region (the roads evaluate the land
cost model themselves), the rest of realmgen inside a region, and region
serialization (fields, rivers and roads are rebuilt on every visit; only the
refined town and city positions are saved with the world). Non-refined layers
are still sampled per global texel, so their coast and lakes are the globe's.
See the design doc's phase list.
