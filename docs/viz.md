# Visualization Layer

The Qt viewers. Stages never import Qt: they fill a `DisplayRegistry`
([core.md](core.md)) and `viz/mapview/registry.py` wires it into the window. The
zoomed region / local-map view: [viz_regionview.md](viz_regionview.md).

## Files
- `viz/mapview/mapview.py` — `MapWidget` (world map), `MapWindow` (app window), `_WorldProjector`, `_split_at_antimeridian`
- `viz/mapview/regionview.py` — `RegionView`, `_RegionProjector` ([viz_regionview.md](viz_regionview.md))
- `viz/mapview/overlays.py` — `OverlayProjector`, `draw_overlays`, per-feature renderers (shared by every view)
- `viz/mapview/legend.py` — `legend_size`, `draw_legend`
- `viz/mapview/projections.py` — four map projections
- `viz/mapview/registry.py` — `apply_registry` (Qt adapter for `DisplayRegistry`)
- `viz/mapview/main.py` — dev entry point
- `viz/mesh/gui.py` — `MeshViewer`, `MainWindow` (3-D OpenGL globe)
- `viz/params_dialog.py` — `ParamsDialog` (generic params UI)
- `viz/color_stops.py` — colour interpolation utilities

Antimeridian choice: `core/antimeridian.py` ([core.md](core.md)).

**Face rendering priority** (both viewers, falling back when absent): per-face
textures → named colour schemes (callables over `face_metadata`) → static
`face_colors`.

---

## `viz/mapview/mapview.py`

### `_split_at_antimeridian(face_ecef, face_uv=None, cut_lon=π)`

Splits a unit-sphere polygon at meridian `cut_lon` (shifts longitudes so it maps
to ±π, splits crossing edges). Returns a list of `(M, 2)` lonlat arrays (or
`(lonlat, uv)` tuples when `face_uv` given): one element, or two if it crosses.

### `MapWidget` (`QWidget`)

`QPainter` 2-D projected polygon renderer (Robinson default).

- **Caches** (rebuilt on `set_mesh` / `set_globe`): `_faces_lonlat` / `_face_src`
  (split polygons + source face), `_proj_faces` / `_face_bbox_min` / `_max`
  (projected + bboxes for hit-testing), `_texture_qimages` (per-face `QImage`
  when a texture scheme is active).
- **Pipeline**: ECEF → `_split_at_antimeridian` at the world's `antimeridian`
  (with UVs if any textures) → lon/lat → projection → paint. Map body (faces +
  overlays) cached as a pixmap; hover-only changes (region tint) re-blit it.
- **Texture rendering**: per-fan-triangle affine UV mapping via
  `QBrush(QImage)` + `QTransform`. `associate_scheme_texture(scheme, key)` binds a
  `mesh.face_textures` key to a scheme (read by `_rebuild_textures`).
- **Region mode** (`set_region_mode`, **Regions** toggle): tints the cell under
  the cursor, draws cell borders (`ecef_to_cell`, `cell_border_polylines`); click
  → `set_region_click_callback`'s callback with the region id.
- **Legend**: bottom-left via `legend.py` (also used by `RegionView`).
  `register_legend(scheme, kind, entries)` (`"gradient"`/`"categorical"`,
  `(label, rgba)` list); `legend_for(name)`.
- **Hover**: vectorised bbox prefilter → ray-casting PIP → barycentric UV in the
  hit fan triangle. `set_hover_callback(cb)` → `cb(face_src_or_None, uv_or_None)`.
- **Pan/zoom**: drag; wheel zoom 1×–50× anchored to cursor; `zoom_in()` /
  `zoom_out()` / `reset_view()`.

API: `set_globe(world)` / `set_mesh(mesh, antimeridian=None)` (bare mesh uses
`find_ideal_antimeridian`), `set_projection(proj)` (no reload),
`set_color_schemes(dict)` (activates first), `set_color_scheme(name)`,
`associate_scheme_texture`, `set_overlay_visible(name, visible)`,
`set_hover_callback` / `set_region_click_callback`.

### `MapWindow` (`QMainWindow`)

- **Toolbar**: Save, Coloration combo, Overlays menu (checkable `QAction` per
  overlay), Regions toggle, Back ("← World Map" / "← Region"), **Local maps**
  toggle (in a region: click a hex to open its local map), zoom −/+/Reset (routed
  to the visible view).
- **Views**: `QStackedWidget` of world map, region `RegionView`, local-map
  `RegionView` ([viz_regionview.md](viz_regionview.md)).
- **Bottom bar**: status, Parameters, Re-run.

API:
- `set_globe(world)` / `set_mesh(mesh)` — new globe returns to world map, clears local-map cache
- `set_color_schemes(schemes)` (keeps active scheme if still present), `set_active_scheme(name)` / `active_scheme`
- `add_overlay(name, label, checked)` — idempotent; one toggle drives all three views
- `register_legend`, `associate_scheme_texture` — forwarded to `MapWidget`
- `associate_scheme_recipe(scheme, recipe)` — `TextureRecipe` the zoomed views rerun on refined data
- `set_world_params` / `set_realm_params` — the `GlobeParams` / `RealmParams` the world was built with (zoomed views refine with them)
- `set_region_params` / `set_local_params` — view-side tuning (`AppParams.region` / `.local`); rebuilds the map on screen; a region change also rebuilds an open local map and drops cached ones
- `set_hover_parts(fn)` — `fn(face, uv, local) -> list[str]`, joined into the status bar; zoomed views pass `local` to substitute refined values for contributors tagged `refined=` (`DisplayRegistry.hover`)
- `set_rebuild_callback` / `set_params_callback` / `set_save_callback`, `set_status`, `set_rebuild_enabled`

---

## `viz/mapview/overlays.py` — feature overlays

`draw_overlays(painter, globe, visible, projector, refined=None)` draws every
visible overlay with data, in fixed order. Each view supplies an
`OverlayProjector`: `zoom` (world-map-equivalent, for sizing), `view_zoom` and
`px_per_km` (river widths), `regional` (towns only when True), `on_land`
(settlements moved off refined water; identity on world map), `points`,
`segments` (with keep mask), `split_segment` (trade routes), `polygons`.
`_WorldProjector` projects and splits/drops at the antimeridian;
`_RegionProjector` uses the cell's local frame. `refined` substitutes a zoomed
view's finer feature ([viz_regionview.md](viz_regionview.md#overlays-in-a-zoomed-view)).
Marker sizes, line widths and arrow lengths scale with `zoom`.

Drawing order (feature key, registering stage):

| Feature key | Renderer |
|---|---|
| `"rivers"` (hydrology) | Polylines, in-lake hidden. World/region: width ∝ √(flow / world max) (0.5–4 px) × `view_zoom`^0.5 (× 1.5 in a region) — each level on its *own* zoom; ≤ 1 px not drawn. Local map (`px_per_km` set): physical width via `worldgen/hydrology/discharge.py` (discharge = flow × mean precip × runoff 0.36; width = 7·Q^0.5 m), ≥ hairline |
| `"lakes"` | Filled blue polygons. No stage registers this toggle today |
| `"winds_season_N"` (climate, debug) | Stubs per centroid, length ∝ unit arc; antimeridian-crossing skipped |
| `"currents"` (climate, debug) | Stubs, length ∝ speed, blue→grey→red by temp_delta (17-bucket QPen cache) |
| `"deltas"` (hydrology, debug) | Green circles |
| `"volcanoes"` (elevation, debug) | Red/orange circles |
| `"minerals"` (resources, debug) | Squares by type (`minerals.MINERAL_PALETTE`), size ∝ abundance |
| `"rare_bio"` (resources, debug) | Diamonds by type (`rare_bio.RARE_BIO_PALETTE`), size ∝ abundance |
| `"danger_sources"` (danger, debug) | Down-triangles by source category |
| `"trade_routes"` (population) | Yellow, hops with flow ≥ world median; width ∝ √(flow / max) (0.4–4 px) × zoom^0.4; ends follow `on_land` |
| `"villages"` (local only) | 1.3–2.4 px dots by kind (farming, river, fishing, roadside), from `refined["villages"]` |
| `"forts"` (`Forts`, all views) | Crenellated keep (`_KEEP_OUTLINE`) in the kind colour, dark outline over a white halo; half-size `max(3, 3.8 · zoom^0.3)` px, × 1.5 in region/local; `on_land`-snapped |
| `"towns"` (towns; region/local only) | Kind-coloured dots 0.9–1.6 px × zoom^0.3; ringed dark-then-white on local maps |
| `"settlements"` (population) | Tier circles, radius `TIER_RADII[tier] · zoom^0.3 · (√(pop/max_pop)·0.4 + 0.6)`. Region: × 1.5, ≥ 2 × largest town, ringed. Local: built-up footprint (`localmap/cities.py`, texels unioned via `QRegion`, tier colour, dark outline); smaller than the floor marker (2 × largest town radius) → "walled" marker (dark outline, white band, dark ring, dark centre dot), never mistaken for a town |
| `"polity_borders"` (polity) | Black by depth (2.0 / 1.2 / 0.7 px × zoom^0.3, fainter deeper), sovereign on top |
| farmland | Local only, no toggle ([viz_regionview.md](viz_regionview.md)) |

---

## `viz/mapview/projections.py`

`Projection` base: `project(ecef)`, `project_lonlat(lonlat)`, `unproject_lonlat(uv)`
(NaN outside; region picking). ECEF: Z north, lon 0 at +X. Output `float32 (N, 2)`
normalised.

- **Equirectangular**: u = lon/π, v = lat/(π/2).
- **Mercator**: latitudes clamped ±85.05°.
- **Mollweide**: equal-area; θ by 10 Newton iterations.
- **Robinson** (default): table-interpolated at 5° steps.

`ALL_PROJECTIONS`; `projection_by_name(name)` (case-insensitive) resolves
`main.py --projection=NAME` → `MapWindow(projection=...)`.

---

## `viz/mapview/registry.py`

`apply_registry(reg, window, world)` — the only code knowing both
`DisplayRegistry` and `MapWindow` (keeps `core`/stages Qt-free):
1. `set_color_schemes` with all `reg.schemes` in registration order.
2. Per scheme: `associate_scheme_texture` / `associate_scheme_recipe` /
   `register_legend` as present.
3. Per overlay: `add_overlay` + legend.
4. Hover: `reg.hover(face_src, uv, world, local)` → `set_hover_parts`.

---

## `viz/mapview/main.py` — dev entry point

`icosphere(depth=4)` → deregularise → dual → radial-gradient textures tinted per
face → `MapWindow`. Tests projection/rendering without the pipeline.

---

## `viz/mesh/gui.py` — 3-D OpenGL viewer

`MeshViewer` (`QOpenGLWidget`): `PolygonMesh` via VBOs + `glDrawArrays` (GL 3.3
compat), rotating 0.5°/frame at ~60 FPS. With `face_textures`: packed into a
square atlas (`_pack_texture_atlas`, UVs account for the numpy-vs-GL v-flip);
else `mesh.render_colors`. `MainWindow` hosts it; `set_mesh(mesh)` swaps it.

---

## `viz/params_dialog.py`

`ParamsDialog` (`QDialog`) for any `@paramclass` tree: reflects field `metadata`
into spinboxes (float/int), checkboxes (bool), "Fixed" checkbox + spinbox for
`kind: "optional_int"` (seeds), with labels/ranges/steps. `hidden` fields/sections
skipped; nested params → collapsible sections; wrapped in a `QScrollArea`.
`result_params(base) → T` returns an updated copy. `main.py` uses
`AppParamsDialog(AppParams)`.

---

## `viz/color_stops.py`

Stop list: `[(anchor, (r,g,b,a)), ...]`, ascending.

- `interp_stops_scalar(value, stops)` → RGBA tuple (scheme callables)
- `interp_stops(values, stops)` → `(…, 4) float32` (texture builders)
- `gradient_scheme(meta_key, stops, default=0.0)` → `SchemeFn` over `meta[meta_key]`
- `categorical_scheme(meta_key, palette, n, default=0)` → `SchemeFn` palette lookup
- `gradient_texture(channel_tex, stops)` → `(N,H,W,4)` from `(N,H,W,1)`
- `categorical_texture(idx_tex, palette, n)` → `(N,H,W,4)` from int index
- `palette_legend_entries(names, palette)` → `(name, rgba)` legend pairs
