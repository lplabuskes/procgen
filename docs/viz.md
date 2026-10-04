# Visualization Layer

The Qt viewers. Stages never import Qt: they fill a `DisplayRegistry`
([core.md](core.md)) and `viz/mapview/registry.py` wires it into the window. The
zoomed region / local-map view has its own doc: [viz_regionview.md](viz_regionview.md).

## Files
- `viz/mapview/mapview.py` — `MapWidget` (world map), `MapWindow` (the app window), `_WorldProjector`, `_split_at_antimeridian`
- `viz/mapview/regionview.py` — `RegionView`, `_RegionProjector` ([viz_regionview.md](viz_regionview.md))
- `viz/mapview/overlays.py` — `OverlayProjector`, `draw_overlays` and the per-feature renderers, shared by every view
- `viz/mapview/legend.py` — `legend_size`, `draw_legend`
- `viz/mapview/projections.py` — four map projections
- `viz/mapview/registry.py` — `apply_registry` (Qt adapter for `DisplayRegistry`)
- `viz/mapview/main.py` — dev entry point
- `viz/mesh/gui.py` — `MeshViewer`, `MainWindow` (3-D OpenGL globe viewer)
- `viz/params_dialog.py` — `ParamsDialog` (generic params UI)
- `viz/color_stops.py` — color interpolation utilities

The antimeridian choice lives in `core/antimeridian.py` (see [core.md](core.md)).

---

## `viz/mapview/mapview.py`

### `_split_at_antimeridian(face_ecef, face_uv=None, cut_lon=π)`

Splits a unit-sphere polygon at a cut meridian.

- `face_ecef: (N, 3)` ECEF vertices. `face_uv: (N, 2)` per-vertex UVs (optional).
- `cut_lon`: longitude in radians to treat as the antimeridian (default π = standard ±180°).
- Shifts all vertex longitudes so `cut_lon` maps to ±π, then splits edges that cross.
- Returns a list of `(M, 2)` lonlat arrays; or `((M, 2) lonlat, (M, 2) uv)` tuples when `face_uv` is given. One element if no crossing, two if crossing.

### `MapWidget` (`QWidget`)

`QPainter`-based 2-D projected polygon renderer (Robinson projection by default). Internally maintains parallel lists after each `set_mesh` or `set_globe` call:
- `_faces_lonlat` / `_face_src`: post-antimeridian-split lon/lat polygons and their source face index.
- `_proj_faces` / `_face_bbox_min` / `_face_bbox_max`: projected (u, v) polygons + bounding boxes for hover hit-testing.
- `_texture_qimages`: pre-built `QImage` per face when a texture scheme is active.

**Data pipeline per mesh load**: ECEF → `_split_at_antimeridian` at the world's `antimeridian` (per-vertex UVs included when any textures exist) → lon/lat storage → projection → screen paint. The map body (faces + overlays) is cached as a pixmap, so hover-only changes (the region tint) just re-blit it.

**Rendering modes** (priority order, same stack as face rendering in general):
1. Per-face texture: per-fan-triangle affine UV mapping via `QBrush(QImage)` + `QTransform`. Solve image-pixel → screen coords for each triangle, then `drawPolygon`.
2. Named color scheme: callable `(face_metadata dict → RGBA tuple)` evaluated once per polygon.
3. Static `face_colors` array.

**Texture scheme association**: `associate_scheme_texture(scheme_name, texture_key)` binds a named `mesh.face_textures` key to a color scheme. When that scheme is active, `_rebuild_textures` reads the bound texture.

**Region mode** (`set_region_mode`, the **Regions** toggle): tints the region cell under the cursor and draws the cell borders (`region.geometry.ecef_to_cell`, `cell_border_polylines`); a click calls `set_region_click_callback`'s callback with the region id.

**Legend**: gradient or categorical, rendered bottom-left by the shared `viz/mapview/legend.py` (`legend_size` / `draw_legend`, also used by `RegionView`). `register_legend(scheme_name, kind, entries)` where `kind` is `"gradient"` or `"categorical"` and `entries` is a list of `(label, rgba)`; `legend_for(name)` reads it back.

**Hover hit-testing**: vectorised bbox prefilter on all projected polygons → ray-casting PIP → barycentric UV interpolation within the hit fan triangle. `set_hover_callback(cb)` registers `cb(face_src_or_None, uv_or_None)`.

**Pan/zoom**: drag to pan; scroll wheel to zoom (1×–50×, anchored to cursor); `zoom_in()` / `zoom_out()` / `reset_view()`.

**Key public API**:
- `set_globe(world)` / `set_mesh(mesh, antimeridian=None)` — loads mesh (cut at `world.antimeridian`, or `find_ideal_antimeridian` for a bare mesh), rebuilds all caches
- `set_projection(proj)` — reprojects without mesh reload
- `set_color_schemes(schemes: dict[str, SchemeFn])` — replaces full scheme map; activates first scheme
- `set_color_scheme(name)` — switch active scheme, rebuilds textures + colors
- `associate_scheme_texture(scheme_name, texture_key)` — bind texture to scheme
- `set_overlay_visible(name, visible)` — toggle one overlay
- `set_hover_callback(cb)` / `set_region_click_callback(cb)`

### `MapWindow` (`QMainWindow`)

Hosts the views with a toolbar and bottom bar.

**Toolbar**: Save button, Coloration combo (scheme names), Overlays dropdown menu (one checkable `QAction` per overlay), Regions toggle, Back button ("← World Map" in a region, "← Region" on a local map), **Local maps** toggle (in a region: click a hex to open its local map), zoom −/+/Reset (routed to whichever view is showing).

**Views**: a `QStackedWidget` of the world map, a `RegionView` for the region and a second `RegionView` for local maps — see [viz_regionview.md](viz_regionview.md) for how they are entered, kept in sync and tuned.

**Bottom bar**: status label, Parameters button, Re-run button.

**Key public API**:
- `set_globe(world)` / `set_mesh(mesh)` — forwarded to `MapWidget`; a new globe drops back to the world map and clears the local-map cache
- `set_color_schemes(schemes)` — updates combo; preserves previously active scheme if still present
- `set_active_scheme(name)` — programmatic scheme switch; `active_scheme` property returns it
- `add_overlay(name, label, checked)` — idempotent; adds to Overlays menu on first call; one toggle drives the world map, the region view and the local view
- `register_legend(scheme_name, kind, entries)` / `associate_scheme_texture(scheme_name, texture_key)` — forwarded to `MapWidget`
- `associate_scheme_recipe(scheme_name, recipe)` — the scheme's `TextureRecipe`, which the zoomed views rerun on refined data
- `set_world_params(params)` / `set_realm_params(params)` — the tuning the world was built with (`GlobeParams` / `RealmParams`), which the zoomed views refine with
- `set_region_params(params)` / `set_local_params(params)` — tune the region / local tiers (`AppParams.region` / `AppParams.local`, view-side only); the map on screen is rebuilt, and a region change also rebuilds a local map on screen and drops cached ones
- `set_hover_parts(fn)` — `fn(face, uv, local) -> list[str]` status fragments, joined into the status bar by every view; the zoomed views pass `local` (see `DisplayRegistry.hover`) to substitute refined texel values for contributors tagged `refined=`
- `set_rebuild_callback(cb)` / `set_params_callback(cb)` / `set_save_callback(cb)` — wire Re-run / Parameters / Save buttons
- `set_status(text)` / `set_rebuild_enabled(enabled)` — status bar and button-enable helpers

---

## `viz/mapview/overlays.py` — feature overlays

The renderers are shared by the world map and the zoomed views: `draw_overlays(painter, globe, visible, projector, refined=None)` draws every visible overlay the globe has data for, in a fixed order, and each view supplies an `OverlayProjector` — `zoom` (world-map-equivalent magnification, for sizing), `view_zoom` and `px_per_km` (river widths), `regional` (towns draw only when True), `on_land` (settlements moved off refined water; identity on the world map), `points`, `segments` (with a keep mask), `split_segment` (trade routes) and `polygons`. The world map's `_WorldProjector` projects through the map projection and drops/splits at the antimeridian; `_RegionProjector` goes through the cell's local frame. `refined` replaces a feature with a zoomed view's finer version (see [viz_regionview.md](viz_regionview.md#overlays-in-a-zoomed-view)). Marker sizes, line widths and wind/current arrow lengths scale with the projector's `zoom`.

Drawing order and renderers (feature key → toggle registered by):

| Feature key | Renderer |
|---|---|
| `"rivers"` (hydrology) | Polyline segments; in-lake segments hidden; antimeridian skip. World map and region: width ∝ √(flow / world max) (0.5–4 px) × `view_zoom`^0.5 (× 1.5 in a region) — each level on its *own* zoom, not the equivalent world zoom; ≤ 1 px not drawn. Local map (projector `px_per_km` set): physical channel width from `worldgen/hydrology/discharge.py` (flow → discharge via mean precipitation × runoff 0.36; width = 7·Q^0.5 m), at least a hairline |
| `"lakes"` | Filled polygons (blue), antimeridian-split. No stage registers this toggle today |
| `"winds_season_N"` (climate, debug) | Line stubs per face centroid; length ∝ unit arc; skips antimeridian-crossing stubs |
| `"currents"` (climate, debug) | Line stubs; length ∝ speed; colour diverging blue→grey→red by temp_delta (17-bucket QPen cache) |
| `"deltas"` (hydrology, debug) | Green filled circles |
| `"volcanoes"` (elevation, debug) | Red/orange filled circles |
| `"minerals"` (resources, debug) | Coloured squares per mineral type, size ∝ abundance; type-palette from `minerals.MINERAL_PALETTE` |
| `"rare_bio"` (resources, debug) | Coloured diamonds per type, size ∝ abundance; type-palette from `rare_bio.RARE_BIO_PALETTE` |
| `"danger_sources"` (danger, debug) | Downward triangles coloured by source category |
| `"trade_routes"` (population) | Yellow lines for hops with flow ≥ the world median; width ∝ √(flow / max) (0.4–4 px) × zoom^0.4; antimeridian-split; ends follow settlements moved by `on_land` |
| `"villages"` (local maps only) | 1.3–2.4 px dots coloured by kind (farming, river, fishing, roadside), from `refined["villages"]` |
| `"towns"` (towns; region and local views only) | Kind-coloured dots, 0.9–1.6 px × zoom^0.3. Local map: ringed dark-then-white, as cities are in a region |
| `"settlements"` (population) | Tier-coloured circles; radius = `TIER_RADII[tier] · zoom^0.3 · (√(pop/max_pop)·0.4 + 0.6)`. Region: × 1.5, at least twice the largest town, ringed dark-then-white. Local map: each city's built-up footprint (`localmap/cities.py`, texels unioned on the pixel grid via `QRegion`, tier colour, dark outline); a footprint smaller than the floor marker (2 × the largest town radius) is drawn as that marker, "walled" — dark outline, white band, second dark ring, dark centre dot — so it never reads as a ringed town |
| `"polity_borders"` (polity) | Black lines by hierarchy depth (2.0 / 1.2 / 0.7 px × zoom^0.3, fainter deeper), sovereign borders on top |
| farmland | Local map only, no toggle of its own — see [viz_regionview.md](viz_regionview.md) |

---

## `viz/mapview/projections.py`

All projections share a `Projection` base class with `project(ecef_vertices)`, `project_lonlat(lonlat)` and `unproject_lonlat(uv)` (NaN outside the map; used for region picking). ECEF convention: Z is north, lon=0 at +X. Output is `float32 (N, 2)` in normalised map coordinates.

- **`EquirectangularProjection`**: u = lon/π, v = lat/(π/2). Output in [−1, 1]².
- **`MercatorProjection`**: Conformal cylindrical; latitudes clamped at ±85.05° so v stays in [−1, 1].
- **`MollweideProjection`**: Equal-area pseudocylindrical; auxiliary angle θ solved by 10 Newton iterations; u = √2·(lon/π)·cos θ, v = √2·sin θ.
- **`RobinsonProjection`**: Table-based compromise; parallel lengths and y-offsets interpolated from Robinson lookup tables at 5° latitude steps. The default.

`ALL_PROJECTIONS` lists them; `projection_by_name(name)` (case-insensitive) resolves `main.py --projection=NAME`, which is passed to `MapWindow(projection=...)` (Robinson when omitted).

---

## `viz/mapview/registry.py`

**`apply_registry(reg: DisplayRegistry, window: MapWindow, world: World)`**

The one place that knows both the `DisplayRegistry` shape and the `MapWindow` API. Stages contribute to `DisplayRegistry` (Qt-free); this function performs the actual wiring, keeping `core` and all stages free of Qt dependencies.

What it does:
1. `window.set_color_schemes(...)` — all schemes from `reg.schemes`, in registration order.
2. For each scheme: `associate_scheme_texture(...)` (with a `texture`), `associate_scheme_recipe(...)` (with a `recipe`), `register_legend(...)` (with a `legend`).
3. For each overlay in `reg.overlays`: `window.add_overlay(...)`, plus its legend.
4. Builds a decomposed hover function: `reg.hover(face_src, uv, world, local)` (concatenates all stage contributors' fragments), handed to `window.set_hover_parts(...)`.

---

## `viz/mapview/main.py` — Dev entry point

Creates an `icosphere(depth=4)`, deregularises it, computes its dual, builds a radial-gradient per-face texture tinted by each face's random colour, and opens a `MapWindow`. Used for testing the projection and rendering stack without running the full pipeline.

---

## `viz/mesh/gui.py` — 3-D OpenGL viewer

**`MeshViewer`** (`QOpenGLWidget`): Renders a `PolygonMesh` via VBOs and `glDrawArrays` (OpenGL 3.3 compatibility profile).

- Rotates 0.5°/frame via a `QTimer` at ~60 FPS.
- When the mesh has `face_textures`: packs all per-face textures into a square grid atlas (`_pack_texture_atlas`), uploads as a single `GL_TEXTURE_2D`, and uses a UV VBO. Atlas UV coordinates account for the v-flip between numpy (row 0 = top) and OpenGL (v=0 = bottom).
- Without textures: uses the color-array VBO path (`mesh.render_colors`).

**`MainWindow`** (`QMainWindow`): Hosts `MeshViewer`; exposes `set_mesh(mesh)` to swap the displayed mesh at runtime.

---

## `viz/params_dialog.py`

**`ParamsDialog`** (`QDialog`): Generic UI dialog for any `@paramclass` dataclass tree.

Reflects over the dataclass fields and their `metadata` dict to render spinboxes (float / int), checkboxes (bool), and a "Fixed" checkbox + spinbox for `kind: "optional_int"` fields (the seeds), with labels, ranges, and step sizes. Fields or sections with `hidden` metadata are skipped. Nested sub-params become collapsible sections (arrow-toggle). The whole form is wrapped in a `QScrollArea`.

`result_params(base: T) → T`: returns an updated copy of the params dataclass with values read from the dialog widgets.

Used by `main.py` via `AppParamsDialog(AppParams)`.

---

## `viz/color_stops.py`

Stop-based color interpolation utilities. A *stop list* is `[(anchor_float, (r,g,b,a)), ...]` in ascending anchor order.

- **`interp_stops_scalar(value, stops) → tuple`**: Linear interpolation for a scalar; used in per-face color scheme callables (returns RGBA tuple).
- **`interp_stops(values: ndarray, stops) → (…, 4) float32`**: Vectorised version for arrays; used in texture builders.
- **`gradient_scheme(meta_key, stops, default=0.0)`**: Returns a `SchemeFn` that reads `meta[meta_key]` as float and interpolates through stops. Convenience wrapper used across `display.py` modules.
- **`categorical_scheme(meta_key, palette, n, default=0)`**: Returns a `SchemeFn` that reads `meta[meta_key]` as int and looks up in `palette`.
- **`gradient_texture(channel_tex, stops) → (N,H,W,4) float32`**: `(N,H,W,1)` single-channel → RGBA via stops. Used in texture-based display (e.g. elevation, biome RGBA textures).
- **`categorical_texture(idx_tex, palette, n) → (N,H,W,4) float32`**: `(N,H,W,1)` int index → RGBA via palette lookup.
- **`palette_legend_entries(names, palette)`**: `(name, rgba)` legend pairs for a categorical palette.

---

## Face rendering priority

Three layers in priority order (same in both viewers): per-face textures → named color schemes (callables over `face_metadata`) → static `face_colors` array. Both viewers fall back down this stack when the higher layer is absent.
