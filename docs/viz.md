# Visualization Layer

## Files
- `viz/mapview/mapview.py` — `MapWidget`, `MapWindow`, `_split_at_antimeridian`
- `viz/mapview/antimeridian.py` — `find_ideal_antimeridian`
- `viz/mapview/projections.py` — four map projections
- `viz/mapview/registry.py` — `apply_registry` (Qt adapter for `DisplayRegistry`)
- `viz/mapview/main.py` — dev entry point
- `viz/mesh/gui.py` — `MeshViewer`, `MainWindow` (3-D OpenGL globe viewer)
- `viz/params_dialog.py` — `ParamsDialog` (generic params UI)
- `viz/color_stops.py` — color interpolation utilities

---

## `viz/mapview/mapview.py`

### `_split_at_antimeridian(face_ecef, face_uv=None, cut_lon=π)`

Splits a unit-sphere polygon at a cut meridian.

- `face_ecef: (N, 3)` ECEF vertices. `face_uv: (N, 2)` per-vertex UVs (optional).
- `cut_lon`: longitude in radians to treat as the antimeridian (default π = standard ±180°).
- Shifts all vertex longitudes so `cut_lon` maps to ±π, then splits edges that cross.
- Returns a list of `(M, 2)` lonlat arrays; or `((M, 2) lonlat, (M, 2) uv)` tuples when `face_uv` is given. One element if no crossing, two if crossing.

### `MapWidget` (`QWidget`)

`QPainter`-based 2-D projected polygon renderer. Internally maintains two parallel lists after each `set_mesh` or `set_globe` call:
- `_faces_lonlat` / `_face_src`: post-antimeridian-split lon/lat polygons and their source face index.
- `_proj_faces` / `_face_bbox_min` / `_face_bbox_max`: projected (u, v) polygons + bounding boxes for hover hit-testing.
- `_texture_qimages`: pre-built `QImage` per face when a texture scheme is active.

**Data pipeline per mesh load**: ECEF → `_split_at_antimeridian` (per-vertex UVs included when any textures exist) → lon/lat storage → projection → screen paint.

**Rendering modes** (priority order, same stack as face rendering in general):
1. Per-face texture: per-fan-triangle affine UV mapping via `QBrush(QImage)` + `QTransform`. Solve image-pixel → screen coords for each triangle, then `drawPolygon`.
2. Named color scheme: callable `(face_metadata dict → RGBA tuple)` evaluated once per polygon.
3. Static `face_colors` array.

**Texture scheme association**: `associate_scheme_texture(scheme_name, texture_key)` binds a named `mesh.face_textures` key to a color scheme. When that scheme is active, `_rebuild_textures` reads the bound texture rather than the first one.

**Overlays** (drawn in `paintEvent` after faces, with antialiasing on). The renderers live in `viz/mapview/overlays.py` and are shared with `RegionView`: `draw_overlays(painter, globe, visible, projector)` draws every visible overlay in a fixed order, and each view supplies an `OverlayProjector` — `zoom` (for sizing), `view_zoom` and `px_per_km` (river widths), `points`, `segments` (with a keep mask), `split_segment` (trade routes) and `polygons`. The world map's `_WorldProjector` projects through the map projection and drops/splits at the antimeridian; the region view's `_RegionProjector` goes through the cell's local frame. Marker sizes, line widths and wind/current arrow lengths scale with the projector's `zoom`:
| Feature key | Renderer |
|---|---|
| `"rivers"` | Polyline segments; in-lake segments hidden; antimeridian skip. World map and region: width ∝ √(flow / world max) (0.5–4 px) × `view_zoom`^0.5 (× 1.5 in a region) — each level on its *own* zoom, not the equivalent world zoom; ≤ 1 px not drawn. Local map (projector `px_per_km` set): physical channel width from `worldgen/hydrology/discharge.py` (flow → discharge via mean precipitation × runoff 0.36; width = 7·Q^0.5 m), at least a hairline |
| `"lakes"` | Filled polygons (blue), antimeridian-split |
| `"winds_season_N"` | Line stubs per face centroid; length ∝ unit arc; skips antimeridian-crossing stubs |
| `"currents"` | Line stubs; length ∝ speed; colour diverging blue→grey→red by temp_delta (17-bucket QPen cache) |
| `"deltas"` | Green filled circles |
| `"volcanoes"` | Red/orange filled circles |
| `"minerals"` | Coloured squares per mineral type, size ∝ abundance; type-palette from `minerals.MINERAL_PALETTE` |
| `"rare_bio"` | Coloured diamonds per type, size ∝ abundance; type-palette from `rare_bio.RARE_BIO_PALETTE` |
| `"trade_routes"` | Yellow lines, width ∝ √flow (0.4–4 px), antimeridian-split |
| `"settlements"` | Tier-coloured circles; radius = `TIER_RADII[tier] * zoom^0.3 * √(pop/max_pop)*0.4+0.6`. Region: × 1.5, at least twice the largest town, ringed dark-then-white. Local map: each city's built-up footprint (`localmap/cities.py`, texels unioned on the pixel grid via `QRegion`, tier colour, dark outline); a footprint smaller than the floor marker (2 × the largest town radius) is drawn as that marker, "walled" — dark outline, white band, second dark ring, dark centre dot — so it never reads as a ringed town. With it, the city's market-garden ring (farmland, below) |
| `"towns"` | Region and local map only: kind-coloured dots, 0.9–1.6 px × zoom^0.3. Local map: ringed dark-then-white, as cities are in a region, with the town fields (farmland, below) |
| farmland | Local map only, no toggle of its own: each kind shows with its settlements' overlay — village fields with **Villages**, town fields with **Towns**, market gardens with **Settlements**. Baked per hexagon into translucent images (`localmap.farmland.farmland_rgba`: wheat fields, dark-olive gardens, ±10% per-texel patchwork) by `RegionView._rebuild_farm_images` on opening the map or toggling one of those overlays, and drawn over the terrain, under the overlays |

**Legend**: gradient or categorical, rendered bottom-left by the shared `viz/mapview/legend.py` (`legend_size` / `draw_legend`, also used by `RegionView`). `register_legend(scheme_name, kind, entries)` where `kind` is `"gradient"` or `"categorical"` and `entries` is a list of `(label, rgba)`.

**Hover hit-testing**: vectorised bbox prefilter on all projected polygons → ray-casting PIP → barycentric UV interpolation within the hit fan triangle. `set_hover_callback(cb)` registers `cb(face_src_or_None, uv_or_None)`.

**Pan/zoom**: drag to pan; scroll wheel to zoom (1×–50×, anchored to cursor); `zoom_in()` / `zoom_out()` / `reset_view()`.

**Key public API**:
- `set_globe(world)` / `set_mesh(mesh)` — loads mesh, computes ideal antimeridian, rebuilds all caches
- `set_projection(proj)` — reprojects without mesh reload
- `set_color_schemes(schemes: dict[str, SchemeFn])` — replaces full scheme map; activates first scheme
- `set_color_scheme(name)` — switch active scheme, rebuilds textures + colors
- `associate_scheme_texture(scheme_name, texture_key)` — bind texture to scheme
- `set_overlay_visible(name, visible)` — toggle one overlay
- `set_hover_callback(cb)` — register hover handler

### `MapWindow` (`QMainWindow`)

Hosts `MapWidget` with a toolbar and bottom bar.

**Toolbar**: Save button, Projection combo (`ALL_PROJECTIONS`), Coloration combo (scheme names), Overlays dropdown menu (one checkable `QAction` per overlay), Regions toggle, Back button ("← World Map" in a region, "← Region" on a local map), **Local maps** toggle (in a region: click a hex to open its local map), zoom −/+/Reset (routed to whichever zoomed view is showing).

**Views**: a `QStackedWidget` of the world map, a `RegionView` for the region and a second `RegionView` for local maps, so going back from a local map shows the region untouched. `_enter_local(hex)` builds (or takes from the local view's LRU cache) the hex's local map from the region view's fields, rivers, roads and deltas; `_go_back()` steps local → region → world. The **Villages** overlay toggle is added with the first local map.

**Bottom bar**: status label, Parameters button, Re-run button.

**Key public API**:
- `set_globe(world)` / `set_mesh(mesh)` — forwarded to `MapWidget`
- `set_color_schemes(schemes)` — updates combo; preserves previously active scheme if still present
- `set_active_scheme(name)` — programmatic scheme switch
- `add_overlay(name, label, checked)` — idempotent; adds to Overlays menu on first call; one toggle drives the world map, the region view and the local view
- `set_region_params(params)` / `set_local_params(params)` — tune the region / local tiers (`AppParams.region` / `AppParams.local`, view-side only); the map on screen is rebuilt, and a region change also rebuilds a local map on screen and drops cached ones
- `register_legend(scheme_name, kind, entries)` — forwarded to `MapWidget`
- `associate_scheme_texture(scheme_name, texture_key)` — forwarded to `MapWidget`
- `set_hover_parts(fn)` — `fn(face, uv, local) -> list[str]` status fragments, joined into the status bar by both views; the region view passes `local` (see `DisplayRegistry.hover`) to substitute refined texel values for contributors tagged `refined=`
- `set_rebuild_callback(cb)` / `set_params_callback(cb)` / `set_save_callback(cb)` — wire Re-run / Parameters / Save buttons
- `set_status(text)` / `set_rebuild_enabled(enabled)` — status bar and button-enable helpers
- `active_scheme` property — returns current active scheme name

---

## `viz/mapview/antimeridian.py`

**`find_ideal_antimeridian(mesh, n_samples=360) → float`**

Returns the cut longitude (radians) that minimises land-face crossings when splitting polygons at the antimeridian.

Algorithm: samples 360 candidate longitudes uniformly in [−π, π); for each, counts land faces whose vertex longitude span exceeds π after shifting. Among candidates tied for fewest crossings, picks the one that maximises the minimum angular distance to any land face centroid — i.e. the cut farthest from land. Falls back to π when the mesh has no `is_ocean` metadata.

Called by `MapWidget.set_mesh` on every mesh load; the chosen cut is stored as `_antimeridian` and passed to `_split_at_antimeridian` for all subsequent face splits and overlay rendering.

---

## `viz/mapview/projections.py`

All projections share a `Projection` base class with `project(ecef_vertices)` and `project_lonlat(lonlat)` methods. ECEF convention: Z is north, lon=0 at +X. Output is `float32 (N, 2)` in normalised map coordinates.

- **`EquirectangularProjection`**: u = lon/π, v = lat/(π/2). Output in [−1, 1]².
- **`MercatorProjection`**: Conformal cylindrical; latitudes clamped at ±85.05° so v stays in [−1, 1].
- **`MollweideProjection`**: Equal-area pseudocylindrical; auxiliary angle θ solved via Newton iteration. Semi-axes √2 (u) and √2/2·π? — fits an ellipse.
- **`RobinsonProjection`**: Table-based compromise; parallel lengths and y-offsets interpolated from Robinson lookup tables at 5° latitude steps.

`ALL_PROJECTIONS` is the ordered list used to populate the projection combo.

---

## `viz/mapview/registry.py`

**`apply_registry(reg: DisplayRegistry, window: MapWindow, world: World)`**

The one place that knows both the `DisplayRegistry` shape and the `MapWindow` API. Stages contribute to `DisplayRegistry` (Qt-free); this function performs the actual wiring, keeping `core` and all stages free of Qt dependencies.

What it does:
1. `window.set_color_schemes(...)` — all schemes from `reg.schemes`.
2. For each scheme with a `texture` key: `window.associate_scheme_texture(...)`.
3. For each scheme/overlay with a `legend`: `window.register_legend(...)`.
4. For each overlay in `reg.overlays`: `window.add_overlay(...)`.
5. Builds a decomposed hover function: `reg.hover(face_src, uv, world, local)` (concatenates all stage contributors' fragments), handed to `window.set_hover_parts(...)`.

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

Reflects over the dataclass fields and their `metadata` dict to render spinboxes (float / int) and checkboxes (bool) with labels, ranges, and step sizes. Nested sub-params become collapsible sections (arrow-toggle). The whole form is wrapped in a `QScrollArea`.

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

---

## Face rendering priority

Three layers in priority order (same in both viewers): per-face textures → named color schemes (callables over `face_metadata`) → static `face_colors` array. Both viewers fall back down this stack when the higher layer is absent.
