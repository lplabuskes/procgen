# Core — Mesh Data Structures & Visualization

## `core/mesh.py` — Mesh data structures

- **`PolygonMesh`**: Base class for any polygon mesh. Pre-computes `face_adjacency`, `face_centroids`, `face_metadata`, and fan-triangulated render arrays at construction. Key API:
  - `set_face_colors(colors)`
  - `set_face_metadata(face_idx, **kw)`
  - `apply_metadata_channel(key, values)` — bulk-write one metadata key across all faces
  - `set_face_textures(name, textures, n_channels=4)` — attach `(N, H, W, C)` float32; use `n_channels=1` for data layers like elevation, 4 for RGBA display
  - `face_texture_positions(h, w)` — unit-sphere 3D position of every texture pixel; starting point for any per-pixel geographic computation
  - `face_tangent_frames()` — raw per-face tangent-plane data for projecting arbitrary 3D points into a face's UV space

- **`Mesh`** (subclass of `PolygonMesh`): Triangular mesh.
  - `icosphere(depth)` — generates a subdivided icosahedron; vertices sorted along the Z-axis for spatial locality
  - `.deregularize(n_swaps, n_relax, stiffness, outer_iters)` — breaks regularity via random edge-swaps and spring relaxation on the unit sphere
  - `.mesh_dual()` — returns a `PolygonMesh` by placing new vertices at face centroids (normalized to unit sphere), then for each original vertex builds a CCW-wound face from the centroids of its adjacent faces

**Key invariant:** `Mesh` stores only triangular faces; `PolygonMesh` handles arbitrary polygon faces (e.g., the dual has 12 pentagons + many hexagons). The dual normalizes centroids to the unit sphere.

## `viz/mesh/gui.py` — OpenGL/Qt renderer

- **`MeshViewer`** (`QOpenGLWidget`): Renders a `PolygonMesh` using VBOs and `glDrawArrays` (OpenGL 3.3 compatibility). ~60 FPS via a timer, rotates the mesh 0.5°/frame. When the mesh has `face_textures`, `_setup_buffers` packs all per-face textures into a square grid atlas, uploads it as a single `GL_TEXTURE_2D`, and substitutes a UV VBO for the color VBO; atlas UV coordinates account for the v-flip between numpy (row 0 = top) and OpenGL (v=0 = bottom). Without textures, the original color-array path is used.
- **`MainWindow`** (`QMainWindow`): Hosts `MeshViewer`; exposes `set_mesh(mesh)` to swap the displayed mesh at runtime.

## `viz/mapview/projections.py` — Map projections

All projections share a `Projection` base class with `project(ecef_vertices)` and `project_lonlat(lonlat)` methods (ECEF convention: Z is north, lon=0 at +X). Output is `float32 (N, 2)` in normalized map coordinates.

- **`EquirectangularProjection`**: u = lon/π, v = lat/(π/2). Output in [−1, 1]².
- **`MercatorProjection`**: Conformal cylindrical; latitudes clamped at ±85.05° so v stays in [−1, 1].
- **`MollweideProjection`**: Equal-area pseudocylindrical; solved via Newton iteration for auxiliary angle θ. Fits an ellipse with semi-axes √2 (u) and √2 (v).
- **`RobinsonProjection`**: Table-based compromise; parallel lengths and y-offsets interpolated from the original Robinson lookup tables at 5° latitude steps.

## `viz/mapview/mapview.py` — Qt map widget

- **`_split_at_antimeridian(face_ecef, face_uv=None)`**: Splits a unit-sphere polygon at the ±180° meridian; with `face_uv`, also interpolates UVs at crossing points and returns `(lonlat, uv)` pairs.
- **`MapWidget`** (`QWidget`): `QPainter`-based projected polygon renderer with pan/zoom. Renders faces as solid color, named color schemes (callables over `face_metadata`), or per-face textures (affine UV mapping via `QBrush`/`QTransform` per fan triangle). Draws vector overlays from `Globe.features` on top; the `"rivers"` overlay scales line width by flow and skips `in_lake` segments. Hover hit-testing via bbox pre-filter + point-in-polygon + barycentric UV (`set_hover_callback`).
- **`MapWindow`** (`QMainWindow`): Hosts `MapWidget` with projection selector, color scheme selector, overlay checkboxes, legend panel, Re-run/Params/Save buttons, and status bar.

## `viz/mapview/main.py` — Dev entry point

Creates an `icosphere(depth=4)`, deregularizes it, computes its dual, builds a radial-gradient per-face texture tinted by each face's random color, and opens the map viewer.

## Face rendering priority

Three layers in priority order: per-face textures → named color schemes (callables over `face_metadata`) → static `face_colors` array. Both viewers fall back down this stack when the higher layer is absent.

## Tangent frame convention

Per-face tangent-plane axes are `u_axis = cross(z, n)`, `v_axis = cross(n, u_axis)`, so the texture U-axis is geographic east and V-axis is geographic north for ~90% of faces. Any D4 grid-walk algorithm will produce N-S/E-W-biased output; `rivers.py` avoids this with D-infinity accumulation and sub-pixel gradient traces rather than grid-snapped steps.
