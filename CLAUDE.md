# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Style

- Ask for clarification rather than churning your wheels. If the task or approach is ambiguous, ask before attempting.


## Running the Project

```bash
source .venv/bin/activate
python3 -m worldgen.mesh.main       # 3-D mesh viewer
python3 -m worldgen.mapview.main    # 2-D map viewer
python3 -m worldgen.geology.main    # plate tectonics map viewer
```

No build step or linter is configured. Run the test suites with:

```bash
python3 -m pytest                                         # all tests (testpaths in pyproject.toml)
python3 -m pytest worldgen/mesh/test/test_mymesh.py -v
python3 -m pytest worldgen/mapview/test/test_mapview.py -v
```

`mesh/test/test_mymesh.py` covers `Mesh` (icosphere topology, unit-sphere property, outward face normals) and `Mesh.mesh_dual` (dual topology counts, face sizes, and correct CCW winding of all fan triangles).

`mapview/test/test_mapview.py` covers all four projections (equator/meridian anchors, N/S and E/W symmetry, projection-specific bounds, `project` vs `project_lonlat` consistency) and `_split_at_antimeridian` (no-crossing passthrough, two-polygon split, boundary latitude continuity).

## Architecture

This is a procedural world generation project. The core pipeline is: generate a mesh → deregularize it → compute its dual → display it. A geology layer runs on top of the dual mesh to simulate tectonic plates.

### `worldgen/mesh/mymesh.py` — Mesh data structures

- **`PolygonMesh`**: Base class for any polygon mesh. Fan-triangulates faces and pre-computes `render_vertices` and `render_colors` (flat per-vertex arrays) at construction time for direct VBO upload. Also pre-computes `face_adjacency` (edge-sharing neighbor lists), `face_centroids` (unit-sphere centroids, shape `(N, 3)`), and `face_metadata` (list of per-face `dict`s). The triangulation step also builds `render_face_indices` (per-render-vertex face index, shape `(M,)`) and `render_local_uvs` (per-render-vertex UV in [0, 1]², shape `(M, 2)`) using a tangent-plane parameterization at each face centroid — these are consumed by the texture pipeline. Exposes `set_face_colors(colors)` to update colors and rebuild render arrays, `set_face_metadata(face_idx, **kw)` to write arbitrary attributes into a face's metadata dict, `apply_metadata_channel(key, values)` to bulk-write a single key across all faces at once (raises `ValueError` if `len(values)` doesn't match the face count), `set_face_textures(textures)` to attach a `(N, H, W, 4)` float32 RGBA texture array (one per face), and `face_texture_positions(h, w)` to get the unit-sphere 3D position of every pixel in a `(H, W)` texture grid — returns `float32 (N_faces, H, W, 3)` by inverting the same tangent-plane parameterization used for UVs.
- **`Mesh`** (subclass of `PolygonMesh`): Triangular mesh. `icosphere(depth)` generates a subdivided icosahedron; vertices are sorted along the Z-axis for spatial locality. `.deregularize(n_swaps, n_relax, stiffness, outer_iters)` breaks regularity via random edge-swaps and spring relaxation on the unit sphere. `.mesh_dual()` returns a `PolygonMesh` by placing new vertices at face centroids (normalized to unit sphere), then for each original vertex builds a CCW-wound face from the centroids of its adjacent faces.

### `worldgen/mesh/gui.py` — OpenGL/Qt renderer

- **`MeshViewer`** (`QOpenGLWidget`): Renders a `PolygonMesh` using VBOs and `glDrawArrays` (OpenGL 3.3 compatibility). Drives ~60 FPS via a timer and rotates the mesh 0.5°/frame. When the mesh has `face_textures`, `_setup_buffers` packs all per-face textures into a square grid atlas (rows × cols), uploads it as a single `GL_TEXTURE_2D`, and substitutes a UV VBO (`uv_buffer`) for the color VBO; the atlas UV coordinates account for the v-flip between numpy (row 0 = top) and OpenGL (v=0 = bottom). Without textures, the original color-array path is used.
- **`MainWindow`** (`QMainWindow`): Hosts `MeshViewer`; exposes `set_mesh(mesh)` to swap the displayed mesh at runtime (calls `makeCurrent`/`doneCurrent` around buffer setup).

### `worldgen/mesh/main.py` — Entry point

Creates an `icosphere(depth=4)`, deregularizes it, computes its dual, builds a radial-gradient per-face texture tinted by each face's random color, and opens the viewer.

### `worldgen/mapview/projections.py` — Map projections

All projections share a `Projection` base class with `project(ecef_vertices)` and `project_lonlat(lonlat)` methods (ECEF convention: Z is north, lon=0 at +X). Output is `float32 (N, 2)` in normalized map coordinates.

- **`EquirectangularProjection`**: u = lon/π, v = lat/(π/2). Output in [−1, 1]².
- **`MercatorProjection`**: Conformal cylindrical; latitudes clamped at ±85.05° so v stays in [−1, 1].
- **`MollweideProjection`**: Equal-area pseudocylindrical; solved via Newton iteration for auxiliary angle θ. Fits an ellipse with semi-axes √2 (u) and √2 (v).
- **`RobinsonProjection`**: Table-based compromise; parallel lengths and y-offsets interpolated from the original Robinson lookup tables at 5° latitude steps.

### `worldgen/mapview/mapview.py` — Qt map widget

- **`_split_at_antimeridian(face_ecef, face_uv=None)`**: Splits a unit-sphere polygon at the ±180° meridian. Without `face_uv`, returns a list of one or two `(M, 2)` lon/lat arrays. With `face_uv` (`(N, 2)` per-vertex UVs), returns a list of `(lonlat, uv)` tuples — UVs are interpolated at crossing points.
- **`_compute_face_uv(face_ecef)`**: Computes tangent-plane UV coordinates for a face's vertices, returning `(N, 2) float32` in [0, 1]. Used to parameterize polygons for texture mapping.
- **`MapWidget`** (`QWidget`): Paints projected polygons using `QPainter`. Supports pan (drag) and zoom (scroll wheel or ±/Reset buttons). Zoom is clamped to [1×, 50×]. Accepts named color schemes via `set_color_schemes(schemes)` where each scheme is a `dict[str, Callable[[dict], RGBA]]` mapping a display name to a function that receives a face's metadata dict and returns an RGBA tuple. `set_color_scheme(name)` switches the active scheme. When the mesh has `face_textures`, textures are converted to `QImage`s in `_rebuild_textures`, and painting uses per-triangle affine texture mapping via `QBrush` with `QTransform` (solving texture→screen for each fan triangle). Falls back to solid-color `QPolygonF` rendering when no textures are present.
- **`MapWindow`** (`QMainWindow`): Hosts `MapWidget` with a projection combo-box toolbar. `set_color_schemes(schemes)` populates a coloration combo box in the toolbar (hidden when no schemes are registered). `set_rebuild_callback(callback)` wires a Re-run button (hidden by default) that fires `callback` when clicked — used by `geology.main` to regenerate the world in place.

### `worldgen/mapview/main.py` — Entry point

Creates an `icosphere(depth=4)`, deregularizes it, computes its dual, builds a radial-gradient per-face texture tinted by each face's random color, and opens the map viewer.

### `worldgen/geology/elevation.py` — Elevation model

- **`compute_face_elevations(mesh, plate_types_by_face, boundary_data, decay_rate, max_hops, rate_norm)`**: Computes elevation (km) per face. Baseline is set by plate type (`'oceanic'` → −4.0 km, `'continental'` → +0.3 km). Each boundary face receives a delta based on boundary type, convergence rate, and the types of neighboring plates (e.g., continental-continental convergence → +4 km mountain range, oceanic subducting under continental → −2.5 km trench on the oceanic side). Deltas propagate outward from boundary faces via BFS with exponential decay (`exp(-decay_rate * hops)`) up to `max_hops` steps; the maximum-magnitude contribution wins at each face. Returns a `float64` array of shape `(n_faces,)`.

### `worldgen/geology/globe_builder.py` — High-level pipeline orchestration

- **`GlobeParams`** (dataclass): Centralizes every tuning knob in the pipeline. Groups: *mesh generation* (`icosphere_depth`, `n_swaps`, `n_relax`, `stiffness`, `outer_iters`), *plate layout* (`n_plates`, `plate_compactness`, `relax_max_passes`, `oceanic_fraction`), *plate motion* (`drift_scale`, `spin_scale`), *elevation* (`elevation_rate_norm`, `elevation_decay_rate`, `elevation_max_hops`), and *reproducibility* (`seed`). A top-level `seed` fans out to per-stage sub-seeds (`seed+0` for plate generation, `seed+1` for plate types, `seed+2` for velocities); `None` means random each run.
- **`GlobeBuilder`**: Owns a `GlobeParams` and runs the full pipeline when `build()` is called. Returns the dual `PolygonMesh` with all metadata applied and stores intermediate results as attributes: `mesh`, `plate_ids`, `plate_omegas`, `face_velocities`, `boundary_types`, `boundary_data`, `elevations`, `max_speed`. `boundary_data` is the richer `List[dict]` from `classify_plate_boundaries` (with `normal_avg` and `neighbor_plate_types`); `boundary_types` is the `List[str]` derived from it. Callers (e.g. visualization code) read these attributes without re-running any work.

### `worldgen/geology/tectonics.py` — Plate tectonics simulation

- **`generate_plates(mesh, n_plates, seed, compactness)`**: Assigns each face to a plate via per-plate randomized-priority flood fill. Each step picks a plate weighted by a random draw (so no plate gets a compounding size advantage) and expands it using a distance-biased heap, producing compact plates with irregular borders. Post-fill, any plate wholly surrounded by a single other plate is absorbed into it. Returns `List[int]` of plate IDs indexed by face.
- **`relax_plates(mesh, plate_ids, max_passes)`**: Iteratively removes appendages using articulation point detection (`_tarjan_aps_iterative`). Each pass finds faces reachable only via a single articulation point and reassigns them to whichever neighboring plate they border most. Repeats until stable.
- **`generate_plate_velocities(mesh, plate_ids, n_plates, drift_scale, spin_scale, seed)`**: Returns `(n_plates, 3)` angular velocity vectors. Each plate's omega combines a random drift component (tangential to the plate centroid, i.e., pure translation across the sphere) and a spin component (along the centroid axis). The velocity of a face on plate k is `omega[k] × face_centroid`.
- **`assign_plate_types(n_plates, oceanic_fraction, seed)`**: Classifies each plate as `'oceanic'` or `'continental'`. Default `oceanic_fraction=0.7` matches Earth's rough ratio.
- **`classify_plate_boundaries(mesh, plate_ids, plate_omegas)`**: Returns `List[dict]` (one per face) with keys `boundary_type` (`'interior'|'convergent'|'divergent'|'transform'`), `normal_avg` (mean normal velocity component across inter-plate edges; positive = diverging), and `neighbor_plate_types` (set of plate type strings seen across those edges). Interior faces get zeros/empty set.

### `worldgen/geology/main.py` — Entry point

Creates a `GlobeBuilder(GlobeParams())`, calls `build()`, then sets up `MapWindow` with five color schemes: **Elevation** (multi-stop gradient from abyssal ocean blues through beach/lowland/highland to snow-capped peaks, with a hard break at sea level), **Plates** (HSV hue-wheel palette by plate ID), **Plate Type** (blue for oceanic, tan for continental), **Speed** (warm-to-cool by surface speed), and **Boundary** (color-coded by convergent/divergent/transform/interior). The window is created first and a Re-run callback is registered so the world can be regenerated in place without restarting.

## Key Design Notes

- `Mesh` stores only triangular faces; `PolygonMesh` handles arbitrary polygon faces needed for duals (e.g., a depth-1 icosphere dual has 12 pentagons + 30 hexagons).
- The dual construction normalizes centroids to the unit sphere so the dual lives on the same sphere as the original, not scaled by face area.
- Both classes pre-expand geometry into flat `render_vertices`/`render_colors` arrays at construction time; the renderer calls `glDrawArrays` with no index buffer.
- OpenGL context is set to 3.3 compatibility profile (not core) so legacy calls remain available during development.
- Edge-swaps in `deregularize` are rejected if they would create a vertex with degree ≤ 4 (which produces squares in the dual) or ≥ 8 (octagons).
- `PolygonMesh` pre-computes `face_adjacency` and `face_centroids` at construction and after `deregularize`; geology code relies on these without recomputing them.
- Face coloring has two layers: `face_colors` (static RGBA array, used when no active scheme is set) and named color schemes (callables over `face_metadata`, registered via `set_color_schemes`). The color scheme layer takes priority in `MapWidget`.
- Textures are an optional third rendering layer. `set_face_textures(textures)` attaches a `(N, H, W, 4) float32` array. In the 3-D viewer, all face textures are packed into a square atlas and rendered via a UV VBO. In the map viewer, each face texture becomes a `QImage` and is painted per-triangle using affine UV mapping (`QBrush` + `QTransform`). UV coordinates are computed via tangent-plane projection at each face centroid, stored in `render_local_uvs` and passed through `_split_at_antimeridian` when present.
- `face_texture_positions(h, w)` is the canonical way to get the unit-sphere 3D position of every pixel in a face texture. It inverts the tangent-plane UV parameterization (stored as private `_tangent_centroids`, `_u_axes`, `_v_axes`, `_u_offsets`, `_v_offsets`, `_u_ranges`, `_v_ranges` arrays) and returns `float32 (N_faces, H, W, 3)`. Use this as the starting point for any per-pixel geographic or physical computation (e.g. sampling elevation, computing lat/lon, etc.).
- `face_tangent_frames()` exposes the raw per-face tangent-plane data as a 7-tuple `(u_axes, v_axes, u_offsets, v_offsets, u_ranges, v_ranges, vertex_uvs)` — use this to project arbitrary 3D points into a face's UV space without re-deriving the frame. `vertex_uvs` is a list of `(n_v, 2) float32` arrays (one per face). The face centroid maps to UV `(-u_offsets[i]/u_ranges[i], -v_offsets[i]/v_ranges[i])`.
