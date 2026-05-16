# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

```bash
source .venv/bin/activate
cd worldgen && python3 -m mesh.main       # 3-D mesh viewer
cd worldgen && python3 -m mapview.main    # 2-D map viewer
cd worldgen && python3 -m geology.main    # plate tectonics map viewer
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

- **`PolygonMesh`**: Base class for any polygon mesh. Fan-triangulates faces and pre-computes `render_vertices` and `render_colors` (flat per-vertex arrays) at construction time for direct VBO upload. Also pre-computes `face_adjacency` (edge-sharing neighbor lists), `face_centroids` (unit-sphere centroids, shape `(N, 3)`), and `face_metadata` (list of per-face `dict`s). Exposes `set_face_colors(colors)` to update colors and rebuild render arrays, `set_face_metadata(face_idx, **kw)` to write arbitrary attributes into a face's metadata dict, and `apply_metadata_channel(key, values)` to bulk-write a single key across all faces at once (raises `ValueError` if `len(values)` doesn't match the face count).
- **`Mesh`** (subclass of `PolygonMesh`): Triangular mesh. `icosphere(depth)` generates a subdivided icosahedron; vertices are sorted along the Z-axis for spatial locality. `.deregularize(n_swaps, n_relax, stiffness, outer_iters)` breaks regularity via random edge-swaps and spring relaxation on the unit sphere. `.mesh_dual()` returns a `PolygonMesh` by placing new vertices at face centroids (normalized to unit sphere), then for each original vertex builds a CCW-wound face from the centroids of its adjacent faces.

### `worldgen/mesh/gui.py` — OpenGL/Qt renderer

- **`MeshViewer`** (`QOpenGLWidget`): Renders a `PolygonMesh` using VBOs and `glDrawArrays` (OpenGL 3.3 compatibility). Drives ~60 FPS via a timer and rotates the mesh 0.5°/frame.
- **`MainWindow`** (`QMainWindow`): Hosts `MeshViewer`; exposes `set_mesh(mesh)` to swap the displayed mesh at runtime.

### `worldgen/mesh/main.py` — Entry point

Creates an `icosphere(depth=4)`, deregularizes it, computes its dual, and opens the viewer.

### `worldgen/mapview/projections.py` — Map projections

All projections share a `Projection` base class with `project(ecef_vertices)` and `project_lonlat(lonlat)` methods (ECEF convention: Z is north, lon=0 at +X). Output is `float32 (N, 2)` in normalized map coordinates.

- **`EquirectangularProjection`**: u = lon/π, v = lat/(π/2). Output in [−1, 1]².
- **`MercatorProjection`**: Conformal cylindrical; latitudes clamped at ±85.05° so v stays in [−1, 1].
- **`MollweideProjection`**: Equal-area pseudocylindrical; solved via Newton iteration for auxiliary angle θ. Fits an ellipse with semi-axes √2 (u) and √2 (v).
- **`RobinsonProjection`**: Table-based compromise; parallel lengths and y-offsets interpolated from the original Robinson lookup tables at 5° latitude steps.

### `worldgen/mapview/mapview.py` — Qt map widget

- **`_split_at_antimeridian(face_ecef)`**: Splits a unit-sphere polygon at the ±180° meridian. Returns a list of one or two lon/lat polygons; discards sub-polygons with fewer than 3 vertices.
- **`MapWidget`** (`QWidget`): Paints projected polygons using `QPainter`. Supports pan (drag) and zoom (scroll wheel or ±/Reset buttons). Zoom is clamped to [1×, 50×]. Accepts named color schemes via `set_color_schemes(schemes)` where each scheme is a `dict[str, Callable[[dict], RGBA]]` mapping a display name to a function that receives a face's metadata dict and returns an RGBA tuple. `set_color_scheme(name)` switches the active scheme.
- **`MapWindow`** (`QMainWindow`): Hosts `MapWidget` with a projection combo-box toolbar. `set_color_schemes(schemes)` populates a coloration combo box in the toolbar (hidden when no schemes are registered).

### `worldgen/mapview/main.py` — Entry point

Creates an `icosphere(depth=4)`, deregularizes it, computes its dual, and opens the map viewer.

### `worldgen/geology/tectonics.py` — Plate tectonics simulation

- **`generate_plates(mesh, n_plates, seed, compactness)`**: Assigns each face to a plate via per-plate randomized-priority flood fill. Each step picks a plate weighted by a random draw (so no plate gets a compounding size advantage) and expands it using a distance-biased heap, producing compact plates with irregular borders. Post-fill, any plate wholly surrounded by a single other plate is absorbed into it. Returns `List[int]` of plate IDs indexed by face.
- **`relax_plates(mesh, plate_ids, max_passes)`**: Iteratively removes appendages using articulation point detection (`_tarjan_aps_iterative`). Each pass finds faces reachable only via a single articulation point and reassigns them to whichever neighboring plate they border most. Repeats until stable.
- **`generate_plate_velocities(mesh, plate_ids, n_plates, drift_scale, spin_scale, seed)`**: Returns `(n_plates, 3)` angular velocity vectors. Each plate's omega combines a random drift component (tangential to the plate centroid, i.e., pure translation across the sphere) and a spin component (along the centroid axis). The velocity of a face on plate k is `omega[k] × face_centroid`.
- **`assign_plate_types(n_plates, oceanic_fraction, seed)`**: Classifies each plate as `'oceanic'` or `'continental'`. Default `oceanic_fraction=0.7` matches Earth's rough ratio.

### `worldgen/geology/main.py` — Entry point

Builds a deregularized icosphere dual (12 plates by default), runs `generate_plates` → `relax_plates` → `assign_plate_types`, then uses `apply_metadata_channel` to write `plate_id` and `plate_type` onto every face. Opens `MapWindow` with two color schemes: **Plates** (HSV hue-wheel palette by plate ID) and **Plate Type** (blue for oceanic, tan for continental).

## Key Design Notes

- `Mesh` stores only triangular faces; `PolygonMesh` handles arbitrary polygon faces needed for duals (e.g., a depth-1 icosphere dual has 12 pentagons + 30 hexagons).
- The dual construction normalizes centroids to the unit sphere so the dual lives on the same sphere as the original, not scaled by face area.
- Both classes pre-expand geometry into flat `render_vertices`/`render_colors` arrays at construction time; the renderer calls `glDrawArrays` with no index buffer.
- OpenGL context is set to 3.3 compatibility profile (not core) so legacy calls remain available during development.
- Edge-swaps in `deregularize` are rejected if they would create a vertex with degree ≤ 4 (which produces squares in the dual) or ≥ 8 (octagons).
- `PolygonMesh` pre-computes `face_adjacency` and `face_centroids` at construction and after `deregularize`; geology code relies on these without recomputing them.
- Face coloring has two layers: `face_colors` (static RGBA array, used when no active scheme is set) and named color schemes (callables over `face_metadata`, registered via `set_color_schemes`). The color scheme layer takes priority in `MapWidget`.
