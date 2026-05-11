# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

```bash
source .venv/bin/activate
cd worldgen && python3 -m mesh.main       # 3-D mesh viewer
cd worldgen && python3 -m mapview.main    # 2-D map viewer
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

This is a procedural mesh generation and 3D visualization project. The pipeline is: generate a mesh → deregularize it → compute its dual → display it.

### `worldgen/mesh/mymesh.py` — Mesh data structures

- **`PolygonMesh`**: Base class for any polygon mesh. Fan-triangulates faces and pre-computes `render_vertices` and `render_colors` (flat per-vertex arrays) at construction time for direct VBO upload.
- **`Mesh`** (subclass of `PolygonMesh`): Triangular mesh. `icosphere(depth)` generates a subdivided icosahedron; vertices are sorted along the Z-axis for spatial locality. `.deregularize()` breaks regularity via random edge-swaps and spring relaxation on the unit sphere. `.mesh_dual()` returns a `PolygonMesh` by placing new vertices at face centroids (normalized to unit sphere), then for each original vertex builds a CCW-wound face from the centroids of its adjacent faces.

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
- **`MapWidget`** (`QWidget`): Paints projected polygons using `QPainter`. Supports pan (drag) and zoom (scroll wheel or ±/Reset buttons). Zoom is clamped to [1×, 50×].
- **`MapWindow`** (`QMainWindow`): Hosts `MapWidget` with a projection combo-box toolbar.

### `worldgen/mapview/main.py` — Entry point

Creates an `icosphere(depth=4)`, deregularizes it, computes its dual, and opens the map viewer.

## Key Design Notes

- `Mesh` stores only triangular faces; `PolygonMesh` handles arbitrary polygon faces needed for duals (e.g., a depth-1 icosphere dual has 12 pentagons + 30 hexagons).
- The dual construction normalizes centroids to the unit sphere so the dual lives on the same sphere as the original, not scaled by face area.
- Both classes pre-expand geometry into flat `render_vertices`/`render_colors` arrays at construction time; the renderer calls `glDrawArrays` with no index buffer.
- OpenGL context is set to 3.3 compatibility profile (not core) so legacy calls remain available during development.
- Edge-swaps in `deregularize` are rejected if they would create a vertex with degree ≤ 4 (which produces squares in the dual) or ≥ 8 (octagons).
