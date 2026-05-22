# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Style

- Ask for clarification rather than churning your wheels. If the task or approach is ambiguous, ask before attempting.


## Running the Project

```bash
source .venv/bin/activate
python3 -m worldgen.main            # primary entrypoint — full procedural world viewer
python3 -m worldgen.mesh.main       # 3-D mesh viewer (dev)
python3 -m worldgen.mapview.main    # 2-D map viewer (dev)
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

This is a procedural world generation project. The core pipeline is: generate a mesh → deregularize it → compute its dual → run stage-specific generation (geology, etc.) → wrap in a `Globe` → display it.

The top-level orchestrator is `worldgen.globe.GlobeBuilder`, which runs each stage (currently just elevation/tectonics) and returns a `Globe`. The `Globe` holds the surface `mesh` plus a generic `features` registry for vector-style overlays (rivers, coastlines, etc.) that don't fit the per-face rasterized structure.

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
- **`MapWidget`** (`QWidget`): Paints projected polygons using `QPainter`. Supports pan (drag) and zoom (scroll wheel or ±/Reset buttons). Zoom is clamped to [1×, 50×]. Accepts named color schemes via `set_color_schemes(schemes)` where each scheme is a `dict[str, Callable[[dict], RGBA]]` mapping a display name to a function that receives a face's metadata dict and returns an RGBA tuple. `set_color_scheme(name)` switches the active scheme. When the mesh has `face_textures`, textures are converted to `QImage`s in `_rebuild_textures`, and painting uses per-triangle affine texture mapping via `QBrush` with `QTransform` (solving texture→screen for each fan triangle). Falls back to solid-color `QPolygonF` rendering when no textures are present. Vector overlays read from `Globe.features` are drawn on top of the face layer; `set_overlay_visible(name, visible)` toggles each overlay. The `"rivers"` overlay consumes a list of `(positions_3d, flows)` tuples (positions are unit-sphere `(M, 3) float32`, flows are per-segment length `M-1`) and draws each river as a polyline whose width scales with `flows[j-1] / max_flow`; the per-vertex 3D positions are converted to lon/lat and split at the antimeridian before projection.
- **`MapWindow`** (`QMainWindow`): Hosts `MapWidget` with a projection combo-box toolbar. `set_color_schemes(schemes)` populates a coloration combo box in the toolbar (hidden when no schemes are registered). `set_rebuild_callback(callback)` wires a Re-run button (hidden by default) that fires `callback` when clicked — used by `geology.main` to regenerate the world in place.

### `worldgen/mapview/main.py` — Entry point

Creates an `icosphere(depth=4)`, deregularizes it, computes its dual, builds a radial-gradient per-face texture tinted by each face's random color, and opens the map viewer.

### `worldgen/geology/elevation.py` — Elevation model

- **`compute_face_elevations(mesh, plate_types_by_face, boundary_data, decay_rate, max_hops, rate_norm)`**: Computes elevation (km) per face. Baseline is set by plate type (`'oceanic'` → −4.0 km, `'continental'` → +0.3 km). Each boundary face receives a delta based on boundary type, convergence rate, and the types of neighboring plates (e.g., continental-continental convergence → +4 km mountain range, oceanic subducting under continental → −2.5 km trench on the oceanic side). Deltas propagate outward from boundary faces via BFS with exponential decay (`exp(-decay_rate * hops)`) up to `max_hops` steps; the maximum-magnitude contribution wins at each face. Returns a `float64` array of shape `(n_faces,)`.

### `worldgen/geology/elevation_builder.py` — Geology stage orchestration

- **`ElevationParams`** (dataclass): Centralizes every tuning knob in the elevation/tectonics pipeline. Groups: *mesh generation* (`icosphere_depth`, `n_swaps`, `n_relax`, `stiffness`, `outer_iters`), *plate layout* (`n_plates`, `plate_compactness`, `relax_max_passes`, `oceanic_fraction`), *plate motion* (`drift_scale`, `spin_scale`), *elevation* (`elevation_rate_norm`, `elevation_decay_rate`, `elevation_max_hops`), *rivers* (`river_flow_threshold`), and *reproducibility* (`seed`). A top-level `seed` fans out to per-stage sub-seeds (`seed+0` for plate generation, `seed+1` for plate types, `seed+2` for velocities); `None` means random each run.
- **`ElevationBuilder`**: Owns an `ElevationParams` and runs the elevation/tectonics pipeline when `build()` is called. Returns the dual `PolygonMesh` with all metadata applied and stores intermediate results as attributes: `mesh`, `plate_ids`, `plate_omegas`, `face_velocities`, `boundary_types`, `boundary_data`, `elevations`, `volcanism`, `rivers` (`List[River]`), `max_speed`. `boundary_data` is the richer `List[dict]` from `classify_plate_boundaries` (with `normal_avg` and `neighbor_plate_types`); `boundary_types` is the `List[str]` derived from it. Callers (e.g. visualization code) read these attributes without re-running any work.

### `worldgen/geology/rivers.py` — River generation

Per-face grid + Priority-Flood + D-infinity routing + sub-pixel gradient walk. Each face's elevation texture is treated as a `resolution × resolution` grid of nodes; intra-face neighbours form a D8 (8-connected) grid and inter-face neighbours are stitched along shared polygon edges. See [docs/river_generation_spec.md](docs/river_generation_spec.md) for the as-built design and rationale.

- **`RiverNode` / `RiverSegment` / `River`** (dataclasses): A `River` is an ordered list of `RiverSegment`s from `source` to `mouth`; `as_polyline()` returns the 3D positions and `flow_rates()` returns the per-segment upstream flow.
- **`RiverGenerator`** (`mesh, elevation_textures, min_flow=50.0, sea_level=0.0, smoothing_iterations=8, smoothing_alpha=0.5`): End-to-end pipeline. `generate() -> list[River]` runs three stages:
  1. **Priority-Flood with imposed gradient** (`_priority_flood`): seeds every ocean node and propagates filled elevations `eff` outward. A small per-step ε (`1e-6`) is added during the fill so that depression interiors slope gently toward the spill point — otherwise the bilinear gradient on a filled lake would be exactly zero and the sub-pixel walk would coast forever on momentum instead of routing to the outlet.
  2. **D-infinity flow direction** (`_compute_dinf_flow`): for each node, considers the 8 triangular facets formed by adjacent cardinal/diagonal neighbour pairs (Tarboton 1997). The steepest descending facet wins; flow is split between its cardinal and diagonal vertices proportional to the gradient direction angle within the facet. Vectorised one facet at a time across all faces. Boundary nodes with no descending intra-face facet fall back to single-receiver D8 across the stitched inter-face edge (`_inter_face_fallback`). The per-node `flow_angle` is kept for headwater detection and inspection but the *trace walk* uses the bilinear gradient instead (see below).
  3. **Multi-receiver flow accumulation** (`_accumulate_flow`): in `eff`-descending order, each node distributes its flow to up to two receivers per the D-inf fractions. NOTE: with multi-receiver, `flow_accum` can *decrease* downstream at branch points — don't threshold on it mid-trace.
  Headwaters (above-threshold nodes that receive no above-threshold inflow) are then traced downstream via **sub-pixel walks** in face-local UV coordinates (`_trace_river_subpixel`). Each step looks up the local flow direction as the **negative bilinear gradient of `eff`** (`_gradient_angle_at`) — a gradient field is curl-free by construction, so the walk can't get trapped in vortices the way a bilinearly-interpolated angle field can. The new direction is blended 50/50 with the previous step's direction (momentum carry-over) so the walk pushes through (a) face-boundary discontinuities, where per-face textures sample elevation independently and the gradient can flip sign across the seam, and (b) short flats. UV step size is `1/(res-1)` (≈ one grid cell). When the step crosses out of `[0,1]²`, `_3d_to_uv` searches `face_adjacency` for the neighbour face containing the new position and switches into its UV frame. A 40-step stall guard kills walks that genuinely make no 3D progress despite momentum. Final pass applies **`laplacian_smooth_sphere`** (interior points pulled toward neighbour midpoints, re-projected to the unit sphere, endpoints pinned).
- **Inter-face stitching** (`_stitch_face_boundaries`): For each adjacent face pair, finds the two shared mesh vertices, computes both faces' UVs at those vertices via `face_tangent_frames()`, and matches pixels lying within ~½ pixel of the shared edge in each face by parameter `t` along the edge. Matched pixel pairs become bidirectional inter-face edges, stored sparsely in `_inter_adj`; intra-face D8 is derived on the fly in `_neighbors` (used by Priority-Flood). Tangent-frame data (axes, offsets, ranges, `_tangent_centroids`) is cached at init for the trace walk's `_uv_to_3d` / `_3d_to_uv` projections.

### `worldgen/geology/display.py` — Geology-specific visualization

Holds the color schemes (`Elevation`, `Plates`, `Plate Type`, `Speed`, `Boundary`, `Volcanic Activity`), the palette helper, and `build_elevation_display_texture(mesh)`. The entrypoint is `apply_geology_display(window, builder)` which wires the schemes and elevation texture into a `MapWindow` given an `ElevationBuilder` with results already populated.

### `worldgen/globe/` — Top-level orchestration

- **`Globe`** ([globe.py](worldgen/globe/globe.py)): Container for a generated world. Holds `mesh: PolygonMesh` plus a generic feature registry: `add_feature(name, data)`, `get_feature(name)`, `has_feature(name)`, `remove_feature(name)`, `feature_names()`. Use the registry for vector-style overlays (rivers, coastlines) that don't fit the per-face structure.
- **`GlobeParams`** ([builder.py](worldgen/globe/builder.py)): Composes per-stage params dataclasses. Currently exposes `elevation: ElevationParams`; add new stages as new fields.
- **`GlobeBuilder`** ([builder.py](worldgen/globe/builder.py)): Owns a `GlobeParams`, instantiates each stage builder (currently `elevation_builder: ElevationBuilder`), and `build(progress=...)` runs every stage and returns a `Globe`. Stage builders remain accessible as attributes so display code can pull intermediate state.

### `worldgen/geology/tectonics.py` — Plate tectonics simulation

- **`generate_plates(mesh, n_plates, seed, compactness)`**: Assigns each face to a plate via per-plate randomized-priority flood fill. Each step picks a plate weighted by a random draw (so no plate gets a compounding size advantage) and expands it using a distance-biased heap, producing compact plates with irregular borders. Post-fill, any plate wholly surrounded by a single other plate is absorbed into it. Returns `List[int]` of plate IDs indexed by face.
- **`relax_plates(mesh, plate_ids, max_passes)`**: Iteratively removes appendages using articulation point detection (`_tarjan_aps_iterative`). Each pass finds faces reachable only via a single articulation point and reassigns them to whichever neighboring plate they border most. Repeats until stable.
- **`generate_plate_velocities(mesh, plate_ids, n_plates, drift_scale, spin_scale, seed)`**: Returns `(n_plates, 3)` angular velocity vectors. Each plate's omega combines a random drift component (tangential to the plate centroid, i.e., pure translation across the sphere) and a spin component (along the centroid axis). The velocity of a face on plate k is `omega[k] × face_centroid`.
- **`assign_plate_types(n_plates, oceanic_fraction, seed)`**: Classifies each plate as `'oceanic'` or `'continental'`. Default `oceanic_fraction=0.7` matches Earth's rough ratio.
- **`classify_plate_boundaries(mesh, plate_ids, plate_omegas)`**: Returns `List[dict]` (one per face) with keys `boundary_type` (`'interior'|'convergent'|'divergent'|'transform'`), `normal_avg` (mean normal velocity component across inter-plate edges; positive = diverging), and `neighbor_plate_types` (set of plate type strings seen across those edges). Interior faces get zeros/empty set.

### `worldgen/main.py` — Primary entrypoint

Top-level entrypoint. Creates a `GlobeBuilder(GlobeParams())`, calls `build()` to produce a `Globe`, passes it to `MapWindow.set_globe(globe)`, and calls `apply_geology_display(window, builder.elevation_builder)` to wire color schemes and the elevation texture. Available schemes: **Elevation** (multi-stop gradient from abyssal ocean blues through beach/lowland/highland to snow-capped peaks, with a hard break at sea level), **Plates** (HSV hue-wheel palette by plate ID), **Plate Type** (blue for oceanic, tan for continental), **Speed** (warm-to-cool by surface speed), **Boundary** (color-coded by convergent/divergent/transform/interior), and **Volcanic Activity**. A Re-run callback is registered so the world can be regenerated in place without restarting.

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
- Per-face tangent-plane axes are constructed with `u_axis = cross(z, n)` and `v_axis = cross(n, u_axis)`, so for ~90% of faces the texture U-axis is geographic east and V-axis is geographic north. Any algorithm that walks the texture grid in raw 4-connected steps (e.g. D4 flow routing) will therefore produce strongly N-S/E-W-biased output. `rivers.py` avoids this by (a) D-infinity flow accumulation rather than D8 single-receiver, and (b) tracing rivers as continuous sub-pixel walks along the negative bilinear gradient of the filled DEM rather than snapping each step to a grid neighbour. The grid only re-enters the picture when the walk crosses a face boundary and has to switch UV frames. Measured bearing-histogram bias is ~1.2–1.5× peak vs uniform across seeds (down from ~1.7× with D8 grid-snap traces).
