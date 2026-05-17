# procgen

Eventually a super-project for all things procedural generation

## Setup

Requires Python 3.13+ and [Homebrew](https://brew.sh).

```bash
brew install python@3.13
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

After the first time, reactivate with:

```bash
source .venv/bin/activate
```

## `worldgen/mesh`

### Overview

The core pipeline is: build a triangular `Mesh` → deregularize it → compute its dual `PolygonMesh` → display it. The geology pipeline extends this by running tectonic simulation on the dual mesh.

```
Mesh.icosphere(depth)
    └─ .deregularize(...)        # edge-swap + spring relax
        └─ .mesh_dual()          # → PolygonMesh
            ├─ MainWindow(mesh)  # 3-D viewer (mesh module)
            ├─ MapWindow(mesh)   # 2-D map viewer (mapview module)
            └─ generate_plates / relax_plates / ...  # geology module
```

### `mymesh.py`

**`PolygonMesh(vertices, faces)`**

Base class for any polygon mesh. Faces are tuples of vertex indices of arbitrary length.
On construction it fan-triangulates every face and pre-builds flat `render_vertices` and `render_colors` arrays (one random color per face) ready for direct VBO upload. The triangulation also produces `render_face_indices` (per-render-vertex face index) and `render_local_uvs` (per-render-vertex UV in [0, 1]² from a tangent-plane parameterization) for the texture pipeline. Also pre-computes:

- `face_adjacency` — `List[List[int]]` of edge-sharing neighbor face indices per face.
- `face_centroids` — `(N, 3) float64` unit-sphere centroid for each face.
- `face_metadata` — `List[dict]` of per-face key-value stores for downstream annotation.
- `face_textures` — `Optional[(N, H, W, 4) float32]`; `None` until set.

`set_face_colors(colors)` replaces `face_colors` and rebuilds the render arrays. `set_face_metadata(face_idx, **kwargs)` merges kwargs into that face's metadata dict. `apply_metadata_channel(key, values)` bulk-writes a single key across all faces at once; raises `ValueError` if `len(values)` doesn't match the face count. `set_face_textures(textures)` attaches a `(N, H, W, 4) float32` RGBA texture array (one per face).

**`Mesh(vertices, faces)`** — subclass of `PolygonMesh`

Triangular mesh only (faces are 3-tuples). Vertices are re-sorted along the Z-axis at construction for spatial locality.

- `Mesh.icosphere(depth)` — generates a subdivided icosahedron, projecting all vertices onto the unit sphere. Each subdivision level quadruples the face count: depth 0 gives 20 triangles, depth `n` gives `20 × 4ⁿ`.
- `.deregularize(n_swaps, n_relax, stiffness, outer_iters)` — breaks the icosphere's regularity by alternating random edge-swaps with spring relaxation. Edge-swaps are rejected if they would create a vertex with degree ≤ 5 (squares in the dual) or ≥ 7 (octagons). Spring relaxation pulls each vertex toward the mean position of its neighbors and re-projects onto the unit sphere.
- `.mesh_dual()` — builds the dual `PolygonMesh`. Each original face becomes a vertex (its centroid, normalized to the unit sphere). Each original vertex becomes a face whose vertices are the centroids of its adjacent faces, ordered CCW by projecting onto the tangent plane at that vertex.

### `gui.py`

**`MeshViewer(QOpenGLWidget)`**

Renders a `PolygonMesh` via VBOs using `glDrawArrays(GL_TRIANGLES, ...)`. Rotates the mesh 0.5°/frame driven by a 16 ms `QTimer` (~60 FPS). OpenGL 3.3 compatibility profile. When the mesh has `face_textures`, all per-face textures are packed into a square grid atlas, uploaded as a single `GL_TEXTURE_2D`, and rendered via a UV VBO; otherwise the color-array path is used.

**`MainWindow(QMainWindow)`**

Hosts `MeshViewer` in an 800×600 window. Call `set_mesh(mesh)` to swap the displayed mesh at runtime.

### `test/test_mymesh.py`

Covers `Mesh` (vertex/face counts at various subdivision depths, unit-sphere property, outward face normals) and `PolygonMesh.mesh_dual` (dual topology counts, face sizes, outward winding of all fan triangles, convexity of dual faces on the sphere).

## `worldgen/mapview`

### Overview

Renders a `PolygonMesh` as a flat 2-D map with selectable projection and interactive pan/zoom.

```
PolygonMesh (from mesh pipeline)
    └─ MapWindow.set_mesh(mesh)
        └─ MapWidget._rebuild_faces()    # antimeridian split → lon/lat polygons
            └─ MapWidget._reproject()   # Projection.project_lonlat → (u, v)
                └─ MapWidget.paintEvent # QPainter draws filled polygons
```

### `projections.py`

All projections inherit from `Projection` and expose `project(ecef)` and `project_lonlat(lonlat)`, returning `float32 (N, 2)` arrays in normalized map coordinates (ECEF convention: Z is north, lon=0 at +X axis).

**`EquirectangularProjection`** — Linear: u = lon/π, v = lat/(π/2). Output in [−1, 1]².

**`MercatorProjection`** — Conformal cylindrical. Latitudes clamped at ±85.051° so v stays in [−1, 1].

**`MollweideProjection`** — Equal-area pseudocylindrical. Auxiliary angle θ is solved with 10 Newton steps; polar vertices are snapped to θ = ±π/2 to avoid division by zero. Output fits an ellipse with semi-axes √2 in both u and v.

**`RobinsonProjection`** — Compromise projection. Parallel lengths (plen) and y-offsets (pdfe) are interpolated from the classic Robinson lookup table at 5° latitude intervals.

**`ALL_PROJECTIONS`** — List of one instance of each projection, used to populate the viewer combo-box.

### `mapview.py`

**`_split_at_antimeridian(face_ecef, face_uv=None)`** — Takes an `(N, 3)` array of ECEF unit-sphere vertices and returns a list of `(M, 2)` lon/lat arrays. Detects antimeridian crossings (|Δlon| > π on a direct edge), interpolates the crossing latitude, and splits the polygon into eastern and western halves. Sub-polygons with fewer than 3 vertices are discarded. When `face_uv` is supplied, returns `(lonlat, uv)` tuples with UVs interpolated at crossing points.

**`_compute_face_uv(face_ecef)`** — Tangent-plane UV parameterization for a face's vertices. Returns `(N, 2) float32` in [0, 1].

**`MapWidget(QWidget)`** — Paints lon/lat polygons by projecting them on each `_reproject()` call and drawing via `QPainter`. Pan via left-button drag; zoom via scroll wheel (clamped to [1×, 50×]). Call `set_mesh(mesh)` to load a new mesh. Color can be driven by named schemes registered via `set_color_schemes(schemes)` — a `dict[str, Callable[[dict], RGBA]]` mapping display names to functions over face metadata; `set_color_scheme(name)` switches the active scheme. Falls back to `PolygonMesh.face_colors` when no scheme is active. When the mesh has `face_textures`, each face is painted using per-triangle affine texture mapping (`QBrush` with a `QTransform` solved from UV→screen coordinates).

**`MapWindow(QMainWindow)`** — 900×500 window hosting `MapWidget` plus a toolbar with a projection combo-box, a coloration combo-box (shown only when schemes are registered via `set_color_schemes`), and Zoom In / Zoom Out / Reset buttons. Call `set_rebuild_callback(fn)` to show a Re-run button that fires `fn` when clicked.

### `test/test_mapview.py`

Covers all four projections (equator at v=0, prime meridian at u=0, N/S and E/W symmetry, `project` vs `project_lonlat` agreement, projection-specific bounds such as equirectangular unit square, Mercator monotonicity and clamping, Mollweide ellipse containment, Robinson pole positions and parallel-length shrinkage) and `_split_at_antimeridian` (no-crossing passthrough, two-polygon split when two edges cross, hemisphere lon-sign invariant, boundary latitude continuity).

## `worldgen/geology`

### Overview

Tectonic simulation layer that runs on a `PolygonMesh` dual. Assigns faces to plates, relaxes plate boundaries, assigns oceanic/continental types, and annotates face metadata for display via `MapWindow` color schemes.

```
PolygonMesh (dual from mesh pipeline)
    └─ generate_plates(mesh, n_plates)      # flood-fill assignment
        └─ relax_plates(mesh, plate_ids)    # remove appendages via articulation points
            └─ assign_plate_types(n_plates)
                └─ apply_metadata_channel() # bulk-write plate_id / plate_type onto faces
                    └─ MapWindow with color schemes
```

### `elevation.py`

**`compute_face_elevations(mesh, plate_types_by_face, boundary_data, decay_rate, max_hops, rate_norm)`**

Computes elevation (km) for each face. Baseline is set by plate type (`'oceanic'` → −4.0 km, `'continental'` → +0.3 km). Each boundary face receives a delta determined by its boundary type and the types of adjacent plates (continental-continental collision → +4 km, oceanic trench → −2.5 km, mid-ocean ridge → +2 km above oceanic baseline, continental rift → −1.5 km, etc.). Deltas are scaled by convergence rate and propagated outward via BFS with exponential decay (`exp(-decay_rate * hops)`) up to `max_hops` steps. Returns `float64 (n_faces,)`.

### `tectonics.py`

**`generate_plates(mesh, n_plates, seed, compactness)`**

Assigns each face to a plate via per-plate randomized-priority flood fill. At each step a plate is chosen by weighted random draw (plates have random weights so the result has a realistic spread of sizes, not near-uniform areas), then expanded one face using a distance-biased min-heap so cells near the seed are preferred. `compactness` in [0, 1] blends pure distance priority (circular plates) with pure random priority (snaky plates). Post-fill, any plate entirely surrounded by one other plate is absorbed. Returns `List[int]` of plate IDs indexed by face.

**`relax_plates(mesh, plate_ids, max_passes)`**

Iteratively removes appendages — sets of faces connected to the rest of their plate only through a single articulation point. Uses iterative Tarjan's algorithm (`_tarjan_aps_iterative`) to find articulation points per plate, identifies the stranded faces behind each one, then reassigns them to the neighboring plate they border most. Repeats until no reassignments occur or `max_passes` is reached.

**`generate_plate_velocities(mesh, plate_ids, n_plates, drift_scale, spin_scale, seed)`**

Returns `(n_plates, 3) float64` angular velocity vectors. Each plate's omega is the sum of a drift component (a random vector projected onto the tangent plane at the plate's centroid, so the plate translates across the sphere) and a spin component (a random scalar along the centroid axis). The surface velocity of a point `p` on plate `k` is `omega[k] × p`.

**`assign_plate_types(n_plates, oceanic_fraction, seed)`**

Classifies each plate as `'oceanic'` or `'continental'` by Bernoulli draws with probability `oceanic_fraction` (default 0.7).

**`classify_plate_boundaries(mesh, plate_ids, plate_omegas)`**

Returns `List[dict]` (one per face) with keys `boundary_type` (`'interior'|'convergent'|'divergent'|'transform'`), `normal_avg` (mean normal velocity component, positive = diverging), and `neighbor_plate_types` (set of plate type strings across inter-plate edges).

### `main.py`

Builds an `icosphere(4)` dual, runs the full tectonics and elevation pipeline, then opens `MapWindow` with five color schemes: **Elevation** (multi-stop gradient from abyssal blues through lowlands to snow peaks), **Plates** (evenly-spaced HSV hues), **Plate Type** (blue for oceanic, tan for continental), **Speed** (warm-to-cool by surface speed), and **Boundary** (color-coded by type). Registers a Re-run button so the world can be regenerated in place.
