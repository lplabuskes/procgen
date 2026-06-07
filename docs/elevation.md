# Elevation & Tectonics Stage

## Files
- `worldgen/geology/elevation_builder.py` — stage orchestration
- `worldgen/geology/elevation.py` — elevation model
- `worldgen/geology/tectonics.py` — plate tectonics
- `worldgen/geology/display.py` — visualization

## `elevation_builder.py`

**`ElevationParams`** (dataclass): Every tuning knob in the elevation/tectonics pipeline.
- *mesh generation*: `icosphere_depth`, `n_swaps`, `n_relax`, `stiffness`, `outer_iters`
- *plate layout*: `n_plates`, `plate_compactness`, `relax_max_passes`, `oceanic_fraction`
- *plate motion*: `drift_scale`, `spin_scale`
- *elevation*: `elevation_rate_norm`, `elevation_decay_rate`, `elevation_max_hops`
- *noise*, *face erosion*, *volcanism*
- *reproducibility*: `seed` (fans out to per-stage sub-seeds; `None` = random each run)

River parameters live in `HydrologyParams`, not here.

**`ElevationBuilder`**: The mesh-producing `Stage`. `build(ctx)` sets `ctx.world.mesh` and returns the dual `PolygonMesh` with all metadata applied. Stored attributes after build: `mesh`, `plate_ids`, `plate_omegas`, `face_velocities`, `boundary_types`, `boundary_data`, `elevations`, `elevation_textures` (single-channel `(N, H, W, 1)` float32), `volcanism`, `max_speed`. The mesh has a `"elevation"` face texture (single-channel) attached.

## `elevation.py`

**`compute_face_elevations(mesh, plate_types_by_face, boundary_data, decay_rate, max_hops, rate_norm)`**: Computes elevation (km) per face.
- Baseline by plate type: `'oceanic'` → −4.0 km, `'continental'` → +0.3 km
- Boundary faces get a delta based on boundary type, convergence rate, and neighboring plate types (e.g., continental-continental convergence → +4 km mountain range; oceanic subducting under continental → −2.5 km trench on oceanic side)
- Deltas propagate via BFS with exponential decay (`exp(-decay_rate * hops)`) up to `max_hops`; max-magnitude contribution wins at each face
- Returns `float64 (n_faces,)`

Elevation textures are single-channel `(N, H, W, 1)` float32 (channel 0 = elevation in km). Previously 4-channel RGBA; extra channels dropped.

## `tectonics.py`

**`generate_plates(mesh, n_plates, seed, compactness)`**: Per-plate randomized-priority flood fill. Each step picks a plate weighted by a random draw and expands via a distance-biased heap → compact plates with irregular borders. Post-fill, any plate wholly surrounded by a single other plate is absorbed. Returns `List[int]` of plate IDs by face.

**`relax_plates(mesh, plate_ids, max_passes)`**: Iteratively removes appendages via articulation point detection (`_tarjan_aps_iterative`). Repeats until stable.

**`generate_plate_velocities(mesh, plate_ids, n_plates, drift_scale, spin_scale, seed)`**: Returns `(n_plates, 3)` angular velocity vectors. Each plate's omega = random drift (tangential to plate centroid = pure translation) + spin (along centroid axis). Face velocity = `omega[k] × face_centroid`.

**`assign_plate_types(n_plates, oceanic_fraction, seed)`**: Classifies each plate as `'oceanic'` or `'continental'`. Default `oceanic_fraction=0.7`.

**`classify_plate_boundaries(mesh, plate_ids, plate_omegas)`**: Returns `List[dict]` per face with keys `boundary_type` (`'interior'|'convergent'|'divergent'|'transform'`), `normal_avg`, `neighbor_plate_types`. Interior faces get zeros/empty set.

## `display.py`

Holds color scheme callables (`_elevation_color`, `_elevation_color_vec`), RGBA constants (`OCEAN_COLOR`, `LAND_COLOR`, `BOUNDARY_COLORS`), and `_plate_palette` helper. Imported by `worldgen/globe/builder.py` — no standalone entrypoint.
