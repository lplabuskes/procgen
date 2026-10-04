# Elevation & Tectonics Stage

## Files
- `worldgen/geology/elevation_builder.py` — stage orchestration + params
- `worldgen/geology/elevation.py` — elevation model, erosion, rasterisation, noise, seam smoothing, ocean mask
- `worldgen/geology/tectonics.py` — plate tectonics
- `worldgen/geology/volcanism.py` — volcanic activity + volcano peaks
- `worldgen/geology/channels.py` — `IS_OCEAN`, `ELEVATION_FACE`, `PLATE_TYPE`, `BOUNDARY_TYPE`, `VOLCANIC_ACTIVITY`, `ELEVATION`, `OCEAN_MASK`
- `worldgen/geology/display.py` — visualization

## `elevation_builder.py`

**`ElevationParams`** (dataclass): Every tuning knob in the elevation/tectonics pipeline. Structured as nested sub-param dataclasses:
- `plates: PlateParams` — `n_plates`, `plate_compactness`, `relax_max_passes`, `oceanic_fraction`
- `plate_motion: PlateMotionParams` — `drift_scale`, `spin_scale`
- `elevation: ElevationFieldParams` — `elevation_rate_norm`, `elevation_decay_rate`, `elevation_max_hops`, `elevation_tex_size` (the world's texture size, 16)
- `noise: NoiseParams` — `noise_amplitude`, `noise_coast_sigma`
- `face_erosion: FaceErosionParams` — hydraulic/thermal erosion knobs (hidden in params dialog)
- `volcanism: VolcanismParams` — `n_hotspots`, `volcanic_decay_rate`, `volcanic_max_hops`, `volcanic_elevation_scale`, `volcanic_elevation_scale_oceanic`

Mesh generation params (`icosphere_depth`, deregularize args) are not in `ElevationParams` — the mesh is pre-built before this stage runs. The world seed is passed to `ElevationBuilder.__init__(self, params, *, seed)` separately, not stored in `ElevationParams`; each draw takes a name-keyed sub-seed (`_seed("plates")`, …), so adding a draw never shifts the others.

River parameters live in `HydrologyParams`, not here.

**`ElevationBuilder`**: A pipeline `Stage`. `build(ctx)` decorates the already-existing dual `PolygonMesh` in `ctx.mesh`:

plates → relax → plate types + velocities → boundary classification → face elevations → volcanism → face erosion → + volcanism × `volcanic_elevation_scale[_oceanic]` → rasterise to textures (`rasterize_face_values`, below) → fBm simplex noise (`compute_noise_elevation_deltas`, band-limited to the texel grid, × `exp(−|elevation| / noise_coast_sigma)` so it is strongest near sea level and roughens the coastline) → seam smoothing → ocean mask → `world.antimeridian` (`core.antimeridian.find_ideal_antimeridian`).

Stored attributes after build: `mesh`, `plate_ids`, `plate_omegas`, `face_velocities`, `boundary_types`, `boundary_data`, `elevations` (per-face means of the final texture), `elevation_textures` (single-channel `(N, H, W, 1)` float32), `ocean_mask`, `volcanism`, `volcano_positions`, `max_speed`.

- Face textures: `"elevation"` (`(N, H, W, 1)` float32, km) and `"ocean_mask"` (`(N, H, W, 1)` uint8).
- Per-face metadata: `plate_id`, `plate_type`, `boundary_type`, `surface_speed`, `volcanic_activity`, `elevation` (the tectonic + erosion + volcanic face value, before noise — read by the resource placement models), `is_ocean` (more than half the face's texels are ocean).
- Feature: `"volcanoes"` — unit-sphere positions of above-sea-level activity peaks (`generate_volcano_positions`).

## `elevation.py`

**`compute_face_elevations(mesh, plate_types_by_face, boundary_data, decay_rate, max_hops, rate_norm)`**: Computes elevation (km) per face.
- Baseline by plate type: `'oceanic'` → −4.0 km, `'continental'` → +0.3 km
- Boundary faces get a delta × `|normal_avg| / rate_norm` by boundary type and neighbouring plate types: continental-continental convergence +4; continental over subducting oceanic +2.5 (arc), the oceanic side −2.5 (trench); oceanic-oceanic −1.5; mid-ocean ridge +2; continental rift −1.5; transform and interior 0
- Deltas propagate via BFS with exponential decay (`exp(-decay_rate * hops)`) up to `max_hops`; max-magnitude contribution wins at each face
- Returns `float64 (n_faces,)`

**`compute_erosion(mesh, elevations, params)`**: face-level hydraulic (fill sinks → accumulate → erode/deposit) then thermal (talus slumping) passes.

**`rasterize_face_values(mesh, face_values, tex_size, support_spacings=2.0)`**: per-face scalars → `(N, H, W, K)` float32 textures, shared by every stage that rasterises a per-face field (elevation, temperature, humidity, fertility, forest, coast smoothness). Each texel takes the Wendland-C² kernel-weighted mean of the face values whose centroids lie within `support_spacings` mean centroid spacings of its 3-D position (a face's centre keeps ~½ of its own value). The texture is a function of position alone, so faces agree on their seams and over-hangs, and its gradient is continuous. (The earlier piecewise-linear fan interpolation over each polygon creased at every face edge and fan diagonal, and flattened a lone high face into a hexagonal plateau at ⅓ of its height.)

**`compute_ocean_mask_texture(mesh, elevation_textures)`**: pixel-level ocean flood fill. Faces whose max elevation ≤ −0.5 km seed it; it spreads to adjacent faces at or below sea level, then pixel by pixel (4-connected within a face, across faces via matched seam pixel pairs). Sub-sea-level depressions no ocean reaches stay land. The region tier runs a level-constrained version of the same flood over its refined relief.

Elevation textures are single-channel `(N, H, W, 1)` float32 (channel 0 = elevation in km).

## `tectonics.py`

**`generate_plates(mesh, n_plates, seed, compactness)`**: Per-plate randomized-priority flood fill. Each step picks a plate uniformly at random (so frontier size gives no compounding advantage) and expands it one face via a distance-biased heap → compact plates with irregular borders. Post-fill, any plate wholly surrounded by a single other plate is absorbed. Returns `List[int]` of plate IDs by face.

**`relax_plates(mesh, plate_ids, max_passes)`**: Iteratively removes appendages via articulation point detection (`_tarjan_aps_iterative`). Repeats until stable.

**`generate_plate_velocities(mesh, plate_ids, n_plates, drift_scale, spin_scale, seed)`**: Returns `(n_plates, 3)` angular velocity vectors. Each plate's omega = random drift (tangential to plate centroid = pure translation) + spin (along centroid axis). `compute_face_velocities`: face velocity = `omega[k] × face_centroid`.

**`assign_plate_types(n_plates, oceanic_fraction, seed)`**: Classifies each plate as `'oceanic'` or `'continental'`. Default `oceanic_fraction=0.7`.

**`classify_plate_boundaries(mesh, plate_ids, plate_omegas)`**: Returns `List[dict]` per face with keys `boundary_type` (`'interior'|'convergent'|'divergent'|'transform'`), `normal_avg`, `neighbor_plate_types`. Interior faces get zeros/empty set.

## `volcanism.py`

**`compute_volcanism(...)`**: per-face activity in [0, 1] from boundary sources (divergent ridges/rifts, convergent arcs) plus `n_hotspots` random hotspots, spread by BFS with exponential decay (max wins). **`generate_volcano_positions(mesh, volcanism, elevations, threshold=0.3)`**: faces above the threshold, above sea level, and a local activity maximum.

## `display.py`

Holds the per-face scheme callable (`_elevation_color`), `ELEVATION_RECIPE` (the `TextureRecipe` behind the "Elevation" texture, `relief=True`, also run by the region view on refined data), `elevation_legend_entries`, RGBA constants (`OCEAN_COLOR`, `LAND_COLOR`, `BOUNDARY_COLORS`), `_plate_palette`, and the water-aware scheme helpers the other stages reuse (`apply_water_masks`, `ocean_gradient_scheme`, `water_gradient_scheme`, `ocean_categorical_scheme`, `is_water_meta`).

Schemes: **Elevation** always; with `--debug` also Plates, Plate Type, Speed, Boundary, Volcanic Activity, the **Volcanoes** overlay (unchecked), and an `[px: … km]` elevation hover (`refined=("elevation",)`, so a region shows its refined value).
