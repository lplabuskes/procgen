# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Style

- Ask for clarification rather than churning your wheels. If the task or approach is ambiguous, ask before attempting.
- Assume visuals should be limited to debug mode, ask before adding to the main view.


## Running the Project

```bash
source .venv/bin/activate
python3 main.py                     # primary entrypoint — full procedural world viewer
python3 -m worldgen.mesh.main       # 3-D mesh viewer (dev)
python3 -m utils.mapview.main       # 2-D map viewer (dev)
```

No build step or linter is configured. Run the test suites with:

```bash
python3 -m pytest                                       # all tests (testpaths in pyproject.toml)
python3 -m pytest core/test/test_mymesh.py -v
python3 -m pytest utils/mapview/test/test_mapview.py -v
```

`core/test/test_mymesh.py` covers `Mesh` (icosphere topology, unit-sphere property, outward face normals) and `Mesh.mesh_dual` (dual topology counts, face sizes, and correct CCW winding of all fan triangles).

`utils/mapview/test/test_mapview.py` covers all four projections (equator/meridian anchors, N/S and E/W symmetry, projection-specific bounds, `project` vs `project_lonlat` consistency) and `_split_at_antimeridian` (no-crossing passthrough, two-polygon split, boundary latitude continuity).

## Diagnostic Scripts

```bash
python3 scripts/measure_seam_quality.py              # default seeds 42, 137, 999
python3 scripts/measure_seam_quality.py 1 2 3        # specific seeds
python3 scripts/measure_seam_quality.py --random 5   # N random seeds
python3 scripts/measure_seam_quality.py --pre-noise 42  # base texture before noise
```

`scripts/measure_seam_quality.py` — reports mean / median / RMS / max elevation jump (in metres) across all ~350 k matched boundary pixel pairs per seed. Used to quantify face-seam discontinuities in the elevation texture pipeline. The underlying function is `measure_boundary_elevation_jumps` in `worldgen/geology/elevation.py`.

```bash
python3 scripts/measure_zone_biomes.py              # default seeds 42, 137, 999
python3 scripts/measure_zone_biomes.py 1 2 3        # specific seeds
python3 scripts/measure_zone_biomes.py --random 5   # N random seeds
python3 scripts/measure_zone_biomes.py --detail     # include per-seed biome breakdown
```

`scripts/measure_zone_biomes.py` — reports Tropical, Savannah, and Desert biome fractions within tropical latitudes, compared against Earth reference values (~30% of ±23.5° land tropical, ~22% of ±30° land tropical, ~20% savannah, ~18% hot desert). Runs elevation + climate (no hydrology). Results are appended to `scripts/results/zone_biome_results.json` and a human-readable summary is regenerated at `scripts/results/zone_biome_results.md` for longitudinal comparison across parameter tuning sessions.

```bash
python3 scripts/measure_resources.py              # default seeds 42, 137, 999
python3 scripts/measure_resources.py 1 2 3        # specific seeds
python3 scripts/measure_resources.py --random 5   # N random seeds
python3 scripts/measure_resources.py --detail     # include per-biome breakdown per seed
```

`scripts/measure_resources.py` — reports per-biome mean fertility and forest density across land pixels, compared against rough Earth-ish anchors (Tropical forest ≈ 0.85, Temperate forest ≈ 0.75, Tundra fertility ≈ 0.10, Desert fertility ≈ 0.03, …). Runs the full pipeline (elevation + climate + hydrology + lake feedback + resources). Results are appended to `scripts/results/resources_results.json` and the human-readable summary is regenerated at `scripts/results/resources_results.md`.

## Architecture

This is a procedural world generation project. The core pipeline is: generate a mesh → deregularize it → compute its dual → run stage-specific generation (geology, etc.) → wrap in a `Globe` → display it.

The top-level orchestrator is `worldgen.globe.GlobeBuilder`, which runs each stage (elevation/tectonics → climate → hydrology → resources) and returns a `Globe`. The `Globe` holds the surface `mesh` plus a generic `features` registry for vector-style overlays (rivers, coastlines, etc.) that don't fit the per-face rasterized structure.

### `core/mesh.py` — Mesh data structures (was `worldgen/mesh/mymesh.py`)

- **`PolygonMesh`**: Base class for any polygon mesh. Pre-computes `face_adjacency`, `face_centroids`, `face_metadata`, and fan-triangulated render arrays at construction. Key API: `set_face_colors(colors)`, `set_face_metadata(face_idx, **kw)`, `apply_metadata_channel(key, values)` (bulk-write one metadata key across all faces), `set_face_textures(name, textures, n_channels=4)` (attach `(N, H, W, C)` float32; use `n_channels=1` for data layers like elevation, 4 for RGBA display), `face_texture_positions(h, w)` (unit-sphere 3D position of every texture pixel — starting point for any per-pixel geographic computation), `face_tangent_frames()` (raw per-face tangent-plane data for projecting arbitrary 3D points into a face's UV space).
- **`Mesh`** (subclass of `PolygonMesh`): Triangular mesh. `icosphere(depth)` generates a subdivided icosahedron; vertices are sorted along the Z-axis for spatial locality. `.deregularize(n_swaps, n_relax, stiffness, outer_iters)` breaks regularity via random edge-swaps and spring relaxation on the unit sphere. `.mesh_dual()` returns a `PolygonMesh` by placing new vertices at face centroids (normalized to unit sphere), then for each original vertex builds a CCW-wound face from the centroids of its adjacent faces.

### `worldgen/mesh/gui.py` — OpenGL/Qt renderer

- **`MeshViewer`** (`QOpenGLWidget`): Renders a `PolygonMesh` using VBOs and `glDrawArrays` (OpenGL 3.3 compatibility). Drives ~60 FPS via a timer and rotates the mesh 0.5°/frame. When the mesh has `face_textures`, `_setup_buffers` packs all per-face textures into a square grid atlas (rows × cols), uploads it as a single `GL_TEXTURE_2D`, and substitutes a UV VBO (`uv_buffer`) for the color VBO; the atlas UV coordinates account for the v-flip between numpy (row 0 = top) and OpenGL (v=0 = bottom). Without textures, the original color-array path is used.
- **`MainWindow`** (`QMainWindow`): Hosts `MeshViewer`; exposes `set_mesh(mesh)` to swap the displayed mesh at runtime (calls `makeCurrent`/`doneCurrent` around buffer setup).

### `worldgen/mesh/main.py` — Entry point

Creates an `icosphere(depth=4)`, deregularizes it, computes its dual, builds a radial-gradient per-face texture tinted by each face's random color, and opens the viewer.

### `utils/mapview/projections.py` — Map projections

All projections share a `Projection` base class with `project(ecef_vertices)` and `project_lonlat(lonlat)` methods (ECEF convention: Z is north, lon=0 at +X). Output is `float32 (N, 2)` in normalized map coordinates.

- **`EquirectangularProjection`**: u = lon/π, v = lat/(π/2). Output in [−1, 1]².
- **`MercatorProjection`**: Conformal cylindrical; latitudes clamped at ±85.05° so v stays in [−1, 1].
- **`MollweideProjection`**: Equal-area pseudocylindrical; solved via Newton iteration for auxiliary angle θ. Fits an ellipse with semi-axes √2 (u) and √2 (v).
- **`RobinsonProjection`**: Table-based compromise; parallel lengths and y-offsets interpolated from the original Robinson lookup tables at 5° latitude steps.

### `utils/mapview/mapview.py` — Qt map widget

- **`_split_at_antimeridian(face_ecef, face_uv=None)`**: Splits a unit-sphere polygon at the ±180° meridian; with `face_uv`, also interpolates UVs at crossing points and returns `(lonlat, uv)` pairs.
- **`MapWidget`** (`QWidget`): `QPainter`-based projected polygon renderer with pan/zoom. Renders faces as solid color, named color schemes (callables over `face_metadata`), or per-face textures (affine UV mapping via `QBrush`/`QTransform` per fan triangle). Draws vector overlays from `Globe.features` on top; the `"rivers"` overlay scales line width by flow and skips `in_lake` segments. Hover hit-testing via bbox pre-filter + point-in-polygon + barycentric UV (`set_hover_callback`).
- **`MapWindow`** (`QMainWindow`): Hosts `MapWidget` with projection selector, color scheme selector, overlay checkboxes, legend panel, Re-run/Params/Save buttons, and status bar.

### `utils/mapview/main.py` — Entry point

Creates an `icosphere(depth=4)`, deregularizes it, computes its dual, builds a radial-gradient per-face texture tinted by each face's random color, and opens the map viewer.

### `worldgen/geology/elevation.py` — Elevation model

- **`compute_face_elevations(mesh, plate_types_by_face, boundary_data, decay_rate, max_hops, rate_norm)`**: Computes elevation (km) per face. Baseline is set by plate type (`'oceanic'` → −4.0 km, `'continental'` → +0.3 km). Each boundary face receives a delta based on boundary type, convergence rate, and the types of neighboring plates (e.g., continental-continental convergence → +4 km mountain range, oceanic subducting under continental → −2.5 km trench on the oceanic side). Deltas propagate outward from boundary faces via BFS with exponential decay (`exp(-decay_rate * hops)`) up to `max_hops` steps; the maximum-magnitude contribution wins at each face. Returns a `float64` array of shape `(n_faces,)`.
- Elevation textures throughout this module are single-channel `(N, H, W, 1)` float32 arrays (channel 0 = elevation in km). Previously 4-channel RGBA; the extra channels have been dropped.

### `worldgen/geology/elevation_builder.py` — Geology stage orchestration

- **`ElevationParams`** (dataclass): Centralizes every tuning knob in the elevation/tectonics pipeline. Groups: *mesh generation* (`icosphere_depth`, `n_swaps`, `n_relax`, `stiffness`, `outer_iters`), *plate layout* (`n_plates`, `plate_compactness`, `relax_max_passes`, `oceanic_fraction`), *plate motion* (`drift_scale`, `spin_scale`), *elevation* (`elevation_rate_norm`, `elevation_decay_rate`, `elevation_max_hops`), *noise*, *face erosion*, *volcanism*, and *reproducibility* (`seed`). A top-level `seed` fans out to per-stage sub-seeds; `None` means random each run. River parameters live in `HydrologyParams` (hydrology stage).
- **`ElevationBuilder`**: A pipeline `Stage` (the mesh-producing one). Owns an `ElevationParams` and runs the elevation/tectonics pipeline when `build(ctx)` is called; it sets `ctx.world.mesh` and returns the dual `PolygonMesh` with all metadata applied. Stores intermediate results as attributes: `mesh`, `plate_ids`, `plate_omegas`, `face_velocities`, `boundary_types`, `boundary_data`, `elevations`, `elevation_textures` (single-channel `(N, H, W, 1)` float32), `volcanism`, `max_speed`. The mesh has a `"elevation"` face texture (single-channel) attached. Standalone callers (scripts) run it as `builder.build(BuildContext(world=World()))` and then read these attributes. As a `Stage` it also implements `register_display` (Elevation scheme + debug Plates/Boundary/Speed/Volcanic schemes), `serialize`, and `load`.

### `worldgen/hydrology/` — Hydrology stage

- **`HydrologyParams`** (dataclass in `builder.py`): Tuning knobs for the hydrology stage. Fields: `river_flow_threshold` (default `200.0`) — minimum upstream flow accumulation for a node to count as a river headwater; `min_lake_pixels` (default `50`) — minimum flooded-component size to keep as a lake; `pet_baseline_mm_yr` / `pet_per_degree_mm_yr` / `pet_max_mm_yr` — clamped-linear PET model used by water-balance lake shrinking (`PET = clamp(baseline + slope·max(0,T), 0, max)`).
- **`HydrologyBuilder`** (`builder.py`): A pipeline `Stage`. Runs the hydrology pipeline after elevation **and climate** are built (it consumes the `"precipitation"`, `"temperature"`, and optionally `"ocean_mask"` face textures from the climate/elevation stages). `build(ctx)` reads the dual `PolygonMesh` and its single-channel `(N, H, W, 1)` `"elevation"` texture from `ctx.world`, runs `RiverGenerator`, and attaches two new single-channel face textures to the mesh: `"filled_height"` (Priority-Flood filled DEM, rewritten over endorheic basins to reflect the equilibrium lake surface and exposed dry bed) and `"lake_mask"` (1.0 where a pixel is inside the post-shrinking lake footprint, 0.0 otherwise). Stores results as `rivers` (`List[River]`) and `lake_polygons` (`List[np.ndarray]`), and registers the `"rivers"` (flattened polylines for rendering/serialization), `"river_objects"` (the rich `River` objects that Resources/Cost consume), `"lakes"`, and `"deltas"` world features.
- The hydrology module is split into: `types.py` (RiverNode/RiverSegment/River dataclasses, `laplacian_smooth_sphere`), `routing.py` (Planchon-Darboux Priority-Flood, Tarboton D-infinity flow, multi-receiver accumulation), `stitching.py` (connected-component search), `crossings.py` (great-circle arc crossing detection, self-crossing removal, inter-river truncation). `rivers.py` contains only `RiverGenerator`. See [docs/river_generation_spec.md](docs/river_generation_spec.md) for design details.
  - **`RiverGenerator`**: End-to-end pipeline — Priority-Flood fill, D-infinity flow direction, precipitation-weighted accumulation, water-balance lake shrinking (endorheic basins sized by PET vs inflow), flat-region outlet biasing, headwater extraction, sub-pixel gradient trace walks, self-crossing removal, inter-river crossing resolution. Rivers trace continuously through exorheic lakes (`in_lake` flags let the renderer hide those segments); endorheic basins are true sinks. Outputs: `generate() → list[River]`, `get_lake_polygons()`, `get_filled_height_texture()`, `get_lake_mask_texture()`.

### `worldgen/climate/` — Climate stage

- **`ClimateParams`** (dataclass in `builder.py`): Nested sub-params: `temperature` (equator/pole temperatures, lapse rate, ocean/land thermal inertia), `wind` (three-cell speeds, seasonal ITCZ shift, monsoon pressure-gradient scale — set to 0 to disable), `precipitation` (global wetness multiplier, advection iterations, continental drying rate, orographic strength, hidden lake-feedback knobs), `currents` (enabled flag, β scale for western-boundary intensification, temperature/precipitation feedback scales, inland decay). Each leaf field carries `metadata` consumed by the params dialog. `GlobeParams` exposes `climate: ClimateParams`.
- **`ClimateBuilder`** (`builder.py`): A pipeline `Stage`. `build(ctx)` (reading the mesh + its `"elevation"` texture from `ctx.world`) pipeline: insolation → coast distances → surface temperatures → zonal + pressure winds → SST-driven moisture advection → (if currents enabled) Stommel streamfunction, then second-pass temperatures and moisture with current ΔT/ΔP → per-pixel textures → biome classification. Attaches `"temperature"` `(N,H,W,K) float32 °C`, `"precipitation"` `(N,H,W,K) float32 mm/season`, `"humidity"` `(N,H,W,K) float32`, `"biome"` `(N,H,W,1) uint8`. Writes `temperature_annual/warmest/coldest`, `precipitation_annual/seasonality`, `biome`, `distance_to_coast`, `humidity_annual` per-face metadata, and registers the `"winds"`/`"currents"` features. The lake-feedback second pass is a separate node: `lake_feedback_stage()` returns a `ClimateLakeFeedbackStage` (sharing the builder) placed after hydrology, whose `build(ctx)` calls `apply_lake_humidity_feedback(mesh, lake_mask_tex)` to re-run moisture advection with lake evaporation as an extra source.
- **`insolation.py`**: Standard TOA daily-mean insolation with polar-day/polar-night handling; 4 seasons (equinoxes + solstices).
- **`temperature.py`**: Parabolic latitude model for annual mean; seasonal anomaly damped by tanh-based ocean/land thermal inertia; lapse-rate correction broadcast to per-pixel texture.
- **`winds.py`**: Three-cell (trades/westerlies/polar) zonal winds with seasonal ITCZ migration. `pressure_winds` adds a monsoonal land–sea thermal contrast term (pressure-gradient proxy, no Coriolis) on top before moisture transport and currents.
- **`humidity.py`**: SST-driven Jacobi moisture transport — ocean faces emit proportional to SST, each iteration distributes moisture downwind with background decay. Continentality drying, ITCZ convergence, and rain shadows emerge from the transport physics. IDW interpolation to per-pixel grid.
- **`precipitation.py`**: Converts per-pixel humidity to precipitation with an orographic factor (wind dot elevation gradient, clamped to [0.3×, 2.0×]).
- **`classification.py`**: 9 biomes (Icecap → Desert); first-match-wins decision tree on T_warmest, T_coldest, P_annual, P_seasonality. Vectorised `classify_textures` mirrors the scalar `default_biome` exactly (pinned by a unit test).
- **`display.py`**: Per-face color schemes (temperature, log-scale precipitation, humidity, biome categorical) and RGBA texture builders for temperature (annual or single season) and precipitation (annual sum, log scale).
- **`currents.py`**: Barotropic Stommel streamfunction — solves (L − β·D_east)·ψ = −curl(τ) on the ocean face graph by GMRES (Voronoi-weighted Laplacian, Dirichlet BC at basin boundaries), recovers divergence-free currents as `u = k̂ × ∇ψ`, advects insolation as a thermal payload, and decays ΔT inland exponentially. Produces additive temperature and multiplicative precipitation deltas applied in a second climate pass. `beta_scale=0` gives symmetric gyres; default adds mild western-boundary intensification.

### `worldgen/geology/display.py` — Geology-specific visualization

Holds the color scheme callables (`_elevation_color`, `_elevation_color_vec`), RGBA constants (`OCEAN_COLOR`, `LAND_COLOR`, `BOUNDARY_COLORS`), and the `_plate_palette` helper. These are imported by `worldgen/globe/builder.py` to wire display — there is no longer a standalone `apply_geology_display` entrypoint in this module.

### `worldgen/resources/` — Resources stage

See [docs/resources_generation_spec.md](docs/resources_generation_spec.md) for full design details.

- **`ResourcesParams`** (dataclass in `builder.py`): Nested params with `fertility: FertilityParams`, `forest_density: ForestDensityParams`, `minerals: MineralParams`, `rare_bio: RareBioParams`, and an optional `seed`. Biome-indexed lookup tables (`BIOME_FERTILITY_MULT`, `BIOME_FOREST_BASELINE`, `BIOME_PRECIP_REFERENCE_MM`) live as **module-level constants** in `fertility.py` / `forest.py` — they define the model rather than tune it.
- **`ResourcesBuilder`** (`builder.py`): A pipeline `Stage`, runs after climate's lake-feedback pass. `build(ctx)` reads the `"river_objects"` world feature (the hydrology `River` objects) plus the `temperature`, `precipitation`, `elevation`, `ocean_mask`, and `lake_mask` face textures and the per-face `biome`, `volcanic_activity`, `boundary_type`, `plate_type` metadata. Attaches `"fertility"` `(N,H,W,1) float32`, `"forest_density"` `(N,H,W,1) float32`, and `"dominant_crop"` `(N,H,W,1) uint8` face textures; writes `fertility_mean`, `forest_density_mean`, `dominant_crop` per-face metadata. Stores `minerals` and `rare_bio` as `(positions, kinds, abundances)` 3-tuples for the globe builder to register as features.
- **`terrain.py`**: Per-face helpers. `face_elevation_range(elevation_tex) → (N,) float32 km` is the per-face pixel elevation span used as a robust "ruggedness" proxy in place of unreliable per-pixel slope (at default tex_size, pixel-spacing slope is dominated by the noise overlay rather than terrain). `face_hops_from_seeds(mesh, seed_mask, max_hops) → (N,) int32` BFS-counts hops on the face-adjacency graph from any seed face; faces beyond `max_hops` get sentinel value `max_hops+1` (callers' `exp(-hops·decay)` naturally treats this as zero contribution). `river_traversed_faces(rivers, n_faces)` and `lake_containing_faces(lake_mask_tex)` build seed masks for the water-proximity BFS; `volcanic_faces(mesh, threshold)` for the volcanism-proximity BFS.
- **`fertility.py`** — `compute_fertility()`: Multiplicative product of five per-face factors — climate (bell-curve in growing-season T and log-P), terrain (penalizes rugged faces by elevation range), water proximity (floodplain BFS hop-decay bonus), volcanism proximity (volcanic-soil BFS hop-decay bonus), and a biome LUT modifier — plus an optional per-pixel forest penalty. Per-face factors are rasterized via vertex-averaged interpolation when `mesh` is supplied so adjacent faces blend continuously.
- **`forest.py`** — `compute_forest_density()`: Biome baseline × precipitation modulation (tanh centred on per-biome reference P) × treeline taper, plus a riparian bonus and a small per-pixel fBm simplex noise overlay. Per-face inputs are rasterized before combining so biome boundaries blend continuously.
- **`crops.py`** — `classify_crops()`: First-match-wins threshold table (T_grow, P_annual, elevation) over 8 crop types. Fertility gates the whole classifier — pixels below `FERTILITY_GATE` (0.08) yield no crop. Vectorised implementation mirrors the scalar `default_crop` reference exactly.
- **`minerals.py`** — `place_mineral_deposits()`: Hybrid geology + fBm noise model. Nine mineral types (Iron, Copper, Gold, Gems, Coal, Tin, Rare Metals, Salt, Marble) each have a geological candidate mask derived from boundary type, plate type, volcanic activity, and elevation; independent 3-octave fBm noise fields cluster deposits within those candidates. Returns `(positions (M,3), kinds (M,) int8, abundances (M,) float32)`.
- **`rare_bio.py`** — `place_rare_bio_resources()`: Ten rare biological resources (Spices, Tea, Silk, Ivory, Furs, Incense, Dyes, Amber, Rubber, Pearls) placed via narrow ecological criteria (biome, T_annual, P_annual, elevation, distance_to_coast) plus fBm noise. A global per-type cap enforces world-level rarity by keeping only the top-scoring candidates.
- **`display.py`**: Color schemes and texture builders for all five resource layers. Fertility and forest density use multi-stop gradients; crops and resources use categorical palettes. All schemes are registered unconditionally (not gated on `debug`) since resources are world-building outputs.

### `worldgen/cost/` — Cost surface stage

Final physical stage. Converts already-computed data into the **base** land + sea traversal-cost layers consumed by the (future) `realmgen` human-geography layer, which adds its danger field on top (see [docs/settlement_generation_design.md](docs/settlement_generation_design.md), Stage 0). Everything is per-face (faces, not pixels, for logic). The unifying concept is **sailable water = ocean ∪ lake**: lakes are impassable on land and navigable like the ocean. **Navigable rivers** (enough flow, flat enough — below the fall line) extend the sailable network inland, so a river connects to the sea directly; such faces are *amphibious* (passable on both the land and sea surfaces — the only overlap).

- **`CostParams`** (dataclass in `builder.py`): Nested `land: LandCostParams`, `sea: SeaCostParams`, `strait: StraitParams`. Land knobs cover the biome-cost scale, ruggedness cost/km, the river term (ford penalty, corridor discount, flow reference, min/max multiplier), and `navigable_ruggedness_km` (the local-relief limit at which a river stops being navigable — the fall line); sea knobs cover coastal/open-ocean cost, the open-ocean distance ramp, the strait discount, and `navigable_flow_min` (minimum river flow to float a boat); strait knobs are `opposite_dot_threshold` + `min_water_neighbors`.
- **`CostBuilder`** (`builder.py`): A pipeline `Stage`. `build(ctx)` runs after elevation + climate + hydrology + resources (it reads the `biome` and `is_ocean` per-face metadata, the `elevation` + `lake_mask` textures, and the hydrology rivers via the `"river_objects"` world feature). Derives `is_water = is_ocean | is_lake` (a face counts as lake when ≥50% of its pixels are lake). Stores `self.surfaces: CostSurfaces` and writes `land_base_cost` / `sea_base_cost` / `is_strait` / `is_water` / `is_navigable_river` / `river_navigability` per-face metadata (the displayable, serializable summary). The full surfaces — including the per-face river flow field — are registered as the `"cost_surfaces"` globe feature.
- **`surfaces.py`** — the model + pure functions (testable against a lightweight stub mesh exposing only `faces`, `face_adjacency`, `face_centroids`). `CostSurfaces` dataclass holds per-face `land_base_cost` + `land_passable` (= `~is_water`), `river_flow_dir (N,3)` + `river_flow_mag` + `river_navigability` (fall-line taper in [0,1]), `sea_base_cost` + `sea_passable` (= water ∪ navigable river), and `strait_mask`. `compute_land_base_cost` (biome LUT `BIOME_TRAVERSAL_COST` × scale + ruggedness via pixel elevation span, water impassable); `compute_river_flow_field` (per-face downstream unit tangent flow direction + max flow, in-lake segments skipped, `exclude_mask` zeros river terms on water faces); `compute_river_navigability` (linear taper from placid→1 to `navigable_ruggedness_km` of relief→0, the fall line) and `navigable_river_mask` (flow ≥ `navigable_flow_min` ∧ navigability>0 ∧ not water); `build_land_edge_costs` folds the **directional** river term into the per-edge cost — cheap *along* a navigable valley (flow ∥ step, discount tapered by navigability so rugged gorges give no discount), expensive to *ford* (flow ⊥ step, not tapered), scaled by flow magnitude and clamped — so realmgen calls the same function after adding danger to the per-face base. `detect_straits` tags water faces pinched between opposite land neighbours; `compute_sea_distance_to_coast` is a Dijkstra over the sailable sub-graph from coastal water (open-ocean→coast, the opposite of climate's `distance_to_coast`); `compute_sea_cost` ramps coastal→open with the strait discount (lakes/navigable rivers read as coastal). `build_sea_edge_costs` is the isotropic ocean-graph edge averaging.

### `worldgen/globe/` — Top-level orchestration

- **`Globe`** ([globe.py](worldgen/globe/globe.py)): Container for a generated world. Holds `mesh: PolygonMesh` plus a generic feature registry: `add_feature(name, data)`, `get_feature(name)`, `has_feature(name)`, `remove_feature(name)`, `feature_names()`. Use the registry for vector-style overlays (rivers, coastlines) that don't fit the per-face structure.
- **`GlobeParams`** ([builder.py](worldgen/globe/builder.py)): Composes per-stage params dataclasses. Exposes `elevation: ElevationParams`, `climate: ClimateParams`, `hydrology: HydrologyParams`, `resources: ResourcesParams`, and `cost: CostParams`; add new stages as new fields.
- **`GlobeBuilder`** ([builder.py](worldgen/globe/builder.py)): Owns a `GlobeParams`, instantiates `elevation_builder: ElevationBuilder`, `climate_builder: ClimateBuilder`, `hydrology_builder: HydrologyBuilder`, `resources_builder: ResourcesBuilder`, and `cost_builder: CostBuilder` — **each is itself a `Stage`** — and composes them into a `Pipeline([elevation, climate, hydrology, climate.lake_feedback_stage(), resources, cost])`. `build(progress=...)` creates a `World` + `BuildContext` and runs `pipeline.build(ctx)`; each stage reads its inputs from the context and registers its own features (so there is no manual orchestration or hand-registration). Returns the `World`. The climate lake-feedback second pass is the explicit node after hydrology. Stage builders remain accessible as attributes. Also registers wind vectors as a `"winds"` globe feature: a list of `(centroids, wind_vectors)` tuples, one per season; and (when currents are enabled) a `"currents"` feature as a one-element list `[(centroids, current_vectors, temperature_delta)]` (3-tuple carrying the annual-mean current and per-face ΔT). `apply_display(window, debug=False)` builds a `DisplayRegistry` by calling each pipeline stage's `register_display(world, reg, debug=...)`, then hands it to the viz adapter (`utils/mapview/registry.py:apply_registry`) for Qt wiring — call this after `build()`. By default only **Elevation** and **Biome** (plus **Dominant Crop**) schemes are registered; with `debug=True` the full set is added: Plates, Plate Type, Speed, Boundary, Volcanic Activity, Annual Temperature, Summer Temperature, Winter Temperature, Annual Precipitation, Humidity, Fertility, Forest Density, Land Cost, Sea Cost (the last two read the per-face cost metadata; Land Cost treats ocean ∪ lake as water, Sea Cost shows all sailable water — ocean, lakes, navigable rivers — highlighting strait chokepoints), plus hover callback on the status bar and winds/currents overlays. The single-channel→RGBA display-texture helpers `build_elevation_display_texture(mesh)` (in `worldgen/geology/display.py`) and `build_biome_display_texture(mesh)` (in `worldgen/climate/display.py`) are re-exported from `worldgen.globe.builder` for back-compat.

### `worldgen/geology/tectonics.py` — Plate tectonics simulation

- **`generate_plates(mesh, n_plates, seed, compactness)`**: Assigns each face to a plate via per-plate randomized-priority flood fill. Each step picks a plate weighted by a random draw (so no plate gets a compounding size advantage) and expands it using a distance-biased heap, producing compact plates with irregular borders. Post-fill, any plate wholly surrounded by a single other plate is absorbed into it. Returns `List[int]` of plate IDs indexed by face.
- **`relax_plates(mesh, plate_ids, max_passes)`**: Iteratively removes appendages using articulation point detection (`_tarjan_aps_iterative`). Each pass finds faces reachable only via a single articulation point and reassigns them to whichever neighboring plate they border most. Repeats until stable.
- **`generate_plate_velocities(mesh, plate_ids, n_plates, drift_scale, spin_scale, seed)`**: Returns `(n_plates, 3)` angular velocity vectors. Each plate's omega combines a random drift component (tangential to the plate centroid, i.e., pure translation across the sphere) and a spin component (along the centroid axis). The velocity of a face on plate k is `omega[k] × face_centroid`.
- **`assign_plate_types(n_plates, oceanic_fraction, seed)`**: Classifies each plate as `'oceanic'` or `'continental'`. Default `oceanic_fraction=0.7` matches Earth's rough ratio.
- **`classify_plate_boundaries(mesh, plate_ids, plate_omegas)`**: Returns `List[dict]` (one per face) with keys `boundary_type` (`'interior'|'convergent'|'divergent'|'transform'`), `normal_avg` (mean normal velocity component across inter-plate edges; positive = diverging), and `neighbor_plate_types` (set of plate type strings seen across those edges). Interior faces get zeros/empty set.

### `realmgen/` — Human geography layer

Standalone module downstream of worldgen. Takes a fully-built `Globe` (read-only) and attaches human geography features. Integrated via `RealmBuilder.build(globe)` — not wired into `GlobeBuilder`; keep realmgen optional.

- **`RealmParams`** (dataclass in `builder.py`): Nested sub-params — `water: WaterProximityParams`, `climate: ClimateHabitabilityParams`, `defensibility: DefensibilityParams`, `resource: ResourceAccessParams`, `weights: SuitabilityWeightParams`. All leaf fields carry `label`/`min`/`max` metadata.
- **`RealmBuilder`** (`builder.py`): `build(globe, progress=None)` reads `is_water`/`is_ocean` metadata, `lake_mask` texture, `cost_surfaces` feature (for `river_flow_mag` river mask), `fertility_mean`/`temperature_*`/`precipitation_annual`/`distance_to_coast` metadata, `elevation` texture, and `minerals`/`rare_bio` globe features. Stores `self.suitability (N,) float32` and writes `"suitability"` per-face metadata.
- **`suitability.py`** — pure testable functions. Five factors, each in [0, 1]: `compute_water_proximity` (BFS hop-decay from river+lake faces + exponential coastal-access term); `compute_climate_habitability` (piecewise-linear T_warmest/T_coldest/P_annual, multiplied); `compute_defensibility` (tanh of `face_elevation_range`); `compute_resource_proximity` (BFS hop-decay from mineral+rare_bio deposit faces via `_positions_to_face_ids` nearest-centroid lookup); `combine_suitability` (weighted sum, water faces forced to 0). Default weights: water=0.30, climate=0.30, fertility=0.25, resources=0.10, defensibility=0.05.

### `main.py` — Primary entrypoint

Top-level entrypoint. Creates a `GlobeBuilder(GlobeParams())`, calls `build()` to produce a `Globe`, passes it to `MapWindow.set_globe(globe)`, and calls `builder.apply_display(window)` to wire color schemes and the elevation texture. By default (non-debug) the available schemes are **Elevation** (multi-stop gradient from abyssal ocean blues through beach/lowland/highland to snow-capped peaks, with a hard break at sea level; lake pixels rendered in lake blue) and **Biome** (categorical 9-class palette). A Re-run callback is registered so the world can be regenerated in place without restarting.

## Key Design Notes

- `Mesh` stores only triangular faces; `PolygonMesh` handles arbitrary polygon faces (e.g., the dual has 12 pentagons + many hexagons). The dual normalizes centroids to the unit sphere so it stays on the same surface as the original.
- `PolygonMesh` pre-computes `face_adjacency` and `face_centroids` at construction; all downstream stages (geology, climate, hydrology, resources) rely on these without recomputing.
- Face rendering has three layers in priority order: per-face textures → named color schemes (callables over `face_metadata`) → static `face_colors` array. Both viewers fall back down this stack when the higher layer is absent.
- Per-face tangent-plane axes are `u_axis = cross(z, n)`, `v_axis = cross(n, u_axis)`, so the texture U-axis is geographic east and V-axis is geographic north for ~90% of faces. Any D4 grid-walk algorithm will produce N-S/E-W-biased output as a result; `rivers.py` avoids this with D-infinity accumulation and sub-pixel gradient traces rather than grid-snapped steps.
