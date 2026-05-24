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

## Diagnostic Scripts

```bash
python3 scripts/measure_seam_quality.py              # default seeds 42, 137, 999
python3 scripts/measure_seam_quality.py 1 2 3        # specific seeds
python3 scripts/measure_seam_quality.py --random 5   # N random seeds
python3 scripts/measure_seam_quality.py --pre-noise 42  # base texture before noise
```

`scripts/measure_seam_quality.py` — reports mean / median / RMS / max elevation jump (in metres) across all ~350 k matched boundary pixel pairs per seed. Used to quantify face-seam discontinuities in the elevation texture pipeline. The underlying function is `measure_boundary_elevation_jumps` in `worldgen/geology/elevation.py`.

```bash
python3 scripts/measure_tropical_biome.py              # default seeds 42, 137, 999
python3 scripts/measure_tropical_biome.py 1 2 3        # specific seeds
python3 scripts/measure_tropical_biome.py --random 5   # N random seeds
python3 scripts/measure_tropical_biome.py --detail     # include per-seed biome breakdown
```

`scripts/measure_tropical_biome.py` — reports the fraction of tropical-latitude land classified as Tropical biome, compared against Earth reference values (~9% of global land, ~30% of ±23.5° land, ~22% of ±30° land). Runs elevation + climate (no hydrology). Results are appended to `scripts/results/tropical_biome_results.json` and a human-readable summary is regenerated at `scripts/results/tropical_biome_results.md` for longitudinal comparison across parameter tuning sessions.

## Architecture

This is a procedural world generation project. The core pipeline is: generate a mesh → deregularize it → compute its dual → run stage-specific generation (geology, etc.) → wrap in a `Globe` → display it.

The top-level orchestrator is `worldgen.globe.GlobeBuilder`, which runs each stage (currently just elevation/tectonics) and returns a `Globe`. The `Globe` holds the surface `mesh` plus a generic `features` registry for vector-style overlays (rivers, coastlines, etc.) that don't fit the per-face rasterized structure.

### `worldgen/mesh/mymesh.py` — Mesh data structures

- **`PolygonMesh`**: Base class for any polygon mesh. Fan-triangulates faces and pre-computes `render_vertices` and `render_colors` (flat per-vertex arrays) at construction time for direct VBO upload. Also pre-computes `face_adjacency` (edge-sharing neighbor lists), `face_centroids` (unit-sphere centroids, shape `(N, 3)`), and `face_metadata` (list of per-face `dict`s). The triangulation step also builds `render_face_indices` (per-render-vertex face index, shape `(M,)`) and `render_local_uvs` (per-render-vertex UV in [0, 1]², shape `(M, 2)`) using a tangent-plane parameterization at each face centroid — these are consumed by the texture pipeline. Exposes `set_face_colors(colors)` to update colors and rebuild render arrays, `set_face_metadata(face_idx, **kw)` to write arbitrary attributes into a face's metadata dict, `apply_metadata_channel(key, values)` to bulk-write a single key across all faces at once (raises `ValueError` if `len(values)` doesn't match the face count), `set_face_textures(name, textures, n_channels=4)` to attach a named `(N, H, W, n_channels)` float32 texture array (raises `ValueError` if shape doesn't match; use `n_channels=4` for RGBA display textures and `n_channels=1` for single-channel data layers like elevation or lake mask), and `face_texture_positions(h, w)` to get the unit-sphere 3D position of every pixel in a `(H, W)` texture grid — returns `float32 (N_faces, H, W, 3)` by inverting the same tangent-plane parameterization used for UVs.
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
- Elevation textures throughout this module are single-channel `(N, H, W, 1)` float32 arrays (channel 0 = elevation in km). Previously 4-channel RGBA; the extra channels have been dropped.

### `worldgen/geology/elevation_builder.py` — Geology stage orchestration

- **`ElevationParams`** (dataclass): Centralizes every tuning knob in the elevation/tectonics pipeline. Groups: *mesh generation* (`icosphere_depth`, `n_swaps`, `n_relax`, `stiffness`, `outer_iters`), *plate layout* (`n_plates`, `plate_compactness`, `relax_max_passes`, `oceanic_fraction`), *plate motion* (`drift_scale`, `spin_scale`), *elevation* (`elevation_rate_norm`, `elevation_decay_rate`, `elevation_max_hops`), *noise*, *face erosion*, *volcanism*, and *reproducibility* (`seed`). A top-level `seed` fans out to per-stage sub-seeds; `None` means random each run. River parameters live in `HydrologyParams` (hydrology stage).
- **`ElevationBuilder`**: Owns an `ElevationParams` and runs the elevation/tectonics pipeline when `build()` is called. Returns the dual `PolygonMesh` with all metadata applied and stores intermediate results as attributes: `mesh`, `plate_ids`, `plate_omegas`, `face_velocities`, `boundary_types`, `boundary_data`, `elevations`, `elevation_textures` (single-channel `(N, H, W, 1)` float32), `volcanism`, `max_speed`. The mesh has a `"elevation"` face texture (single-channel) attached. Callers (e.g. visualization code and `HydrologyBuilder`) read these attributes without re-running any work.

### `worldgen/hydrology/` — Hydrology stage

- **`HydrologyParams`** (dataclass in `builder.py`): Tuning knobs for the hydrology stage. Currently: `river_flow_threshold` (default `200.0`) — minimum upstream flow accumulation for a node to count as a river headwater.
- **`HydrologyBuilder`** (`builder.py`): Runs the hydrology pipeline after elevation is built. `build(mesh, elevation_textures, progress=...)` takes the dual `PolygonMesh` and the single-channel `(N, H, W, 1)` elevation texture from `ElevationBuilder`, runs `RiverGenerator`, and attaches two new single-channel face textures to the mesh: `"filled_height"` (Priority-Flood filled DEM) and `"lake_mask"` (1.0 where a pixel is inside a lake, 0.0 otherwise). Stores results as `rivers` (`List[River]`) and `lake_polygons` (`List[np.ndarray]`).
- **`rivers.py`** — Per-face grid + Priority-Flood + D-infinity routing + sub-pixel gradient walk. Each face's elevation texture is treated as a `resolution × resolution` grid of nodes; intra-face neighbours form a D8 (8-connected) grid and inter-face neighbours are stitched along shared polygon edges. See [docs/river_generation_spec.md](docs/river_generation_spec.md) for the as-built design and rationale.
  - **`RiverNode` / `RiverSegment` / `River`** (dataclasses): A `River` is an ordered list of `RiverSegment`s from `source` to `mouth`; `as_polyline()` returns the 3D positions and `flow_rates()` returns the per-segment upstream flow.
  - **`RiverGenerator`** (`mesh, elevation_textures, min_flow=50.0, sea_level=0.0, smoothing_iterations=8, smoothing_alpha=0.5`): End-to-end pipeline. `generate() -> list[River]` runs Priority-Flood, D-infinity flow direction, multi-receiver accumulation, headwater extraction, and sub-pixel trace walks followed by Laplacian smoothing. `get_lake_polygons()` returns lake outlines as unit-sphere polylines. `get_filled_height_texture()` and `get_lake_mask_texture()` return `(N, H, W, 1)` float32 arrays.
  - **Inter-face stitching** (`_stitch_face_boundaries`): For each adjacent face pair, finds the two shared mesh vertices, computes both faces' UVs at those vertices via `face_tangent_frames()`, and matches pixels lying within ~½ pixel of the shared edge in each face by parameter `t` along the edge. Matched pixel pairs become bidirectional inter-face edges, stored sparsely in `_inter_adj`.

### `worldgen/climate/` — Climate stage

- **`ClimateParams`** (dataclass in `builder.py`): Top-level params dataclass with nested sub-params: `temperature: TemperatureParams`, `wind: WindParams`, `precipitation: PrecipitationParams`, `currents: CurrentsParams`, and an optional `seed`. Each leaf field carries `metadata` (`label`, `min`, `max`, optional `step`/`hidden`) consumed by the params dialog. `GlobeParams` exposes `climate: ClimateParams`.
  - **`TemperatureParams`**: `equator_annual_c` (27 °C), `pole_annual_c` (−25 °C), `anomaly_scale` (seasonal swing scale, 0.05), `lapse_rate_c_per_km` (6.5), `ocean_inertia` (0.30), `land_inertia` (1.0), `inertia_distance_km` (1500).
  - **`WindParams`**: `trade_speed` (7 m/s), `westerly_speed` (10 m/s), `polar_speed` (5 m/s), `itcz_shift_deg` (10°, amplitude of seasonal ITCZ migration), `pressure_wind_scale` (200 m·s⁻¹·km/°C — strength of the cookbook pressure-gradient wind from land/sea thermal contrast; set to 0 to disable).
  - **`PrecipitationParams`**: `itcz_mm_annual` (2200), `horse_mm_annual` (250), `midlat_mm_annual` (900), `polar_mm_annual` (200), `orographic_k` (0.4), `continentality_decay_km` (1200).
  - **`CurrentsParams`**: `enabled` (True), `beta_scale` (1.0 — scale factor on the planetary vorticity gradient β; 0 gives symmetric gyres, >1 sharpens western-boundary intensification), `advection_alpha` (1.0 — pure Lagrangian transport with source clamping), `advection_iterations` (400), `temperature_scale` (0.06 °C / (W/m²)), `precipitation_scale` (0.03 multiplicative wet delta per °C), `inland_decay_km` (600).
- **`ClimateBuilder`** (`builder.py`): Runs the climate stage on the built elevation mesh. `build(mesh, elevation_textures, progress=...)` takes the dual `PolygonMesh` and the single-channel `(N, H, W, 1)` elevation texture from `ElevationBuilder`. Pipeline steps: (1) `seasonal_insolation` — TOA W/m² per face per season; (2) `_distance_to_coast_km` — Dijkstra on the face graph from ocean seeds, result `(N,)` km; (3) `surface_temperatures` — `(N, K)` °C; (4) `zonal_winds` — `(N, K, 3)` m/s; (5) `surface_precipitation` — `(N, K)` mm/season using per-face elevation gradient; (6) if `params.currents.enabled`, `compute_currents` runs and the `(N,)` annual ΔT and ΔP deltas are broadcast as `(N, 1)` into a second-pass `surface_temperatures` (via `coastal_current_temp_delta`) and `surface_precipitation` (via `current_wet_delta`); (7) `per_pixel_temperatures` — lapse-rate-corrected `(N, H, W, K)` texture; (8) `per_pixel_precipitation` — per-pixel orographic correction → `(N, H, W, K)` texture; (9) `classify_textures` — `(N, H, W, 1) uint8` biome texture. Attaches three named face textures: `"temperature"` `(N,H,W,K) float32 °C`, `"precipitation"` `(N,H,W,K) float32 mm/season`, `"biome"` `(N,H,W,1) uint8`. Writes seven per-face metadata keys: `temperature_annual`, `temperature_warmest`, `temperature_coldest` (°C), `precipitation_annual` (mm/yr), `precipitation_seasonality` (ratio), `biome` (int index), `distance_to_coast` (km). Intermediate arrays are stored as attributes (`insolation`, `distance_to_coast`, `temperature_face`, `precipitation_face`, `winds`, `currents`) for downstream use.
- **`insolation.py`**: `seasonal_insolation(latitudes_rad, declinations_rad) → (N, K) float64` — daily-mean TOA insolation using the standard hour-angle integral, with explicit polar-day/polar-night handling. `SEASON_DECLINATIONS_DEG = [0, 23.5, 0, −23.5]` (equinoxes and solstices). `face_latitudes(face_centroids) → (N,)` — `arcsin(z)` of unit-sphere centroids.
- **`temperature.py`**: `surface_temperatures(latitudes_rad, insolation_w_m2, face_elevations_km, distance_to_coast_km, ...) → (N, K) float32 °C` — annual mean is parabolic in |lat|; seasonal anomaly is proportional to insolation departure from annual mean, damped by a `tanh`-based ocean/land thermal inertia factor; lapse rate applied above sea level only. `per_pixel_temperatures(face_temperatures_c, face_elevations_km, pixel_elevations_km, lapse_rate) → (N, H, W, K) float32` — broadcasts face temperatures to the texture grid and re-applies the lapse-rate correction per pixel (sub-sea-level pixels pinned to sea surface).
- **`winds.py`**: `zonal_winds(face_centroids, latitudes_rad, ...) → (N, K, 3) float32 m/s` — three-cell Hadley/Ferrel/Polar model. Bands: trades (|ϕ_eff| < 30°), westerlies (30–60°), polar easterlies (60°+). The ITCZ shifts seasonally by ±`itcz_shift_deg`·sin(phase). All wind vectors are tangent to the sphere (east/north basis per face); diagonal boundary winds are normalised to avoid √2 speed inflation. `pressure_winds(face_centroids, face_adjacency, temperature_face, *, pressure_wind_scale) → (N, K, 3) float32 m/s` — cookbook pressure-gradient wind from land/sea thermal contrast: treats `-(T - T_annual)` as a pressure proxy, computes its tangent-plane gradient on the face graph via batched normal equations (`pinv` for robustness), and returns `scale · ∇(T_anomaly)` so air blows from cold/high-pressure toward warm/low-pressure (monsoonal mass flux; no Coriolis rotation). The climate builder adds this on top of `zonal_winds` before precipitation and currents, so both consume the combined wind.
- **`precipitation.py`**: `surface_precipitation(latitudes_rad, winds_m_per_s, face_elevation_gradient, distance_to_coast_km, ...) → (N, K) float32 mm/season` — Gaussian-weighted blend of four zonal band baselines (ITCZ/horse/mid-lat/polar), multiplied by an orographic factor (wind·∇h uplift) and a continentality factor (`1 − 0.8·tanh(dist/decay)`) that dries the interior. `per_pixel_precipitation(face_precipitation, face_winds, ..., pixel_elevations_km, orographic_k) → (N, H, W, K) float32` — applies a residual per-pixel orographic correction on top of the face-level value, using the pixel-level elevation gradient from `np.gradient`.
- **`classification.py`**: `BIOME_NAMES` lists 9 biomes indexed 0–8: Icecap, Tundra, Taiga, Temperate, Steppe, Savannah, Tropical, Semi-Arid, Desert. `default_biome(temperature_c, precipitation_mm, lat_deg) → int` — scalar first-match-wins decision tree on T_warmest, T_coldest, P_annual, and P_seasonality. `classify_textures(temperature_tex, precipitation_tex, latitudes_deg) → (N, H, W, 1) uint8` — vectorised version that mirrors `default_biome` exactly (pinned by a unit test).
- **`display.py`**: Per-face color schemes: `temperature_color(meta, key="temperature_annual")`, `precipitation_color(meta)` (log-scale), `biome_color(meta)` (categorical `BIOME_PALETTE`). Texture builders: `build_temperature_display_texture(temperature_tex, season=None, elevation_tex=None) → (N,H,W,4) float32` (annual mean or one season; ocean pixels overwritten), `build_precipitation_display_texture(precipitation_tex, elevation_tex=None)` (annual sum, log scale), `build_biome_display_texture(biome_tex, elevation_tex=None)` (BIOME_PALETTE lookup). `GlobeBuilder.apply_display` pre-builds and attaches named RGBA textures: `"temperature_annual_rgba"`, `"temperature_summer_rgba"`, `"temperature_winter_rgba"`, `"precipitation_rgba"`, `"biome_rgba"`, and registers color schemes `"Annual Temperature"`, `"Summer Temperature"`, `"Winter Temperature"`, `"Annual Precipitation"`, `"Biome"`.
- **`currents.py`**: Annually-averaged surface ocean currents and their climate feedback. `compute_currents(mesh, face_elev_km, face_centroids, winds_per_season, face_insolation, ...) → CurrentsResult` returns the per-face current vector `(N, 3) m/s` (zero on land), the raw streamfunction `streamfunction (N,)` (solver units, zero on land), an additive `temperature_delta (N,) °C`, a multiplicative `precipitation_delta (N,)`, and the post-advection `advected_insolation (N,) W/m²`. The model is a barotropic Stommel streamfunction: (1) `_wind_stress_curl` computes curl(τ) per ocean face via a least-squares tangent-plane gradient fit of the annual-mean wind using quadratic drag (τ = C_D·|u|·u, C_D = 1.3e-3); (2) `_solve_streamfunction` solves (L − β·D_east)·ψ = −curl(τ)·R on the ocean face graph by GMRES — `L` is a Voronoi-weighted graph Laplacian (w = edge_len/centroid_dist, approximates −∇² in FVM sense), the β term adds Stommel planetary-vorticity-gradient forcing for western-boundary intensification, R = EARTH_RADIUS_KM scales the RHS to match the Voronoi Laplacian's dimensionless weights, and `ψ = 0` Dirichlet BC on every ocean–land neighbour pins each basin's circulation closed; (3) `_streamfunction_currents` recovers `u = k̂ × ∇ψ` per ocean face (divergence-free by construction); (4) `_build_advection_edges` projects this velocity onto outgoing ocean-to-ocean edges; (5) `_advect_field` carries insolation as a thermal payload via damped Jacobi with per-destination averaging (`α=1.0` ⇒ pure Lagrangian sampling); (6) ocean ΔT = `temperature_scale · (advected − local)`; (7) a Dijkstra from ocean faces (tracking each land face's parent ocean) carries ΔT inland with `exp(−d / inland_decay_km)` decay; (8) wet delta = `precipitation_scale · ΔT`. The `(N,)` deltas are broadcast across seasons via the existing `coastal_current_temp_delta` / `current_wet_delta` hooks. No seasonal resolution; the model is annual-mean. Typical ocean speeds: ~5e-3 m/s mean, ~0.015 m/s max (depth-5, seed=42). The β east-derivative stencil uses a nearest-east-neighbour finite difference; `beta_scale=0` gives symmetric gyres, `beta_scale=1.0` (default) adds mild western intensification.

### `worldgen/geology/display.py` — Geology-specific visualization

Holds the color scheme callables (`_elevation_color`, `_elevation_color_vec`), RGBA constants (`OCEAN_COLOR`, `LAND_COLOR`, `BOUNDARY_COLORS`), and the `_plate_palette` helper. These are imported by `worldgen/globe/builder.py` to wire display — there is no longer a standalone `apply_geology_display` entrypoint in this module.

### `worldgen/globe/` — Top-level orchestration

- **`Globe`** ([globe.py](worldgen/globe/globe.py)): Container for a generated world. Holds `mesh: PolygonMesh` plus a generic feature registry: `add_feature(name, data)`, `get_feature(name)`, `has_feature(name)`, `remove_feature(name)`, `feature_names()`. Use the registry for vector-style overlays (rivers, coastlines) that don't fit the per-face structure.
- **`GlobeParams`** ([builder.py](worldgen/globe/builder.py)): Composes per-stage params dataclasses. Exposes `elevation: ElevationParams`, `climate: ClimateParams`, and `hydrology: HydrologyParams`; add new stages as new fields.
- **`GlobeBuilder`** ([builder.py](worldgen/globe/builder.py)): Owns a `GlobeParams`, instantiates `elevation_builder: ElevationBuilder`, `climate_builder: ClimateBuilder`, and `hydrology_builder: HydrologyBuilder`. `build(progress=...)` runs all three stages in order (elevation → climate → hydrology) and returns a `Globe`. Stage builders remain accessible as attributes. Also registers wind vectors as a `"winds"` globe feature: a list of `(centroids, wind_vectors)` tuples, one per season; and (when currents are enabled) a `"currents"` feature as a one-element list `[(centroids, current_vectors)]` carrying the annual-mean current. `apply_display(window)` wires all color schemes (Elevation, Plates, Plate Type, Speed, Boundary, Volcanic Activity; plus climate schemes when the climate stage ran), pre-built RGBA display textures, and river/lake/winds/currents overlays into a `MapWindow` — call this after `build()`. `build_elevation_display_texture(mesh)` (module-level helper) converts the `"elevation"` single-channel texture to RGBA and applies the `"lake_mask"` texture if present.

### `worldgen/geology/tectonics.py` — Plate tectonics simulation

- **`generate_plates(mesh, n_plates, seed, compactness)`**: Assigns each face to a plate via per-plate randomized-priority flood fill. Each step picks a plate weighted by a random draw (so no plate gets a compounding size advantage) and expands it using a distance-biased heap, producing compact plates with irregular borders. Post-fill, any plate wholly surrounded by a single other plate is absorbed into it. Returns `List[int]` of plate IDs indexed by face.
- **`relax_plates(mesh, plate_ids, max_passes)`**: Iteratively removes appendages using articulation point detection (`_tarjan_aps_iterative`). Each pass finds faces reachable only via a single articulation point and reassigns them to whichever neighboring plate they border most. Repeats until stable.
- **`generate_plate_velocities(mesh, plate_ids, n_plates, drift_scale, spin_scale, seed)`**: Returns `(n_plates, 3)` angular velocity vectors. Each plate's omega combines a random drift component (tangential to the plate centroid, i.e., pure translation across the sphere) and a spin component (along the centroid axis). The velocity of a face on plate k is `omega[k] × face_centroid`.
- **`assign_plate_types(n_plates, oceanic_fraction, seed)`**: Classifies each plate as `'oceanic'` or `'continental'`. Default `oceanic_fraction=0.7` matches Earth's rough ratio.
- **`classify_plate_boundaries(mesh, plate_ids, plate_omegas)`**: Returns `List[dict]` (one per face) with keys `boundary_type` (`'interior'|'convergent'|'divergent'|'transform'`), `normal_avg` (mean normal velocity component across inter-plate edges; positive = diverging), and `neighbor_plate_types` (set of plate type strings seen across those edges). Interior faces get zeros/empty set.

### `worldgen/main.py` — Primary entrypoint

Top-level entrypoint. Creates a `GlobeBuilder(GlobeParams())`, calls `build()` to produce a `Globe`, passes it to `MapWindow.set_globe(globe)`, and calls `builder.apply_display(window)` to wire color schemes and the elevation texture. Available schemes: **Elevation** (multi-stop gradient from abyssal ocean blues through beach/lowland/highland to snow-capped peaks, with a hard break at sea level; lake pixels rendered in lake blue), **Plates** (HSV hue-wheel palette by plate ID), **Plate Type** (blue for oceanic, tan for continental), **Speed** (warm-to-cool by surface speed), **Boundary** (color-coded by convergent/divergent/transform/interior), **Volcanic Activity**, **Annual Temperature** / **Summer Temperature** / **Winter Temperature** (diverging palette anchored at 0 °C), **Annual Precipitation** (log-scale sequential, ~50–4000 mm/yr), and **Biome** (categorical 9-class palette). A Re-run callback is registered so the world can be regenerated in place without restarting.

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
