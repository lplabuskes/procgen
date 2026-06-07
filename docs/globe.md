# Globe & Orchestration

## `worldgen/globe/globe.py` — Globe container

**`Globe`**: Container for a generated world. Holds `mesh: PolygonMesh` plus a generic feature registry:
- `add_feature(name, data)`
- `get_feature(name)`
- `has_feature(name)`
- `remove_feature(name)`
- `feature_names()`

Use the registry for vector-style overlays (rivers, coastlines) that don't fit the per-face rasterized structure.

## `worldgen/globe/builder.py` — Orchestration

**`GlobeParams`**: Composes per-stage params dataclasses. Fields: `elevation: ElevationParams`, `climate: ClimateParams`, `hydrology: HydrologyParams`, `resources: ResourcesParams`, `cost: CostParams`. Add new stages as new fields.

**`GlobeBuilder`**: Owns a `GlobeParams`, instantiates one builder per stage, and composes them into:
```
Pipeline([elevation, climate, hydrology, climate.lake_feedback_stage(), resources, cost])
```
`build(progress=...)` creates a `World` + `BuildContext` and runs `pipeline.build(ctx)`; each stage reads its inputs from the context and registers its own features. Returns the `World`. Stage builders remain accessible as attributes.

**`apply_display(window, debug=False)`**: Builds a `DisplayRegistry` by calling each pipeline stage's `register_display(world, reg, debug=...)`, then hands it to the viz adapter (`viz/mapview/registry.py:apply_registry`). Call this after `build()`.

- Default (non-debug) schemes: **Elevation**, **Biome**, **Dominant Crop**
- With `debug=True`: adds Plates, Plate Type, Speed, Boundary, Volcanic Activity, Annual/Summer/Winter Temperature, Annual Precipitation, Humidity, Fertility, Forest Density, Land Cost, Sea Cost; plus hover callback on status bar and winds/currents overlays

Also registers:
- `"winds"` feature: list of `(centroids, wind_vectors)` tuples, one per season
- `"currents"` feature (when currents enabled): `[(centroids, current_vectors, temperature_delta)]`

Re-exports `build_elevation_display_texture(mesh)` and `build_biome_display_texture(mesh)` for back-compat.

## Pipeline stage contract

Each stage implements the `Stage` interface:
- `build(ctx)` — reads from and writes to `BuildContext`; sets `ctx.world.mesh` (elevation stage only), attaches face textures, writes per-face metadata, registers globe features
- `register_display(world, reg, debug=False)` — adds color schemes/overlays to `DisplayRegistry`
- `serialize` / `load`

Standalone callers (scripts) run a stage as `builder.build(BuildContext(world=World()))` and then read result attributes directly.

## `main.py` — Primary entrypoint

Creates `GlobeBuilder(GlobeParams())`, calls `build()`, passes the `Globe` to `MapWindow.set_globe(globe)`, and calls `builder.apply_display(window)`. A Re-run callback is registered so the world can be regenerated in place without restarting.
