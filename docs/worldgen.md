# Globe & Orchestration

## `worldgen/builder.py` — Orchestration

**`GlobeParams`**: Composes per-stage params dataclasses. Fields: `mesh: MeshParams`, `elevation: ElevationParams`, `climate: ClimateParams`, `hydrology: HydrologyParams`, `resources: ResourcesParams`, `cost: CostParams`, `seed: Optional[int]`. Add new stages as new fields.

**`GlobePipeline`** (extends `Pipeline`): Owns a `GlobeParams`, instantiates one builder per stage, and composes them into:
```
Pipeline([elevation, climate, hydrology, climate.lake_feedback_stage(), resources, cost])
```
`build(ctx)` runs `pipeline.build(ctx)` and stores the result as `self.globe`. Stage builders remain accessible as attributes (`elevation_builder`, `climate_builder`, etc.).

**`apply_display(window, debug=False)`**: Calls each stage's `register_display(world, reg, debug=...)` to populate a `DisplayRegistry`, then hands it to `viz/mapview/registry.py:apply_registry`.

- Default (non-debug) schemes: **Elevation**, **Biome**, **Dominant Crop**
- With `debug=True`: adds Plates, Plate Type, Speed, Boundary, Volcanic Activity, Annual/Summer/Winter Temperature, Annual Precipitation, Humidity, Fertility, Forest Density, Land Cost, Sea Cost; plus hover callback on status bar and winds/currents overlays

**`save_globe(pipeline, path)`** / **`load_globe(path) → (GlobePipeline, World)`**: Persistence helpers. All four are re-exported from `worldgen/__init__.py`.

## Pipeline stage contract

Each stage implements the `Stage` interface:
- `build(ctx)` — reads from and writes to `BuildContext`; attaches face textures, writes per-face metadata, registers globe features
- `register_display(world, reg, debug=False)` — adds color schemes/overlays to `DisplayRegistry`
- `serialize` / `load`

Standalone callers (scripts) run a stage as `builder.build(BuildContext(world=World()))` and then read result attributes directly.

## `main.py` — Primary entrypoint

`AppParams` combines `globe: GlobeParams` and `realm: RealmParams`. On each run: creates `World(globe_params.mesh, seed=...)`, then `GlobePipeline.build(ctx)`, then `RealmPipeline.build(ctx)` on the same context. Both pipelines call `apply_display(window)`. A Re-run callback regenerates the world without restarting.
