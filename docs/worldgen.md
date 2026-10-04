# Globe & Orchestration

## `worldgen/builder.py` — Orchestration

**`GlobeParams`**: Composes per-stage params dataclasses. Fields: `mesh: MeshParams`, `elevation: ElevationParams`, `climate: ClimateParams`, `hydrology: HydrologyParams`, `delta: DeltaParams`, `coast: CoastParams`, `resources: ResourcesParams`, `danger: DangerParams`, `cost: CostParams`, `seed: Optional[int]` (drives mesh selection and every stage's seeded draws; `None` → random). Add new stages as new fields.

**`GlobePipeline`** (extends `Pipeline`): Owns a `GlobeParams`, instantiates one builder per stage, and composes them into:
```
Pipeline([elevation, climate, hydrology, delta, climate.lake_feedback_stage(), coast, resources, danger, cost])
```
`build(ctx)` (inherited) runs each stage in order; the built world is `self.globe` (an alias of `Pipeline.world`). Stage builders remain accessible as attributes (`elevation_builder`, `climate_builder`, `hydrology_builder`, `delta_builder`, `coast_builder`, `resources_builder`, `danger_builder`, `cost_builder`). Only `ElevationBuilder` takes the world seed directly; see [danger.md](danger.md) and [coast.md](coast.md) for the stages that draw from `ctx.rng`.

**Display** is inherited from `Pipeline.apply_display(window, debug=...)`: each stage's `register_display` fills one `DisplayRegistry`, which `viz/mapview/registry.py:apply_registry` wires into the window.

- Default (non-debug) physical schemes: **Elevation**, **Biome**; overlay **Rivers**
- With `debug=True`: adds Plates, Plate Type, Speed, Boundary, Volcanic Activity, Annual/Summer/Winter Temperature, Annual Precipitation, Humidity, Fertility, Forest Density, Dominant Crop, Delta, Coast Smoothness, Danger, Land Cost, Sea Cost; overlays Volcanoes, Winds (×4 seasons), Currents, River Deltas, Minerals, Rare Resources, Danger Sources; and the per-stage hover fragments

**`save_globe(pipeline, path)`** / **`load_globe(path) → (GlobePipeline, World)`**: physical-world-only persistence over `core/persistence.py` (used by the measurement scripts, which cache the globe and re-run realm). These two plus `GlobeParams` / `GlobePipeline` are re-exported from `worldgen/__init__.py`.

## Pipeline stage contract

Each stage implements the `Stage` interface (see [core.md](core.md)):
- `build(ctx)` — reads from and writes to `BuildContext`; attaches face textures, writes per-face metadata, registers globe features
- `register_display(world, reg, debug=False)` — adds color schemes/overlays/hover to `DisplayRegistry`
- `serialize` / `load`

Standalone callers (scripts) run a stage as `builder.build(BuildContext(world=World()))` and then read result attributes directly.

## `main.py` — Primary entrypoint

`AppParams` combines `globe: GlobeParams`, `realm: RealmParams`, `region: RegionParams` and `local: LocalParams` (the last two are view-side only and never rebuild the world). `AppPipeline` is a `Pipeline` of `[GlobePipeline, RealmPipeline]`.

On each run: creates `World(globe_params.mesh, seed=...)` (a random seed is drawn when `GlobeParams.seed` is `None`), runs `AppPipeline.build(ctx)`, then one `apply_display(window, debug=...)` over the whole tree. It also hands the window `set_world_params` / `set_realm_params` / `set_region_params` / `set_local_params`, which the region and local views refine with.

- **Re-run** regenerates with a fresh random seed.
- **Parameters** opens `AppParamsDialog`; an edit that only touches `region` / `local` re-refines the views, anything else rebuilds the world.
- **Save** writes a `.npz` with `core.persistence.save_world` over the `AppPipeline` + `AppParams`; `python3 main.py path.npz` loads one (`load_world`, then each stage's `load`).
- `--debug` enables the debug layers and overlays.
- `--projection=NAME` picks the world map's projection (`robinson` — the default — `mollweide`, `mercator`, `equirectangular`).
