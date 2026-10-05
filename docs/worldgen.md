# Globe & Orchestration

## `worldgen/builder.py` — Orchestration

**`GlobeParams`**: `mesh`, `elevation`, `climate`, `hydrology`, `delta`, `coast`,
`resources`, `danger`, `cost`, `seed` (drives mesh selection and every stage's
seeded draws; `None` → random). New stages = new fields.

**`GlobePipeline`** (a `Pipeline`):
```
Pipeline([elevation, climate, hydrology, delta, climate.lake_feedback_stage(), coast, resources, danger, cost])
```
Built world = `self.globe` (alias of `Pipeline.world`). Builders exposed as
`elevation_builder`, `climate_builder`, `hydrology_builder`, `delta_builder`,
`coast_builder`, `resources_builder`, `danger_builder`, `cost_builder`. Only
`ElevationBuilder` takes the world seed directly; danger and coast draw from
`ctx.rng` ([danger.md](danger.md), [coast.md](coast.md)).

**Display** via inherited `Pipeline.apply_display(window, debug=...)`.
- Default: schemes **Elevation**, **Biome**; overlay **Rivers**.
- `--debug` adds schemes Plates, Plate Type, Speed, Boundary, Volcanic Activity, Annual/Summer/Winter Temperature, Annual Precipitation, Humidity, Fertility, Forest Density, Dominant Crop, Delta, Coast Smoothness, Danger, Land Cost, Sea Cost; overlays Volcanoes, Winds (×4), Currents, River Deltas, Minerals, Rare Resources, Danger Sources; per-stage hover fragments.

**`save_globe(pipeline, path)`** / **`load_globe(path) → (GlobePipeline, World)`**:
physical-only persistence over `core/persistence.py` (scripts cache the globe and
re-run realm). Re-exported with `GlobeParams` / `GlobePipeline` from
`worldgen/__init__.py`.

## Pipeline stage contract

Each stage implements `Stage` ([core.md](core.md)): `build(ctx)` (attach face
textures, write per-face metadata, register features), `register_display(world,
reg, debug=False)`, `serialize` / `load`. Scripts run a stage standalone as
`builder.build(BuildContext(world=World()))` and read result attributes.

## `main.py` — Primary entrypoint

`AppParams` = `globe`, `realm`, `region`, `local` (last two view-side only, never
rebuild the world). `AppPipeline` = `Pipeline([GlobePipeline, RealmPipeline])`.

Each run: `World(globe_params.mesh, seed=...)` (random if `None`) →
`AppPipeline.build(ctx)` → one `apply_display(window, debug=...)`; also
`set_world_params` / `set_realm_params` / `set_region_params` /
`set_local_params` for the zoomed views.

- **Re-run**: fresh random seed.
- **Parameters**: `AppParamsDialog`; edits touching only `region` / `local` re-refine the views, anything else rebuilds.
- **Save**: `.npz` via `core.persistence.save_world`; `python3 main.py path.npz` loads (`load_world` + each stage's `load`).
- `--debug`: debug layers/overlays. `--projection=NAME`: `robinson` (default), `mollweide`, `mercator`, `equirectangular`.
