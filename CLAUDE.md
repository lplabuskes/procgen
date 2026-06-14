# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Style

- Ask for clarification rather than churning your wheels. If the task or approach is ambiguous, ask before attempting.
- Assume visuals should be limited to debug mode, ask before adding to the main view.
- Don't worry about backwards compatibility for saving and loading worlds


## Running the Project

```bash
source .venv/bin/activate
pip install -e .                    # one-time: installs project packages into the venv
python3 main.py                     # primary entrypoint — full procedural world viewer
python3 -m viz.mapview.main         # 2-D map viewer (dev)
```

No build step. Run the linter, type checker, and test suite with:

```bash
python3 -m flake8                   # lint (config in pyproject.toml: max-line-length=120)
python3 -m pyright                  # static type checking (target: 0 errors)
python3 -m pytest                   # all configured test paths (see pyproject.toml)
```

Each module and sub-module has a `test/` directory with its own tests, all wired into the `pytest` testpaths in `pyproject.toml`.


## Diagnostic Scripts

```bash
python3 scripts/measure_seam_quality.py   [seeds…|--random N|--pre-noise seed]
python3 scripts/measure_climate_seams.py  [seeds…|--random N]
python3 scripts/measure_zone_biomes.py    [seeds…|--random N|--detail]
python3 scripts/measure_resources.py      [seeds…|--random N|--detail]
python3 scripts/measure_population.py     [seeds…|--random N|--detail|--rebuild-cache]
python3 scripts/measure_population_absolute.py [seeds…|--random N|--detail|--rebuild-cache]
python3 scripts/measure_rivers.py         [seeds…|--random N|--detail]
python3 scripts/measure_polities.py       [seeds…|--random N|--detail|--rebuild-cache]
python3 scripts/profile_pipeline.py       [seeds…|--repeat N|--top N|--cprofile [--module M]]
```

- `measure_seam_quality.py` — mean/median/RMS/max elevation jump (km) across ~350k matched boundary pixel pairs per seed.
- `measure_climate_seams.py` — mean/median/RMS/max temperature (°C) and precipitation (mm/season) boundary jumps after IDW smoothing.
- `measure_zone_biomes.py` — Tropical, Savannah, Desert fractions within tropical latitudes vs Earth reference values. Runs elevation + climate (no hydrology). Results appended to `scripts/results/zone_biome_results.json`.
- `measure_resources.py` — per-biome mean fertility and forest density vs Earth-ish anchors. Runs the full pipeline. Results appended to `scripts/results/resources_results.json`.
- `measure_population.py` — city-size distribution shape (rank-size slope, R², Gini) vs high-medieval Europe. Caches the physical world build; re-runs only the realm stage.
- `measure_population_absolute.py` — absolute population grounding (world total ~300–400 M people, top-city 50k–500k band, trade-fed-city test) vs medieval Earth c.1300. Same world cache; default seeds 4/31/35. Results appended to `scripts/results/population_absolute_results.json`.
- `measure_rivers.py` — river source-to-mouth length distribution (longest, percentiles, count over 1/2/3/4k km, % reaching ocean) vs Earth's great rivers. Runs elevation + climate + hydrology. Results appended to `scripts/results/rivers_results.json`. Default seeds (4, 31, 35) are chosen for Earth-like land fraction (~29%) and a continent-scale largest landmass.
- `measure_polities.py` — polity-forest structure: sovereign/vassal counts, largest-domain face fraction (consolidation), max tree depth, and `corr(infrastructure, vassal_fraction)` (should be negative — high-infra realms hold directly, low-infra ones fragment). Caches the physical world; re-runs only the realm stage. Default seeds 4/31/35.
- `profile_pipeline.py` — per-phase wall-clock breakdown of the full globe pipeline. Use `--cprofile` for function-level drill-down.


## Architecture

Pipeline: mesh → deregularize → dual → **elevation/tectonics** → **climate** → **hydrology** → **climate lake-feedback** → **resources** → **cost** → *(optional)* **realmgen**

Top-level orchestrator: `GlobePipeline` in `worldgen/builder.py`. Each stage is a `Stage` that reads from and writes to a `BuildContext`. Realmgen is a separate `RealmPipeline` with sub-stages: scoring → placement → trade → population → polity. Both pipelines are wired together in `main.py` via `AppParams`.

Stage reference docs (read the relevant one before working on a stage):

- [docs/core.md](docs/core.md) — `core/` infrastructure (pipeline, channels, params, display, mesh cache, mesh data structures)
- [docs/viz.md](docs/viz.md) — `viz/` layer (MapWidget, MapWindow, projections, antimeridian, registry adapter, MeshViewer, ParamsDialog, color_stops)
- [docs/worldgen.md](docs/worldgen.md) — `worldgen/builder.py`, `main.py`, pipeline stage contract
- [docs/elevation.md](docs/elevation.md) — `worldgen/geology/`
- [docs/climate.md](docs/climate.md) — `worldgen/climate/`
- [docs/hydrology.md](docs/hydrology.md) — `worldgen/hydrology/`
- [docs/resources.md](docs/resources.md) — `worldgen/resources/`
- [docs/cost.md](docs/cost.md) — `worldgen/cost/`
- [docs/realmgen.md](docs/realmgen.md) — `realmgen/`

Historical design specs live in `docs/plans/`.


## World Scale (default params)

Icosphere depth 5 → **10,242 dual faces** (hexagons + 12 pentagons).

| Unit | Value |
|---|---|
| Face area | ~50,000 km² (~size of Costa Rica) |
| Pixel area at `tex_size=16` (default) | ~195 km² · ~14 km/side |

The pixel figures are naive averages (face area ÷ tex_size²); corner pixels may fall outside the face polygon.

## Key Design Notes

- **Stage contract**: each `Stage` receives a `BuildContext` containing the `World` (mesh + seed) and all outputs written by prior stages. Stages communicate exclusively through the context — no direct stage-to-stage calls.

- **Face-centric data model**: the mesh dual (hexagons + 12 pentagons) is the primary working surface for all worldgen and realmgen stages. Face indices are the universal key. Elevation, climate, hydrology, resources, and cost data are all stored per-face (or as per-face textures).

- **Rendering layers**: three layers in priority order for both viewers — per-face textures → named color schemes (callables over `face_metadata`) → static `face_colors` array. Viewers fall back down the stack when the higher layer is absent.

- **Elevation textures**: single-channel `(N, H, W, 1)` float32 where `N` = number of faces and channel 0 = elevation in km. U-axis is geographic east, V-axis is geographic north for ~90% of faces.

- **Water semantics**: sailable water = ocean ∪ lake. Lakes are impassable on land but navigable like ocean. Navigable rivers extend the sailable network inland. The land cost surface treats all water as fully impassable (`land_passable = ~is_water`); only the population hinterland stage opts into a limited coastal-water band for fishing grounds.

- **Debug flag**: `python3 main.py --debug` enables extra diagnostic overlays. New visuals should go behind this flag unless there is a specific reason to add them to the main view.
