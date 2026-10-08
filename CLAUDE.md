# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Style

- Ask for clarification rather than churning your wheels. If the task or approach is ambiguous, ask before attempting.
- Assume visuals should be limited to debug mode, ask before adding to the main view.
- Don't worry about backwards compatibility for saving and loading worlds
- Be picky about which params are visible in basic mode (`BASIC_PARAMS` in `main.py`; everything else needs `--advanced`). New params default to advanced; only add a headline knob a casual user would actually change, prefer one master toggle over per-item settings, and ask before adding to the basic list.


## Running the Project

```bash
source .venv/bin/activate
pip install -e .                    # one-time: installs project packages into the venv
python3 main.py                     # primary entrypoint — full procedural world viewer
python3 main.py --advanced          # same, with every param in the Parameters dialog (default: main.BASIC_PARAMS)
python3 -m viz.mapview.main         # 2-D map viewer (dev)
```

No build step. Run the linter, type checker, and test suite with the venv's interpreter (the system `python3` lacks these tools):

```bash
.venv/bin/python -m flake8          # lint (config in pyproject.toml: max-line-length=120)
.venv/bin/python -m pyright         # static type checking (target: 0 errors)
.venv/bin/python -m pytest          # all configured test paths (see pyproject.toml)
```

Each module and sub-module has a `test/` directory with its own tests, all wired into the `pytest` testpaths in `pyproject.toml`.


## Script Utilities

Shared helpers for all `measure_*.py` scripts live in `scripts/_common.py`:

- `load_history(path)` / `save_history(history, path)` — JSON append-log read/write
- `add_seed_args(parser)` / `resolve_seeds(args, defaults)` — standard `seeds` + `--random N` CLI args
- `get_physical_world(seed, rebuild, cache_dir=CACHE_DIR)` — build-or-load the cached physical world (saves to `saved_worlds/`)
- `get_realm_world(seed, rebuild, cache_dir=CACHE_DIR)` — the same with the realm stage run on it (settlements, towns, rural density, routes, danger) — what the region/local tiers read
- `avg(rows, key)` — NaN-safe mean over a list of result dicts
- `CACHE_DIR` — default world cache path (`saved_worlds/`)


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
python3 scripts/debug_borders.py          [seed|--face F|--lat/--lon|--radius D|--depth N|--show-overhang|--seams|--vertices]
python3 scripts/debug_region.py           [seed|--region R|--layer L|--tex-size N|--n-cols N|--relief F]
python3 scripts/debug_local.py            [seed|--region R|--hex H|--layer L|--relief F]
python3 scripts/measure_villages.py       [seeds…|--random N|--hexes N|--rebuild-cache]
python3 scripts/measure_wetlands.py       [seeds…|--random N|--rebuild-cache]
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
- `debug_borders.py` — visual border-drawing troubleshooter. Renders a zoomed, faithful raster of one region: the texel polity fill as actual corner-quads (tiles at any orientation) with the border polylines overlaid, both from the current `fill.py`/`borders.py`. Draws **only visible (in-`face_texture_in_polygon`) texels** by default; `--show-overhang` reveals the clipped over-hang texels (faded, red-edged). `--seams`/`--vertices` overlay hexagon seams and border vertices. Prints open endpoints per depth (should all be coastal). Auto-picks a border-rich centre; `--face`/`--lat`/`--lon`/`--radius` aim it. Writes `scripts/results/borders_<seed>.png`.
- `debug_region.py` — regional upsampling check. Renders one region's hex textures side by side — left with detail off and the water masks copied verbatim (the global field, upsampled only), right with the real params — and prints the added-relief statistics, how far the refined shoreline moved (flooded/dried), and the **conservation report** (per-global-texel residual after downsampling; must stay at float32 noise). Auto-picks the most land-rich region; `--region` aims it. Writes `scripts/results/region_<seed>_<region>_<layer>.png`. See [docs/region_fields.md](docs/region_fields.md).
- `debug_local.py` — local-map check. Renders one regional hex's local map side by side — left the region's fields upsampled only, right the real local map with its rivers, roads, towns, cities and villages — and prints the added relief, shoreline movement, the **conservation report** against the region (per regional texel; float32 noise), and the **village budget report** (rural + villages per global texel ∩ map; exact) with village counts, sizes, spacing and kinds. Defaults to the hex of the densest land; `--layer "Rural Density"` shows the countryside net of villages. Writes `scripts/results/local_<seed>_<region>_<hex>_<layer>.png`. See [docs/local.md](docs/local.md) and [docs/local_settlements.md](docs/local_settlements.md).
- `measure_villages.py` — village statistics on local maps of hexes sampled across the rural-density range (face-density quantiles 0.5–1.0): villages per 100 km² vs the budget's prediction, size, nearest-neighbour spacing, share of rural people in villages (target `village_fraction`), share stranded in no-village budgets, kinds, conservation residual. Default seeds 4/31/35 (~85 s). Results appended to `scripts/results/village_results.json`.
- `measure_wetlands.py` — global Wetland biome: share of land (whole-texel target ~4–6%; Earth ~6–9% finer-grained), share per climate biome, rain-fed share, largest connected complexes vs West Siberia / Pantanal / Sudd. Re-scores the cached physical world with current `WetlandParams` (no rebuild needed). Results appended to `scripts/results/wetland_results.json`.


## Architecture

Pipeline: mesh → deregularize → dual → **elevation/tectonics** → **climate** → **hydrology** → **delta** → **climate lake-feedback** → **wetland** → **coast** → **resources** → **danger** → **cost** → *(optional)* **realmgen**

Top-level orchestrator: `GlobePipeline` in `worldgen/builder.py`. Each stage is a `Stage` that reads from and writes to a `BuildContext`. Realmgen is a separate `RealmPipeline` with sub-stages: scoring → placement → population (incl. trade) → polity → refine → towns. Both pipelines are wired together in `main.py` (`AppPipeline` + `AppParams`). The region and local tiers (`region/`, `localmap/`) are view-side: built on demand when a region or hex is opened, never part of the world build.

Stage reference docs (read the relevant one before working on a stage):

- [docs/core.md](docs/core.md) — `core/` infrastructure (pipeline, channels, params, display, mesh cache, mesh data structures)
- [docs/viz.md](docs/viz.md) — `viz/` layer (MapWidget, MapWindow, overlay renderers, projections, registry adapter, MeshViewer, ParamsDialog, color_stops)
- [docs/viz_regionview.md](docs/viz_regionview.md) — `RegionView`, the zoomed region / local-map viewer (wiring, drawing, hover, refined overlays)
- [docs/region.md](docs/region.md) — `region/` regional maps overview (cubed-sphere cells, hex grid, global sampling, parent sources, params, display)
- [docs/region_fields.md](docs/region_fields.md) — `region/upsample.py`, `region/resources.py`: the conserving field refinement (elevation, climate, water, coast style and shore reshaping, resources, rural density, danger)
- [docs/region_features.md](docs/region_features.md) — `region/corridor.py`, `rivers.py`, `distributaries.py`, `streams.py`, `roads.py`, `towns.py`, `cities.py`: rivers (with delta distributaries), roads, towns and cities re-sited on refined terrain; the region's own streams for the local maps
- [docs/local.md](docs/local.md) — `localmap/` local maps (one regional hex refined onto its own hex grid: geometry, fields, rivers, roads)
- [docs/local_settlements.md](docs/local_settlements.md) — local-map towns and cities, city footprints, farmland, villages
- [docs/worldgen.md](docs/worldgen.md) — `worldgen/builder.py`, `main.py`, pipeline stage contract
- [docs/elevation.md](docs/elevation.md) — `worldgen/geology/`
- [docs/climate.md](docs/climate.md) — `worldgen/climate/`
- [docs/hydrology.md](docs/hydrology.md) — `worldgen/hydrology/`
- [docs/delta.md](docs/delta.md) — `worldgen/delta/` (river deltas: deposited plains shaped bird's-foot / arcuate / cuspate by river vs wave power, rivers carried across them, fertility / apex consumers; marshes are the wetland stage's)
- [docs/wetland.md](docs/wetland.md) — `worldgen/wetland/`, `region/wetlands.py` (wetness score from slope, rain/PET and height above nearest drainage; the Wetland biome on the globe, refined with lower thresholds and riparian marsh on region/local maps)
- [docs/coast.md](docs/coast.md) — `worldgen/coast/` (coast smoothness: the shore style the region/local tiers draw)
- [docs/resources.md](docs/resources.md) — `worldgen/resources/`
- [docs/danger.md](docs/danger.md) — `worldgen/danger/`
- [docs/cost.md](docs/cost.md) — `worldgen/cost/`
- [docs/realmgen.md](docs/realmgen.md) — `realmgen/` orchestration (`RealmParams`, `DangerInfluenceParams`, `RealmPipeline`)
- [docs/realm_scoring.md](docs/realm_scoring.md) — `realmgen/scoring/`
- [docs/realm_placement.md](docs/realm_placement.md) — `realmgen/placement/`
- [docs/realm_population.md](docs/realm_population.md) — `realmgen/population/`, `realmgen/trade/`
- [docs/realm_polity.md](docs/realm_polity.md) — `realmgen/polity/`
- [docs/realm_refine.md](docs/realm_refine.md) — `realmgen/refine/`
- [docs/realm_towns.md](docs/realm_towns.md) — `realmgen/towns/`

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
