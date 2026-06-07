# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Style

- Ask for clarification rather than churning your wheels. If the task or approach is ambiguous, ask before attempting.
- Assume visuals should be limited to debug mode, ask before adding to the main view.
- Don't worry about backwards compatibility for saving and loading worlds


## Running the Project

```bash
source .venv/bin/activate
python3 main.py                     # primary entrypoint — full procedural world viewer
python3 -m viz.mapview.main         # 2-D map viewer (dev)
```

No build step or linter is configured. Run the test suites with:

```bash
python3 -m pytest                                       # all tests (testpaths in pyproject.toml)
python3 -m pytest core/test/test_mymesh.py -v
python3 -m pytest viz/mapview/test/test_mapview.py -v
```

`core/test/test_mymesh.py` covers `Mesh` (icosphere topology, unit-sphere property, outward face normals) and `Mesh.mesh_dual` (dual topology counts, face sizes, and correct CCW winding of all fan triangles).

`viz/mapview/test/test_mapview.py` covers all four projections (equator/meridian anchors, N/S and E/W symmetry, projection-specific bounds, `project` vs `project_lonlat` consistency) and `_split_at_antimeridian` (no-crossing passthrough, two-polygon split, boundary latitude continuity).


## Diagnostic Scripts

```bash
python3 scripts/measure_seam_quality.py [seeds…|--random N|--pre-noise seed]
python3 scripts/measure_zone_biomes.py  [seeds…|--random N|--detail]
python3 scripts/measure_resources.py    [seeds…|--random N|--detail]
```

- `measure_seam_quality.py` — mean/median/RMS/max elevation jump (metres) across ~350k matched boundary pixel pairs per seed. Underlying function: `measure_boundary_elevation_jumps` in `worldgen/geology/elevation.py`.
- `measure_zone_biomes.py` — Tropical, Savannah, Desert fractions within tropical latitudes vs Earth reference values. Runs elevation + climate (no hydrology). Results appended to `scripts/results/zone_biome_results.json` and summarized in `scripts/results/zone_biome_results.md`.
- `measure_resources.py` — per-biome mean fertility and forest density vs Earth-ish anchors. Runs the full pipeline. Results appended to `scripts/results/resources_results.json` and summarized in `scripts/results/resources_results.md`.


## Architecture

Pipeline: mesh → deregularize → dual → **elevation/tectonics** → **climate** → **hydrology** → **climate lake-feedback** → **resources** → **cost** → *(optional)* **realmgen**

Top-level orchestrator: `GlobeBuilder` in `worldgen/globe/builder.py`. Each stage is a `Stage` that reads from and writes to a `BuildContext`. Realmgen is standalone — not wired into `GlobeBuilder`.

Stage reference docs (read the relevant one before working on a stage):

- [docs/core.md](docs/core.md) — `core/mesh.py`, `viz/mesh/gui.py`, `viz/mapview/`
- [docs/globe.md](docs/globe.md) — `worldgen/globe/`, `main.py`, pipeline stage contract
- [docs/elevation.md](docs/elevation.md) — `worldgen/geology/`
- [docs/climate.md](docs/climate.md) — `worldgen/climate/`
- [docs/hydrology.md](docs/hydrology.md) — `worldgen/hydrology/`
- [docs/resources.md](docs/resources.md) — `worldgen/resources/`
- [docs/cost.md](docs/cost.md) — `worldgen/cost/`
- [docs/realmgen.md](docs/realmgen.md) — `realmgen/`

Historical design specs live in `docs/plans/`.


## Key Design Notes

- `Mesh` stores only triangular faces; `PolygonMesh` handles arbitrary polygon faces (the dual has 12 pentagons + many hexagons). The dual normalizes centroids to the unit sphere.
- `PolygonMesh` pre-computes `face_adjacency` and `face_centroids` at construction; all downstream stages rely on these without recomputing.
- Face rendering has three layers in priority order: per-face textures → named color schemes (callables over `face_metadata`) → static `face_colors` array. Both viewers fall back down this stack when the higher layer is absent.
- Per-face tangent-plane axes are `u_axis = cross(z, n)`, `v_axis = cross(n, u_axis)` — texture U-axis is geographic east and V-axis is geographic north for ~90% of faces. D4 grid-walk algorithms produce N-S/E-W-biased output; `rivers.py` avoids this with D-infinity accumulation and sub-pixel gradient traces.
- Elevation textures are single-channel `(N, H, W, 1)` float32 (channel 0 = elevation in km).
- **Sailable water = ocean ∪ lake**: lakes are impassable on land and navigable like the ocean. Navigable rivers extend the sailable network inland. The land cost surface treats water as fully impassable (`land_passable = ~is_water`); only the population hinterland stage opts into a limited coastal-water band for fishing grounds.
