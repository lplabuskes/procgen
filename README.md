# procgen

A procedural planet generator. Given a seed, it builds a full globe: tectonic plates, elevation, climate, hydrology (rivers and lakes), biomes, and natural resources. An optional second pass places medieval-style settlements, trade networks, and political territories.

The output is an interactive 2-D map viewer with selectable projections and color schemes.

The eventual goal is to be a one-stop shop for all things RPG procgen.

## Setup

Requires Python 3.13+ and [Homebrew](https://brew.sh).

```bash
brew install python@3.13
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

After the first time, reactivate with:

```bash
source .venv/bin/activate
```


## Running

```bash
python3 main.py               # full world viewer (globe + settlements)
python3 main.py --debug       # same, with extra diagnostic color schemes
python3 main.py --projection=mollweide   # robinson (default), mollweide, mercator, equirectangular
python3 main.py world.npz     # load a previously saved world
```

A parameters dialog opens before each build. Click **Re-run** to regenerate with new or changed parameters. Worlds can be saved and loaded from the File menu.

The 2-D map viewer can also be launched standalone for development:

```bash
python3 -m viz.mapview.main
```


## Pipeline

```
mesh → elevation/tectonics → climate → hydrology → resources → cost
                                                                  └─ (optional) realmgen
```

1. **Mesh** — subdivided icosphere, deregularized and dualized to a hex/pentagon grid
2. **Elevation/tectonics** — plate generation, boundary classification, elevation textures
3. **Climate** — temperature, precipitation, humidity, biome classification
4. **Hydrology** — river routing, lake detection, lake–climate feedback
5. **Resources** — fertility, forest density, terrain features (volcanic, lake shores, river valleys)
6. **Cost** — land and sea movement cost surfaces
7. **Realmgen** — settlement placement, trade networks, population, polity territories


## Tests & Diagnostics

```bash
python3 -m pytest                                       # full test suite
python3 scripts/measure_seam_quality.py   [seeds…]     # elevation texture seam quality
python3 scripts/measure_climate_seams.py  [seeds…]     # climate texture seam quality
python3 scripts/measure_zone_biomes.py    [seeds…]     # tropical biome coverage vs Earth
python3 scripts/measure_resources.py      [seeds…]     # per-biome fertility/forest density
python3 scripts/measure_population.py     [seeds…]     # settlement population distribution
python3 scripts/profile_pipeline.py       [seeds…]     # per-phase wall-clock breakdown
```

Each diagnostic script accepts `--random N` to run over N random seeds. See the script docstrings for full usage.
