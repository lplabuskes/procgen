# Climate Stage

Original design: `docs/plans/climate_generation_spec.md`.

## Files
`worldgen/climate/`: `builder.py` (stage + params), `insolation.py`,
`temperature.py`, `winds.py`, `humidity.py`, `precipitation.py`,
`classification.py`, `currents.py`, `_gradient.py` (shared tangent frames +
least-squares gradients), `channels.py`, `display.py`.

## `builder.py`

**`ClimateParams`**:
- `temperature` — equator/pole temperatures, lapse rate, ocean/land thermal inertia
- `wind` — three-cell speeds, seasonal ITCZ shift, monsoon pressure-gradient scale (0 disables)
- `precipitation` — global wetness, advection iterations, continental drying, orographic strength, hidden lake-feedback knobs
- `currents` — enabled, β scale (western-boundary intensification), T/P feedback scales, inland decay
- `seed` — unused (model is deterministic)

**`ClimateBuilder.build(ctx)`**: insolation → coast distances (Dijkstra from
ocean) → surface temperatures → zonal + pressure winds → SST-driven moisture
advection + smoothing → (currents) Stommel streamfunction, second-pass
temperatures and moisture with current ΔT → per-pixel temperature (lapse rate) →
current ΔP on humidity → per-pixel humidity → per-pixel precipitation
(orographic) → biomes.

Textures (K = 4 seasons): `"temperature"` `(N,H,W,K) float16 °C`,
`"precipitation"` `(N,H,W,K) float16 mm/season`, `"humidity"` `(N,H,W,K)
float16`, `"biome"` `(N,H,W,1) uint8`.

Per face: `temperature_annual/warmest/coldest`, `precipitation_annual`, `biome`
(mode of texels), `distance_to_coast`, `humidity_annual`. Features `"winds"`
(per season: centroids + vectors), `"currents"`.

**Lake-feedback pass**: `lake_feedback_stage()` → `ClimateLakeFeedbackStage`
(shares the builder), placed after hydrology **and delta**, so delta-built land
gets its own precipitation and biome. `apply_lake_humidity_feedback(mesh,
lake_mask_tex)`: lake evaporation (SST formula × lake area fraction ×
`lake_evap_scale`) advected `lake_feedback_iter` steps and added to humidity;
humidity, precipitation, biome and face aggregates re-rasterised. Temperature
unchanged. No-op without lakes or with either knob 0.

## Individual modules

- **`insolation.py`** — TOA daily-mean insolation with polar day/night; 4 seasons (equinoxes + solstices).
- **`temperature.py`** — parabolic latitude annual mean; seasonal anomaly damped by tanh ocean/land inertia; lapse rate per pixel.
- **`winds.py`** — three-cell zonal winds with seasonal ITCZ; `pressure_winds` adds monsoonal land–sea contrast (pressure-gradient proxy, no Coriolis) before moisture and currents.
- **`humidity.py`** — SST-driven Jacobi moisture transport (ocean emits ∝ SST, downwind spread with decay); continentality, ITCZ convergence and rain shadows emerge. IDW to pixels.
- **`precipitation.py`** — humidity × orographic factor (wind · elevation gradient, clamped [0.3×, 2.0×]).
- **`classification.py`** — 9 climate biomes (Icecap → Desert) plus `WETLAND_BIOME` (9, never classified — stamped by the [wetland stage](wetland.md)), first-match decision tree on T_warmest, T_coldest, P_annual, P_seasonality. Vectorised `classify_textures` mirrors scalar `default_biome` exactly (unit-tested). Thresholds calibrated to Earth's tropical-zone fractions (`scripts/measure_zone_biomes.py`).
- **`currents.py`** — barotropic Stommel: solves `(L − β·D_east)·ψ = −curl(τ)` on the ocean face graph by GMRES (Voronoi-weighted Laplacian, Dirichlet at basin boundaries); `u = k̂ × ∇ψ`; advects insolation as thermal payload, ΔT decays inland. Additive temperature / multiplicative precipitation deltas for the second pass. `beta_scale=0` → symmetric gyres.
- **`display.py`** — per-face schemes (temperature, log precipitation, humidity, biome) and recipes `BIOME_RECIPE` (`relief=True`), `ANNUAL_/SUMMER_/WINTER_TEMPERATURE_RECIPE` (seasons 1 / 3), `PRECIPITATION_RECIPE` (annual, log); builder runs each on `mesh.face_textures`, region view reruns on refined data.

**Biome** always; `--debug`: Annual/Summer/Winter Temperature, Annual
Precipitation, Humidity, four **Winds** overlays and **Currents** (unchecked),
biome/T/P hover (`refined=("biome", "temperature", "precipitation")`).
