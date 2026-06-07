# Climate Stage

See also: `docs/plans/climate_generation_spec.md` for the original design.

## Files
- `worldgen/climate/builder.py` — stage orchestration + params
- `worldgen/climate/insolation.py`
- `worldgen/climate/temperature.py`
- `worldgen/climate/winds.py`
- `worldgen/climate/humidity.py`
- `worldgen/climate/precipitation.py`
- `worldgen/climate/classification.py`
- `worldgen/climate/currents.py`
- `worldgen/climate/display.py`

## `builder.py`

**`ClimateParams`** (dataclass): Nested sub-params:
- `temperature`: equator/pole temperatures, lapse rate, ocean/land thermal inertia
- `wind`: three-cell speeds, seasonal ITCZ shift, monsoon pressure-gradient scale (set to 0 to disable)
- `precipitation`: global wetness multiplier, advection iterations, continental drying rate, orographic strength, hidden lake-feedback knobs
- `currents`: enabled flag, β scale for western-boundary intensification, temperature/precipitation feedback scales, inland decay

Each leaf field carries `metadata` consumed by the params dialog. `GlobeParams` exposes `climate: ClimateParams`.

**`ClimateBuilder`**: A pipeline `Stage`. `build(ctx)` pipeline: insolation → coast distances → surface temperatures → zonal + pressure winds → SST-driven moisture advection → (if currents enabled) Stommel streamfunction, then second-pass temperatures and moisture with current ΔT/ΔP → per-pixel textures → biome classification.

Attaches face textures:
- `"temperature"` `(N,H,W,K) float32 °C`
- `"precipitation"` `(N,H,W,K) float32 mm/season`
- `"humidity"` `(N,H,W,K) float32`
- `"biome"` `(N,H,W,1) uint8`

Writes per-face metadata: `temperature_annual/warmest/coldest`, `precipitation_annual/seasonality`, `biome`, `distance_to_coast`, `humidity_annual`. Registers `"winds"` / `"currents"` features.

**Lake-feedback second pass**: `lake_feedback_stage()` returns a `ClimateLakeFeedbackStage` (sharing the builder) placed after hydrology in the pipeline. Its `build(ctx)` calls `apply_lake_humidity_feedback(mesh, lake_mask_tex)` to re-run moisture advection with lake evaporation as an extra source.

## Individual modules

**`insolation.py`**: Standard TOA daily-mean insolation with polar-day/polar-night handling; 4 seasons (equinoxes + solstices).

**`temperature.py`**: Parabolic latitude model for annual mean; seasonal anomaly damped by tanh-based ocean/land thermal inertia; lapse-rate correction broadcast to per-pixel texture.

**`winds.py`**: Three-cell (trades/westerlies/polar) zonal winds with seasonal ITCZ migration. `pressure_winds` adds a monsoonal land–sea thermal contrast term (pressure-gradient proxy, no Coriolis) on top before moisture transport and currents.

**`humidity.py`**: SST-driven Jacobi moisture transport — ocean faces emit proportional to SST, each iteration distributes moisture downwind with background decay. Continentality drying, ITCZ convergence, and rain shadows emerge from the transport physics. IDW interpolation to per-pixel grid.

**`precipitation.py`**: Converts per-pixel humidity to precipitation with an orographic factor (wind dot elevation gradient, clamped to [0.3×, 2.0×]).

**`classification.py`**: 9 biomes (Icecap → Desert); first-match-wins decision tree on T_warmest, T_coldest, P_annual, P_seasonality. Vectorised `classify_textures` mirrors the scalar `default_biome` exactly (pinned by a unit test). Calibrated thresholds documented in memory.

**`currents.py`**: Barotropic Stommel streamfunction — solves `(L − β·D_east)·ψ = −curl(τ)` on the ocean face graph by GMRES (Voronoi-weighted Laplacian, Dirichlet BC at basin boundaries), recovers divergence-free currents as `u = k̂ × ∇ψ`, advects insolation as a thermal payload, decays ΔT inland exponentially. Produces additive temperature and multiplicative precipitation deltas applied in a second climate pass. `beta_scale=0` gives symmetric gyres; default adds mild western-boundary intensification.

**`display.py`**: Per-face color schemes (temperature, log-scale precipitation, humidity, biome categorical) and RGBA texture builders for temperature (annual or single season) and precipitation (annual sum, log scale).
