# Climate Generation

**Module:** `worldgen/climate/` (`builder.py`, `insolation.py`, `temperature.py`, `winds.py`, `humidity.py`, `precipitation.py`, `classification.py`, `currents.py`, `display.py`)
**Purpose:** Produce per-face, per-season temperature & precipitation textures, a biome classification, and a prevailing-wind vector field on top of an already-generated elevation mesh.

Inspired by Geoff's [Climate Cookbook](https://worldbuildingworkshop.com/2015/11/27/climate/), rendered as continuous (lat, season) → (°C, mm/season) functions. Precipitation emerges from SST-driven moisture advection along seasonal wind fields rather than zonal lookup tables, so continental drying, ITCZ convergence, and rain shadows arise from the transport physics.

---

## Pipeline overview

```
elevation textures, mesh
        │
        ▼
[ Seasonal insolation Q(lat, season) ]              # insolation.py
        │  Q: (N_faces, n_seasons) W/m² daily mean
        ▼
[ Distance-to-coast field ]                         # builder, Dijkstra on face graph
        │  d_coast: (N_faces,) km
        ▼
[ Surface temperature ]                             # temperature.py
        │  T_face: (N_faces, n_seasons) °C   (annual mean + damped seasonal anomaly,
        │                                     lapse rate)
        ▼
[ Prevailing winds (zonal Hadley/Ferrel/Polar + pressure-gradient) ]    # winds.py
        │  W_face: (N_faces, n_seasons, 3) tangent vectors, m/s
        ▼
[ SST-driven moisture advection ]                   # humidity.py
        │  q_air: (N_faces, n_seasons) dimensionless specific humidity
        ▼
[ Ocean currents — second-pass T and humidity ]     # currents.py (if enabled)
        │  ΔT, ΔP deltas broadcast across seasons
        ▼
[ Texture rasterisation ]                           # builder
        │  per-pixel T, humidity via vertex-averaged barycentric interpolation
        │  per-pixel P via humidity × orographic factor × precipitation_scale
        ▼
[ Classification ]                                  # classification.py
        │  biome: (N_faces, H, W) uint8
        ▼
Globe with climate textures + winds/currents features
```

---

## Inputs

| Source | Field | Used by |
|---|---|---|
| `mesh.face_centroids` | 3D unit-sphere positions | latitude, wind tangent frames |
| `mesh.face_adjacency` | edge-sharing neighbours | distance-to-coast BFS, gradient |
| `mesh.face_metadata['elevation']` | per-face elevation (km) | temperature lapse, ocean/land mask |
| `mesh.face_textures['elevation']` | per-pixel elevation (km) | per-pixel T lapse, orographic uplift |
| `mesh.face_textures['ocean_mask']` | per-pixel ocean flag (uint8) | ocean/land mask override when present |

Sea level threshold (elev ≤ 0 km) defines the ocean mask, unless the `ocean_mask` texture is present.

---

## Data model

### Face textures (attached to mesh)

| Key | Shape | Dtype | Meaning |
|---|---|---|---|
| `temperature` | `(N, H, W, 4)` | `float16` | Channel `j` = mean surface temperature in °C during season `j` |
| `precipitation` | `(N, H, W, 4)` | `float16` | Channel `j` = total precipitation in mm during season `j` |
| `humidity` | `(N, H, W, 4)` | `float16` | Channel `j` = dimensionless specific humidity (q_air) during season `j` |
| `biome` | `(N, H, W, 1)` | `uint8` | Biome index (0–8 for the default 9-biome classifier) |

### Per-face metadata (for color schemes & quick display)

| Key | Type | Meaning |
|---|---|---|
| `temperature_annual` | `float` | Annual mean °C |
| `temperature_warmest` | `float` | Warmest season mean °C |
| `temperature_coldest` | `float` | Coldest season mean °C |
| `precipitation_annual` | `float` | Annual total mm |
| `biome` | `int` | Dominant biome index (mode across the face's pixels) |
| `distance_to_coast` | `float` | km along great-circle from the nearest ocean face |
| `humidity_annual` | `float` | Annual mean specific humidity (mean of q_air across the 4 seasons) |

### Globe features (vector overlays)

| Name | Structure | Meaning |
|---|---|---|
| `winds` | `list[(positions_3d, vectors_3d)]` length 4 | One entry per season; `positions_3d` is `(N_faces, 3)` face centroids, `vectors_3d` is `(N_faces, 3)` wind velocity (m/s) tangent to the sphere |
| `currents` | `[(positions_3d, current_vectors_3d, temperature_delta)]` length 1 | Annual-mean current velocity `(N_faces, 3)` and per-face ΔT `(N_faces,)`. Present only when `params.currents.enabled` |

---

## Season indexing

Four seasons anchored to astronomical events; declination drives insolation:

| Index | Event | Solar declination δ (deg) | Northern Hemisphere | Southern Hemisphere |
|---|---|---|---|---|
| 0 | March equinox  | 0     | spring | autumn |
| 1 | June solstice  | +23.5 | summer | winter |
| 2 | September equinox | 0  | autumn | spring |
| 3 | December solstice | −23.5 | winter | summer |

Display labels follow NH convention; SH viewers will see "wrong" labels (a known concession to keep the index single-valued globally).

---

## Stage 1 — Seasonal insolation

Daily-mean top-of-atmosphere insolation for latitude φ and declination δ:

```
H₀ = arccos( clamp(−tan φ · tan δ, −1, 1) )       # half-day length (rad)

if polar night (tan φ · tan δ ≥ 1):  Q = 0
if polar day   (tan φ · tan δ ≤ −1): Q = S₀ · sin φ · sin δ
otherwise:                            Q = (S₀ / π) · (cos φ · cos δ · sin H₀ + H₀ · sin φ · sin δ)
```

`S₀ = 1361 W/m²` is the solar constant. Returns `Q[face, season]` array, shape `(N, 4)`, in W/m².

Vectorised: precompute `φ = arcsin(centroid_z)` once and broadcast over the four declination values.

---

## Stage 2 — Surface temperature

Built as **annual mean + seasonal anomaly**, the anomaly damped by ocean thermal inertia.

```
T_annual(φ)            = T_eq − (T_eq − T_pole) · (φ/(π/2))²         # parabolic in |lat|
T_eq_season(φ, season) = a + b · Q(φ, season)                         # b tuned so anomaly amplitude is right
ΔT_season              = T_eq_season − ⟨T_eq_season⟩_year             # zero-mean over the 4 seasons
T_face(face, season)   = T_annual(φ)
                          + inertia_factor(face) · ΔT_season(φ, season)
                          − Γ · max(elevation_km(face), 0)             # lapse rate above sea level only
                          + current_delta(face)                        # broadcast from annual ΔT; 0 when currents disabled
```

Tunables (`TemperatureParams`):

| Name | Default | Effect |
|---|---|---|
| `equator_annual_c` | `27.0` °C | Equatorial annual mean |
| `pole_annual_c` | `-25.0` °C | Polar annual mean |
| `anomaly_scale` (`b`) | `0.05` | converts insolation anomaly to °C anomaly |
| `lapse_rate_c_per_km` (`Γ`) | `6.5` °C/km | dry adiabatic-ish, applied above sea level only |
| `ocean_inertia` | `0.3` | factor for face entirely over ocean (damps seasonal swing) |
| `land_inertia` | `1.0` | factor for face deep inland |
| `inertia_distance_km` | `1500.0` | tanh saturation distance (km) |

`inertia_factor(face)` interpolates between `ocean_inertia` and `land_inertia` by `distance_to_coast` (saturating at ~1500 km inland with `tanh(d_coast / 1500)`).

### Per-pixel temperature texture

Per-face temperatures are averaged onto shared vertices via `rasterize_face_values` + `smooth_texture_seams` (from `elevation.py`) so adjacent faces interpolate from the same boundary values and face-edge seams are eliminated. The per-pixel lapse-rate correction is then applied relative to the vertex-averaged reference elevation:

```
T_pixel[f, h, w, j] = T_interp[f, h, w, j] + Γ · (ref_elev[f, h, w] − pix_elev[f, h, w])
```

where `ref_elev` is the rasterized per-face elevation (on the same vertex-averaged grid), and both elevations are clamped at 0 km for lapse purposes.

---

## Stage 3 — Prevailing winds (zonal + pressure-gradient)

The wind field is computed at face centroids; per-pixel winds are not stored.

### Zonal winds

Bands (NH; mirror for SH by reversing meridional sign):

| Latitude band | Surface wind | Speed default |
|---|---|---|
| 0° – 30° | NE trades — blows toward (W, equator-ward) | 7 m/s |
| 30° – 60° | Westerlies — blows toward (E, pole-ward) | 10 m/s |
| 60° – 90° | Polar easterlies — blows toward (W, equator-ward) | 5 m/s |

**ITCZ shift:** the boundaries shift toward the summer hemisphere. Effective latitude used for the band lookup:

```
φ_eff = φ − itcz_shift_amplitude · sin(season_phase)
season_phase ∈ {0, π/2, π, 3π/2} for seasons {0, 1, 2, 3}
```

`itcz_shift_amplitude` default = 10°.

**3D vector at a face:** at the face centroid, build the tangent frame
- `east_hat = normalise( ẑ × centroid )`     (zero at poles — handled with a fallback frame)
- `north_hat = centroid × east_hat`

then `wind_3d = u_east · east_hat + v_north · north_hat`. The (u_east, v_north) direction is normalised to unit magnitude before multiplying by the band speed.

### Pressure-gradient wind from land/sea contrast

A second wind component is added on top of the zonal base to capture seasonal pressure systems (continental highs in winter, lows in summer → monsoonal reversal). It is a cookbook proxy, not a real pressure-tendency model: the seasonal temperature anomaly itself is treated as a (sign-flipped) pressure field, and wind blows from high to low.

```
anomaly[f, j]   = T_face[f, j] − T_annual[f]                  # °C, near-zero over ocean
∇a[f, j]        = least-squares tangent-plane gradient of anomaly on the face graph
                 (neighbour offsets in km, RHS = anomaly differences)
W_pressure[f,j] = pressure_wind_scale · ∇a[f, j]              # m/s, blows toward warm/low
W_total[f, j]   = W_zonal[f, j] + W_pressure[f, j]
```

The wind is **pure cross-isobar** — no Coriolis rotation. The ocean's large thermal inertia (`ocean_inertia ≈ 0.3`) keeps the ocean's anomaly near-zero, so the contribution is largest on continental coasts and inland. Tunable on `WindParams.pressure_wind_scale` (m·s⁻¹·km/°C, default 600, set to 0 to disable).

---

## Stage 4 — Atmospheric moisture and precipitation

Precipitation emerges from SST-driven evaporation advected along seasonal winds. All logic lives in `humidity.py` and `precipitation.py`.

### Evaporation source

Ocean faces emit moisture proportional to SST; land faces are zeroed:

```
evap[f, j] = evaporation_scale · clamp((T_face[f, j] + 25) / 52, 0, 1)   (ocean faces)
           = 0                                                              (land faces)
```

Warm tropical oceans produce strong emission; cold polar seas produce negligible evaporation.

### Moisture advection (Jacobi)

An iterative transport model runs to near-steady-state (`moisture_advection_iter` steps):

1. Each face sends `0.5 × q` to its downwind neighbours, distributed proportional to the dot product of wind with the direction to each neighbour (normalised to sum to 1 per source face).
2. Background precipitation removes `moisture_decay_rate × q` from every face.
3. Ocean faces are clamped back to their evaporation source level (continuous replenishment).

`moisture_decay_rate` controls continental drying: higher values keep moisture close to the ocean; lower values allow wetter continental interiors. Convergence zones (ITCZ) accumulate moisture from multiple source directions; divergence zones (horse latitudes) dry out because outflow exceeds inflow. This eliminates the need for explicit continentality and zonal-base parameters.

Followed by Laplacian graph smoothing (`humidity_smoothing_passes` passes of neighbour-mean blending with weight `humidity_smoothing_alpha`) to soften face-edge discontinuities.

### Per-pixel humidity

Face humidity values are rasterized to per-pixel grids via vertex-averaged barycentric interpolation (same `rasterize_face_values` + `smooth_texture_seams` pipeline as temperature), eliminating hard face-edge seams.

### Per-pixel precipitation

```
P_pixel[f, h, w, j] = q_pixel[f, h, w, j] · oro_factor[f, h, w, j] · (precipitation_scale / 4)

oro_factor = clamp(1 + orographic_k · uplift, 0.3, 2.0)
uplift = wind_u · (∂h/∂u) + wind_v · (∂h/∂v)    # m/s in face tangent-frame UV coordinates
```

The per-pixel elevation gradient `(∂h/∂u, ∂h/∂v)` is computed from `np.gradient` over the face's pixel elevation texture; wind components are projected into the face's UV tangent frame via `face_u_axes` and `face_v_axes`. `precipitation_scale` (mm/yr) sets the global precipitation level at unit moisture; the spatial distribution is shaped entirely by the humidity field.

---

## Stage 5 — Classification

`classification.py` exposes:

```python
BIOME_NAMES: list[str] = [
    "Icecap", "Tundra", "Taiga", "Temperate",  # 0–3
    "Steppe", "Savannah", "Tropical",          # 4–6
    "Semi-Arid", "Desert",                     # 7–8
]

def default_biome(
    temperature_c: np.ndarray,    # shape (4,), per-season mean temperature in °C
    precipitation_mm: np.ndarray, # shape (4,), per-season precipitation in mm
    lat_deg: float,
) -> int: ...
```

Internally the function derives:

```
t_warmest     = temperature_c.max()
t_coldest     = temperature_c.min()
p_annual      = precipitation_mm.sum()
p_seasonality = (precipitation_mm.max() - precipitation_mm.min()) / max(p_annual, 1e-6)
```

Decision tree (in order; first match wins):

1. `t_warmest < 0`                                                          → **Icecap (0)**
2. `t_warmest < 10`                                                         → **Tundra (1)**
3. `t_coldest < -10`                                                        → **Taiga (2)**
4. `p_annual < 300`  and `t_warmest > 18`                                   → **Desert (8)**
5. `p_annual < 500`  and `t_warmest > 18`                                   → **Semi-Arid (7)**
6. `p_annual < 400`                                                         → **Steppe (4)**
7. `t_warmest > 22` and `p_annual > 750`  and `p_seasonality < 0.5`        → **Tropical (6)**
8. `t_warmest > 21` and `p_annual > 350`                                    → **Savannah (5)**
9. fallthrough                                                               → **Temperate (3)**

The classifier runs **per pixel** (using the per-pixel temperature/precip textures) so we get sub-face transitions like "snow line on a tropical peak". For per-face metadata, the dominant biome is the integer mode over the face's pixels.

`default_biome` is the canonical scalar reference. `classification.py` additionally exposes `classify_textures(temperature_tex, precipitation_tex, lat_tex) -> biome_tex` which implements the same decision tree fully vectorised over `(N, H, W, 4)` arrays — this is what the builder calls. The two must agree pixel-for-pixel; a small test pins this invariant.

### Swappability

`ClimateBuilder.__init__` takes `classifier: Callable[..., np.ndarray] = classify_textures`. Replacing it (e.g. Köppen-Geiger later) requires only matching the array-in / array-out signature; no other code in the module reads biome indices semantically.

---

## Module layout

```
worldgen/climate/
    __init__.py
    builder.py            # ClimateParams + ClimateBuilder
    insolation.py         # seasonal_insolation(latitudes, declinations) -> Q[face, season]
    temperature.py        # surface_temperatures(...) -> T_face; per_pixel_temperatures(...)
    winds.py              # zonal_winds(...) + pressure_winds(...) -> W[face, season, 3]
    humidity.py           # evaporation_source, advect_moisture, smooth_humidity, per_pixel_humidity
    precipitation.py      # per_pixel_precipitation(...) -> P[face, px, px, season]
    classification.py     # BIOME_NAMES, default_biome, classify_textures
    currents.py           # CurrentsResult + compute_currents(mesh, ...) → annual ΔT and ΔP deltas
    display.py            # color schemes for T/P/humidity/biome + legend entries
```

Per-stage functions are pure (input arrays → output arrays) so they can be tested in isolation. The builder is responsible for plumbing.

### `ClimateBuilder`

```python
class ClimateBuilder:
    def __init__(
        self,
        params: ClimateParams | None = None,
        classifier: Callable[..., np.ndarray] = classify_textures,
    ): ...

    def build(
        self,
        mesh: PolygonMesh,
        elevation_textures: np.ndarray,        # (N, H, W, 1)
        progress: Optional[Callable[[str], None]] = None,
    ) -> None: ...

    def apply_lake_humidity_feedback(
        self,
        mesh: PolygonMesh,
        lake_mask_tex: np.ndarray,             # (N, H, W, 1) float32
        progress: Optional[Callable[[str], None]] = None,
    ) -> None: ...
```

`build()` attaches `temperature`, `precipitation`, `humidity`, and `biome` textures to the mesh and stores intermediate results as attributes:

| Attribute | Type | Notes |
|---|---|---|
| `insolation` | `(N, 4) float64` | W/m², daily mean |
| `distance_to_coast` | `(N,) float32` | km |
| `temperature_face` | `(N, 4) float64` | °C |
| `precipitation_face` | `(N, 4) float32` | mm/season (pixel averages) |
| `humidity_face` | `(N, 4) float32` | dimensionless q_air |
| `winds` | `(N, 4, 3) float32` | m/s tangent |
| `currents` | `CurrentsResult \| None` | populated when `params.currents.enabled` |

`apply_lake_humidity_feedback(mesh, lake_mask_tex)` is called after `HydrologyBuilder.build()` completes. It adds lake evaporation as an extra moisture source (proportional to lake area fraction × SST), then re-runs humidity advection (`lake_feedback_iter` steps) and refreshes precipitation and biome textures. This single-correction pass approximates the hydrology→climate feedback without re-running the full pipeline.

---

## Pipeline integration

`GlobeBuilder.build()` order:

```
elevation_builder.build()
climate_builder.build(mesh, elevation_textures)
hydrology_builder.build(mesh, elevation_textures)    # reads "precipitation" and "temperature" textures
climate_builder.apply_lake_humidity_feedback(mesh, lake_mask_tex)    # if lake_mask present
resources_builder.build(mesh, rivers=rivers)
```

`GlobeParams` exposes `climate: ClimateParams`. `GlobeBuilder.apply_display` registers climate color schemes; in debug mode this includes temperature (annual + seasonal slices), precipitation, humidity, and wind/currents overlays.

---

## Ocean currents (annual mean)

Currents are computed as an annual-mean second pass using a **barotropic Stommel streamfunction** model.

### Model (`worldgen/climate/currents.py`)

```
annual-mean wind ──▶ quadratic-drag wind stress τ ──▶ curl(τ) per face
                                                              │
                                                              ▼
                    GMRES solve: (L − β·D_east)·ψ = −curl(τ)
                    L = Voronoi-weighted graph Laplacian (FVM ≈ −∇²)
                    β·D_east = Stommel planetary vorticity gradient term
                    ψ = 0 on land (Dirichlet BC per basin)
                                                              │
                                                              ▼
                              u = k̂ × ∇ψ  (divergence-free currents)
                                                              │
                                                              ▼
                   Jacobi advection of annual-mean insolation along currents
                                                              │
                                                              ▼
                        ΔT_ocean = temperature_scale × (advected_Q − local_Q)
                                                              │
                                                              ▼
          Dijkstra from ocean faces → ΔT_land = ΔT_ocean[parent] · exp(-d/decay)
                                                              │
                                                              ▼
                            precip_delta = precipitation_scale × ΔT  (multiplicative)
```

The `CurrentsResult` dataclass holds:
- `currents` `(N, 3) float32` m/s — annual-mean current vector per face (zero on land)
- `streamfunction` `(N,) float32` — ψ in solver units
- `temperature_delta` `(N,) float32` — °C, broadcast across all K seasons
- `precipitation_delta` `(N,) float32` — multiplicative wet delta applied to q_air before precipitation conversion
- `advected_insolation` `(N,) float32` — W/m² post-advection (for inspection)

### Hooks

1. **`temperature.surface_temperatures(...)`** accepts `coastal_current_temp_delta: (N, 1) | None`; the builder passes the per-face annual ΔT broadcast across seasons.
2. **`humidity`** `current_wet_delta` is applied as `q_air × (1 + delta)` after humidity advection and before the humidity→precipitation conversion.
3. **`ClimateParams.currents: CurrentsParams`** has `enabled: bool = True`. When disabled the second pass is skipped and `self.currents` stays `None`.

### Tunables (`CurrentsParams`)

| Field | Default | Effect |
|---|---|---|
| `enabled` | `True` | Master switch for the second pass |
| `beta_scale` | `1.0` | Scale on β·D_east: 0 = symmetric gyres, ~1 = western-boundary intensification |
| `advection_alpha` | `1.0` | Per-iter inflow mixing fraction |
| `advection_iterations` | `400` | Jacobi sweeps; ~400 is enough for depth-4 |
| `temperature_scale` | `0.06` | °C per W/m² of advected-vs-local insolation gap |
| `precipitation_scale` | `0.03` | Multiplicative wet delta per °C of anomaly |
| `inland_decay_km` | `600.0` | e-folding distance for coastal → inland decay |

### Known shortcomings

- **No proper Lagrangian particle tracking.** The advection is a Markov diffusion on the face graph, which dilutes signal at convergence points. Effective range is ~±3 °C ocean / ~±2 °C land, vs. ~±10 °C for Gulf-Stream-class anomalies.
- **No full gyre structure beyond Stommel + wind curl.** The β-plane term captures western-boundary intensification, but not boundary-current intensification at Gulf-Stream scale.
- **Inland propagation is Dijkstra from the nearest ocean face**, not upwind-traced. The coastal parent is the closest ocean face, not the one the prevailing wind would carry air from.

---

## Tuning knobs (`ClimateParams`)

Sub-dataclasses mirror `ElevationParams` style (`_f(default, label=..., min=..., max=...)` factory for dialog metadata).

| Group | Field | Default | Range |
|---|---|---|---|
| `TemperatureParams` | `equator_annual_c` | 27.0 | -10, 40 |
| | `pole_annual_c` | -25.0 | -60, 10 |
| | `anomaly_scale` | 0.05 | 0.0, 0.5 |
| | `lapse_rate_c_per_km` | 6.5 | 0.0, 15.0 |
| | `ocean_inertia` | 0.30 | 0.0, 1.0 |
| | `land_inertia` | 1.0 | 0.0, 2.0 |
| | `inertia_distance_km` | 1500.0 | 100, 5000 |
| `WindParams` | `trade_speed` | 7.0 m/s | 0, 30 |
| | `westerly_speed` | 10.0 m/s | 0, 30 |
| | `polar_speed` | 5.0 m/s | 0, 30 |
| | `itcz_shift_deg` | 10.0 | 0, 30 |
| | `pressure_wind_scale` | 600.0 m·s⁻¹·km/°C | 0, 1000 |
| `PrecipitationParams` | `evaporation_scale` | 1.0 | 0.1, 5.0 |
| | `precipitation_scale` | 2000.0 mm/yr | 100, 10000 |
| | `moisture_advection_iter` | 200 | 50, 600 |
| | `moisture_decay_rate` | 0.03 | 0.005, 0.2 |
| | `orographic_k` | 0.4 | 0, 2 |
| | `humidity_smoothing_passes` | 10 (hidden) | 0, 30 |
| | `humidity_smoothing_alpha` | 0.4 (hidden) | 0.0, 1.0 |
| | `lake_evap_scale` | 1.0 (hidden) | 0, 5 |
| | `lake_feedback_iter` | 100 (hidden) | 0, 600 |
| `CurrentsParams` | `enabled` | `True` | — |
| | `beta_scale` | `1.0` | 0, 5 |
| | `advection_alpha` | `1.0` | 0, 1 |
| | `advection_iterations` | `400` | 10, 2000 |
| | `temperature_scale` | `0.06` | 0, 0.5 |
| | `precipitation_scale` | `0.03` | 0, 0.2 |
| | `inland_decay_km` | `600` | 50, 5000 |
| (root) | `seed` | `None` | Optional[int] |

---

## Display additions

Color schemes wired by `GlobeBuilder.apply_display`:

| Scheme name | Source | Mode |
|---|---|---|
| Biome | `biome` texture, categorical 9-class palette | default |
| Annual Temperature | `temperature` texture (mean of 4 seasons), blue→white→red diverging at 0 °C | debug |
| Summer Temperature | `temperature` texture season 1 (NH summer) | debug |
| Winter Temperature | `temperature` texture season 3 (NH winter) | debug |
| Annual Precipitation | `precipitation` texture (4-season sum), log scale mm | debug |
| Humidity | `humidity_annual` metadata (per-face, arid tan → moist blue-green) | debug |

Wind overlays (`winds` globe feature, 4 seasons) and current overlay (`currents` feature) are drawn as line segments at face centroids, length scaled by speed; both rendered in debug mode only.

---

## Known limitations / future work

- **Currents are a Stommel barotropic model** — captures gyre structure and western-boundary asymmetry, but peak anomalies are ~±3 °C rather than the ~±10 °C real western boundary currents produce.
- **No deflection around mountains or continents.** Surface winds pass straight through terrain. The pressure-anomaly wind component captures continental highs/lows but produces no anticyclonic/cyclonic rotation around them.
- **Per-face wind, per-pixel everything else.** Orographic precipitation uses the per-face wind vector even when sampling the per-pixel elevation gradient. Cross-face wind variation inside a single face is below the modelling resolution.
- **No snowpack or clouds.** Precipitation is the only hydrological output; the temperature field does not distinguish rain from snow (the classifier infers it from temperature).
- **Lapse rate is uniform.** Wet vs dry adiabatic distinction is ignored; coastal humidity does not damp the lapse rate.
- **Distance-to-coast is great-circle Dijkstra on the face graph.** Approximates surface distance to ~one-face resolution; fine for continentality damping but coarse for, e.g., island chains.
