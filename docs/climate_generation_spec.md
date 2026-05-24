# Climate Generation — Intended Design

**Modules (to be created):** `worldgen/climate/` (`builder.py`, `insolation.py`, `temperature.py`, `winds.py`, `precipitation.py`, `classification.py`, `currents.py`, `display.py`)
**Purpose:** Produce per-face, per-season temperature & precipitation textures, a biome classification, and a prevailing-wind vector field on top of an already-generated elevation mesh.

This is a forward-looking spec written before implementation. It is intentionally a *minimum viable climate*: zonal winds (no pressure-cell modulation), no ocean currents, quantitative but heuristic. Hooks are reserved for currents to be added later without changing the call shape of temperature/precipitation.

Inspired by Geoff's [Climate Cookbook](https://worldbuildingworkshop.com/2015/11/27/climate/), but rendered as continuous (lat, season) → (°C, mm/season) functions rather than lookup tables.

---

## Pipeline overview

```
elevation textures, mesh
        │
        ▼
[ Seasonal insolation Q(lat, season) ]              # insolation.py
        │  Q: (N_faces, n_seasons) W/m² daily mean
        ▼
[ Distance-to-coast field ]                         # builder, BFS on face graph
        │  d_coast: (N_faces,) km
        ▼
[ Surface temperature ]                             # temperature.py
        │  T_face: (N_faces, n_seasons) °C   (annual mean + damped seasonal anomaly,
        │                                     lapse rate, optional current delta)
        ▼
[ Prevailing winds (zonal Hadley/Ferrel/Polar) ]    # winds.py
        │  W_face: (N_faces, n_seasons, 3) tangent vectors, m/s
        ▼
[ Precipitation ]                                   # precipitation.py
        │  P_face: (N_faces, n_seasons) mm/season   (zonal base × orographic × continentality)
        ▼
[ Texture rasterisation ]                           # builder
        │  per-pixel T, P from per-face values via elevation-aware downsample
        ▼
[ Classification ]                                  # classification.py
        │  biome: (N_faces, H, W) uint8
        ▼
Globe with climate textures + winds feature
```

---

## Inputs

| Source | Field | Used by |
|---|---|---|
| `mesh.face_centroids` | 3D unit-sphere positions | latitude, wind tangent frames |
| `mesh.face_adjacency` | edge-sharing neighbours | distance-to-coast BFS, gradient |
| `mesh.face_metadata['elevation']` | per-face elevation (km) | temperature lapse, ocean/land mask |
| `mesh.face_textures['elevation']` | per-pixel elevation (km) | per-pixel T downscale, orographic uplift |
| `ElevationFieldParams.elevation_tex_size` | H = W = tex_size | shared texture resolution |

Sea level threshold (elev ≤ 0 km) defines the ocean mask, identically to hydrology.

---

## Data model

### Face textures (attached to mesh)

| Key | Shape | Dtype | Meaning |
|---|---|---|---|
| `temperature` | `(N, H, W, 4)` | `float32` | Channel `j` = mean surface temperature in °C during season `j` |
| `precipitation` | `(N, H, W, 4)` | `float32` | Channel `j` = total precipitation in mm during season `j` |
| `biome` | `(N, H, W, 1)` | `uint8` | Biome index (0–8 for the default 9-biome classifier) |

> `PolygonMesh.set_face_textures` currently hard-codes `float32`. To support the `uint8` biome channel, the signature needs an optional `dtype` parameter (or to infer dtype from the array). The texture-atlas packing in `MeshViewer._setup_buffers` and the `QImage` conversion in `MapWidget._rebuild_textures` already pass through `uint8` for display purposes; the change is local to validation and storage. Spec'd as a small precursor to the climate work.

### Per-face metadata (for color schemes & quick display)

| Key | Type | Meaning |
|---|---|---|
| `temperature_annual` | `float` | Annual mean °C |
| `temperature_warmest` | `float` | Warmest season mean °C |
| `temperature_coldest` | `float` | Coldest season mean °C |
| `precipitation_annual` | `float` | Annual total mm |
| `precipitation_seasonality` | `float` | (max − min seasonal P) / annual P, in [0, 1] |
| `biome` | `int` | Dominant biome index (mode across the face's pixels) |
| `distance_to_coast` | `float` | km along great-circle from the nearest ocean face |

### Globe features (vector overlays)

| Name | Structure | Meaning |
|---|---|---|
| `prevailing_winds` | `list[(positions_3d, vectors_3d)]` length 4 | One entry per season; `positions_3d` is `(N_faces, 3)` face centroids, `vectors_3d` is `(N_faces, 3)` wind velocity (m/s) tangent to the sphere |

Rendered as four toggleable map overlays (`Winds — Spring`, `Summer`, `Autumn`, `Winter`), all hidden by default. Drawing strategy: short line segments at each face centroid, length scaled by wind speed.

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
                          − Γ · elevation_km(face)
                          + current_delta(face, season)               # 0 in MVP
```

Tunables (`TemperatureParams`):

| Name | Default | Effect |
|---|---|---|
| `T_eq` | `27.0` °C | Equatorial annual mean |
| `T_pole` | `-25.0` °C | Polar annual mean |
| `anomaly_scale` (`b`) | tuned to give ~3 °C amplitude at the equator and ~25 °C at 60° land | converts insolation anomaly to °C anomaly |
| `lapse_rate` (`Γ`) | `6.5` °C/km | dry adiabatic-ish, applied uniformly |
| `ocean_inertia` | `0.3` | factor for face entirely over ocean (damps seasonal swing) |
| `land_inertia` | `1.0` | factor for face deep inland |

`inertia_factor(face)` interpolates between `ocean_inertia` and `land_inertia` by `distance_to_coast` (saturating at ~1500 km inland with `tanh(d_coast / 1500)`).

### Per-pixel temperature texture

Lapse-rate scaling is applied per-pixel using the face's pixel-elevation texture so high terrain inside an otherwise-warm face gets cooled. All other terms are face-uniform (the per-face value is broadcast to every pixel of the face).

```
T_pixel[f, h, w, j] = T_face[f, j] + Γ · elev_face[f] − Γ · elev_pixel[f, h, w]
```

This keeps face-scale physics where it belongs and adds local terrain detail "for free".

---

## Stage 3 — Prevailing winds (zonal)

The wind field is computed at face centroids; per-pixel winds are not stored.

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

then `wind_3d = u_east · east_hat + v_north · north_hat`. The sign of `u_east` and `v_north` follows the band lookup above; magnitude is the band's speed default.

The mesh is rigid (no deflection around landmasses in MVP). This is the single largest concession to the article's model and is the obvious next thing to add when currents go in: a divergence-free correction that routes winds around mountains and coastlines.

---

## Stage 4 — Precipitation

```
P_face[f, j] = P_zonal(φ_eff[f, j])
               · orographic_factor(f, j)
               · continentality_factor(f)
               · (1 + current_wet_delta(f, j))      # 0 in MVP
```

### Zonal base

`P_zonal(φ_eff)` is a piecewise function (mm/season):

| Latitude (|φ_eff|) | Annual baseline | Per-season (÷4) |
|---|---|---|
| 0° (ITCZ) | 2200 mm | 550 mm |
| 30° (horse latitudes) | 250 mm | 62 mm |
| 50°–60° (mid-lat) | 900 mm | 225 mm |
| 75°+ (polar) | 200 mm | 50 mm |

Smoothly interpolated (Gaussian sum centred on each band, see `PrecipitationParams.zonal_*` knobs).

Because `φ_eff` shifts seasonally, the same face gets a different zonal base per season — monsoonal latitudes (~10–20°) get the ITCZ band in summer and the horse latitudes in winter.

### Orographic uplift

```
∇h[f] = projected elevation gradient at face centroid, units km / km
        (3D vector tangent to sphere, computed from elevation differences with face neighbours)

uplift[f, j] = wind_3d[f, j] · ∇h[f]            # m/s · (km/km) = m/s, positive on windward side

orographic_factor[f, j] = clamp(1 + K_orographic · uplift[f, j], 0.1, 3.0)
```

`K_orographic` default = 0.4 (tuneable). Positive uplift wets the windward side; negative uplift (leeward) drops precipitation by up to 10×. This single term captures both rain shadow and mountain enhancement.

### Continentality

```
continentality_factor[f] = lerp(1.0, 0.2, tanh(d_coast[f] / 1200 km))
```

Deep interior gets ~20% of the coastal precipitation. Distance is in km along the surface.

### Per-pixel precipitation texture

Like temperature, the per-face value is broadcast; we additionally re-apply a per-pixel orographic correction using the pixel-elevation gradient inside the face. The pixel uplift is computed from finite differences on the per-face elevation texture (the same neighbours the river code uses, via D8 grid + inter-face stitching is overkill — plain interior-only `np.gradient` is enough since orographic effects don't need cross-face precision).

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
t_warmest    = temperature_c.max()
t_coldest    = temperature_c.min()
p_annual     = precipitation_mm.sum()
p_seasonality = (precipitation_mm.max() - precipitation_mm.min()) / max(p_annual, 1e-6)
```

Decision tree (in order; first match wins):

1. `t_warmest < 0`                                                    → **Icecap (0)**
2. `t_warmest < 10`                                                    → **Tundra (1)**
3. `t_coldest < -10`                                                   → **Taiga (2)**
4. `p_annual < 250`  and `t_warmest > 18`                              → **Desert (8)**
5. `p_annual < 500`  and `t_warmest > 18`                              → **Semi-Arid (7)**
6. `p_annual < 400`                                                    → **Steppe (4)**
7. `t_warmest > 22` and `p_annual > 1500` and `p_seasonality < 0.4`    → **Tropical (6)**
8. `t_warmest > 22` and `p_seasonality ≥ 0.4`                          → **Savannah (5)**
9. fallthrough                                                          → **Temperate (3)**

Thresholds are intentionally loose. The classifier runs **per pixel** (using the per-pixel temperature/precip textures) so we get sub-face transitions like "snow line on a tropical peak". For per-face metadata, the dominant biome is the integer mode over the face's pixels.

`default_biome` is the canonical scalar reference. `classification.py` additionally exposes `classify_textures(temperature_tex, precipitation_tex, lat_tex) -> biome_tex` which implements the same decision tree fully vectorised over `(N, H, W, 4)` arrays — this is what the builder calls. The two must agree pixel-for-pixel; a small test pins this invariant.

### Swappability

`ClimateBuilder.__init__` takes `classifier: Callable[..., np.ndarray] = classify_textures`. Replacing it (e.g. Köppen-Geiger later) requires only matching the array-in / array-out signature; no other code in the module reads biome indices semantically. Users who only want to override the per-point logic can pass `lambda t, p, lat: vectorise_pointwise(my_default_biome, t, p, lat)`.

---

## Module layout

```
worldgen/climate/
    __init__.py
    builder.py            # ClimateParams + ClimateBuilder
    insolation.py         # seasonal_insolation(latitudes, declinations) -> Q[face, season]
    temperature.py        # surface_temperatures(...) -> (T_face, T_pixel)
    winds.py              # zonal_winds(centroids, season_phases, params) -> W[face, season, 3]
    precipitation.py      # surface_precipitation(...) -> (P_face, P_pixel)
    classification.py     # BIOME_NAMES, default_biome, classify_textures
    currents.py           # stub: compute_currents(...) -> None  (raises NotImplementedError)
    display.py            # color schemes for T/P/biome + wind overlay rendering helpers
```

Per-stage functions are pure (input arrays → output arrays) so they can be tested in isolation. The builder is responsible for plumbing.

### `ClimateBuilder`

```python
class ClimateBuilder:
    def __init__(
        self,
        params: ClimateParams,
        classifier: Callable[..., int] = default_biome,
    ): ...

    def build(
        self,
        mesh: PolygonMesh,
        elevation_textures: np.ndarray,        # (N, H, W, 1)
        progress: Optional[Callable[[str], None]] = None,
    ) -> None: ...
```

Attaches `temperature`, `precipitation`, and `biome` textures to the mesh and stores intermediate results as attributes:

| Attribute | Type | Notes |
|---|---|---|
| `insolation` | `(N, 4) float32` | W/m², daily mean |
| `distance_to_coast` | `(N,) float32` | km |
| `temperature_face` | `(N, 4) float32` | °C |
| `precipitation_face` | `(N, 4) float32` | mm/season |
| `winds` | `(N, 4, 3) float32` | m/s tangent |
| `currents` | `None` | reserved; set by future currents stage |

---

## Pipeline integration

`GlobeBuilder.build()` order becomes:

```
elevation_builder.build()
climate_builder.build(mesh, elevation_textures)        # ← new
hydrology_builder.build(mesh, elevation_textures)
```

Hydrology does not consume climate output in this iteration — the river network is still driven by elevation alone. A future revision can switch `flow_accum[k] = 1.0` to `flow_accum[k] = precipitation_pixel[k] / reference_P` (see `river_generation_spec.md` § "Current limitations"), which is the natural place for climate → hydrology coupling.

`GlobeParams` gains `climate: ClimateParams`. `GlobeBuilder.apply_display` is extended to register temperature/precip/biome color schemes and the four wind overlays.

---

## Currents-later seams

Three explicit hooks let currents drop in without re-shaping the public API:

1. **`temperature.surface_temperatures(...)` signature** accepts an optional `coastal_current_temp_delta: np.ndarray | None = None` of shape `(N, 4)`. In MVP this is `None` and adds zero. Currents will populate it from advected SST anomalies.
2. **`precipitation.surface_precipitation(...)` signature** accepts an optional `current_wet_delta: np.ndarray | None = None` of shape `(N, 4)`, treated as a multiplicative correction `(1 + delta)`. MVP passes `None`. Warm currents → positive delta on downwind coasts; cold currents → negative.
3. **`ClimateParams.currents: CurrentsParams`** dataclass exists with `enabled: bool = False`. `ClimateBuilder.build` early-returns the currents stage when disabled; the attribute `self.currents` stays `None`.

Future currents work will:
- Add `worldgen/climate/currents.py::compute_currents(mesh, temperature_face, winds) -> CurrentsResult`
- Populate `self.currents` and derive the two delta arrays
- Re-call temperature & precipitation with the deltas (a second pass), or carry them through the first call if the dependency order can be untangled

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
| `PrecipitationParams` | `itcz_mm_annual` | 2200 | 0, 5000 |
| | `horse_mm_annual` | 250 | 0, 2000 |
| | `midlat_mm_annual` | 900 | 0, 3000 |
| | `polar_mm_annual` | 200 | 0, 2000 |
| | `orographic_k` | 0.4 | 0, 2 |
| | `continentality_decay_km` | 1200 | 100, 5000 |
| `CurrentsParams` | `enabled` | `False` | bool (MVP placeholder) |
| (root) | `seed` | `None` | Optional[int] |

---

## Display additions

New color schemes wired by `GlobeBuilder.apply_display`:

| Scheme name | Source | Style |
|---|---|---|
| Annual Temperature | `temperature_annual` metadata + per-pixel temperature texture | blue→white→red diverging, anchored at 0 °C |
| Summer Temperature | warmest-season slice of `temperature` texture | same palette, season 1 (NH summer) |
| Winter Temperature | coldest-season slice of `temperature` texture | same palette, season 3 (NH winter) |
| Annual Precipitation | sum of seasonal precipitation channels | yellow→green→deep blue, log scale (mm) |
| Biome | `biome` texture | 9 categorical swatches, dark grey for unset |

Per-scheme texture associations follow the existing `window.associate_scheme_texture(name, texture_key)` pattern.

Four wind overlays (`Winds — Spring/Summer/Autumn/Winter`) draw as line segments at face centroids, length scaled by wind speed; toggled like the rivers overlay.

---

## Known limitations / future work

- **No coastal current effects.** Coastal climate asymmetry (e.g. west-coast vs east-coast at the same latitude) is absent until the currents stage lands.
- **Zonal winds only.** No deflection around mountains or continents, no monsoonal reversal beyond the ITCZ shift, no continental low/high cells.
- **Hemispheric symmetry in the wind model.** A real world has hemispheric asymmetry because land/sea distribution is asymmetric. MVP winds are purely zonal so any seed with strong land/ocean asymmetry will look "too clean".
- **Per-face wind, per-pixel everything else.** Orographic precipitation uses the per-face wind vector even when sampling the per-pixel elevation gradient. Cross-face wind variation inside a single face is below the modelling resolution.
- **No snowpack, no humidity, no clouds.** Precipitation is the only hydrological output; the temperature field does not distinguish rain from snow (the classifier infers it from temperature).
- **Lapse rate is uniform.** Wet vs dry adiabatic distinction is ignored; coastal humidity does not damp the lapse rate.
- **Distance-to-coast is great-circle on the face graph.** It approximates surface distance to ~one-face resolution; fine for continentality damping but coarse for, e.g., island chains.
- **Texture dtype precursor.** Requires `PolygonMesh.set_face_textures` to accept `uint8` for the biome channel — straightforward but a prerequisite change to land before climate.
