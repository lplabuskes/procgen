# Resources Generation — As-Built Design

**Modules:** `worldgen/resources/` (`builder.py`, `terrain.py`, `fertility.py`, `forest.py`, `crops.py`, `minerals.py`, `rare_bio.py`, `display.py`)
**Purpose:** Produce per-pixel fertility and forest-density textures, a dominant-crop classification, and point-feature deposits (minerals and rare biological resources) on top of a fully-built elevation + climate + hydrology mesh.

Runs after `HydrologyBuilder.build()` and after `ClimateBuilder.apply_lake_humidity_feedback()`, so biome metadata reflects the final precipitation field including lake evaporation.

---

## Pipeline overview

```
elevation, temperature, precipitation, ocean_mask, lake_mask textures
per-face biome, volcanic_activity, boundary_type, plate_type metadata
rivers (list[River]), mesh
        │
        ▼
[ Terrain helpers ]                     # terrain.py
        │  face_elev_range (N,) km      — per-face pixel elevation span ("ruggedness")
        │  water_hops (N,) int           — BFS hops from river/lake seed faces
        │  volcanism_hops (N,) int       — BFS hops from volcanic seed faces
        ▼
[ Forest density ]                      # forest.py
        │  forest (N, H, W, 1) float32  — biome baseline × P_modulation × treeline + riparian + noise
        ▼
[ Fertility ]                           # fertility.py
        │  fertility (N, H, W, 1) float32 — climate × terrain × water × geology × biome × forest_penalty
        ▼
[ Crop classification ]                 # crops.py
        │  dominant_crop (N, H, W, 1) uint8 — first-match threshold table over T_grow, P_annual, elevation
        ▼
[ Mineral deposits ]                    # minerals.py
        │  minerals (M, 3) positions, (M,) kinds, (M,) abundances — geology-gated fBm noise
        ▼
[ Rare biological resources ]           # rare_bio.py
        │  rare_bio (M, 3) positions, (M,) kinds, (M,) abundances — ecology-gated fBm noise + global cap
        ▼
Globe with resources textures + "minerals" and "rare_bio" features
```

---

## Inputs

| Source | Field | Used by |
|---|---|---|
| `mesh.face_textures['elevation']` | `(N, H, W, 1)` float32 km | ruggedness, treeline, coal/salt elevation gate |
| `mesh.face_textures['temperature']` | `(N, H, W, K)` float32 °C | fertility climate term, crop classifier |
| `mesh.face_textures['precipitation']` | `(N, H, W, K)` float32 mm/season | fertility climate term, forest density, crop classifier |
| `mesh.face_textures['ocean_mask']` | `(N, H, W, 1)` uint8 | zero out ocean pixels in all outputs |
| `mesh.face_textures['lake_mask']` | `(N, H, W, 1)` float32 | water-proximity seed mask; lake exclusion for rare_bio |
| `mesh.face_metadata['biome']` | `int` per face | forest baseline LUT, fertility biome multiplier, rare_bio ecological gates |
| `mesh.face_metadata['volcanic_activity']` | `float` per face | volcanism seed mask, mineral gold/gem/rare_metals gates |
| `mesh.face_metadata['boundary_type']` | `str` per face | mineral iron/copper/tin/marble/coal gates |
| `mesh.face_metadata['plate_type']` | `str` per face | mineral coal/salt candidate mask |
| `mesh.face_metadata['elevation']` | `float` per face | mineral coal/salt elevation gate; rare_bio ecological gates |
| `mesh.face_metadata['temperature_annual']` | `float` per face | rare_bio ecological gates |
| `mesh.face_metadata['precipitation_annual']` | `float` per face | rare_bio ecological gates |
| `mesh.face_metadata['distance_to_coast']` | `float` per face | amber coastal gate |
| `mesh.face_metadata['is_ocean']` | `bool` per face | mineral/rare_bio land gates; pearl ocean gate |
| `rivers` | `list[River]` | water-proximity seed mask |

---

## Data model

### Face textures (attached to mesh)

| Key | Shape | Dtype | Meaning |
|---|---|---|---|
| `fertility` | `(N, H, W, 1)` | `float16` | Agricultural fertility in [0, 1] |
| `forest_density` | `(N, H, W, 1)` | `float16` | Forest cover fraction in [0, 1] |
| `dominant_crop` | `(N, H, W, 1)` | `uint8` | Best-fit crop index per pixel (0 = none) |

### Per-face metadata

| Key | Type | Meaning |
|---|---|---|
| `fertility_mean` | `float` | Mean fertility across all pixels in the face |
| `forest_density_mean` | `float` | Mean forest density across all pixels |
| `dominant_crop` | `int` | Modal crop index across non-ocean pixels |

### Globe features (point overlays)

| Name | Structure | Meaning |
|---|---|---|
| `minerals` | `(positions (M,3), kinds (M,) int8, abundances (M,) float32)` | Unit-sphere face centroids, `MINERAL_*` index, normalised deposit richness |
| `rare_bio` | `(positions (M,3), kinds (M,) int8, abundances (M,) float32)` | As above, `RARE_BIO_*` index |

---

## Stage 1 — Terrain helpers

### Ruggedness (`face_elevation_range`)

Per-face pixel elevation span in km: `max(elev_pixels) − min(elev_pixels)`. Used as a robust proxy for mountainous terrain. Per-pixel slope from `np.gradient(elevation_tex)` is explicitly avoided — at default texture resolution the noise overlay dominates the gradient signal and would inject spurious ruggedness everywhere.

### BFS hop counts (`face_hops_from_seeds`)

Breadth-first search on the face adjacency graph from a boolean seed mask. Returns `(N,) int32` capped at `max_hops + 1` for unreachable faces. Two hop arrays are built:

- **`water_hops`**: seeds are faces traversed by non-lake river segments plus faces whose lake-mask coverage exceeds 5%. Used for both the floodplain fertility bonus and the riparian forest bonus. `max_hops = max(floodplain_max_hops, riparian_max_hops)` so a single BFS pass satisfies both consumers.
- **`volcanism_hops`**: seeds are faces with `volcanic_activity ≥ volcanism_threshold`. Used for the volcanic-soil fertility bonus and the mineral gold/gem geological candidate mask.

---

## Stage 2 — Forest density

```
biome → BIOME_FOREST_BASELINE[biome]   (N,) face baseline in [0, 1]
biome → BIOME_PRECIP_REFERENCE_MM[biome]  (N,) per-biome reference P (mm/yr)
        │
        ▼
[ Rasterize per-face values → (N, H, W) ]   # rasterize_face_values; blends across face boundaries
        │  baseline_tex, p_ref_tex, riparian_tex
        ▼
P_modulation = clip(1 + strength × tanh((P_annual − P_ref) / P_ref), 0.7, 1.3)
treeline     = clip((treeline_km − elevation) / falloff_km, 0, 1)
riparian_tex = riparian_bonus × exp(−riparian_hops × decay) [capped at max_hops]
noise        = fBm simplex (N, H, W)  × noise_amplitude   [skipped if amplitude == 0]
        │
        ▼
density = clip(baseline_tex × P_modulation × treeline + riparian_tex + noise, 0, 1)
ocean pixels → 0
```

**Biome LUTs** (`BIOME_FOREST_BASELINE`, `BIOME_PRECIP_REFERENCE_MM`) are module-level constants. The baseline encodes the biome's structural forest potential independent of local precipitation; the reference centres the P_modulation so the tanh is ±0 at a biome-typical wetness.

---

## Stage 3 — Fertility

Fertility is a multiplicative product of five per-face factors, optionally docked by a per-pixel forest penalty:

```
fertility = clip(climate × terrain × water × geology × biome_mod
                 × (1 − forest_penalty_strength × forest_density), 0, 1)
```

All five factors are computed as `(N,)` per-face scalars, combined into a single `face_mult`, then rasterized via `rasterize_face_values` (vertex-averaged barycentric interpolation) so adjacent faces blend continuously.

### Climate factor

Bell-curve response to growing-season temperature and log-precipitation:

```
T_grow  = mean per-pixel T over seasons where T > 5°C (falls back to annual mean)
t_term  = exp(−0.5 × ((T_grow − T_opt) / σ_T)²)
p_term  = exp(−0.5 × ((log(P_annual) − log(P_opt)) / σ_P)²)
climate = t_term × p_term
```

Using log-precipitation reflects the diminishing returns of additional rain at high P; using growing-season T avoids penalising faces that are cold but not frozen during their growth window.

### Terrain factor

```
terrain = 1 − clip(face_elev_range_km / threshold_km, 0, 1)
```

Smooth ramp from 1 (flat) to 0 (rugged). Uses the same ruggedness proxy as terrain helpers.

### Water and geology factors (hop-decay bonuses)

```
water   = 1 + floodplain_bonus × exp(−hops × floodplain_decay)   [beyond max_hops → 1]
geology = 1 + volcanic_bonus   × exp(−hops × volcanic_decay)      [beyond max_hops → 1]
```

Floodplain bonus models alluvial soil enrichment near rivers and lakes. Volcanic-soil bonus models nutrient-rich basaltic soils downslope of volcanic fields.

### Biome multiplier

`BIOME_FERTILITY_MULT` (module-level constant, indexed by the 9-class biome):

| Biome | Multiplier |
|---|---|
| Icecap | 0.00 |
| Tundra | 0.20 |
| Taiga | 0.60 |
| Temperate | 1.00 |
| Steppe | 0.50 |
| Savannah | 0.80 |
| Tropical | 0.95 |
| Semi-Arid | 0.30 |
| Desert | 0.05 |

### Forest penalty

Dense tropical forest sequesters nutrients in biomass, leaving soils poor for agriculture once cleared. A small multiplicative deduction proportional to the per-pixel `forest_density` texture captures this; controlled by `forest_penalty_strength` (default 0.30).

---

## Stage 4 — Crop classification

Mirrors `worldgen/climate/classification.py`: a first-match-wins threshold table applied per pixel. Inputs are growing-season T (`T > 5°C` seasons averaged, falling back to annual mean), annual P, elevation, and a fertility gate.

```
if fertility < FERTILITY_GATE (0.08): → CROP_NONE
else: scan table in order, return first matching crop
```

**Crop table** (ordered from most specific to least specific):

| Crop | T_grow (°C) | P_annual (mm) | Elevation (km) |
|---|---|---|---|
| Sugarcane | 22–50 | > 1800 | < 1.0 |
| Coffee | 15–25 | > 2000 | 0.9–1.9 |
| Rice | 20–50 | > 1000 | < 2.0 |
| Millet | 22–50 | 200–750 | < 2.0 |
| Cotton | 18–50 | 500–1500 | < 2.0 |
| Maize | 16–50 | 400–2500 | < 3.0 |
| Wheat | 5–28 | 250–1500 | < 3.5 |
| Barley | 3–26 | 150–1200 | < 4.5 |

The vectorised `classify_crops` applies the table in reverse order (broadcasting masks) so earlier, more-specific entries overwrite later ones in a single pass — semantically identical to the scalar `default_crop` reference implementation.

Per-face `dominant_crop` metadata is the mode of land pixels using `np.bincount`.

---

## Stage 5 — Mineral deposits

A hybrid geology + fBm noise model. Geology constrains *where* each mineral type can appear; fBm noise clusters actual deposit sites and sets abundance.

### Geological candidate masks

| Mineral | Candidate mask |
|---|---|
| Iron | land ∩ convergent boundary |
| Copper | land ∩ convergent boundary (independent noise draw) |
| Gold | land ∩ volcanic_activity ≥ `volcanic_threshold_gold` |
| Gems | land ∩ volcanic_activity ≥ `volcanic_threshold_gems` (rarer threshold) |
| Coal | continental ∩ non-ocean ∩ elevation < `coal_max_elevation_km` ∩ boundary_hops ≥ `coal_min_boundary_hops` |
| Tin | land ∩ divergent boundary |
| Rare Metals | land ∩ (divergent ∪ volcanic_activity ≥ `volcanic_threshold_rare`) |
| Salt | continental ∩ non-ocean ∩ arid biome ∩ elevation < 1.5 km ∩ boundary_hops ≥ `coal_min_boundary_hops` |
| Marble | continental ∩ non-ocean ∩ convergent boundary |

Coal and salt share a "tectonically quiet sedimentary basin" gate via BFS hops from any non-interior boundary face. Salt additionally requires an arid biome (Semi-Arid or Desert) as a proxy for evaporite conditions.

### Deposit placement

For each mineral type, a 3-octave fBm simplex noise field is sampled at face centroids (seed offset by `kind_index × 1000` to keep types spatially independent). A face becomes a deposit when `noise > threshold AND candidate_mask`. Abundance is the normalised excess above threshold:

```
abundance = clip((noise − threshold) / (1 − threshold), 0, 1)
```

### Output

`(positions (M,3), kinds (M,) int8, abundances (M,) float32)` — stored as `ResourcesBuilder.minerals`, registered as a `"minerals"` globe feature by `GlobeBuilder`.

---

## Stage 6 — Rare biological resources

Ten rare biological resources placed using narrow ecological criteria plus independent fBm noise draws. A global per-type cap enforces world-level rarity.

### Land resources

| Resource | Key ecological criteria |
|---|---|
| Spices | Tropical, T_annual > 22°C, P > 1500 mm, elevation < 1.0 km |
| Tea | Tropical or Temperate, T 13–24°C, P > 1200 mm, elevation 0.5–2.5 km |
| Silk | Temperate/Tropical/Savannah, T 14–28°C, P 400–1800 mm, elevation < 2.0 km |
| Ivory | Savannah/Tropical/Steppe, T > 18°C, P 350–1600 mm, elevation < 1.5 km |
| Furs | Tundra or Taiga, T_annual < 5°C |
| Incense | Desert or Semi-Arid, elevation 0.3–2.5 km |
| Dyes | Tropical/Savannah/Temperate/Steppe, T 14–30°C, P 400–2500 mm, elevation < 2.5 km |
| Amber | Temperate or Taiga, T −5–15°C, distance_to_coast < 400 km, elevation < 0.8 km |
| Rubber | Tropical, T_annual > 22°C, P > 1800 mm, elevation < 0.8 km |

Lake faces are excluded from land-resource eligibility alongside ocean faces.

### Ocean resource

| Resource | Key ecological criteria |
|---|---|
| Pearls | ocean ∩ coastal_hops ≤ 2 ∩ T_annual > 18°C |

`coastal_hops` is a BFS from all non-ocean faces (capped at 3), so pearls are placed only in near-coastal warm ocean faces.

### Placement algorithm

For each type:
1. Compute fBm noise at face centroids (3 octaves, seed `base_seed + kind × 1000 + 77777`).
2. Deposit where `noise > base_threshold AND ecological_mask`.
3. Apply global cap: keep only the top `max_N` faces by noise score (discards lower-scoring sites even above threshold).
4. Abundance = normalised excess above threshold.

The global cap is the primary mechanism for world-level rarity — `base_threshold` controls density within the eligible zone, the cap sets an absolute ceiling regardless of how large the eligible zone is.

### Output

`(positions (M,3), kinds (M,) int8, abundances (M,) float32)` — stored as `ResourcesBuilder.rare_bio`, registered as a `"rare_bio"` globe feature by `GlobeBuilder`.

---

## Display

All five resource schemes are registered by `GlobeBuilder.apply_display` regardless of debug mode, since these are world-building outputs players interact with directly.

| Scheme | Type | Notes |
|---|---|---|
| Fertility | per-pixel texture | 6-stop gradient: gray → tan → light green → deep green |
| Forest Density | per-pixel texture | 5-stop gradient: tan → light green → dark forest |
| Dominant Crop | per-pixel texture | categorical palette per `CROP_NAMES`; muted gray for none |
| Minerals | point overlay | `MINERAL_PALETTE` RGBA per type; drawn as markers in `MapWidget` |
| Rare Resources | point overlay | `RARE_BIO_PALETTE` RGBA per type |

Ocean pixels in texture schemes are overridden with `OCEAN_COLOR`; lake pixels with `LAKE_COLOR`.

---

## Key design decisions

**Per-face factors rasterized to pixels, not computed per-pixel.** Climate, terrain, water proximity, volcanism proximity, biome, forest baseline, and riparian bonus are all computed as per-face scalars and then rasterized via `rasterize_face_values` (vertex-averaged barycentric interpolation). This gives continuous blending across face boundaries without face-blocking artifacts, and avoids computing expensive per-pixel climate quantities a second time.

**Ruggedness via elevation range, not per-pixel slope.** At default texture resolution, `np.gradient(elevation_tex)` is dominated by the fBm noise overlay rather than actual terrain shape, making slope an unreliable proxy for "mountainous." Elevation range (max − min within a face) correlates with true terrain relief at the face scale and is immune to per-pixel noise.

**Biome LUTs are module-level constants.** `BIOME_FERTILITY_MULT`, `BIOME_FOREST_BASELINE`, and `BIOME_PRECIP_REFERENCE_MM` define the model rather than tune it. They are not exposed in `ResourcesParams` and should only be exposed if per-world model variation becomes a real need.

**Minerals: geology-first, noise-second.** Pure noise placement would produce geologically implausible distributions; pure geology would be too deterministic. The hybrid approach gives plausible regional distributions (iron near subduction zones, coal in quiet basins) while noise supplies the clustering and rarity variation that makes exploration interesting.

**Rare resources: ecological gating + noise + global cap.** The cap is the primary rarity knob — it guarantees world-level scarcity independent of how large the eligible habitat zone is. `base_threshold` and `noise_frequency` control within-zone density and cluster size.

**Forest penalty on fertility.** Dense tropical forest sequesters nutrients in its canopy biomass; soils under rainforest are notoriously poor for agriculture despite a lush biome score. The multiplicative penalty `(1 − strength × forest_density)` captures this without adding a separate lookup table.
