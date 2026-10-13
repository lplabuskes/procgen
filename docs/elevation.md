# Elevation & Tectonics Stage

## Files
- `worldgen/geology/elevation_builder.py` — stage + params
- `worldgen/geology/elevation.py` — elevation model, erosion, rasterisation, noise, seam smoothing, ocean mask
- `worldgen/geology/tectonics.py` — plates
- `worldgen/geology/volcanism.py` — activity + volcano peaks
- `worldgen/geology/ruggedness.py` — the ruggedness seed (`RuggednessParams`, `ruggedness_seed`, `ruggedness_texture`; shared `decayed_max` / `diffuse`)
- `worldgen/geology/jaggedness.py` — the jaggedness (terrain character) seed (`JaggednessParams`, `jaggedness_seed`, `NEUTRAL_JAGGEDNESS`)
- `worldgen/geology/channels.py` — `IS_OCEAN`, `ELEVATION_FACE`, `PLATE_TYPE`, `BOUNDARY_TYPE`, `VOLCANIC_ACTIVITY`, `ELEVATION`, `OCEAN_MASK`, `RUGGEDNESS_FACE`, `RUGGEDNESS`, `JAGGEDNESS_FACE`, `JAGGEDNESS`
- `worldgen/geology/display.py`

## `elevation_builder.py`

**`ElevationParams`**:
- `plates` — `n_plates`, `plate_compactness`, `relax_max_passes`, `oceanic_fraction`
- `plate_motion` — `drift_scale`, `spin_scale`
- `elevation` — `elevation_rate_norm`, `elevation_decay_rate`, `elevation_max_hops`, `elevation_tex_size` (world texture size, 16)
- `noise` — `noise_amplitude`, `noise_coast_sigma`
- `face_erosion` — hydraulic/thermal knobs (hidden)
- `volcanism` — `n_hotspots`, `volcanic_decay_rate`, `volcanic_max_hops`, `volcanic_elevation_scale`, `volcanic_elevation_scale_oceanic`
- `ruggedness` — the seed: `base` (the master knob, in `BASIC_PARAMS` as `globe.elevation.ruggedness.base`) and the term weights ([below](#ruggednesspy))
- `jaggedness` — the character seed: `base` (neutral 0.5) and the term weights ([below](#jaggednesspy)); all advanced

Mesh params are not here (mesh is pre-built); river params are in
`HydrologyParams`. The seed goes to `ElevationBuilder(params, *, seed)`; each draw
takes a name-keyed sub-seed (`_seed("plates")`, …) so adding a draw never shifts
others.

**`ElevationBuilder.build(ctx)`** decorates the dual mesh:

plates → relax → types + velocities → boundary classification → face elevations
→ volcanism → face erosion → + volcanism × `volcanic_elevation_scale[_oceanic]` →
ruggedness and jaggedness seeds → `rasterize_face_values` → fBm simplex noise (`compute_noise_elevation_deltas`,
band-limited to the texel grid, × `exp(−|elevation| / noise_coast_sigma)`, so
strongest near sea level, roughening coasts) → seam smoothing → ocean mask →
`world.antimeridian` (`find_ideal_antimeridian`).

Attributes: `mesh`, `plate_ids`, `plate_omegas`, `face_velocities`,
`boundary_types`, `boundary_data`, `elevations` (face means of final texture),
`elevation_textures`, `ocean_mask`, `volcanism`, `volcano_positions`, `max_speed`.

- Textures: `"elevation"` (`(N, H, W, 1)` float32, km), `"ocean_mask"` (`(N, H, W, 1)` uint8), `"ruggedness"`, `"jaggedness"` (`(N, H, W, 1)` float32 in [0, 1]).
- Per face: `plate_id`, `plate_type`, `boundary_type`, `surface_speed`, `volcanic_activity`, `elevation` (tectonic + erosion + volcanic value *before noise* — read by resource placement), `is_ocean` (> half texels ocean).
- Per face: `ruggedness`, `jaggedness` (seed indices in [0, 1]). The ruggedness stage rewrites both fields later and restores the per-face values on load ([ruggedness.md](ruggedness.md)).
- Feature `"volcanoes"`: unit-sphere positions of above-sea-level activity peaks.

## `elevation.py`

**`compute_face_elevations(...)`** (km per face):
- Baseline: oceanic −4.0, continental +0.3.
- Boundary delta × `|normal_avg| / rate_norm`: continental–continental convergence +4; continental over subducting oceanic +2.5 (arc), oceanic side −2.5 (trench); oceanic–oceanic −1.5; ridge +2; continental rift −1.5; transform/interior 0.
- BFS spread with `exp(-decay_rate * hops)` up to `max_hops`; max magnitude wins.

**`compute_erosion(mesh, elevations, params)`**: face-level hydraulic (fill
sinks → accumulate → erode/deposit), then thermal (talus).

**`rasterize_face_values(mesh, face_values, tex_size, support_spacings=2.0)`** →
`(N, H, W, K)` float32; shared by every stage rasterising a per-face field
(elevation, temperature, humidity, fertility, forest, ruggedness). Each
texel = Wendland-C² weighted mean of face values whose centroids lie within
`support_spacings` mean centroid spacings (a face's centre keeps ~½ its own
value). A function of position alone → faces agree on seams/over-hangs, gradient
continuous. (Replaced piecewise-linear fan interpolation, which creased at face
edges and flattened lone high faces into ⅓-height hexagonal plateaus.)

**`compute_ocean_mask_texture(mesh, elevation_textures)`**: pixel-level flood.
Seeds: faces with max elevation ≤ −0.5 km; spreads to adjacent faces at/below sea
level, then per pixel (4-connected in a face, across faces via seam pixel pairs).
Unreached sub-sea depressions stay land. The region tier runs a level-constrained
version over refined relief.

## `tectonics.py`

- `generate_plates(mesh, n_plates, seed, compactness)` — randomized-priority flood: each step picks a plate uniformly (no compounding frontier advantage) and grows it one face via a distance-biased heap → compact, irregular plates; plates wholly surrounded by one other are absorbed.
- `relax_plates(mesh, plate_ids, max_passes)` — removes appendages via articulation points (`_tarjan_aps_iterative`) until stable.
- `generate_plate_velocities(...)` → `(n_plates, 3)` angular velocities = drift (tangential at plate centroid) + spin (about centroid axis). `compute_face_velocities`: `omega[k] × face_centroid`.
- `assign_plate_types(n_plates, oceanic_fraction, seed)` — default 0.7 oceanic.
- `classify_plate_boundaries(...)` → per-face dict: `boundary_type` (`interior|convergent|divergent|transform`), `normal_avg`, `neighbor_plate_types`.

## `volcanism.py`

`compute_volcanism(...)` — activity ∈ [0, 1] from divergent ridges/rifts,
convergent arcs and `n_hotspots` random hotspots, BFS with exponential decay (max
wins). `generate_volcano_positions(mesh, volcanism, elevations, threshold=0.3)` —
above threshold, above sea level, local maximum.

## `ruggedness.py`

**Ruggedness** ∈ [0, 1]: how rough the ground is *below* the globe's resolution
(0 floodplain-smooth, 1 alpine). The globe's relief doesn't read it; it is meant
to set the sub-grid relief the region, local and settlement tiers draw, and to
replace the slope / elevation-span proxies elsewhere
([plans/ruggedness.md](plans/ruggedness.md)). An index, not a physical quantity:
terms add, and relief will map from it logarithmically, so `base` shifts sub-grid
relief by a uniform factor. The ruggedness stage refines it in place
([ruggedness.md](ruggedness.md)).

`ruggedness_seed(mesh, params, seed, *, plate_types, boundary_data, rate_norm,
volcanism, elevation)` sums, per face:

| term | contribution | params |
|---|---|---|
| base | constant | `base` 0.3 |
| orogenic | convergent faces at `min(|normal| / rate_norm, 1)` (× `arc_factor` unless continent meets continent), max-decayed `exp(−hops / decay)` | `orogenic_weight` 0.5, `arc_factor` 0.7, `orogenic_decay_hops` 2 |
| rift | divergent faces on continental plates, likewise | `rift_weight` 0.25, `rift_decay_hops` 1.5 |
| volcanic | `volcanic_activity` | `volcanic_weight` 0.3 |
| uplift | `clip(face elevation / uplift_ref_km, 0, 1)` (pre-noise tectonic value) — high plateaus are dissected | `uplift_weight` 0.3, `uplift_ref_km` 2 |
| interior | **subtracted**: `1 − exp(−hops / decay)` from any plate boundary (shields, cratons) | `interior_weight` 0.2, `interior_decay_hops` 3 |
| noise | 2-octave simplex on centroids, ~[−1, 1] | `noise_weight` 0.3, `noise_frequency` 5 |

then `smoothing_passes` (3) of neighbour-mean diffusion at `smoothing_alpha`
(0.5), clipped, and rasterised as a smooth function of position with one
seam-averaging pass (`ruggedness_texture`), so a bilinear read is continuous
across face edges. Seeds 4/31: land p5/p50/p95 ≈ 0.14 / 0.34 / 0.67, corr with
face elevation ≈ 0.68; collision belts brightest, plate interiors darkest.

**Face relief** (`face_relief_km`, `face_relief(mesh)`): the face-scale relief in
km the index stands for, for the world-build consumers that think in km (land
cost and navigability, suitability, fertility, danger sources):
`FACE_RELIEF_SMOOTH_KM` 0.04 × (5.0 / 0.04) ** ρ, log-linear like the finer
tiers' spectrum and matched to the land quantiles of the per-face elevation span
it replaced (seeds 4/31/35: span ≈ exp(4.9 (ρ − 0.64)) km; corr ~0.5, so where
relief falls changes, not how much). A mesh without the field falls back to the span.

## `jaggedness.py`

**Jaggedness** ∈ [0, 1]: the terrain's *character* beside ruggedness's amount
(0 rolling, 0.5 neutral, 1 jagged). It sets the finer tiers' relief spectrum's
Hurst exponent ([region_fields.md](region_fields.md#detail-amplitude-the-relief-spectrum)):
jagged ground keeps its relief at fine scales, rolling ground puts it in the
long wavelengths. The ruggedness stage refines it with climate
([ruggedness.md](ruggedness.md#jaggedness)). Nothing in the world build reads it.

`jaggedness_seed(mesh, params, *, plate_types, boundary_data, rate_norm)` sums, per face:

| term | contribution | params |
|---|---|---|
| base | constant; above neutral since most land is plate interior | `base` 0.62 |
| young | convergent faces as ruggedness's orogenic term (`boundary_strengths`), max-decayed — active orogens are sharp | `young_weight` 0.4, `arc_factor` 0.7, `young_decay_hops` 2 |
| old | **subtracted**: `1 − exp(−hops / decay)` from any plate boundary — interiors are worn and rounded | `old_weight` 0.35, `old_decay_hops` 3 |

then diffused (`smoothing_passes` 2 at 0.5), clipped and rasterised like ruggedness.

## `display.py`

`_elevation_color`, `ELEVATION_RECIPE` (`relief=True`; rerun by the region view),
`elevation_legend_entries`, `OCEAN_COLOR`, `LAND_COLOR`, `BOUNDARY_COLORS`,
`_plate_palette`, and water-aware helpers reused by other stages
(`apply_water_masks`, `ocean_gradient_scheme`, `water_gradient_scheme`,
`ocean_categorical_scheme`, `is_water_meta`); `RUGGEDNESS_RECIPE`,
`ruggedness_color`, `ruggedness_legend_entries`, `JAGGEDNESS_RECIPE`, `jaggedness_color`,
`jaggedness_legend_entries` (registered by the ruggedness stage).

**Elevation** always; `--debug`: Plates, Plate Type, Speed, Boundary, Volcanic
Activity, **Volcanoes** overlay (unchecked), `[px: … km]` hover
(`refined=("elevation",)`).
