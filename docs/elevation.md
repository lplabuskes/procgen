# Elevation & Tectonics Stage

## Files
- `worldgen/geology/elevation_builder.py` — stage + params
- `worldgen/geology/elevation.py` — elevation model, erosion, rasterisation, noise, seam smoothing, ocean mask
- `worldgen/geology/tectonics.py` — plates
- `worldgen/geology/volcanism.py` — activity + volcano peaks
- `worldgen/geology/ruggedness.py` — the ruggedness seed (`RuggednessParams`, `ruggedness_seed`, `ruggedness_texture`; shared `decayed_max` / `diffuse`)
- `worldgen/geology/channels.py` — `IS_OCEAN`, `ELEVATION_FACE`, `PLATE_TYPE`, `BOUNDARY_TYPE`, `VOLCANIC_ACTIVITY`, `ELEVATION`, `OCEAN_MASK`, `RUGGEDNESS_FACE`, `RUGGEDNESS`
- `worldgen/geology/display.py`

## `elevation_builder.py`

**`ElevationParams`**:
- `plates` — `n_plates`, `plate_compactness`, `relax_max_passes`, `oceanic_fraction`
- `plate_motion` — `drift_scale`, `spin_scale`
- `elevation` — `elevation_rate_norm`, `elevation_decay_rate`, `elevation_max_hops`, `elevation_tex_size` (world texture size, 16)
- `noise` — `noise_amplitude`, `noise_coast_sigma`
- `face_erosion` — hydraulic/thermal knobs (hidden)
- `volcanism` — `n_hotspots`, `volcanic_decay_rate`, `volcanic_max_hops`, `volcanic_elevation_scale`, `volcanic_elevation_scale_oceanic`
- `ruggedness` — `base` (the master knob) and the seed's term weights ([below](#ruggednesspy))

Mesh params are not here (mesh is pre-built); river params are in
`HydrologyParams`. The seed goes to `ElevationBuilder(params, *, seed)`; each draw
takes a name-keyed sub-seed (`_seed("plates")`, …) so adding a draw never shifts
others.

**`ElevationBuilder.build(ctx)`** decorates the dual mesh:

plates → relax → types + velocities → boundary classification → face elevations
→ volcanism → face erosion → + volcanism × `volcanic_elevation_scale[_oceanic]` →
ruggedness seed → `rasterize_face_values` → fBm simplex noise (`compute_noise_elevation_deltas`,
band-limited to the texel grid, × `exp(−|elevation| / noise_coast_sigma)`, so
strongest near sea level, roughening coasts) → seam smoothing → ocean mask →
`world.antimeridian` (`find_ideal_antimeridian`).

Attributes: `mesh`, `plate_ids`, `plate_omegas`, `face_velocities`,
`boundary_types`, `boundary_data`, `elevations` (face means of final texture),
`elevation_textures`, `ocean_mask`, `volcanism`, `volcano_positions`, `max_speed`.

- Textures: `"elevation"` (`(N, H, W, 1)` float32, km), `"ocean_mask"` (`(N, H, W, 1)` uint8), `"ruggedness"` (`(N, H, W, 1)` float32 in [0, 1]).
- Per face: `plate_id`, `plate_type`, `boundary_type`, `surface_speed`, `volcanic_activity`, `elevation` (tectonic + erosion + volcanic value *before noise* — read by resource placement), `is_ocean` (> half texels ocean).
- Per face: `ruggedness` (seed index in [0, 1]; on load, the texture's face means).
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
(elevation, temperature, humidity, fertility, forest, coast smoothness). Each
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
relief by a uniform factor. Later stages refine it in place.

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

## `display.py`

`_elevation_color`, `ELEVATION_RECIPE` (`relief=True`; rerun by the region view),
`elevation_legend_entries`, `OCEAN_COLOR`, `LAND_COLOR`, `BOUNDARY_COLORS`,
`_plate_palette`, and water-aware helpers reused by other stages
(`apply_water_masks`, `ocean_gradient_scheme`, `water_gradient_scheme`,
`ocean_categorical_scheme`, `is_water_meta`).

**Elevation** always; `--debug`: Plates, Plate Type, Speed, Boundary, Volcanic
Activity, **Ruggedness** (`RUGGEDNESS_RECIPE`, water flat, hillshaded; + hover), **Volcanoes** overlay (unchecked), `[px: … km]` hover
(`refined=("elevation",)`).
