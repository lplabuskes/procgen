# Terrain Ruggedness — Plan

Maps are flat, worst at the settlement tier. The cause is the detail amplitude
rule every tier shares (`region/upsample.py::refine_terrain`):

    amplitude = relief_factor × feature wavelength × effective slope   (clamped)

where the slope is the **parent's**. The chain starts at the globe, which is
smooth at ~14 km (mountains ~0.007 km/km), and each tier adds its own clamps:

| tier | feature wavelength | slope_reference | slope_floor | max_relief_km |
|---|---|---|---|---|
| region | ~28 km | 0.005 | 0.10 | 1.0 |
| local | ~9 km | 0.03 | 0.03 | 0.6 |
| settlement | ~0.73 km | 0.1 | 0.03 | **0.03** |

So a settlement map gets ±2 m of detail on flat ground and ±30 m at most. Raising
the clamps only goes so far: a 14 km slope doesn't know what the ground is like
below it. A dissected range and a high plateau can have the same coarse slope.
Each tier also restarts a normalised fBm whose amplitude grows linearly with
wavelength (an effective Hurst exponent H = 1), which is as smooth at fine scales
as fBm gets.

**Ruggedness** is a new world field that says how rough the ground is below the
globe's resolution. It sets the sub-grid relief spectrum directly, and it replaces
the places that currently use slope or elevation range as a stand-in for it.

## Decisions (2026-10-11)

1. **One channel now, with room for a second.** The first channel is
   ruggedness (relief amplitude). The second, to come later, is character:
   jagged vs rolling, which is the spectrum's Hurst exponent `H`. For now `H` is a
   single param, read through one function so a per-texel channel can replace it
   without touching the callers.
2. **Plate tectonics seeds it, later stages refine it.** The elevation stage
   writes the seed. Later stages edit it in place, the same way the delta stage
   already rewrites `ELEVATION`.
3. **Sub-grid only.** The global elevation texture, and so climate, hydrology,
   biomes and every calibration, doesn't change.
4. **Slope users reviewed one by one** (table below). Most wanted ruggedness.
   Danger and buildability use both.
5. **Ruggedness drives coast style.** `coast_smoothness` goes; the coast stage's
   terms move into the ruggedness field.
6. **Texture, not per-face.** Rasterised with `rasterize_face_values`
   (position-only), so the amplitude doesn't step at face edges. It is carried
   down the tiers as a bilinear copy, the way `coast_smoothness` is now: not
   conserved, no noise of its own. Elevation stays the only noise source.
7. **Realm consumers come in a later phase**, after the terrain looks right.
8. **Settlements build on hills.** That's the point of the change. The footprint
   and anchor knobs get retuned so hill towns happen, not just tolerated.
9. **One basic knob** for ruggedness. Everything else is advanced.
10. **Rivers and lakes on rough relief are deferred** until we've seen the
    results. The hope is that hydrology refining ruggedness (smooth valley floors
    and floodplains) does most of the work.
11. **The basic knob acts in the world build** (2026-10-11): it scales the field
    itself, so the view tiers and the realm consumers read the same thing.
    Changing it needs a rebuild.
12. **The stage is renamed** `worldgen/coast/` → `worldgen/ruggedness/` (and
    `globe.coast` → `globe.ruggedness`) in Phase 2.

## The field

`ruggedness` ∈ [0, 1], a texture `(N, H, W, 1)` float32 plus a per-face mean. It
is an index, not a physical quantity, so terms combine and refine additively the
way the coast stage's do now. It maps to relief through a log interpolation,
because relief spans orders of magnitude (metres on a floodplain, hundreds of
metres in a range):

    relief_ref(ρ) = relief_flat × (relief_rugged / relief_flat) ** ρ     # km at λ_ref

### Seed (elevation stage)

From what `ElevationBuilder` already has. Each term is clipped and combined like
the coast stage's, with low-frequency noise so the field isn't just a function of
elevation:

| term | source |
|---|---|
| base | constant |
| orogenic | `exp(−hops / decay)` from convergent boundaries (from the coast stage) |
| rift | the same from divergent continental boundaries (weaker) |
| volcanic | `volcanic_activity` (from the coast stage) |
| interior | plate-interior distance lowers it (cratons, shields) |
| noise | 2-octave simplex on face centroids (from the coast stage) |

Erosion output and height above sea level are candidate terms; we'll decide from
the debug layer.

### Refinement (later stages)

The coast stage becomes the **ruggedness refinement stage**. It runs at the same
point (after wetland, before resources, danger and cost, so every world-build
consumer sees the final field):

| term | source |
|---|---|
| glacial | + ramp on annual mean temperature (from the coast stage) |
| delta | − at river mouths by `sqrt(flow / max)` (from the coast stage) |
| wetland | − on the Wetland biome (wetlands are flat by definition) |
| floodplain / valley floor | − along big rivers. **Deferred** (Decision 10) |

Diffused over the face graph, then rasterised seam-free, as the coast stage does
now.

## The spectrum (all tiers)

`refine_terrain` stops normalising the fBm. Octave `k`, at wavelength `λ_k`, gets
an absolute amplitude:

    A_k = relief_ref(ρ) × (λ_k / λ_ref) ** H        (each octave capped)

with `ρ` read bilinearly from the parent's `ruggedness`. Each tier contributes
only its own band (parent Nyquist to own Nyquist, as now), so the relief at any
scale is the same whichever tier draws it. A settlement map ends up with every
band from the globe down. The per-tier `relief_factor` / `slope_reference` /
`slope_floor` / `max_relief_km` knobs go away, replaced by one set of spectrum
params shared by all tiers (`relief_flat`, `relief_rugged`, `λ_ref` = 1 km, `H`,
a per-octave cap).

What stays:
- shallow-water damping (lake beds),
- the coast blend (persistence and wander near water), now lerped on ruggedness
  instead of smoothness,
- conservation (unchanged, since the noise is still added before
  `conserve_smooth`).

Targets for calibration: local relief over ~1 km windows of a few tens of metres
on plains, ~100–300 m in hills, several hundred metres and more in mountains.

## Slope and proxy users, case by case

**→ ruggedness** (were stand-ins):

| where | now | becomes |
|---|---|---|
| `region/upsample.py` detail amplitude | parent slope | the spectrum |
| `region/upsample.py` coast wander / persistence | slope, `coast_smoothness` | ruggedness |
| `worldgen/coast/` | its own ruggedness → smoothness | the refinement stage |
| `worldgen/cost/surfaces.py` land cost | `face_elevation_range` | ruggedness (Phase 5) |
| `worldgen/cost/surfaces.py` navigability | `face_elevation_range` | ruggedness (Phase 5) |
| `realmgen/scoring/suitability.py` penalty | `face_elevation_range` | ruggedness (Phase 5) |
| `worldgen/danger/builder.py` | `face_elevation_range` | ruggedness (Phase 5) |
| polity seeding, forts | `land_base_cost` | follow the cost change automatically |
| `settlegen/profile.py` stone share | the frame's 2–98% relief | ruggedness |

**→ slope + ruggedness** (both mean something):

| where | why |
|---|---|
| `region/resources.py` danger terrain nudge | rough country hides danger even where the texel itself is flat |
| ~~`settlegen/footprint.py`, `localmap/villages.py`, `localmap/farmland.py`, `settlegen/hinterland.py`~~ | **slope only** (user, 2026-10-11): with real relief the local and settlement slopes see the hills; ruggedness on top would penalise flat valley floors in rugged country. Ruggedness goes where the measuring tier is coarser than the relief (Phase 5, region danger nudge). |

**→ slope stays** (a real gradient):
hydrology routing, orographic precipitation, wetland flatness (global and
riparian), refined fertility slope penalty, settlement arterial slope cost,
anchor flatness and citadels, road channel climb, small-lake breach
(`region/lakes.py`, revisit after Phase 3).

## Phases

### 0. Baseline harness
`scripts/measure_relief.py` reports, per tier (region, local, settlement), on
seeds 4/31/35:
- local relief in ~1 km windows, by ruggedness class (and by biome until the
  field exists), as p10/p50/p90,
- slope percentiles,
- conservation max.

It records today's numbers before anything changes and gives the calibration
targets above. Results go to `scripts/results/relief_results.json`.

### 1. Seed field
- `RUGGEDNESS_FACE` / `RUGGEDNESS` channels in `worldgen/geology/channels.py`.
- Seed terms in `ElevationBuilder`, with params under `ElevationParams.ruggedness`.
- A `--debug` Ruggedness layer with a texture recipe, so the region view draws
  its own copy.
- Nothing reads it yet.

### 2. Refinement stage, coast smoothness retired
- The coast stage becomes the refinement stage: glacial, delta and wetland terms,
  diffusion, rasterisation.
- `coast_smoothness` and its channels, display and persistence are deleted
  (backwards compatibility isn't needed).
- The `globe.coast.base_ruggedness` basic param is replaced by the single basic
  knob.

### 3. Spectrum at every tier
- `refine_terrain` reads `ruggedness` from the parent and builds the absolute
  spectrum.
- Each tier copies `ruggedness` down bilinearly.
- The per-tier slope knobs are deleted.
- Calibrate with `measure_relief` plus `debug_region` / `debug_local` /
  `debug_settlement`. Conservation must stay at float noise.
- Watch, but don't fix yet (Decision 10): small-lake counts, cut-off sea
  channels, rivers climbing out of their valleys at the local and settlement
  tiers, wetland share.

### 4. Settlement consumers
- Footprint, anchors and profile read ruggedness, per the table above.
- Retune `slope_free` / `slope_weight` / `buildable_max_slope` so towns climb
  hills.
- `measure_settlements` footprint error and pieces will move; record the new
  baseline.

### 5. Realm consumers
- Replace `face_elevation_range` in cost, navigability, suitability and danger.
- Retune `ruggedness_cost_per_km`, `penalty_scale_km`, `navigable_ruggedness_km`
  and danger weights.
- Re-run `measure_population_absolute`, `measure_polities` and
  `measure_borders`, keeping the calibrated bands (world 300–400 M; polity
  priorities largest > count > corr).

### Deferred
- Hydrology refining ruggedness (valley floors, floodplains), and rivers/lakes on
  rough relief.
- Second channel: per-texel `H` (jagged vs rolling).
- Wetland flatness reading the seed ruggedness.

## Status

Planned 2026-10-11.

**Phase 0 done (2026-10-11).** `scripts/measure_relief.py`. Baseline, seeds
4/31/35, median over seeds of each class's window relief (m, p10 / p50 / p90),
by span tercile:

| tier (window) | flat | hilly | rugged |
|---|---|---|---|
| region (3 texels, ~13 km) | 6 / 12 / 20 | 10 / 18 / 31 | 17 / 35 / 71 |
| local (~1 km) | 1.6 / 3.2 / 5.9 | 2.0 / 4.2 / 8.0 | 4.2 / 9.6 / 20 |
| settlement (1 km) | 5.0 / 7.2 / 10 | 5.1 / 7.8 / 13 | 6.9 / 11 / 19 |
| settlement (250 m) | 1.9 / 3.0 / 4.8 | 1.9 / 3.1 / 5.7 | 2.4 / 4.6 / 8.0 |

Rugged ground gets ~10 m per km where Earth's mountains have hundreds. The
settlement frames sit on towns, so they lean flat within each class.

**Phase 1 done (2026-10-11).** Seed in `worldgen/geology/ruggedness.py`, written
by `ElevationBuilder` (`RUGGEDNESS_FACE` / `RUGGEDNESS`), `--debug` Ruggedness
layer. Added an **uplift** term (height above sea level) beyond the planned
ones: without it, high plateaus far from a boundary came out smooth. Erosion
stays a candidate. Nothing reads the field yet. The basic knob
(`globe.elevation.ruggedness.base`) joins `BASIC_PARAMS` in Phase 2, replacing
`globe.coast.base_ruggedness`, so there is never more than one basic ruggedness
knob, and never one that does nothing.

**Phase 2 done (2026-10-11).** `worldgen/coast/` → `worldgen/ruggedness/`
(`RuggednessBuilder`, `globe.ruggedness` = `RuggednessRefineParams`): glacial,
river-mouth and wetland terms added to the seed in place; `coast_smoothness`
and its channels, display and tests are gone. The region tier's coast style reads
`ruggedness` (rugged 1 = old smoothness 0) and copies it down every tier, ready
for Phase 3. The Ruggedness debug layer moved to the stage, with a refined hover
on region and local views. `globe.elevation.ruggedness.base` replaced
`globe.coast.base_ruggedness` in `BASIC_PARAMS`. Coastal land is ~0.2 smoother
than the old coast field (seed 4 p5–p95 0.14–0.74 vs 0.31–0.90): plains coasts
draw smoother. Left as is for review; `coast_*_smooth` params absorb it if wanted.

**Phase 3 done (2026-10-11).** `refine_terrain` builds the detail from the
spectrum (`spectrum_amplitudes`: `relief_flat_km` 2.5 m → `relief_rugged_km`
250 m at 1 km, `hurst` 0.75, `octave_cap_km` 1, `relief_factor` kept as an
advanced overall multiplier), the octaves summed un-normalised. The params live
on `RegionUpsampleParams` and the local and settlement tiers share them.
`slope_reference` / `slope_floor` / `max_relief_km` / `persistence` are gone at
every tier, and `relief_factor` left `BASIC_PARAMS`. `hurst_exponent` is the
per-texel handle for the character channel. Medians of 1 km window relief
(seeds 4/31/35):

| ruggedness | local | settlement | (baseline, by span tercile: local 3–10 m, settlement 7–11 m) |
|---|---|---|---|
| < 0.2 | 13 m | 21 m | |
| 0.2–0.4 | 28 m | 47 m | |
| 0.4–0.6 | 64 m | 100 m | |

Regions over ~13 km at ρ ≥ 0.6: 465 m (p90 910 m). Conservation still at float
noise on every tier.

Two things broke and were fixed in this phase rather than deferred:
- **Small lakes flooded the plains** (seed 4's densest hex: 22% lake on the
  region, 26% on its local map): noise pits now run tens of metres deep, beyond
  the slope × length breach. Added `lake_breach_relief` (0.3 × the tier's summed
  octave amplitudes, `detail_relief`): region 29 back to 0.02% (baseline 0.04%),
  cold rugged region 49 keeps ~2.5% (glacial lake country).
- **Shore reshaping drew straight cliffs** along parent-texel edges (local and
  settlement tiers): it put each parent mean back by scaling the parent texel's
  ground, which steps at the parent's edge once relief is tall.
  `reshape_to_target(taper=...)` now weights each texel's share by
  `anchor_edge_taper` (0 along parent edges), so the absorption is continuous.

Still open (Decision 10, watching): rivers on the rougher relief (none seen
climbing out of valleys in the renders checked), small lakes on local maps (seed 4, the
same 40 maps as docs/local.md: 319 local-only lakes, 0.19% of land, vs 577 /
0.27% before), regional texture reads as fBm rather than drainage (the character
channel's job), `connect_ocean` channels (no visible artefacts found).

**Phase 4 (2026-10-11).** `measure_settlements.py` reports the hills (site
ruggedness, frame and footprint relief and slope, steep share, `hill_ratio` =
footprint slope ÷ the land's, stone share). On the new terrain (seeds 4/31/35,
6 cities / 3 towns / 3 villages each), footprint |error| 3.8% / 2.5% / 5.7%
(baseline 3.1 / 2.2 / 5.4), pieces and sparse share unchanged. Footprints
already climb: `hill_ratio` ~0.96 overall, 0.76–1.0 on the ρ ≈ 0.4 sites with
10–40% of their ground steeper than 0.15. `slope_free` / `slope_weight` /
`buildable_max_slope` left as they are. The sampled settlements sit on smooth
ground (mean ρ ~0.2 vs land median 0.3): that's their placement in the realm
stage (Phase 5's suitability penalty), not the footprint. The stone share reads
`ruggedness` at the settlement (`stone_ruggedness_low` 0.25 → `_high` 0.65;
replaces `rugged_relief_m`, which the real relief saturated everywhere).
Decided with the user: settlement- and local-tier siting (footprint, buildable
ground, villages, farmland, hinterland) stays slope-only. **Phase 4 done.**

**Phase 5 done (2026-10-11).** `worldgen.geology.ruggedness.face_relief_km` maps
ruggedness to a face-scale relief in km (0.04 → 5.0 km, log-linear, matched to
the old per-face elevation span's land quantiles), so every km-denominated knob
kept its meaning. It replaces `face_elevation_range` in land cost and river
navigability (`worldgen/cost`), suitability (`realmgen/scoring`), the fertility
terrain factor (`worldgen/resources`) and danger-source placement
(`worldgen/danger`); polity seeding and forts follow through land cost.
`face_elevation_range` remains only as `face_relief`'s fallback for meshes
without the field. The region danger nudge is left on slope + forest: it is
renormalised per global texel, and ruggedness is nearly constant within one.

Fixed on the way: per-face ruggedness is now the texture's face means at build
too (it was the raw values at build and the face means after load, so a saved
world disagreed with its own build).

Realm shift (baseline → ruggedness, same code otherwise): cities 395 / 360 / 377 →
273 / 319 / 287, sovereigns (6 seeds) 53.8 → 39.2 mean, mean largest domain
23.9 → 29.2% at the old `sea_range_factor`. An ablation (rebuilding with the old
proxy in one consumer at a time; all four old reproduces the baseline exactly)
put the polity change on land cost and the city count on suitability; fertility
and danger barely matter. Retuned: `sea_range_factor` 1.4 → 1.2 (mean largest
23.3%, corr −0.49), `rural_density_max` 14 → 13 (world 343 / 401 / 354 M). Seed
31's top city is 572k, over the 50k–500k band, as seed 4's 588k was before.
Borders: river share 2.1 → 2.4%, ridge share 3.1 → 2.8%, sovereign exclaves
3.0 → 5.7 per seed (mostly land-cut). Settlements sit at about the land's mean
ruggedness, as before (cities ρ 0.34 / 0.36 / 0.30 vs land 0.35 / 0.35 / 0.31).

Noticed, not fixed (pre-existing, outside this work): re-running the cost stage
on a *loaded* world doesn't reproduce its build (seed 4: 428 vs 273 cities), so
some cost input restores differently on load.

**All five phases done.** Still open: rivers and lakes against the rough relief
beyond the breach fix (Decision 10), hydrology refining ruggedness, and the
character (Hurst) channel.
