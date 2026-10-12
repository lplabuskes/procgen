# Ruggedness Stage

## Files
- `worldgen/ruggedness/builder.py` — `RuggednessRefineParams`, `RuggednessBuilder` (stage, display, persistence), `refine_ruggedness`, `river_mouth_strength`
- The field, its channels (`RUGGEDNESS_FACE`, `RUGGEDNESS`), the seed and the display recipe live in `worldgen/geology/` ([elevation.md](elevation.md#ruggednesspy)).

## Concept

**Ruggedness** ∈ [0, 1] is how rough the ground is below the globe's resolution
(0 floodplain-smooth, 1 alpine). The elevation stage seeds it from plates,
volcanism and height; this stage refines it **in place** once climate, rivers,
deltas and wetlands exist. The globe's relief doesn't read it. The region and
local tiers style their shores by it ([region_fields.md](region_fields.md#coast-style)),
and later phases of [plans/ruggedness.md](plans/ruggedness.md) drive the
sub-grid relief and the realm's terrain costs from it.

Runs after the wetland stage and before resources, danger and cost, so every
later consumer reads the final field. (Replaced the coast stage and its
`coast_smoothness`, whose tectonic, volcanic and noise terms moved to the seed.)

## `RuggednessBuilder`

Reads the seed `ruggedness` (required: raises without the elevation stage's),
`temperature_annual`, `"river_objects"` and `wetland_fraction` (each optional;
missing → contributes nothing). Adds:

| term | contribution | params |
|---|---|---|
| glacial | + ramp of annual mean T, 0 at warm limit → 1 at cold | `glacial_weight` 0.3, `glacial_warm_c` 4 °C, `glacial_cold_c` −6 °C |
| delta | − `sqrt(flow / largest flow)` at each river mouth face, spread `max · exp(−hops / decay)` | `delta_weight` 0.6, `delta_decay_hops` 1.5 |
| wetland | − the face's Wetland share | `wetland_weight` 0.4 |

The terms' sum is diffused (`smoothing_passes` 2 at `smoothing_alpha` 0.5; the
seed is diffused already), added to the seed, clipped to [0, 1] and rasterised
seam-free (`ruggedness_texture`).

Seeds 4/31/35: land p5/p50/p95 ≈ 0.06–0.10 / 0.27–0.32 / 0.66–0.74; coastal
land a little rougher (p50 0.33–0.41). That is ~0.2 smoother at the coast than
the old coast-style ruggedness (`1 − coast_smoothness`, seed 4 p5–p95 0.31–0.90),
so plains coasts draw smoother than they did.

## Display

`--debug`: **Ruggedness** scheme (`RUGGEDNESS_RECIPE`: pale green smooth →
olive → brown → slate alpine, water flat, hillshaded where relief) + hover
(`refined=("ruggedness",)`, so region and local views show their copy).

## Persistence

Texture saved with the mesh (`serialize` adds nothing). `load` restores the
per-face values from texture face means (close to, not exactly, the built ones).
