# Ruggedness Stage

## Files
- `worldgen/ruggedness/builder.py` — `RuggednessRefineParams`, `RuggednessBuilder` (stage, display, persistence), `refine_ruggedness`, `river_mouth_strength`; `JaggednessRefineParams` (`globe.ruggedness.jaggedness`), `refine_jaggedness`
- The field, its channels (`RUGGEDNESS_FACE`, `RUGGEDNESS`), the seed and the display recipe live in `worldgen/geology/` ([elevation.md](elevation.md#ruggednesspy)); likewise jaggedness's ([elevation.md](elevation.md#jaggednesspy)).

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
| floodplain | − per **texel** along the rivers (`floodplain_strength`): `weight · strength · exp(−d / width)` from each stream-network channel texel at ≥ `floodplain_min_flow`; strength = `min(√(flow / floodplain_full_flow), 1) × 1 / (1 + (slope / floodplain_slope)²)` (the globe's slope at the river: gorges stay rough); width = `floodplain_width_km · (flow / min flow) ** exponent` | `floodplain_weight` 0.35, `floodplain_min_flow` 500, `floodplain_full_flow` 20,000, `floodplain_width_km` 6, `floodplain_width_exponent` 0.4, `floodplain_slope` 0.001 |

The face terms' sum is diffused (`smoothing_passes` 2 at `smoothing_alpha` 0.5; the
seed is diffused already), added to the seed, clipped to [0, 1] and rasterised
seam-free (`ruggedness_texture`); the floodplain comes off the texture (it is far
narrower than a face: a few to ~40 km); the per-face values are then the
texture's face means.

Floodplain, seeds 4/31/35: ruggedness on river texels of flow 500–2k 0.24–0.28 →
0.19–0.23, 2k–10k 0.16–0.22 → 0.06–0.11, ≥ 10k 0.09–0.20 → ~0; the land mean
moves < 0.005; ~3% of land is touched at all. The hydrology → ruggedness
feedback of [plans/ruggedness.md](plans/ruggedness.md).

Seeds 4/31/35: land p5/p50/p95 ≈ 0.06–0.10 / 0.27–0.32 / 0.66–0.74; coastal
land a little rougher (p50 0.33–0.41). That is ~0.2 smoother at the coast than
the old coast-style ruggedness (`1 − coast_smoothness`, seed 4 p5–p95 0.31–0.90),
so plains coasts draw smoother than they did.

## Jaggedness

The stage also refines the **jaggedness** seed (terrain character: 0 rolling,
1 jagged; [elevation.md](elevation.md#jaggednesspy)) in place, after ruggedness,
with climate (`refine_jaggedness`). It needs `jaggedness` on the mesh (else it
is skipped) and reads `temperature_annual`, `precipitation_annual` and the
refined per-face ruggedness:

| term | contribution | params (`globe.ruggedness.jaggedness`) |
|---|---|---|
| aridity | + ramp of P / PET (PET as hydrology's, `RuggednessBuilder(pet=HydrologyParams)`): 0 at `aridity_humid`, full at `aridity_arid` — dry ground keeps sharp scarps and badlands, humid ground creeps round | `aridity_weight` 0.4, `aridity_humid` 1.0, `aridity_arid` 0.3 |
| glacial | ± the glacial ramp of annual T × (2 · ramp(ρ, `alpine_low`, `alpine_high`) − 1): cold rough ground goes jagged (arêtes, horns), cold smooth ground rolling (ice-scoured, hummocky lowlands) | `glacial_weight` 0.3, `glacial_warm_c` 4 °C, `glacial_cold_c` −6 °C, `alpine_low` 0.35, `alpine_high` 0.65 |

The terms are diffused (`smoothing_passes` 2 at 0.5), added to the seed,
clipped, rasterised; the per-face values are the texture's face means.
Seed 4's dry biomes (steppe, shrubland, desert) sit at P / PET ≈ 0.2–0.6, the
rest at ~1.5–2 and above (PET bottoms out in the cold, so boreal ratios run high).

## Display

`--debug`: **Ruggedness** scheme (`RUGGEDNESS_RECIPE`: pale green smooth →
olive → brown → slate alpine, water flat, hillshaded where relief) + hover
(`refined=("ruggedness",)`, so region and local views show their copy).

`--debug`: **Jaggedness** scheme (`JAGGEDNESS_RECIPE`: green rolling → sand
neutral → rust jagged, water flat, hillshaded) + hover (`refined=("jaggedness",)`).

## Persistence

Textures (`ruggedness`, `jaggedness`) saved with the mesh (`serialize` adds nothing); `load` restores both per-face fields as face means. The per-face values are
the texture's face means, both when built and when `load` restores them, so a
saved world reads exactly as built (a rerun of the cost stage on a loaded world
otherwise gave seed 4 428 cities instead of 275).
