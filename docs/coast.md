# Coast Stage

## Files
- `worldgen/coast/builder.py` — `CoastParams`, `CoastBuilder` (stage, display, persistence), `coast_smoothness`, `river_mouth_strength`
- `worldgen/coast/channels.py` — `COAST_SMOOTHNESS_FACE` (per face), `COAST_SMOOTHNESS` (texture)
- `worldgen/coast/display.py` — `coast_color`, `coast_legend_entries`, `COAST_RECIPE`

## Concept

Per-face **coast smoothness** ∈ [0, 1]: the *style* of shorelines the region and
local tiers draw — 1 long sweeping beaches/deltas, 0 crenellated, fjord-like. The
globe's own coast is unchanged (no shape to style at ~14 km texels); the field
only steers sub-grid detail one and two tiers down
([region_fields.md](region_fields.md#coast-style)). Defined everywhere, so lake
shores read it too. Runs after climate lake feedback, before resources; nothing
in the world build reads it.

## `CoastBuilder`

Reads `boundary_type`, `volcanic_activity`, `temperature_annual` metadata and
`"river_objects"` (each optional; missing → contributes nothing). Ruggedness =
sum of:

| term | ruggedness | params |
|---|---|---|
| base | constant | `base_ruggedness` 0.4 |
| noise | 2-octave simplex on centroids, ~[−1, 1] × weight | `noise_weight` 0.35, `noise_frequency` 5 (~1,000–2,000 km features) |
| tectonic | `exp(−hops / decay)` from convergent-boundary faces (BFS ≤ 12 hops) | `tectonic_weight` 0.4, `tectonic_decay_hops` 2 |
| volcanic | `volcanic_activity` | `volcanic_weight` 0.3 |
| glacial | ramp of annual mean T, 0 at warm limit → 1 at cold | `glacial_weight` 0.5, `glacial_warm_c` 4 °C, `glacial_cold_c` −6 °C |
| delta | **subtracted**: `sqrt(flow / largest flow)` at each river mouth face, spread `max · exp(−hops / decay)` | `delta_weight` 0.6, `delta_decay_hops` 1.5 |

Smoothness = `clip(1 − ruggedness, 0, 1)`, diffused (`smoothing_passes` 3 of
neighbour-mean blending at `smoothing_alpha` 0.5), rasterised as a smooth field
of position + one seam-averaging pass (`rasterize_face_values` /
`smooth_texture_seams`) — no steps at face edges, which the bilinear reads below
rely on.

Seed 4: coastal land 0.10–0.69 (p5–p95); tectonic contributes most, then noise
and volcanism; glacial confined to high latitudes.

Outputs: `coast_smoothness` texture `(N, H, W, 1)` float32 and per-face metadata.

**Reproducibility.** Noise seed from `ctx.rng("coast")`, keyed on
`BuildContext.seed`, not `World.seed` (as danger).

## Display

`--debug`: **Coast Smoothness** scheme (rugged slate → smooth sand, water flat,
hillshaded where relief) + hover. Has a `TextureRecipe`, so the region view
redraws it from its own copy.

## Persistence

Texture saved with the mesh (`serialize` adds nothing). `load` restores per-face
metadata from texture face means (close to, not exactly, the built values).
Worlds without the field → finer tiers draw unstyled shores.
