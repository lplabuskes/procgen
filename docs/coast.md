# Coast Stage

## Files
- `worldgen/coast/builder.py` — `CoastParams`, `CoastBuilder` (stage orchestration, display, persistence), `coast_smoothness`, `river_mouth_strength`
- `worldgen/coast/channels.py` — `COAST_SMOOTHNESS_FACE` (per-face metadata) and `COAST_SMOOTHNESS` (the `coast_smoothness` face texture)
- `worldgen/coast/display.py` — `coast_color`, `coast_legend_entries`, `COAST_RECIPE`

## Concept

A per-face **coast smoothness** in [0, 1]: the *style* of the shorelines the
region and local tiers draw. 1 is long, sweeping beaches and deltas; 0 is
crenellated, rugged, fjord-like shores. The globe's own coast is not changed:
at ~14 km texels there is no shoreline shape to style. The field only steers
the sub-grid detail one and two tiers down (see
[region_fields.md](region_fields.md#coast-style)). It is defined everywhere,
not only on coastal faces, so lake shores read it too.

It runs after the climate lake-feedback pass and before resources. Nothing in
the world build reads it.

## `CoastBuilder`

`build(ctx)` reads `boundary_type`, `volcanic_activity` and
`temperature_annual` metadata and the `"river_objects"` feature (each optional;
a missing input contributes nothing), and calls `coast_smoothness`.
Ruggedness is a sum of terms:

| term | ruggedness | params |
|---|---|---|
| base | `base_ruggedness` everywhere | 0.4 |
| noise | 2-octave simplex on the face centroids, ~[−1, 1] × weight | `noise_weight` 0.35, `noise_frequency` 5 (features ~1,000–2,000 km) |
| tectonic | `exp(−hops / decay)` from convergent-boundary faces (BFS, capped at 12 hops) | `tectonic_weight` 0.4, `tectonic_decay_hops` 2 |
| volcanic | `volcanic_activity` | `volcanic_weight` 0.3 |
| glacial | linear ramp of annual mean temperature, 0 at the warm limit → 1 at the cold one | `glacial_weight` 0.5, `glacial_warm_c` 4 °C, `glacial_cold_c` −6 °C |
| delta | **subtracted**: `sqrt(flow / largest flow)` at each river's mouth face, spread as `max · exp(−hops / decay)` | `delta_weight` 0.6, `delta_decay_hops` 1.5 |

Smoothness is `clip(1 − ruggedness, 0, 1)`, diffused over the face graph
(`smoothing_passes` 3 of neighbour-mean blending at `smoothing_alpha` 0.5),
and rasterised as a smooth field of position plus one seam-averaging pass
(`rasterize_face_values` / `smooth_texture_seams`, as for humidity). The
texture therefore has no steps at face edges, which the bilinear reads one
tier down rely on.

On seed 4, coastal land runs from 0.10 to 0.69 (5th–95th percentile). The
tectonic term is the largest contributor on coasts, then noise and volcanism;
the glacial term is confined to high latitudes.

Outputs: the `coast_smoothness` face texture `(N, H, W, 1)` float32, and
per-face `coast_smoothness` metadata.

**Reproducibility.** The noise seed comes from `ctx.rng("coast")`. As with
danger, that is keyed on `BuildContext.seed`, not `World.seed`.

## Display

Debug only: the **Coast Smoothness** scheme (rugged slate → smooth sand, water
flat, hillshaded where there is sub-texel relief) and a hover line. The scheme
has a `TextureRecipe`, so the region view redraws it from the region's own copy
of the field.

## Persistence

The texture is saved with the mesh, so `serialize` contributes nothing.
`load` restores the per-face metadata from the texture's face means, which are
close to, but not exactly, the values the build wrote. Worlds saved before this
stage existed have no field; the finer tiers then draw unstyled shores.
