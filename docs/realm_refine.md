# Realm Refine Stage

Part of `realmgen/`. [Back to realmgen overview](realmgen.md)

## Overview

The final realm stage. A **purely geometric** refinement pass that lifts realm
render geometry from the face level to the texel level — it recomputes *positions
and border curves only*, never population or polity structure (those already ran
on the face graph). All outputs are **mutated in place** on the existing world
features, so the existing Settlements / Trade Routes / Polity Borders overlays
render the refined geometry without any new display wiring.

Runs after `PolityStage`; reads the textures attached by worldgen (`ocean_mask`,
`lake_mask`) and the per-face `authority` channel written by `PolityStage`.

## Files

- `realmgen/refine/builder.py` — `RefineStage`, `RefineParams`
- `realmgen/refine/settlements.py` — settlement snapping + trade-route remapping
- `realmgen/refine/borders.py` — control-biased weighted-Voronoi border refinement

## `RefineParams`

- `river_threshold_factor` (default 1.5) — a candidate texel counts as "on a river"
  if it lies within `factor × texel_spacing` of a river polyline vertex.
- `border_samples` (default 5) — sample points per shared edge; the refined border
  emits `border_samples − 1` segments per face-pair.
- `border_control_floor` (hidden, default 1e-3) — floor on per-face authority so a
  zero-authority (unclaimed-adjacent) face still yields a finite Voronoi weight.

## Settlement snapping (`settlements.py`)

`refine_settlement_positions` moves each settlement from its face centroid to the
best texel **within the same face**, in priority order:

1. **Coast** — a land texel orthogonally adjacent to a water texel, preferring
   ocean-adjacent over lake-adjacent. (A fully-interior face has no coast texels,
   so the settlement stays inland.)
2. **River** — restrict to candidates within `river_threshold_factor × texel_spacing`
   of a river polyline vertex, when any candidate qualifies.
3. **Fertility** — highest per-face fertility among remaining candidates. Fertility
   is a per-face scalar, so within a single face this is uniform; the step is kept
   for correctness and ranks first only if candidates ever span faces.
4. **Nearest centroid** — deterministic final tiebreak; keeps the marker central
   and stable among equally-ranked candidates.

Water/coast tests use the per-texel `ocean_mask` / `lake_mask` textures (which are
continuous across faces, so a coastal land face genuinely contains the texel-level
coastline). Rivers are matched against a KD-tree of all river polyline vertices.

The function returns `{old_position_key: new_position}` for moved settlements.
`remap_routes` then rewrites any trade-route endpoint matching a moved settlement's
old centroid to its refined position — same connections, adjusted geometry. (Route
endpoints are face centroids bit-identical to the pre-refine settlement positions,
so the match is exact; deposit-side and intermediate endpoints are untouched.)

## Border refinement (`borders.py`)

`refine_polity_borders` replaces each straight shared-edge border segment with a
control-biased, texel-resolution polyline. Because the cost surface is face-level,
the boundary is refined *geometrically* as a multiplicatively-weighted Voronoi
boundary between the two faces' centroids:

- The ancestor-depth table mirrors `PolityStage._compute_polity_borders` (depth 0 =
  sovereign borders, depth d > 0 = internal subtree borders), so the refined
  `{depth: (M, 6)}` structure is a drop-in replacement.
- For each adjacent differing-polity face pair, the shared edge is sampled at
  `border_samples` points. At each sample the boundary is offset across the edge to
  the point where `dist / control` equalises between the two centroids
  (`_weighted_voronoi_offset` solves the quadratic `w_b²·|x−a|² = w_a²·|x−b|²`,
  picking the smallest-magnitude root, clamped to ±0.4·|a−b|). Control is the
  per-face `authority` value, so the higher-authority polity bulges into the
  weaker one. Equal control → the boundary tracks the geometric bisector near the
  edge. Each sampled point is renormalised onto the unit sphere.

## Outputs

Mutates in place: `settlements` (`.position`), `trade_routes` (endpoint geometry),
`polity_borders` (`{depth: (M, 6)}` refined segments). No new channels, features,
color schemes, or overlays.

## Validation

`realmgen/refine/test/test_refine.py` — stub-mesh snapping (coast/ocean-vs-lake/
river), route remapping, and weighted-Voronoi offset sign/clamp behaviour.
