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
- `realmgen/refine/borders.py` — continuous border polylines traced on the texel polity classification
- `realmgen/refine/fill.py` — texel polity classification (`compute_texel_labels`) + authority-weighted fill texture

## `RefineParams`

- `river_threshold_factor` (default 1.5) — a candidate texel counts as "on a river"
  if it lies within `factor × texel_spacing` of a river polyline vertex.
- `border_smoothing_iterations` (default 2) — Chaikin corner-cutting passes applied
  to the chained border polylines. Higher = rounder, denser lines.
- `border_control_floor` (hidden, default 1e-3) — floor on per-face authority so a
  zero-authority (unclaimed-adjacent) face still yields a finite Voronoi weight.
- `boundary_warp_amplitude` (default 0.012) — magnitude (unit-sphere units, ~⅓ of a
  face) of the fBm domain warp applied to the texel positions fed to the polity
  Voronoi. `0` disables the warp (boundaries fall on the bare face-centroid
  bisectors — the polygonal staircase). The warp is seeded from `World.seed`.
- `boundary_warp_frequency` (default 45) — base frequency of the warp noise; higher =
  finer, busier wiggles. Pushed too high it aliases against the texel grid.
- `boundary_warp_octaves` (hidden, default 3) — fBm octave count for the warp.

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

## Texel polity classification (`fill.py`)

`compute_texel_labels` is the **single source of truth** for both the fill colours
and the border lines. It returns an `(N, H, W)` int32 label grid where each texel
is a polity id (`≥ 0`), `UNCLAIMED` (-1) for claimed-by-nobody land, or `WATER`
(-2) for ocean/lake:

- Each face starts uniform: claimed land → its polity id; everything else → -1.
- **Border faces** (claimed land with a differing claimed-land neighbour) have
  their texels reclassified by an authority-weighted (multiplicatively-weighted)
  Voronoi over the face's own centroid and its claimed-land neighbours
  (`argmin |x−c|² / control²`), so the stronger polity bulges into the weaker one.
  Interior claimed-land faces stay uniform.
- **Boundary erosion (domain warp).** The texel positions `x` fed to that Voronoi
  are first displaced by a coherent fBm-simplex vector field (`domain_warp`,
  `boundary_warp_*`). Without it the boundary is the bare perpendicular bisector
  between two face centroids — a straight line / circular arc that hugs the
  hexagon edges, so the map reads as polygonal. The warp wiggles the boundary
  organically. It is a pure function of 3D position, so coincident shared-edge
  texels get the *same* displacement (it never introduces seams of its own);
  residual cross-face label disagreements are still reconciled below. Only the
  *classification* is warped — the border tracer still uses the true texel
  positions, so the lines land exactly on the warped fill's region edges.
- **Orphan land in water faces.** A face counts as water at *face* level once
  **more than half** its texels are water (`is_ocean_per_face`), so a coastal
  "water" face can still hold a large minority of genuine land texels. Those
  faces, when they have a claimed-land neighbour, are run through the same
  weighted Voronoi over their claimed neighbours — so their land texels take the
  winning polity instead of being painted ocean. This closes the coastal gap
  where the face-level land map disagreed with the texel map.
- Ocean and lake texels (per-texel `ocean_mask`/`lake_mask`) override any land
  label with `WATER`.
- **Shared-edge reconciliation.** `face_texture_positions` makes shared-edge
  texels coincident in 3D across adjacent faces; `compute_texel_labels` uses
  `mesh.boundary_pixel_pairs` to force those coincident texels to agree (land-vs-
  land disagreements only). This is what lets the border tracer produce lines that
  are continuous across face seams.

`build_polity_fill_texture` colourises the labels into the `(N, H, W, 4)` float16
RGBA "Polities" fill texture: polity ids → the exact `make_polity_tree_color`
palette, negative labels → unclaimed grey, then ocean/lake texels are repainted
their own blues (matching the elevation view's `apply_water_masks`).

## Border refinement (`borders.py`)

`PolityStage` emits borders as a soup of independent hexagon edges — angular and,
read as a whole, a staircase, and they stop at hexagon corners rather than the
texel coastline. `smooth_polity_borders` instead traces the borders **directly on
the texel label grid**, so the lines are exactly the edges of the fill's regions:

- For each hierarchy depth (depth 0 = sovereign borders, *including* polity vs.
  unclaimed land; depth d > 0 = internal subtree borders), an edge between two
  adjacent texels is a border at the **shallowest** depth where their depth-d
  ancestors differ (`_edge_border_depth`, vectorised over an ancestor LUT). The
  resulting `{depth: (M, 6)}` structure is a drop-in replacement. Coastlines
  (land vs. `WATER`) are *not* borders, so a polity-vs-polity line simply
  terminates at the texel waterline.
- Within each 2×2 texel cell the crossings around the four cell edges are joined
  marching-squares style: exactly two crossings → a straight link between their
  edge midpoints; otherwise each crossing links to the cell centre (handles 3+
  region junctions and coastal terminations). Even-parity is broken only at the
  coast, which is the desired dangling endpoint.
- **Saddle resolution.** A 2×2 *checkerboard* cell (all four edges are borders but
  only two regions are present, `TL==BR ≠ TR==BL`) would, under the centre-hub
  rule, route four spokes through the centre — a self-crossing X that Chaikin
  rounds into a visible knot/loop. Such cells are instead split into the two
  non-crossing links that keep the equal-label diagonal connected. Genuine 3-/4-
  region junctions still use the centre hub.
- **Continuity across seams is free**: a crossing midpoint on a shared mesh edge
  is the average of two coincident edge texels, i.e. the *same 3D point in both
  faces*. Reconciled edge labels (above) make both faces emit that crossing, so
  the per-face pieces chain through the shared point. Chaining (`_chain_edges`,
  keyed on quantised 3D positions) therefore yields lines continuous across faces;
  on seed 4 ~97% of endpoints are degree-2 (interior chain), the rest coastal
  terminations and junction hubs.
- Each chain is **Chaikin-smoothed** (`border_smoothing_iterations` passes), with
  open-chain endpoints held fixed so chains meeting at a junction stay joined, then
  renormalised onto the unit sphere — rounding off the texel staircase.

`RefineStage` computes the labels once and feeds them to both the fill and the
border tracer, then stores the fill via `mesh.set_face_textures("polity_fill", …)`,
and
`PolityStage.register_display` attaches it to the "Polities" scheme (via
`add_scheme(..., texture="polity_fill")`) when present — so it renders per-pixel
and only when that mode is active.  `RealmPipeline.apply_display` then calls
`window.associate_scheme_texture` for every scheme carrying a texture (it merges
into worldgen's schemes rather than going through `apply_registry`, so it has to
do this wiring itself); without that the Polities mode falls back to flat
per-face colours.

## Outputs

Mutates in place: `settlements` (`.position`), `trade_routes` (endpoint geometry),
`polity_borders` (`{depth: (M, 6)}` smoothed segments). Adds one face texture,
`polity_fill`, consumed by the "Polities" scheme. No new features or overlays.

## Validation

`realmgen/refine/test/test_refine.py` — stub-mesh snapping (coast/ocean-vs-lake/
river), route remapping, Chaikin corner-cutting + edge chaining (open/closed),
texel border tracing (two-polity split, coastline ignored, saddle resolved
without a centre X), domain-warp identity/continuity, and fill-texture
shape/coastline/orphan-land behaviour.
