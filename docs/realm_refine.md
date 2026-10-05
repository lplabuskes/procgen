# Realm Refine Stage

Part of `realmgen/`. [Back to realmgen overview](realmgen.md)

## Overview

Fifth realm stage (after `PolityStage`, before `TownStage`). **Purely
geometric**: lifts realm render geometry from face level to texel level —
positions and border curves only, never population or polity structure. Outputs
are **mutated in place** on existing features, so the Settlements / Trade Routes
/ Polity Borders overlays draw refined geometry with no new wiring.

Reads textures `ocean_mask`, `lake_mask`, `fertility`; feature `rivers`; per-face
`polity_id`, `is_water`, `authority`; features `settlements`, `trade_routes`,
`polities`, `polity_borders`. No-op without settlements and an ocean mask. Warp
and city jitter seeded from `World.seed`.

## Files

- `realmgen/refine/builder.py` — `RefineStage`, `RefineParams`
- `realmgen/refine/sites.py` — `SiteField`: per-texel site features (coast, river, confluence, local fertility, face ownership), shared with the town stage
- `realmgen/refine/settlements.py` — city siting (`refine_settlement_positions`, `city_site_scores`, `SITE_*` weights reused by region/local tiers) + `remap_routes`
- `realmgen/refine/fill.py` — `PolityLabelField` (polity at any sphere point, incl. domain warp), texel labels, fill texture, `polity_fill_recipe`
- `realmgen/refine/borders.py` — border polylines contoured from the same field on a seamless icosphere lattice (marching triangles + bisection + Chaikin)

## `RefineParams`

- `river_threshold_factor` (1.5) — texel is "on a river" within `factor × texel_spacing` of a river polyline (densified to half a texel).
- `city_min_spacing_km` (120) — no city closer than this to a bigger one (centroids ~200 km apart, so adjacent-face cities could otherwise share an edge).
- `border_smoothing_iterations` (2) — Chaikin passes on border polylines.
- `border_lattice_offset` (hidden, 0) — lattice levels finer (+) / coarser (−) than texel-fine (`lattice_depth_for`, depth 9 at defaults). −1: ~0.6 s faster per build (~1.1 s on first), but wiggles/enclaves under ~2 texels vanish from lines while the fill still shows them.
- `border_control_floor` (hidden, 1e-3) — authority floor so a zero-authority face still has a finite Voronoi weight.
- `boundary_warp_amplitude` (0.012, unit-sphere ≈ ⅓ face) — fBm domain warp on positions fed to the polity Voronoi; `0` → bare centroid bisectors (polygonal staircase).
- `boundary_warp_frequency` (45) — warp base frequency; too high aliases against the texel grid.
- `boundary_warp_octaves` (hidden, 3).

## Site features (`sites.py`)

`SiteField(mesh, ocean_tex, lake_tex, fertility_tex, rivers, *,
river_threshold_factor, streams=None)` answers per texel of a face's grid: land?
**owned** by the face (nearest centroid — same rule as `GlobalSampler` and the
label field, so faces partition the sphere)? ocean-/lake-coast (4-adjacent)? on a
river / at a confluence (within `river_threshold_factor × texel_spacing`)? 3×3
land fertility? With `streams` (town stage only): stream channel?
`FaceSites.sites()` = owned land texels. `prefetch(faces)` batches many faces
(one KD-tree query per feature).

A confluence = a river mouth within `confluence_search_radius` of an interior
vertex of a *different* river (Hack decomposition: every tributary ends on its
stem).

## City siting (`settlements.py`)

`refine_settlement_positions` moves each settlement from its face centroid to
its face's best candidate site:

| term | weight |
|---|---|
| ocean coast (else lake coast) | 3.0 (2.0) |
| confluence (else river) | 2.0 (1.5) |
| local fertility, 0–1 | 1.0 |
| distance from centroid / face radius | −0.5 |
| jitter, uniform 0–1, keyed `(seed, face)` | 0.3 |

Terms add (river mouth > bare shore; harbour > inland river). Jitter stops
uniform inland faces all picking the same grid spot (a visible lattice at region
scale).

**Spacing.** Largest first; sites within `city_min_spacing_km` of a sited city
are rejected (all crowded → least crowded). Seeds 4/31/35 have no pair < 150 km
either way — it's a guard for unlucky coasts.

**Site kind.** The chosen texel's features → `Settlement.site_kind`
(`SITE_KIND_OCEAN_COAST | _LAKE_COAST | _RIVER | _CONFLUENCE` bit flags), texel →
`site_position` (`Settlement.set_site`). Region/local tiers re-site from there,
keeping what the site offered
([region_features.md](region_features.md#towns-and-cities)).

Returns `{old_position_key: new_position}`; `remap_routes` rewrites trade-route
endpoints matching a moved settlement's old centroid (bit-identical, so exact;
deposit-side and intermediate endpoints untouched).

## Polity label field (`fill.py`)

`PolityLabelField` is the **single source of truth** for fill and borders: the
polity at any sphere point as a function of **position alone**, never of which
face's grid holds it — so over-hanging face textures agree by construction and
seams need no reconciling.

Labels: polity id (`≥ 0`), `UNCLAIMED` (-1), `WATER` (-2).

- **Owner face** = nearest centroid.
- **Weighted Voronoi.** Claimed-land owner: point split among the owner's polity
  and its claimed-land neighbours by `argmin |x−c|² / control²` (authority), so
  stronger polities bulge. Single-polity candidate sets skip the vote.
- **Domain warp.** `x` is displaced by a coherent fBm-simplex field
  (`domain_warp`, `boundary_warp_*`), pure function of position — otherwise
  borders are straight bisectors hugging hexagon edges.
- **Orphan land.** A face is water at face level when > half its texels are, so
  coastal "water" faces can hold real land; a water owner votes among its
  claimed-land neighbours only, so that land isn't painted ocean.
- **Unclaimed**: unclaimed land owner, or water owner with no claimed-land
  neighbour.
- **Water**: `land_labels` ignores it; `texel_labels` overlays per-texel
  `ocean_mask`/`lake_mask` so the fill runs to the texel coastline.

Kept on the world as feature `polity_label_field`; not serialized
(`RefineStage.load` rebuilds it from persisted polity channels).

`colorize_labels` → RGBA (polity ids in the exact `make_polity_tree_color`
palette, unclaimed grey, ocean/lake blues), stored as face texture `polity_fill`
(float16), attached by `PolityStage.register_display` to "Polities" along with
`polity_fill_recipe`, which colours any texel positions — so the region view
evaluates fill at its own texels with refined shoreline
([region.md](region.md#display-regiondisplaypy)).

## Border refinement (`borders.py`)

Replaces `PolityStage`'s soup of hexagon edges with continuous polylines
contoured from the label field. Face textures can't be contoured (grids don't
line up across seams; a border *along* a seam is in neither), so the field is
sampled on a fine seamless icosphere (`_icosphere_lattice`):

- **Lattice.** Depth from `lattice_depth_for` so an edge ≤ a face texel: depth 9
  (~2.6 M vertices, ~15 km edges) by default; vectorised, cached per process.
  Vertex water = bilinear ocean/lake mask via `GlobalSampler`.
- **Crossings.** An edge whose two non-water labels differ is a border at the
  **shallowest** depth where their ancestors differ (`_edge_border_depth`: 0 =
  sovereign, incl. polity vs unclaimed; d > 0 = internal). Coastlines aren't
  borders.
- **Bisection** against the label field (`bisect_steps` 5) → within ~0.5 km of
  the true boundary.
- **Marching triangles.** Two crossings → linked; three (junction) or one (meets
  coast) → each linked to the triangle centre. No ambiguous cases, each crossing
  shared by exactly two triangles → chains are continuous by construction (no
  welding, de-doubling or seam repair).
- **Smoothing.** Chaikin (`border_smoothing_iterations`); open ends and
  junctions pinned, so vassal lines end exactly on the line they meet.

Lines end only at junctions or coast. `scripts/debug_borders.py` prints open
endpoints per depth; seeds 4/31/35: every depth-0 open end within one lattice
edge of water.

## Outputs

Mutates: `settlements` (`Settlement.set_site`: `.position`, `.site_position`,
`.region_position`, `.site_kind`), `trade_routes` (endpoints), `polity_borders`
(`{depth: (M, 6)}` smoothed segments). Adds face texture `polity_fill`
("Polities") and in-memory feature `polity_label_field` (read by the recipe). No
new overlays.

Persistence: `RefineStage` owns final `trade_routes` and `polity_borders`
(`pack_routes` / `pack_borders`); `polity_fill` persists as a mesh texture.

## Validation

`realmgen/refine/test/test_refine.py`: city siting (coast, ocean-vs-lake, river,
owned texels, spacing), route remapping, Chaikin and chaining, icosphere lattice,
lattice tracing (loops, bisection, coastlines ignored, coast endings, junctions,
vassal lines ending on sovereign lines), label field (unclaimed, orphan land,
interiors, authority bulge, face-independence), domain warp, end-to-end tracer on
a real dual mesh, fill recipe and fill texture behaviour.
