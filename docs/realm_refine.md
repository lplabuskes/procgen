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
- `realmgen/refine/sites.py` — `SiteField`: per-texel site features (coast, river, confluence, local fertility, face ownership), shared with the town stage
- `realmgen/refine/settlements.py` — city siting + trade-route remapping
- `realmgen/refine/fill.py` — `PolityLabelField` (the polity at any point of the sphere, incl. domain-warp erosion), the texel labels sampled from it, the fill texture, and `polity_fill_recipe`
- `realmgen/refine/borders.py` — border polylines contoured from the same field on a seamless icosphere lattice (marching triangles + bisection + Chaikin)

## `RefineParams`

- `river_threshold_factor` (default 1.5) — a candidate texel counts as "on a river"
  if it lies within `factor × texel_spacing` of a river polyline (densified to
  half a texel).
- `city_min_spacing_km` (default 120) — no city is sited closer than this to a
  bigger one. Face centroids are ~200 km apart, so without it two cities on
  adjacent faces could both take the shared edge.
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

## Site features (`sites.py`)

`SiteField` reads the global face textures and answers, per texel of a face's
grid: is it land, is it **owned** by the face (its nearest face centroid — the
rule `GlobalSampler` and the label field use, so faces partition the sphere),
is it ocean- or lake-coast (4-adjacent to that water), is it on a river or at a
confluence (within `river_threshold_factor × texel_spacing`), and how fertile is
its 3×3 land neighbourhood. `FaceSites.sites()` is a face's candidate sites: its
owned land texels. `prefetch(faces)` computes many faces in one batch (one
KD-tree query per feature), which is what both callers do.

A confluence is a river mouth that lands within `confluence_search_radius` of an
interior vertex of a *different* river (in the Hack decomposition every
tributary ends on its stem).

## City siting (`settlements.py`)

`refine_settlement_positions` moves each settlement from its face centroid to
the best of its face's candidate sites, by a weighted score:

| term | weight |
|---|---|
| ocean coast (else lake coast) | 3.0 (2.0) |
| confluence (else river) | 2.0 (1.5) |
| local fertility, 0–1 | 1.0 |
| distance from the centroid, as a fraction of the face's radius | −0.5 |
| jitter, uniform 0–1, keyed on `(seed, face)` | 0.3 |

Terms add, so a river mouth beats a bare shore, and a harbour beats an inland
river. The jitter keeps uniform inland faces from all choosing the same spot of
the texel grid (a visible lattice at region scale).

**Spacing.** Cities are sited largest first; a site within
`city_min_spacing_km` of an already-sited city is rejected, so the bigger city
gets its pick and a smaller neighbour takes its best remaining site (or, if every
site is crowded, the least crowded one). On seeds 4/31/35 no two cities end up
closer than 150 km, with or without the constraint — it is a guard for the
unlucky coast, not something the scores routinely fight.

**Site kind.** The chosen texel's features are kept as `Settlement.site_kind`
(`SITE_KIND_OCEAN_COAST | _LAKE_COAST | _RIVER | _CONFLUENCE` bit flags), and
the texel as `site_position` (`Settlement.set_site`). The region and local tiers
re-site the city on their finer coast and rivers from there, keeping what the
site offered (see [region.md](region.md#towns-and-cities-regiontownspy-regioncitiespy)).

The function returns `{old_position_key: new_position}` for moved settlements.
`remap_routes` then rewrites any trade-route endpoint matching a moved settlement's
old centroid to its refined position — same connections, adjusted geometry. (Route
endpoints are face centroids bit-identical to the pre-refine settlement positions,
so the match is exact; deposit-side and intermediate endpoints are untouched.)

## Polity label field (`fill.py`)

`PolityLabelField` is the **single source of truth** for both the fill colours and
the border lines. It gives the polity at any point of the sphere, as a function of
**position alone** — never of which face's texture grid holds the point. That is
what makes the result seamless: each face's square texture over-hangs its hexagon,
and wherever two faces' textures sample the same ground they now agree by
construction, so there is nothing to reconcile at the seams.

Labels: a polity id (`≥ 0`), `UNCLAIMED` (-1) for claimed-by-nobody land, or
`WATER` (-2) for ocean/lake.

- **Owner face.** A point belongs to its nearest face centroid.
- **Weighted Voronoi.** If the owner is claimed land, the point is split among the
  owner's polity and its claimed-land neighbours by an authority-weighted
  (multiplicatively-weighted) Voronoi (`argmin |x−c|² / control²`), so the
  stronger polity bulges into the weaker one. Owners whose whole candidate set is
  one polity skip the vote.
- **Boundary erosion (domain warp).** The positions `x` fed to that Voronoi are
  displaced by a coherent fBm-simplex vector field (`domain_warp`,
  `boundary_warp_*`). Without it the boundary is the bare bisector between two
  face centroids — a straight line that hugs the hexagon edges, so the map reads
  as polygonal. The warp is a pure function of position.
- **Orphan land.** A face counts as water at *face* level once more than half its
  texels are water, so a coastal "water" face can still hold a large minority of
  genuine land. A water owner votes among its claimed-land neighbours only, so
  that land takes a neighbouring polity instead of being painted ocean.
- **Unclaimed.** An unclaimed land owner, or a water owner with no claimed-land
  neighbour, is `UNCLAIMED`.
- **Water.** `land_labels` ignores water. `texel_labels` overlays the per-texel
  `ocean_mask`/`lake_mask`, so the fill runs to the texel coastline.

The field is kept on the world as the `polity_label_field` feature. It is not
serialized. `RefineStage.load` rebuilds it from the persisted polity channels.

`colorize_labels` turns labels into RGBA: polity ids take the exact
`make_polity_tree_color` palette, unclaimed land is grey, and ocean/lake texels
their own blues. The refine stage stores the result as the `polity_fill` face
texture (float16), which `PolityStage.register_display` attaches to the
"Polities" scheme when present. It also attaches `polity_fill_recipe`, which
colours any grid of texel positions, so the region view evaluates the fill at its
own finer texels and with its refined shoreline (see [region.md](region.md)).

## Border refinement (`borders.py`)

`PolityStage` emits borders as a soup of independent hexagon edges. The refine
stage replaces them with continuous polylines contoured from the label field.

Contouring the face textures themselves does not work. No two faces' grids line
up, so a border would have to be stitched across every seam, and a border running
*along* a seam lives in neither grid. Instead the field is sampled on the vertices
of a fine icosphere (`_icosphere_lattice`), one triangle mesh with no seams:

- **Lattice.** Its depth is chosen by `lattice_depth_for` so an edge is no longer
  than a face texel: depth 9 (~2.6 M vertices, ~15 km edges) on the default world.
  It is generated vectorised and cached per process. Water at the vertices is the
  bilinear read of the ocean/lake masks through `GlobalSampler`.
- **Crossings.** A lattice edge whose two non-water labels differ is a border at
  the **shallowest** hierarchy depth where their ancestors differ
  (`_edge_border_depth`: depth 0 = sovereign borders, including polity vs.
  unclaimed land; depth d > 0 = internal subtree borders). Coastlines are not
  borders.
- **Bisection.** Each crossing is placed by bisecting its edge against the label
  field (`bisect_steps`, default 5), so the line sits on the true boundary within
  ~0.5 km instead of at the edge midpoint.
- **Marching triangles.** A triangle with two crossings links them. One with
  three crossings (a three-region junction) or one crossing (a border meeting the
  coast) links each crossing to the triangle's centre. Triangles have no ambiguous
  case, and each crossing is shared by exactly the two triangles on its edge, so
  the pieces chain into continuous lines by construction. There is no welding,
  de-doubling or seam repair.
- **Smoothing.** Chains are Chaikin-smoothed (`border_smoothing_iterations`).
  Open-chain ends and three-region junctions stay fixed, so a vassal line always
  ends exactly on the line it meets.

A line can only end at a junction or at the coast. `scripts/debug_borders.py`
prints the open-endpoint count per depth. On seeds 4/31/35 every depth-0 open end
lies within one lattice edge of water.

## Outputs

Mutates in place: `settlements` (`.position`, `.site_position`, `.site_kind`), `trade_routes` (endpoint geometry),
`polity_borders` (`{depth: (M, 6)}` smoothed segments). Adds one face texture,
`polity_fill`, consumed by the "Polities" scheme, and one in-memory feature,
`polity_label_field`, which the scheme's recipe reads. No new overlays.

## Validation

`realmgen/refine/test/test_refine.py` covers city siting (coast,
ocean-vs-lake, river, owned texels only, minimum spacing with the bigger city first), route remapping, Chaikin smoothing and edge chaining, the
icosphere lattice (closed mesh, edge ordering, depth for a texel size), lattice
tracing (a closed loop between hemispheres, bisection onto the true boundary,
coastlines ignored, a border ending at the coast, three-way junctions joining, a
vassal line ending on the sovereign line), the label field (unclaimed, orphan
land, uniform interiors, authority bulge, labels independent of which face holds
the texel), the domain warp, the end-to-end tracer on a real dual mesh, the fill
recipe, and fill-texture shape/coastline/orphan-land/lake behaviour.
