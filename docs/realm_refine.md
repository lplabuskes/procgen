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
- `realmgen/refine/river_follow.py` — borders moved onto rivers run by run (`RiverSwaps`, `build_river_swaps`)
- `realmgen/refine/barriers.py` — `pack_river_segments` (river polylines → segment rows, the form views pass under `RIVER_SEGMENTS_KEY`), `flow_ramp`
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
- `border_river_reach_km` (120) — how far from a big river a border is moved onto it ([River following](#river-following-river_followpy)); 0 = off. Face-level borders run along cell edges up to ~half a face (~100 km) off a river; 70 left many of those (region-scale borders 100–200 km beside a river).
- `border_river_min_flow` (200) / `border_river_full_flow` (4000) — river flow at which the reach starts / is full (log ramp; world rivers start at 500, median ~1200, p90 ~7600 — 1000/8000 left the mid-size rivers that border most realms almost inert). Below 500 it reaches the stream stems: 300 → 200 let streams carry borders (reach ≤ ~37 km at 500).
- `border_river_min_run_km` (40) — shortest river stretch a run is moved onto.

Rivers play no part in claiming (the polity stage's cost surface has no river step), so
this is the only river handling of borders.

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
- **Power diagram.** Claimed-land owner: point split among the owner's polity
  and its claimed-land neighbours by `argmin |x−c|² − ½·L²·ln(control)` (authority;
  L the mean neighbour centroid spacing), so stronger polities bulge and every
  two-polity boundary is a straight line before the warp. The shift matches the old
  multiplicative `|x−c|² / control²` near the bisector, but that curved boundaries
  into arcs round the weaker centroids, which met in long thin cusps at three-polity
  corners (seed 525099826 region 11 hex 151); ~0.5% of land changed owner.
  Single-polity candidate sets skip the vote.
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
- **Rivers.** `land_labels` applies the river swaps after these plain labels
  (`plain_land_labels`); `with_rivers(segments)` moves them onto a view's own rivers
  ([River following](#river-following-river_followpy)).

Kept on the world as feature `polity_label_field`; not serialized
(`RefineStage.load` rebuilds it from persisted polity channels).

`colorize_labels` → RGBA (polity ids in the exact `make_polity_tree_color`
palette, unclaimed grey, ocean/lake blues), stored as face texture `polity_fill`
(float16), attached by `PolityStage.register_display` to "Polities" along with
`polity_fill_recipe`, which colours any texel positions — so the region view
evaluates fill at its own texels with refined shoreline
([region.md](region.md#display-regiondisplaypy)).

## River following (`river_follow.py`)

Decided once on the globe, as "border points
13–20 become river points 1–5":

1. **Plain chains.** The field without any river handling (`plain_land_labels`)
   is traced on the global lattice into chains between junctions / coasts, each
   with its leaf label pair (`borders.plain_border_chains` → `trace_lattice_chains`).
2. **Runs.** Per chain, each world river (land segments, reach = `reach_km` × the
   flow ramp) gives runs of chain points within reach (gaps of ≤ 2 points bridged).
   "Rivers" are the world `"rivers"` then the stream network's channels below them as
   main stems (`hydrology.rivers.stream_stems`, ids numbered on from the rivers) — the
   region tier draws those too, so a border may follow a stream; the flow ramp keeps
   the smallest out (streams run 100–500; at `min_flow` 200 those over 200, reach
   ≤ ~37 km). Seeds 4 / 31 / 35: swaps 88 / 55 / 63 without stems (min_flow 300) →
   147 / 91 / 107, 7533 / 4581 / 6195 km of border on stems, river stretches about
   the same; min_flow 300 gave 114 / 66 / 78, 100 gave 228 / 138 / 163.
   In-lake segments count (a border reaching a lake follows the river's course across
   it; only land is labelled). Every segment within the largest reach is tested
   against its own: a fixed k-nearest search missed a big river's lower course looping
   back past its own headwaters, whose small segments crowded the nearest, so runs
   began deep inside reach and jumped onto the river on a straight edge (70 km at
   seed 1583294000 region 51 hex 63).
   Consecutive runs of a chain along one river are joined when the river runs on the
   same way from the first's core to the second's, so the border follows the river past
   a bulge beyond reach in between (judged run by run, the return stretch was often too
   short and the bulge stayed).
   A run's **core** spans its first to last point within `0.6 × reach`; it is kept
   if the core's river stretch is ≥ `min_run_km` and ≥ half the straight distance
   between the core's ends (else it's a crossing; measuring against the border's
   *length* rejected borders that wander between two rivers — exactly the case to fix).
3. **One river per run, one border per river.** Runs from all chains are taken
   sovereign first, then longest river stretch; a run overlapping a taken one on its
   chain is dropped, so a border never alternates between two parallel rivers, and so
   is one whose river stretch already carries a border (both borders of a strip a
   river runs down can be within reach of it; their claims fought).
4. **New line.** The core is replaced by the river between the feet of its end
   points; outside the core each point is pulled toward its foot by
   `w = (reach − d) / (0.4 · reach)`, held non-decreasing toward the core, so the
   line eases onto and off the river along the border's own course (straight 45°
   connectors flipped large wedges).
5. **Claim by side.** The run claims the polygon of border run + new line
   (even–odd, so a river crossing the plain border gives alternating lobes), a
   margin of `edge_km` (8) round the old border and a band of `clean_km` (20)
   either side of the new line. There a point of either of the run's polities
   takes the one on its side of the new line — the polity that was left of the
   border run (probed 2 km either side, majority) is left of the line. Third
   polities are untouched; the first run claiming a point wins. Trading the two
   labels inside the polygon instead cut the field's enclaves beside the river in
   two and left specks wherever the traced border cut a corner the lattice can't
   resolve (a tongue tip under a lattice edge); the margin is about a lattice edge
   for that reason. Side at a line vertex uses the averaged normal. Past the
   line's ends nothing is claimed, except inside the polygon past an end cut short at a
   junction swap (where the border bulges past the line's end), on the side of the end
   segment extended; elsewhere that side rule gave wrong-bank islands where a border
   criss-crosses its river. The border edge is still densified to 1 km and
   bisected onto the plain boundary.
6. **Checks.** A run is dropped if > 5% of its polygon's land holds a third
   polity (not counting land within `clean_km` of the run's two ends: a run starting
   at a junction takes in a corner of the third polity there, which is never
   relabelled; counting it dropped e.g. a 91 km run on seed 525099826 region 11 hex 151), and split 25 km either side of any settlement of either polity it would
   hand over.
7. **Junctions.** Two accepted runs of *different* chains, consecutive along one
   river, with a polity P on the same bank in both (e.g. 7 | 29 then 29 | 82, 29 south
   of the river) become one line: P's boundary from the first run's core end is
   walked through up to 4 junctions (`_walk`: at each, the one other chain holding P)
   to the second's, and a `JunctionSwap` replaces it with the river between them.
   Where it claims (polygon ∪ margin ∪ band, as above), P takes its neighbours' land
   on its bank and gives its own on the far bank to whoever would hold it without P
   (`PolityLabelField.labels_without`: the warped power diagram over the 12 nearest
   other-polity claimed-land faces), so the junctions move onto the river. The two runs
   lose their eased lead-out into the gap. Unlike a run's, its claim covers all of its
   polygon, past the line's ends too (all of it is far bank; else strips of P stayed
   where its boundary leaves the river beyond the stretch).
   **Run ends at a junction.** A run whose chain ends at a junction while still
   beside the river (its end jumping ≥ `edge_km` onto the river) moves that junction
   too: from the run's core end, either of its polities' boundary is walked through up
   to 4 junctions (`_walk_to_river`) until it comes back within `clean_km` of the same
   river further along, and that stretch becomes a junction swap as above. Coast ends
   (no other chain there) are left: most long jumps are there, where the river reaches
   the water some way along the shore from the border, and no river point lies closer.
   Seeds 4 / 31 / 1583294000: 7 / 3 / 2 run ends moved.
   **Bridgeheads.** A settlement within 25 km (`settlement_clear_km`) of the river that
   it would hand over keeps its polity's land on its bank within its distance from
   the river + 25 km of its nearest river point (a half-disc, its radius roughened ±35% by fBm noise of position; the bank by owner: P's
   city on the far bank, a neighbour's on P's), measured against the drawn line on
   region / local maps. A river city holding both banks, as on the 29 | 7/81/82
   junction of seed 1926295024 region 16. A settlement further from the river, a walk
   through border points another run already holds (on a parallel river their claims
   alternated, leaving islands), a walk that fails (a coast, a boundary that runs elsewhere), or > 5% of the polygon's
   land held by a polity not on the walk, drops the swap (a radius from a city deep in
   the lump drew a disc hundreds of km across). Seeds 4 / 31 / 35: 8 / 5 / 0 junctions
   moved (walks failing 4 / 4 / 3, far settlement 0 / 0 / 1); 1926295024: 7, 3 with a
   bridgehead.

`PolityLabelField.land_labels` = plain labels, then the swaps. The region and
local views swap in their own drawing of the very river each run follows
(`RiverSwaps.with_rivers`): the view packs its rivers with their world river / stem ids
(`pack_river_segments(..., river_ids=RegionRivers.river_ids())`, column 7; see
[region_features.md](region_features.md#rivers-regionriverspy)). For each swap,
that river's drawn path alongside the core (segments within `view_reach_km` (30)
of it) is cut at its nearest points to the core's two ends; an end the drawing
doesn't reach (off the view) keeps the world line from the path's end on. Blend
feet move onto that river within the same reach. Before the link, a stand-in was
guessed by flow and distance: feet hopped onto tributaries (spurs that broke the
side test — fans of the wrong polity) and rivers re-routed far off their world
line weren't found. The plain border is the same at every tier, so only the
river edge moves. Old saved worlds lack `"river_texels"`: views then keep the
world line (`--rebuild-cache`).

Of sovereign border length within 60 km of a river at flow ≥ 2000, the share
within 8 km of it, seeds 4 / 31 / 35 / 369893336 / 1926295024: 32 / 60 / 61 / 52 / 67%
(mean 54%; ~12% with no river handling); 1.4% of all sovereign border length lies on a
river. Joining runs across a bulge raised the first four from a mean of 44% to 51%;
moving junctions takes the five to 38 / 64 / 61 / 53 / 69% (mean 57%).
Reach 70 gave ~31% here. This replaced a per-point snap (each point near a river read its
label from further out on its own bank, plus a 2-ring vote penalising candidates
across a river): it reached ~64% but alternated between parallel rivers, left
fans at bends and slivers along cell edges. Refine stage +2.2 s (seed 4; a second lattice trace), also paid on load
(the field is rebuilt); region / local Polities fill ~0.85 / 0.95 s (vs ~0.15 s
without swaps) — big batches go through a corner-classified grid over the query
points (`_SideGrid`, identical to the exact test), the rest are exact. Known gaps:
a run whose river ends (its source) inside it leaves a thin wedge; where no drawn
river stands in, the line follows the coarse world river.

The region and local views pass the rivers they draw to the "Polities" recipe under
`RIVER_SEGMENTS_KEY` (`pack_river_segments` → `(M, 8)` `[a, b, flow, river id]`,
in-lake segments included) and contour their own lines from that field
(`region/borders.py`, [viz_regionview.md](viz_regionview.md)).
`scripts/debug_borders.py --pick-river K` shows one.

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
