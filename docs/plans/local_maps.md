# Local Maps — Plan

The third and finest tier: **each regional hexagon becomes a local map**, fine
enough to site villages of 200–1000 people. It refines the region exactly the way
a region refines the globe — bilinear read of the parent, detail above the
parent's Nyquist, per-parent-texel conservation — so the machinery in
`region/upsample.py` is generalised rather than copied.

## Scale (defaults)

| | Region | Local |
|---|---|---|
| Map covers | ~3,070 km cell | one regional hex, ~286 km across (~53k km²) |
| Hex | ~286 km, ~53k km² | ~23 km, ~355 km² (`n_cols=16`) |
| Texel | ~4.5 km (~20 km²) | ~365 m (~0.13 km², `tex_size=64`) |
| Hexes | ~250 | ~200 (only those overlapping the parent hexagon) |

A village is about one local texel across, so it is drawn as a **dot**. Villages
per local hex: up to ~14 in the best land (16 people/km²), 0–2 in marginal land.
The detail band gains ~3–4 octaves (a 12× refinement, against the region's 3×).

## Decisions

1. **Village population is conserved per (global texel ∩ local map).** A
   regional texel (~20 km², ~320 people at best) is smaller than one village, so
   it cannot be the unit; a global texel (~195 km²) holds several. Villages come
   *out of* the local rural density, so rural + villages inside each global
   texel ∩ map is unchanged, and summing over the local maps a global texel
   straddles recovers it.
2. **Fixed village fraction**, `village_fraction = 0.7` by default: that share of
   each budget goes into villages of `[200, 1000]`; the rest stays as the rural
   density texture (hamlets, farmsteads).
3. **Streams: the world's network only** in the first version. The code marks
   where a finer local network (flow accumulation over the local relief) would
   plug in.
4. **Villages exist in the local view only** — not on the region or world maps.

## Phases

Status: phases 1–4 done (2026-09-27); 5–6 to do.

1. **Geometry and grid** — `localmap/geometry.py`: `LocalGeometry(region
   geometry, hex index)` maps local `[0, 1]²` affinely onto the parent hex's
   texture square in the region's `(s, t)` frame (which extends smoothly past the
   cell, so edge hexes work, and handedness / `up_direction` carry over).
   `RegionHexGrid` is reused unchanged apart from a filter keeping only local
   hexes that overlap the parent hexagon. `in_map` marks texels inside the
   parent hexagon: the aggregation weight, so neighbouring local maps never
   double-count.
2. **Tier-agnostic refinement** — a *parent source* interface
   (`locate`, `coarse`, `anchor_values`, `slope`, water surfaces, danger) with a
   `GlobalSource` over the mesh and a `RegionSource` over `RegionFields`.
   `build_region_fields` and `build_local_fields` become thin wrappers over one
   `refine_fields`. Region output stays bit-identical; local output conserves
   every regional texel at float noise.
3. **Features** — the region's rivers and roads re-routed through
   `region/corridor.py` with a corridor of ~one regional texel; towns and cities
   keep their regional sites (small snap onto local streams/roads).
4. **Villages** — `localmap/villages.py`: per-budget power-law sizes in
   `[village_min, village_max]` with the town stage's stochastic remainder;
   siting weights on fertility/density plus stream, river and road bonuses;
   spacing ~2–3 km × (pop/200)^0.25 and a town/city clearance; seeded
   deterministically per local map.
5. **Viewer** — `RegionView` generalised to either tier; a "local" toggle in the
   region view enters the clicked hex, with a back button; villages as dots with
   hover (population, site reason); an in-memory LRU of visited local maps.
6. **Tooling** — `docs/local.md`, `scripts/debug_local.py` (side-by-side +
   conservation report), `scripts/measure_villages.py` (count, spacing, size
   distribution, share of rural people in villages).
