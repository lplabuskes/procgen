# Danger Stage

See also: `docs/plans/danger.md` for the original design.

## Files
- `worldgen/danger/builder.py` — `DangerBuilder` (stage orchestration, display, persistence)
- `worldgen/danger/sources.py` — `DangerParams`, `DangerSourceParams`, `DangerTerrain`, the source registry (`_SOURCE_DEFS`), `source_contribution`
- `worldgen/danger/channels.py` — `DANGER` (per-face metadata) and `DANGER_ARRAY` (the `"danger"` world feature)
- `worldgen/danger/display.py` — `danger_color`, `danger_legend_entries`, `DANGER_RECIPE`

## Concept

A per-face **danger** scalar in [0, 1]: monster lairs, planar rifts, undead and the like, seeded on the terrain that suits each. It runs in the globe pipeline between resources and cost (cost does not read it). Realmgen is the consumer: `DangerInfluenceParams` (see [realmgen.md](realmgen.md)) turns it into a suitability penalty, a placement block, and extra land traversal cost; the region tier makes it a continuous field (see [region_fields.md](region_fields.md)).

## `DangerBuilder`

`build(ctx)` reads `volcanic_activity`, `is_ocean`, `biome`, `boundary_type`, `plate_type`, `fertility_mean`, `forest_density_mean` metadata (and the delta stage's `marsh_fraction` when present), the `elevation` and `lake_mask` textures and the `"river_objects"` feature, and packs the derived inputs into a `DangerTerrain`: water (ocean ∪ faces containing lake), mean elevation and elevation span, BFS hop counts for ocean depth (from coastal ocean), distance to coast (from coastal land) and distance to fresh water (from river and lake faces), and the river-face mask.

Each enabled source then:
1. builds its candidate mask from the terrain (`DangerSource.mask`),
2. draws `count` seed faces from it without replacement (`ctx.rng("danger")`),
3. adds `intensity · exp(−hops · decay_per_hop)` to every face within `hop_radius` hops of a seed.

The sum is clamped to [0, 1].

Outputs: per-face `danger` metadata and the `"danger"` feature (the same `(N,)` float32 array), the `"danger_sources"` feature (`(positions, kinds)` of every seed face), and per-face `danger_sources` metadata (the names of the sources seeded on that face).

**Reproducibility.** The draws come from `ctx.rng("danger")`, which is keyed on `BuildContext.seed`, not `World.seed`. `main.py` builds its context without a seed, so the same world seed can give a different danger field on each run.

## Sources (`sources.py`)

`DangerParams` holds one `DangerSourceParams` per source (`count`, `hop_radius`, `intensity`, `decay_per_hop`, `enabled`). The candidate masks use module-level threshold constants (`HARPY_ELEV_MIN`, `KRAKEN_DEPTH_MIN`, …):

| Category | Sources (candidate terrain) |
|---|---|
| Volcanic | Dragon Lair (any land), Fire Elemental Rift (volcanic land) |
| Ocean | Kraken Domain (deep ocean), Deadwater Zone (open ocean), Sea Serpent Territory (ocean near the coast) |
| Highland | Harpy Peak (high, moderately rugged), Troll Hold (river faces), Giant Lair (high ground) |
| Forest | Goblin Warband (forest), Werewolf Range (wooded steppe/savannah), Basilisk Territory (semi-arid/desert), Cursed Forest (dense forest far from fresh water), Lizardfolk Fen (river-delta marsh: `marsh_fraction ≥ 0.05`; last in `_SOURCE_DEFS` so the others keep their kind indices) |
| Planar Rift | Fire (volcanic), Earth (active continental boundaries), Air (very high), Water (open ocean) |
| Undead | Lich Stronghold (high, barren, treeless), Barrow Field (low, flat, dry), Haunt (barren coast) |

The order of `_SOURCE_DEFS` defines the source kind indices stored in `"danger_sources"`.

## Display (`--debug` only)

- **Danger** scheme (per-face colour function). `DANGER_RECIPE` exists for the region view only: the globe has no danger texture, but a region refines one.
- **Danger Sources** overlay (unchecked), with a legend by category.
- Hover: `Danger: 0.42` plus the names of the sources seeded on the face.

## Persistence

`danger`, `danger_source_positions`, `danger_source_kinds`. On load, the per-face `danger_sources` names are rebuilt by matching each seed position to its nearest face centroid.
