# Danger Stage

Original design: `docs/plans/danger.md`.

## Files
- `worldgen/danger/builder.py` — `DangerBuilder` (stage, display, persistence)
- `worldgen/danger/sources.py` — `DangerParams`, `DangerSourceParams`, `DangerTerrain`, source registry `_SOURCE_DEFS`, `source_contribution`
- `worldgen/danger/channels.py` — `DANGER` (per face), `DANGER_ARRAY` (`"danger"` feature)
- `worldgen/danger/display.py` — `danger_color`, `danger_legend_entries`, `DANGER_RECIPE`

## Concept

Per-face **danger** ∈ [0, 1]: monster lairs, planar rifts, undead…, seeded on
suitable terrain. Runs between resources and cost (cost doesn't read it).
Consumers: realmgen's `DangerInfluenceParams` ([realmgen.md](realmgen.md)) —
suitability penalty, placement block, extra land traversal cost; the region tier
makes it continuous ([region_fields.md](region_fields.md)).

## `DangerBuilder`

Reads metadata `volcanic_activity`, `is_ocean`, `biome`, `boundary_type`,
`plate_type`, `fertility_mean`, `forest_density_mean` (+ `wetland_fraction` if
present), textures `elevation`, `lake_mask`, feature `"river_objects"`. Packs a
`DangerTerrain`: water (ocean ∪ lake faces), mean elevation and span, BFS hops for
ocean depth (from coastal ocean), distance to coast (from coastal land), distance
to fresh water (from river/lake faces), river-face mask.

Per enabled source:
1. candidate mask (`DangerSource.mask`);
2. draw `count` seed faces without replacement (`ctx.rng("danger")`);
3. add `intensity · exp(−hops · decay_per_hop)` within `hop_radius` hops.

Sum clamped to [0, 1].

Outputs: per-face `danger` + `"danger"` feature (same `(N,)` float32),
`"danger_sources"` (`(positions, kinds)` of seed faces), per-face
`danger_sources` (names seeded there).

**Reproducibility.** `ctx.rng("danger")` is keyed on `BuildContext.seed`, not
`World.seed`; `main.py` builds its context without a seed, so the same world seed
can give a different danger field per run.

## Sources (`sources.py`)

`DangerParams`: a master `enabled` (off → no source is seeded, danger is zero),
then one `DangerSourceParams` per source (`count`, `hop_radius`,
`intensity`, `decay_per_hop`, `enabled`). Masks use module threshold constants
(`HARPY_ELEV_MIN`, `KRAKEN_DEPTH_MIN`, …).

| Category | Sources (candidate terrain) |
|---|---|
| Volcanic | Dragon Lair (any land), Fire Elemental Rift (volcanic land) |
| Ocean | Kraken Domain (deep ocean), Deadwater Zone (open ocean), Sea Serpent Territory (near coast) |
| Highland | Harpy Peak (high, moderately rugged: face relief from the world's ruggedness), Troll Hold (river faces), Giant Lair (high ground) |
| Forest | Goblin Warband (forest), Werewolf Range (wooded steppe/savannah), Basilisk Territory (semi-arid/desert), Cursed Forest (dense forest far from fresh water), Lizardfolk Fen (warm wetland, `wetland_fraction ≥ 0.2`, not Icecap/Tundra/Taiga) |
| Planar Rift | Fire (volcanic), Earth (active continental boundaries), Air (very high), Water (open ocean) |
| Undead | Lich Stronghold (high, barren, treeless), Barrow Field (low, flat, dry), Haunt (barren coast) |

`_SOURCE_DEFS` order defines the kind indices in `"danger_sources"` — **append new
sources** (Lizardfolk Fen is last for this reason).

## Display (`--debug` only)

- **Danger** scheme (per-face colour function). `DANGER_RECIPE` is region-only (globe has no danger texture; regions refine one).
- **Danger Sources** overlay (unchecked), legend by category.
- Hover: `Danger: 0.42` + source names on the face.

## Persistence

`danger`, `danger_source_positions`, `danger_source_kinds`; on load per-face
`danger_sources` names are rebuilt by nearest-centroid matching.
