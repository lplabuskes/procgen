# Fantasy Danger Layer — Implementation Brief

## Overview

A new DangerBuilder stage slots into GlobePipeline between resources and cost. It seeds danger values onto faces based on terrain proxies already available from geology, climate, hydrology, and resources stages, producing a per-face (N,) float32 danger field in [0, 1]. The cost stage is unchanged; realmgen (or future stages) consume the danger channel directly.

## Pipeline Position

In `worldgen/builder.py`, add `self.danger_builder` between `self.resources_builder` and `self.cost_builder` in the stage list. Add `DangerParams` as a nested field in `GlobeParams`.

## Output

- Write (N,) float32 clamped to [0, 1] via `mesh.apply_metadata_channel("danger", values.tolist())`
- Define `DANGER = MetaChannel("danger", float, 0.0)` in a new `worldgen/danger/channels.py`
- Register the array as a FeatureChannel via `DANGER.write(ctx.world, values)` so realmgen can read it as a numpy array without parsing face metadata

## File Layout

```
worldgen/danger/
    __init__.py
    builder.py     — DangerParams, DangerBuilder (Stage)
    channels.py    — DANGER MetaChannel + FeatureChannel
    sources.py     — pure placement functions
    display.py     — color scheme + hover registration
```

## RNG

Use `ctx.rng("danger")` — no seed field in `DangerParams`. This gives a deterministic `np.random.Generator` keyed to `(world_seed, "danger")`, independent of call order.

For each source type, draw chosen faces with `rng.choice(candidates, size=n, replace=False)`. If `len(candidates) < n`, seed all candidates without error.

## Distance Mechanism

All radial falloff uses `face_hops_from_seeds(mesh, seed_mask, max_hops)` from `worldgen/resources/terrain.py`. Radii are hop counts (1 hop ≈ 220 km at depth-5 icosphere). Contribution formula:

```python
hops = face_hops_from_seeds(mesh, seed_mask, max_hops=hop_radius)
contrib = np.where(hops <= hop_radius,
                   intensity * np.exp(-hops * decay_per_hop), 0.0)
danger += contrib
```

`hop_radius=0` is a valid degenerate case: only seed faces receive intensity, all others 0.

## Terrain Proxies

All proxies are derived from data available before the cost stage. Named threshold constants (all-caps) live at module level in `sources.py`.

Derived inputs the danger stage must compute locally (not exposed as channels by prior stages):

```python
is_water = is_ocean | (lake_mask_tex[...,0].mean(axis=(1,2)) >= 0.5)
# ocean_depth_hops — recompute via coastal_ocean_faces(mesh, is_ocean) + face_hops_from_seeds
# coast_hops (land-side) — land faces adjacent to ocean via symmetric BFS
# water_hops — face_hops_from_seeds from river + lake seeds (same computation as resources, not re-exported)
elev_mean = elevation_tex[...,0].mean(axis=(1,2))
elev_range = face_elevation_range(elevation_tex)  # from terrain.py
```

All other inputs read from mesh metadata or channels: `BIOME`, `IS_OCEAN`, `ELEVATION`, `VOLCANIC_ACTIVITY`, `LAKE_MASK`, `RIVER_OBJECTS`, `fertility_mean`, `forest_density_mean`.

## Sources

### Volcanic / Tectonic

| Source | Candidate proxy | Count | Hop radius | Intensity | Decay/hop |
|---|---|---|---|---|---|
| Dragon lairs | `volcanic_activity >= DRAGON_VOLCANIC_MIN` | 3–5 | 1–2 | 0.8 | 0.3 |
| Fire elemental rifts | `volcanic_activity >= FIRE_RIFT_VOLCANIC_MIN` (active only) | 4–8 | 0 | 0.9 | — |
| Salamander colonies | `volcanic_activity >= SALAMANDER_VOLCANIC_MIN` | 6–10 | 0 | 0.5 | — |

### Coastal / Ocean

| Source | Candidate proxy | Count | Hop radius | Intensity | Decay/hop |
|---|---|---|---|---|---|
| Kraken domains | `is_ocean AND ocean_depth_hops >= KRAKEN_DEPTH_MIN` | 3–6 | 2–3 | 0.7 | 0.15 |
| Deadwater zones | `is_ocean AND ocean_depth_hops >= DEADWATER_DEPTH_MIN` | 5–10 | 2–3 | 0.4 | 0.15 |
| Sea serpent territories | `~is_ocean AND coast_hops <= SEA_SERPENT_COAST_MAX` | 8–15 | 1 | 0.6 | 0.4 |

### Highland / Pass

| Source | Candidate proxy | Count | Hop radius | Intensity | Decay/hop |
|---|---|---|---|---|---|
| Harpy peaks | `elev_mean >= HARPY_ELEV_MIN AND PASS_RUGG_MIN <= elev_range <= PASS_RUGG_MAX` | 6–12 | 0 | 0.8 | — |
| Troll holds | river-traversed land faces (`river_traversed_faces` filtered by `~is_water`) | 4–8 | 0 | 0.95 | — |
| Giant lairs | `elev_mean >= GIANT_ELEV_MIN` | 5–10 | 1–2 | 0.6 | 0.25 |

Pass geometry proxy: moderate ruggedness band (`PASS_RUGG_MIN <= elev_range <= PASS_RUGG_MAX`) at high elevation. No explicit pass detection exists before cost stage; this is the best available signal.

### Forest / Wilderness

| Source | Candidate proxy | Count | Hop radius | Intensity | Decay/hop |
|---|---|---|---|---|---|
| Goblin warbands | `forest_density_mean >= GOBLIN_FOREST_MIN AND ~is_ocean` | 15–25 | 2–3 | 0.35 | 0.2 |
| Werewolf ranges | `forest_density_mean >= WEREWOLF_FOREST_MIN AND biome in GRASSLAND_BIOMES` | 8–15 | 2–3 | 0.35 | 0.2 |
| Basilisk territories | `biome in SCRUB_BIOMES AND ~is_ocean` | 6–12 | 1–2 | 0.55 | 0.3 |
| Cursed forests | `forest_density_mean >= CURSED_FOREST_DENSITY_MIN AND water_hops >= CURSED_FOREST_ISOLATION_MIN` | 4–8 | 2–3 | 0.6 | 0.2 |

### Planar Intrusions

| Source | Candidate proxy | Count | Hop radius | Intensity | Decay/hop |
|---|---|---|---|---|---|
| Fire rifts | same as fire elemental rifts | 3–6 | 0 | 0.7 | — |
| Earth rifts | `boundary_type in ACTIVE_BOUNDARY_TYPES AND plate_type in CONTINENTAL_TYPES` | 4–8 | 0 | 0.7 | — |
| Air rifts | `elev_mean >= AIR_RIFT_ELEV_MIN` | 4–8 | 0 | 0.6 | — |
| Water rifts | `is_ocean AND ocean_depth_hops >= WATER_RIFT_DEPTH_MIN` | 4–8 | 0 | 0.6 | — |

Where fire rifts and fire elemental rifts share a candidate mask, compute it once.

### Undead / Necromantic

| Source | Candidate proxy | Count | Hop radius | Intensity | Decay/hop |
|---|---|---|---|---|---|
| Lich strongholds | `elev_mean >= LICH_ELEV_MIN AND fertility_mean <= LICH_FERTILITY_MAX AND forest_density_mean <= LICH_FOREST_MAX` | 2–4 | 0 | 0.9 | — |
| Barrow fields | `elev_mean <= BARROW_ELEV_MAX AND elev_range <= BARROW_RUGG_MAX AND water_hops >= BARROW_ISOLATION_MIN AND ~is_ocean` | 8–15 | 0–1 | 0.6 | 0.5 |
| Haunt concentrations | `coast_hops <= HAUNT_COAST_MAX AND fertility_mean <= HAUNT_FERTILITY_MAX AND ~is_ocean` | 10–20 | 0 | 0.5 | 0.5 |

## DangerParams Structure

```python
@paramclass
class DangerSourceParams:
    count: int        # number of seed faces to place
    hop_radius: int   # BFS depth cap
    intensity: float  # peak contribution
    decay_per_hop: float
    enabled: bool = True

@paramclass
class DangerParams:
    dragon_lairs: DangerSourceParams = ...
    fire_elemental_rifts: DangerSourceParams = ...
    # ... one field per source type
```

All defaults in `DangerSourceParams` match the table above. The `enabled` flag zeros out a source type for testing without removing it from the params tree.

## Combining Contributions

Sum all enabled source contributions additively, then clamp:

```python
danger = np.zeros(n, dtype=np.float32)
for source in enabled_sources:
    danger += compute_source_contribution(...)
danger = np.clip(danger, 0.0, 1.0)
```

## Display

Register behind the debug flag (per CLAUDE.md convention). Color scheme: low danger = neutral land color, high danger = deep red. Hover fragment: `f"Danger: {meta['danger']:.2f}"`. Use `reg.add_scheme("Danger", ...)` gated on `debug`.

## Reuse from terrain.py

Import and reuse directly from `worldgen/resources/terrain.py`:

- `face_hops_from_seeds`
- `face_elevation_range`
- `river_traversed_faces`
- `coastal_ocean_faces`
- `volcanic_faces`
- `lake_containing_faces`
