# Resources Stage

Original design: `docs/plans/resources_generation_spec.md`.

## Files
`worldgen/resources/`: `builder.py` (stage + params), `terrain.py` (per-face
helpers), `fertility.py`, `forest.py`, `crops.py`, `minerals.py`, `rare_bio.py`,
`display.py`, `channels.py` (`FERTILITY_MEAN`, `MINERALS`, `MINERAL_SITES`,
`RARE_BIO`).

## `builder.py`

**`ResourcesParams`**: `fertility`, `forest_density`, `minerals`, `rare_bio`,
optional `seed` (forest noise, minerals, rare resources; 0 when unset — **not**
the world seed). Biome LUTs (`BIOME_FERTILITY_MULT`, `BIOME_FOREST_BASELINE`,
`BIOME_PRECIP_REFERENCE_MM`) are module constants in `fertility.py` / `forest.py`
— they define the model, not tune it.

**`ResourcesBuilder`** (after climate lake feedback). Reads features
`"river_objects"`, `"currents"` (aquatic fertility); textures `temperature`,
`precipitation`, `elevation`, `ocean_mask`, `lake_mask`; per-face `biome`,
`is_ocean`, and for minerals `volcanic_activity`, `boundary_type`, `plate_type`,
`elevation`.

Order: forest → fertility (with forest penalty) → delta silt bonus
(`apply_delta_fertility`, if `delta` texture; [delta.md](delta.md)) → aquatic
fertility over ocean/lake texels → crops → minerals + workings → rare resources.

- Textures: `"fertility"` `(N,H,W,1) float16`, `"forest_density"` float16, `"dominant_crop"` uint8.
- Per face: `fertility_mean`, `forest_density_mean`, `dominant_crop` (mode over non-ocean texels).
- Features: `"minerals"`, `"rare_bio"` (`(positions, kinds, abundances)`), `"mineral_sites"` (workings).

## Individual modules

**`terrain.py`**:
- `face_elevation_range(elevation_tex)` — per-face pixel elevation span as ruggedness (per-pixel slope is noise-dominated at default tex_size)
- `face_hops_from_seeds(mesh, seed_mask, max_hops)` — BFS hops; beyond → `max_hops+1`
- `river_traversed_faces` / `lake_containing_faces` — water-proximity seeds
- `coastal_ocean_faces(mesh, is_ocean)` — ocean faces next to land (depth BFS seeds)
- `volcanic_faces(mesh, threshold)`

**`fertility.py`**:
- `compute_fertility()` — product of five per-face factors: climate (bell in growing-season T and log-P), terrain (elevation-range penalty), water proximity (floodplain hop-decay bonus), volcanism proximity (volcanic-soil bonus), biome LUT; plus optional per-pixel forest penalty. Factors rasterised with `rasterize_face_values` so faces blend continuously.
- `compute_aquatic_fertility()` — per-face scalar for ocean/lake faces from SST, ocean depth hops (upwelling proxy) and lake river-proximity; written over water pixels so fishing enters carrying capacity.
- `neighbour_water_fertility()` — land faces have no aquatic value, so their sea pixels take the mean of neighbouring ocean faces' and lake pixels of neighbouring lake faces' (0 if none): a coast's shallows are the same fishing grounds, feeding its fishers ([realm_population.md](realm_population.md#farmers-and-fishers)).
- `apply_delta_fertility()` — raises delta plains toward `delta_fertility` (0.9) × growing-season T bell (the river, not rain, waters a delta), marsh docked by `delta_marsh_penalty` (0.4). Only raises.

**`forest.py`** — `compute_forest_density()`: biome baseline × precipitation
modulation (tanh about per-biome reference P) × treeline taper, + riparian bonus
+ small per-pixel fBm noise (`noise_amplitude`); 0 on ocean. Inputs rasterised
before combining so biome boundaries blend.

**`crops.py`** — `classify_crops()`: first-match threshold table (T_grow,
P_annual, elevation) over 8 crops; pixels below `FERTILITY_GATE` (0.08) get none.
Vectorised version mirrors scalar `default_crop` exactly.

**`minerals.py`**:
- `place_mineral_deposits()` — geology + fBm hybrid. Nine types (Iron, Copper, Gold, Gems, Coal, Tin, Rare Metals, Salt, Marble), each with a geological candidate mask (boundary type, plate type, volcanism, elevation) and independent 3-octave fBm clustering. Returns `(positions (M,3), kinds int8, abundances float32)`, one deposit per (face, kind) at the centroid.
- `place_mineral_sites()` — splits each deposit into **workings**: `min_sites`–`max_sites` (2–5) on distinct owned land texels (nearest centroid; interior before coastal), at a random point in an interior texel (coastal keeps its centre, off the sea), Dirichlet(`site_share_concentration` = 1) shares (a few workings carry most). Returns `(positions (K,3), deposit int32, share float32)`, shares sum to 1 per deposit; deterministic per deposit. No owned land → one working at the centroid. Seed 4: 415 deposits → 1,416 workings, ~80 km from face centre on average.
- `mineral_workings(world)` reads them back (one per deposit at the deposit if absent); `site_workers(deposit, share, mine_workers)` gives each working its share of miners.
- World map draws one marker per deposit; region/local maps draw workings ([realm_towns.md](realm_towns.md), [region_fields.md](region_fields.md)).

**`rare_bio.py`** — `place_rare_bio_resources()`: ten resources (Spices, Tea,
Silk, Ivory, Furs, Incense, Dyes, Amber, Rubber, Pearls) via narrow ecological
criteria (biome, T, P, elevation, distance_to_coast) + fBm; a global per-type cap
keeps only the top candidates (world-level rarity).

**`display.py`** — schemes, legends, recipes (`FERTILITY_RECIPE`,
`FOREST_DENSITY_RECIPE`, `DOMINANT_CROP_RECIPE`; rerun by the region view). All
`--debug`: **Fertility**, **Forest Density**, **Dominant Crop** schemes;
**Minerals**, **Rare Resources** overlays (unchecked); fertility/forest/crop,
mineral and rare-resource hovers.
