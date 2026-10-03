# Resources Stage

See also: `docs/plans/resources_generation_spec.md` for the original design.

## Files
- `worldgen/resources/builder.py` — stage orchestration + params
- `worldgen/resources/terrain.py` — per-face helper functions
- `worldgen/resources/fertility.py`
- `worldgen/resources/forest.py`
- `worldgen/resources/crops.py`
- `worldgen/resources/minerals.py`
- `worldgen/resources/rare_bio.py`
- `worldgen/resources/display.py`
- `worldgen/resources/channels.py` — `FERTILITY_MEAN`, `MINERALS`, `RARE_BIO`

## `builder.py`

**`ResourcesParams`** (dataclass): Nested params — `fertility: FertilityParams`, `forest_density: ForestDensityParams`, `minerals: MineralParams`, `rare_bio: RareBioParams`, optional `seed` (forest noise, minerals and rare resources are seeded from it, 0 when unset — not from the world seed). Biome-indexed lookup tables (`BIOME_FERTILITY_MULT`, `BIOME_FOREST_BASELINE`, `BIOME_PRECIP_REFERENCE_MM`) live as **module-level constants** in `fertility.py` / `forest.py` — they define the model, not tune it.

**`ResourcesBuilder`**: A pipeline `Stage`, runs after climate's lake-feedback pass. Reads: `"river_objects"` and (for aquatic fertility) `"currents"` world features, plus `temperature`, `precipitation`, `elevation`, `ocean_mask`, `lake_mask` face textures, and per-face `biome`, `is_ocean`, `volcanic_activity`, `boundary_type`, `plate_type`, `elevation` metadata (the last four by the mineral model).

Order: forest density → fertility (with the forest penalty) → the delta silt bonus (`apply_delta_fertility`, when the `delta` texture is present; see [delta.md](delta.md)) → aquatic fertility written over ocean/lake texels → crops → minerals → rare resources.

Attaches face textures:
- `"fertility"` `(N,H,W,1) float16`
- `"forest_density"` `(N,H,W,1) float16`
- `"dominant_crop"` `(N,H,W,1) uint8`

Writes per-face metadata: `fertility_mean`, `forest_density_mean`, `dominant_crop` (modal crop over the face's non-ocean texels). Stores `minerals` and `rare_bio` as `(positions, kinds, abundances)` 3-tuples and registers them directly as `"minerals"` and `"rare_bio"` globe features.

## Individual modules

**`terrain.py`**: Per-face helpers.
- `face_elevation_range(elevation_tex) → (N,) float32 km` — per-face pixel elevation span as a ruggedness proxy (per-pixel slope is dominated by noise at default tex_size)
- `face_hops_from_seeds(mesh, seed_mask, max_hops) → (N,) int32` — BFS hop count from any seed face; faces beyond `max_hops` get sentinel `max_hops+1`
- `river_traversed_faces(rivers, n_faces)` / `lake_containing_faces(lake_mask_tex)` — seed masks for water-proximity BFS
- `coastal_ocean_faces(mesh, is_ocean) → (N,) bool` — ocean faces adjacent to land, used as seeds for ocean depth BFS
- `volcanic_faces(mesh, threshold)` — seed mask for volcanism-proximity BFS

**`fertility.py`** — `compute_fertility()`: Multiplicative product of five per-face factors — climate (bell-curve in growing-season T and log-P), terrain (penalizes rugged faces by elevation range), water proximity (floodplain BFS hop-decay bonus), volcanism proximity (volcanic-soil BFS hop-decay bonus), and a biome LUT modifier — plus an optional per-pixel forest penalty. Per-face factors rasterized via vertex-averaged interpolation so adjacent faces blend continuously. Also exports `compute_aquatic_fertility()` — a separate per-face scalar for ocean/lake faces, driven by SST (temperature), ocean depth hops (upwelling proxy), and lake river-proximity; the builder writes these values over ocean/lake pixels in the fertility texture so coastal fishing is captured in carrying capacity. `apply_delta_fertility()` raises the river deltas' plains toward `delta_fertility` (0.9) × the growing-season temperature bell (the river, not the rain, waters a delta), docked by `delta_marsh_penalty` (0.4) on the marsh. It only ever raises fertility.

**`forest.py`** — `compute_forest_density()`: Biome baseline × precipitation modulation (tanh centred on per-biome reference P) × treeline taper, plus a riparian bonus and a small per-pixel fBm simplex noise overlay (`noise_amplitude`); zero on ocean. Per-face inputs rasterized before combining so biome boundaries blend continuously.

**`crops.py`** — `classify_crops()`: First-match-wins threshold table (T_grow, P_annual, elevation) over 8 crop types. Fertility gates the whole classifier — pixels below `FERTILITY_GATE` (0.08) yield no crop. Vectorised implementation mirrors the scalar `default_crop` reference exactly.

**`minerals.py`** — `place_mineral_deposits()`: Hybrid geology + fBm noise model. Nine mineral types (Iron, Copper, Gold, Gems, Coal, Tin, Rare Metals, Salt, Marble) each have a geological candidate mask derived from boundary type, plate type, volcanic activity, and elevation; independent 3-octave fBm noise fields cluster deposits within those candidates. Returns `(positions (M,3), kinds (M,) int8, abundances (M,) float32)`.

**`rare_bio.py`** — `place_rare_bio_resources()`: Ten rare biological resources (Spices, Tea, Silk, Ivory, Furs, Incense, Dyes, Amber, Rubber, Pearls) placed via narrow ecological criteria (biome, T_annual, P_annual, elevation, distance_to_coast) plus fBm noise. A global per-type cap enforces world-level rarity by keeping only the top-scoring candidates.

**`display.py`**: Color schemes, legends and the `TextureRecipe`s (`FERTILITY_RECIPE`, `FOREST_DENSITY_RECIPE`, `DOMINANT_CROP_RECIPE`) for the resource layers; the region view reruns them on its refined textures. Fertility and forest density use multi-stop gradients; crops and resources use categorical palettes. Registration gating: everything is `--debug` only — the **Fertility**, **Forest Density** and **Dominant Crop** schemes, the **Minerals** and **Rare Resources** overlays (unchecked), and the fertility / forest / crop, mineral and rare-resource hovers.
