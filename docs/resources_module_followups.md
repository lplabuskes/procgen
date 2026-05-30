# Resources Module — Deferred Work

The first slice of `worldgen/resources/` ships **fertility** and **forest density**
only. Three categories are intentionally deferred. This note captures the design
decisions reached during scoping so they don't have to be re-litigated when each
slice is picked up.

## 1. Crop suitability (next slice)

- **Output shape**: categorical per-pixel texture, mirroring `biome`. A single
  `"dominant_crop"` `(N, H, W, 1) uint8` channel naming the best-fit crop per
  pixel. Per-face metadata: `dominant_crop` (int index).
- **Model**: simple threshold table (`min/max T_growing`, `P_annual range`,
  `max_elevation_km`, `max_slope_proxy`). First match wins, ordered by crop
  "preference". Mirrors `worldgen/climate/classification.py`.
- **Inputs**: same as fertility (T, P, elevation, biome) plus the freshly-built
  `fertility` texture as a multiplicative gate (low fertility → no crop wins).
- **Why simple thresholds, not GDD/water-balance**: easier to tune, matches the
  granularity of the existing biome classifier. Upgrade only if game mechanics
  demand it.

## 2. Mineral deposits

- **Output shape**: point features on `Globe`, e.g. `globe.add_feature("minerals",
  [(position_3d, kind, abundance), ...])`. Drawn as overlay markers in `MapWidget`
  (new infrastructure — no existing point-feature overlay style other than
  rivers/deltas/volcanoes).
- **Model**: **hybrid** — geology picks seed faces, noise modulates abundance and
  adds intra-region variation.
  - Iron/copper → convergent boundary faces.
  - Gold/gems → volcanic faces (`volcanic_activity > threshold`).
  - Coal → ancient sedimentary basins (need a proxy: low-elevation continental
    faces far from boundaries; revisit when implementing).
  - Tin/rare metals → divergent boundary faces / hot-spot faces.
  - For each kind, run a Worley/Perlin field over candidate faces to cluster
    deposits and assign abundance.
- **Why hybrid, not pure geology or pure noise**: geology gives plausible
  distribution; noise gives the clustering and rarity texture that pure
  geology-driven placement lacks.

## 3. Rare resources (spices, dyes, exotic flora)

- **Output shape**: same point-feature infrastructure as minerals.
- **Model**: biome-gated, noise-modulated. Spices in tropical biomes, certain
  dyes in temperate forests, etc. Reuse the mineral-deposit placement code.
- **Order**: implement after minerals so the point-feature rendering and feature
  registry conventions are already in place.

## Cross-cutting decisions already made

- **No new per-pixel cost-bearing computations**: terrain steepness is per-face
  (elevation range), water proximity is per-face (graph BFS hops), volcanism is
  per-face (graph BFS hops). Pixel slope from `np.gradient(elevation_tex)` is
  dominated by noise at default resolution and should not be used for fertility
  or any future resource term without first low-pass-filtering the input.
- **Biome LUTs are module-level constants**, not exposed in `ResourcesParams`.
  They define the model rather than tune it; expose them as params only if
  per-world tuning becomes a real need.
- **Display schemes registered unconditionally** (not gated on `debug`), since
  resources are world-building features users will care about by default.
- **Diagnostic script** `scripts/measure_resources.py` reports per-biome means
  vs Earth-ish anchors, longitudinally appended to
  `scripts/results/resources_results.json` + `.md`, mirroring
  `measure_zone_biomes.py`. Extend as each slice lands.
