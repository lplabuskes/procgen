# Wetland Stage

Where water stands on the ground: a per-texel **wetness score** in [0, 1], and the
**Wetland** biome (index 9, `WETLAND_BIOME`) stamped where it is high. The globe
marks whole ~14 km texels (Everglades, Pantanal, Sudd, boreal peatland); the
region and local tiers refine the score and mark at lower thresholds (smaller fens,
riparian marsh). Delta marshes are the wetland stage's too (the delta stage has no
marsh of its own).

## Files
- `worldgen/wetland/builder.py` — `WetlandParams`, `WetlandBuilder`, `wetland_score`, `stamp_wetland`, `in_polygon_fraction`
- `worldgen/wetland/score.py` — the terms: `drainage_targets`, `flood_height_above`, `flood_term`, `rain_term`, `flatness`, `combine`
- `worldgen/wetland/channels.py` — `WETLAND` (texture), `WETLAND_FRACTION` (per face)
- `worldgen/wetland/display.py` — `WETLAND_RECIPE`, `wetland_color`, `wetland_legend_entries`
- `region/wetlands.py` — tier refinement: `refine_wetland`, `stamp`, `add_riparian_wetland`
- `scripts/measure_wetlands.py` — share of land, per climate biome, rain-fed share, largest complexes

## Place in the pipeline

`… → delta → climate lake feedback → wetland → coast → resources → …`. After the
lake feedback (which reclassifies biomes; wetland is stamped onto its result) on the
final relief, water and streams (deltas included). Shares hydrology's PET params
(`GlobePipeline` passes `HydrologyParams`), so lakes and wetlands see one water balance.

## Score

`score = flatness × (1 − (1 − rain) · (1 − flood))`, 0 on water.

- **flatness** `= 1 / (1 + (slope / slope_scale)²)` (`slope_scale_m_per_km` 0.4;
  global slope from `worldgen.resources.terrain.global_slope`).
- **rain** `= smoothstep(rain_low, rain_high, P / PET)` (4, 12). PET is hydrology's
  `clamp(100 + 60·max(0, T), 0, 2500)` mm/yr — it bottoms out in the cold, so boreal
  ratios run high and the thresholds sit well above 1. Rain-fed bogs and fens; ~60%
  of global wetland, mostly taiga/tundra lowland.
- **flood** — *height above nearest drainage*. Each texel follows steepest descent on
  the depression-filled surface (hydrology's `priority_flood`, seeded at all water)
  to the first stream-network channel or water texel (`drainage_targets`, pointer
  doubling). That drainage floods to its ground + `flood_height_m · (flow /
  REFERENCE_FLOW) ** flood_exponent` (4 m, 1000, 0.45 → ~4 m at flow 1k, ~23 m at
  50k), or for a lake / the sea its surface + `shore_flood_height_m` (3 m).
  `flood = ramp(−bank_m, full_depth_m, flood level − ground)` (3 m, 3 m). Flat
  ground draining to a big river sits barely above it and floods wide; valley sides
  rise fast and stay dry; the bank just above gets a sub-threshold share — what the
  finer tiers' lower thresholds pick up.

  (A spreading flood surface was tried first: it runs downhill from every river and
  drowned 96% of land. Drainage-based HAND keeps each texel to its own river.)

**Stamp**: Wetland where `score ≥ threshold` (0.5) on land that is not frozen
(warmest season < 0 °C).

Defaults, seeds 4/31/35: 4.6 / 4.3 / 5.2% of land; rain-fed 63 / 53 / 60%; taiga
87 / 59 / 23%, tundra 13–16%, temperate 3–4%, tropical 4–5%, savannah 3%. Largest
complex is the flattest cold lowland (seed 4: ~3.4 M km², 3× West Siberia); river
wetlands are dendritic strips and lake-margin swamps. Earth: ~6–9% of land at fine
scale. ~1.5 s per build.

## Outputs

| Output | Kind | What |
|---|---|---|
| `wetland` | texture `(N, H, W, 1)` float16 | the score |
| `biome` | texture (edited) | `WETLAND_BIOME` where stamped |
| `wetland_fraction` | per-face meta | in-polygon share of the face's texels in Wetland |

The face's `biome` stays its **climate** biome (the lake feedback's mode); per-face
consumers read wetland from `wetland_fraction`, per-texel ones from the biome texture.

Consumers:
- **Fertility** (`resources`, `apply_wetland_fertility`, after the delta raise): wetland
  ground is never short of water but is waterlogged — Wetland texels are raised toward
  `wetland_fertility` (0.8) × the growing-season temperature bell (rain no longer
  limits them), then × `(1 − wetland_penalty)` (0.4). Arid wetlands are river-fed
  oases (~0.48 vs desert ~0.03: Inner Niger Delta, Mesopotamian marshes); humid ones
  already above the target only lose the penalty; cold bogs stay poor. Region/local
  maps inherit it through fertility conservation (their own finer wetlands don't
  change fertility).
- **Land cost**: × `(1 + wetland_cost_strength · wetland_fraction)` (3.0).
- **Danger**: **Lizardfolk Fen** on warm faces (not Icecap/Tundra/Taiga) with
  `wetland_fraction ≥ 0.2`.
- **Region/local roads**: per-texel `BIOME_TRAVERSAL_COST[Wetland]` (4.0).
- **Biome tables**: a Wetland row in `BIOME_FERTILITY_MULT`, `BIOME_FOREST_BASELINE`,
  `BIOME_PRECIP_REFERENCE_MM` (unused per face, kept for table size), and the
  biome palette (teal).

## Region and local tiers (`region/wetlands.py`)

Not conserved (a classification, like `biome`). In `refine_fields`, after the biome
reclassification:

- **relief**: `score = parent score (bilinear) + wetland_relief_gain · tanh((parent
  elevation − elevation) / wetland_relief_scale_m)` — hollows wetter, rises drier;
- **stamp** at the tier's `wetland_threshold`.

After the tier's rivers are routed (`RegionView._refine_features`,
`build_local_map`, and the debug/measure scripts), `add_riparian_wetland` adds
`wetland_riparian_gain · (1 − d / width) · flatness` along every path and stream,
`width = max(wetland_riparian_width_km · (flow / ref) ** exponent, min_width_texels ·
texel)`, flatness over `wetland_riparian_slope_m_per_km`, and re-stamps. It keeps
the pre-riparian score/biome under `_pre_riparian_*` texture keys, so re-routing
rivers over the same fields is idempotent.

| | threshold | relief gain / scale | riparian width at flow 1000 | riparian slope scale |
|---|---|---|---|---|
| globe | 0.5 | — | — | — |
| region (~4 km) | 0.4 | 0.1 / 10 m | 5 km | 2 m/km |
| local (~0.36 km) | 0.3 | 0.08 / 3 m | 1.5 km | 8 m/km |

The relief gain stays ≤ the gap between the parent's threshold and the tier's
(unit-tested), so the parent's wetlands stay wetland bar bilinear softening at their
edges; the smaller ones appear around them. Riparian gain is 0.3, min width 1.5
texels, exponent 0.3 on both tiers.

## Display

The Wetland biome shows in the **Biome** scheme (all tiers). `--debug`: **Wetland
Score** scheme (texture recipe, refined on region/local maps) and the hover
`Wetland: 35% of face`.

## Persistence

`wetland` and `biome` textures go with the mesh; `wetland_fraction` is re-derived
from the biome texture on load.
