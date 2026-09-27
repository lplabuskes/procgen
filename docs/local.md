# Local Maps

Qt-free package for the finest tier: **each regional hexagon becomes a local
map**, refined from the region exactly the way a region is refined from the
globe. Plan and status: [plans/local_maps.md](plans/local_maps.md). Built so far:
geometry, grid and field refinement (phases 1–2); features, villages and the
viewer are still to come.

## Files
- `localmap/geometry.py` — `LocalGeometry`: the parent hex's texture square, mapped affinely onto `[0, 1]²`
- `localmap/grid.py` — `LocalHexGrid`: `RegionHexGrid` kept to the parent hexagon; `texture_in_map`
- `localmap/upsample.py` — `LocalUpsampleParams`, `build_local_fields`, `local_conservation_report`

## Scale (defaults)

| | Region | Local |
|---|---|---|
| Map covers | ~3,070 km cell | one regional hex, ~286 km across (~53k km²) |
| Hex | ~286 km | ~23 km (`n_cols=16`) |
| Texel | ~4.5 km | ~330–365 m (`tex_size=64`) |
| Hexes | ~250 | ~180 |
| Detail octaves | 2 | 4 |

About 1.4 s per local map on seed 4.

## Geometry (`localmap/geometry.py`)

`LocalGeometry(region_geometry, region_grid, hex_index)` is a `RegionGeometry`
subclass whose `local_to_ecef` / `ecef_to_local` go through the parent region's
`(s, t)` frame: local `(a, b)` ↦ `origin + 2R·(a, b)`, `origin` being the hex
centre minus `R`. That frame extends smoothly past the region cell, so a hex
hanging over the cell edge works, and handedness, `up_direction`,
`local_to_lonlat` and `center_ecef` carry over unchanged. `region_id` is the
parent region's. The parent hexagon is flat-top, circumradius 0.5, centred on
`(0.5, 0.5)`.

## Grid (`localmap/grid.py`)

`LocalHexGrid` is `RegionHexGrid` with the layout filtered to hexes overlapping
the parent hexagon (~2/3 of the square; the corners belong to neighbouring local
maps). `texture_in_map(T)` marks the texels inside the parent hexagon: the
map's **domain** (`RegionFields.domain` = own hex ∩ parent hexagon) is the
aggregation weight, so two neighbouring local maps never count a texel twice.

## Refinement (`localmap/upsample.py`)

`build_local_fields(globe, region_fields, hex_index, params) → RegionFields`
runs `region.upsample.refine_fields` with `RegionSource(region_fields)` as the
parent:

- **Locating.** A point is placed in the region's `(s, t)` frame, owned by the
  regional hex with the nearest centre (a KD-tree over hex centres), and read
  from that hex's texture, bilinearly, through its over-hang.
- **Anchors** are regional texels (`hex·T·T + row·T + col`), stored as
  `parent_anchors`. `anchors` / `faces` / `face_uv` stay **global**, so the
  display, hover and corridor code read a local map as they read a region.
- **Slope** is the region's refined slope; **lakes** use the region's exported
  `filled_height`; **danger** is the region's (already continuous) danger read
  bilinearly and conserved per regional texel.
- **Noise** is seeded `world.seed XOR 0x10CA`, shared by every local map of a
  world, so neighbouring local maps of one region agree where they meet.

`local_conservation_report(local, region)` is `conservation_report` one tier
down: per regional texel inside the parent hexagon, local mean vs regional
value. On seed 4, every field is at float32 noise (elevation ~2e-8 km).

Rural people in a local map match the parent hex's to ~0.03% at `T=64`; the
two tiers count the regional texels on the parent hexagon's rim slightly
differently (a ~1/T effect).

### `LocalUpsampleParams`

A `RegionUpsampleParams` subclass with local defaults: `n_cols=16`,
`slope_reference=0.03`, `slope_floor=0.03`, `max_relief_km=0.6`,
`water_spread_texels=12` (about one regional texel), `biome_smoothing=13`
(about one regional texel, to hide the per-texel conservation blocks), and
`slope_saturation=0.1`. The relief numbers are a first cut and not yet tuned
(seed 4 region 34: ~50 m std of land relief against the parent texel).
