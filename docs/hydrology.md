# Hydrology Stage

See also: `docs/plans/river_generation_spec.md` for the original design.

## Files
- `worldgen/hydrology/builder.py` — stage orchestration + params
- `worldgen/hydrology/rivers.py` — `RiverGenerator` (end-to-end pipeline)
- `worldgen/hydrology/types.py` — dataclasses + `laplacian_smooth_sphere`
- `worldgen/hydrology/routing.py` — Priority-Flood, D-infinity flow, accumulation
- `worldgen/hydrology/stitching.py` — connected-component search
- `worldgen/hydrology/crossings.py` — arc crossing detection, self-crossing removal, inter-river truncation

## `builder.py`

**`HydrologyParams`** (dataclass):
- `river_flow_threshold` (default `200.0`) — minimum upstream accumulation for a river headwater
- `min_lake_pixels` (default `50`) — minimum flooded-component size to keep as a lake
- `pet_baseline_mm_yr` / `pet_per_degree_mm_yr` / `pet_max_mm_yr` — clamped-linear PET model for water-balance lake shrinking (`PET = clamp(baseline + slope·max(0,T), 0, max)`)

**`HydrologyBuilder`**: A pipeline `Stage`. Runs after elevation **and** climate. Consumes `"precipitation"`, `"temperature"`, and optionally `"ocean_mask"` face textures.

`build(ctx)` runs `RiverGenerator` and attaches two new single-channel face textures:
- `"filled_height"` — Priority-Flood filled DEM, rewritten over endorheic basins to reflect equilibrium lake surface and exposed dry bed
- `"lake_mask"` — 1.0 where pixel is inside post-shrinking lake footprint, 0.0 otherwise

Stored attributes: `rivers` (`List[River]`), `lake_polygons` (`List[np.ndarray]`).

Registered features: `"rivers"` (flattened polylines), `"river_objects"` (rich `River` objects consumed by Resources/Cost), `"lakes"`, `"deltas"`.

## `rivers.py` — RiverGenerator

End-to-end pipeline:
1. Priority-Flood fill (Planchon-Darboux)
2. D-infinity flow direction (Tarboton)
3. Precipitation-weighted accumulation
4. Water-balance lake shrinking (endorheic basins sized by PET vs inflow)
5. Flat-region outlet biasing
6. Headwater extraction
7. Sub-pixel gradient trace walks
8. Self-crossing removal
9. Inter-river crossing resolution

Rivers trace continuously through exorheic lakes (`in_lake` flags let the renderer hide those segments); endorheic basins are true sinks.

Outputs: `generate() → list[River]`, `get_lake_polygons()`, `get_filled_height_texture()`, `get_lake_mask_texture()`.

## Design notes

- D-infinity accumulation and sub-pixel gradient traces are used instead of D4 grid steps to avoid N-S/E-W bias from the tangent-frame convention (see `docs/core.md` — tangent frame).
- The `"river_objects"` feature (rich `River` objects) is what downstream stages (Resources, Cost) consume; `"rivers"` is the flattened polyline form for rendering/serialization.
