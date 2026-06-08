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
- `river_flow_threshold` (default `35.0`) — minimum upstream accumulation for a river headwater. Applied to the single-receiver **tree accumulation** (the network actually drawn), not the D-inf field; the lower default vs the old D-inf threshold (50) preserves channel density because single-receiver flow concentrates more than D-inf.
- `min_lake_pixels` (default `50`) — minimum flooded-component size to keep as a lake
- `pet_baseline_mm_yr` / `pet_per_degree_mm_yr` / `pet_max_mm_yr` — clamped-linear PET model for water-balance lake shrinking (`PET = clamp(baseline + slope·max(0,T), 0, max)`)
- `delta_min_flow` (default `1000.0`) — minimum mouth flow for a river to be tagged as having a delta
- `delta_min_length_km` (default `500.0`) — minimum river length for delta tagging

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
3. Precipitation-weighted D-infinity accumulation (chooses each node's single primary receiver = the larger-accumulation side at confluences)
4. Water-balance lake shrinking (endorheic basins sized by PET vs inflow)
5. Single-receiver **tree accumulation** along the primary-receiver graph — the network actually drawn
6. Channel thresholding + main-stem (Hack) decomposition: each river is a chain from a channel headwater (a tree leaf) down to the sea, an endorheic sink, or the confluence where it meets a larger stem
7. Receiver-graph routing + Laplacian smoothing of each chain
8. Self-crossing removal
9. Inter-river crossing resolution

`extract_rivers_graph` is the live extractor. The older bilinear-gradient tracer (`extract_rivers` / `_trace_river_subpixel`) is retained but no longer wired into `generate()`.

Rivers route continuously through exorheic lakes (`in_lake` flags let the renderer hide those segments); endorheic basins are true sinks.

Outputs: `generate() → list[River]`, `get_lake_polygons()`, `get_filled_height_texture()`, `get_lake_mask_texture()`.

## Design notes

- Rivers are routed along the single-receiver tree (each node's primary D-infinity receiver), not the bilinear gradient, so the drawn polyline coincides with the accumulation channel and the painted flow is monotonic — stepping up only at *drawn* confluences. D-infinity (multi-receiver) still picks the direction at confluences; a separate single-receiver accumulation along those edges supplies the magnitude and the channel threshold. This replaced a gradient trace whose path was decoupled from the accumulation field (off-channel sampling painted a flat-then-leap staircase, and flow could appear to jump or "drop" between overlapping headwater traces).
- The `"river_objects"` feature (rich `River` objects) is what downstream stages (Resources, Cost) consume; `"rivers"` is the flattened polyline form for rendering/serialization.
