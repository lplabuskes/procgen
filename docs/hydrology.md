# Hydrology Stage

See also: `docs/plans/river_generation_spec.md` for the original design.

## Files
- `worldgen/hydrology/builder.py` — stage orchestration + params
- `worldgen/hydrology/rivers.py` — `RiverGenerator` (end-to-end pipeline)
- `worldgen/hydrology/types.py` — dataclasses + `laplacian_smooth_sphere`
- `worldgen/hydrology/routing.py` — Priority-Flood, D-infinity flow, accumulation
- `worldgen/hydrology/stitching.py` — connected-component search
- `worldgen/hydrology/crossings.py` — arc crossing detection, self-crossing removal

## `builder.py`

**`HydrologyParams`** (dataclass):
- `stream_flow_threshold` (default `100.0`) — the same tree accumulation cut much lower: the saved stream network (towns favour its texels; a region draws it as its tributaries)
- `river_flow_threshold` (default `500.0`) — minimum upstream accumulation for a river headwater; a user-tunable dial for how dense the river network is (higher = fewer, larger rivers). Applied to the single-receiver **tree accumulation** (the network actually drawn), not the multi-receiver D-inf field — single-receiver flow concentrates more, so this threshold's scale differs from a D-inf threshold.
- `min_lake_pixels` (default `50`) — minimum flooded-component size to keep as a lake
- `pet_baseline_mm_yr` / `pet_per_degree_mm_yr` / `pet_max_mm_yr` — clamped-linear PET model for water-balance lake shrinking (`PET = clamp(baseline + slope·max(0,T), 0, max)`)
- `delta_min_flow` (default `1000.0`) — minimum mouth flow for a river to be tagged as having a delta
- `delta_min_length_km` (default `500.0`) — minimum river length for delta tagging

**`HydrologyBuilder`**: A pipeline `Stage`. Runs after elevation **and** climate. Consumes `"precipitation"`, `"temperature"`, and optionally `"ocean_mask"` face textures.

`build(ctx)` runs `RiverGenerator` and attaches two new single-channel face textures:
- `"filled_height"` — Priority-Flood filled DEM, rewritten over endorheic basins to reflect equilibrium lake surface and exposed dry bed
- `"lake_mask"` — 1.0 where pixel is inside post-shrinking lake footprint, 0.0 otherwise

Stored attributes: `rivers` (`List[River]`), `lake_polygons` (`List[np.ndarray]`), `streams` (`StreamNetwork`).

Registered features: `"rivers"` (flattened polylines), `"river_objects"` (rich `River` objects consumed by Resources/Cost), `"streams"` (the `StreamNetwork`), `"lakes"`, `"deltas"`.

**`StreamNetwork`** (`types.py`, from `RiverGenerator.stream_network(threshold)`): the single-receiver accumulation tree the rivers are cut from, kept down to `stream_flow_threshold` — every channel texel plus the sea / endorheic texels they drain into, as sparse parallel arrays: flat texel id `nodes` (`face·H·W + row·W + col`), `positions`, `receiver` (index into the arrays, -1 at an end), `flow` (tree accumulation, river flow units), `channel`, `stop` (sea or endorheic: a path ends there). Rivers are exactly its part above `river_flow_threshold`. Saved with the world (`stream_*` arrays; ~27k nodes on seed 4). Consumers: the town stage's stream bonus, and the region tier's rivers (`region/rivers.py`).

## `rivers.py` — RiverGenerator

End-to-end pipeline:
1. Priority-Flood fill (Planchon-Darboux)
2. D-infinity flow direction (Tarboton)
3. Precipitation-weighted D-infinity accumulation (chooses each node's single primary receiver = the larger-accumulation side at confluences)
4. Water-balance lake shrinking (endorheic basins sized by PET vs inflow)
5. Single-receiver **tree accumulation** along the primary-receiver graph — the network actually drawn
6. Channel thresholding + main-stem (Hack) decomposition: each river is a chain from a channel headwater (a tree leaf) down to the sea, an endorheic sink, or the confluence where it meets a larger stem
7. Receiver-graph routing + Laplacian smoothing of each chain
8. Self-crossing removal (loops a single river's own smoothing can introduce)

`extract_rivers_graph` is the live extractor; the earlier bilinear-gradient tracer has been removed.

`main_stem_paths(receiver, accum, is_channel, is_stop, min_nodes)` is that decomposition as a module-level function over any single-receiver tree; the region tier runs it over the stream network (`region/rivers.py`).

Rivers route continuously through exorheic lakes (`in_lake` flags let the renderer hide those segments); endorheic basins are true sinks.

Outputs: `generate() → list[River]`, `get_lake_polygons()`, `get_filled_height_texture()`, `get_lake_mask_texture()`.

## Design notes

- Rivers are routed along the single-receiver tree (each node's primary D-infinity receiver), not the bilinear gradient, so the drawn polyline coincides with the accumulation channel and the painted flow is monotonic — stepping up only at *drawn* confluences. D-infinity (multi-receiver) still picks the direction at confluences; a separate single-receiver accumulation along those edges supplies the magnitude and the channel threshold. This replaced a gradient trace whose path was decoupled from the accumulation field (off-channel sampling painted a flat-then-leap staircase, and flow could appear to jump or "drop" between overlapping headwater traces).
- Distinct rivers' arcs occasionally cross (~25% of rivers touch one crossing on a production mesh): tributaries route down the accumulation tree and each river is smoothed independently, so two smoothed arcs can intersect even though they share no grid node. The earlier inter-river truncation that removed these was dropped — it truncated the lower-flow river at each crossing, chopping long modest-flow main stems wherever a short high-flow river crossed (backwards for "great rivers"). `resolve_self_crossings` still removes loops within a single river. `test_inter_river_crossings_within_baseline` (`worldgen/hydrology/test/test_rivers.py`) is now a per-seed ratchet that only fails if crossings get *worse*.
- The `"river_objects"` feature (rich `River` objects) is what downstream stages (Resources, Cost) consume; `"rivers"` is the flattened polyline form for rendering/serialization.
