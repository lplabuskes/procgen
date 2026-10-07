# Hydrology Stage

Original design: `docs/plans/river_generation_spec.md`.

## Files
- `worldgen/hydrology/builder.py` — stage + params
- `worldgen/hydrology/rivers.py` — `RiverGenerator`
- `worldgen/hydrology/types.py` — dataclasses, `laplacian_smooth_sphere`
- `worldgen/hydrology/routing.py` — Priority-Flood, D-infinity, accumulation
- `worldgen/hydrology/stitching.py` — connected components
- `worldgen/hydrology/crossings.py` — arc crossings, self-crossing removal
- `worldgen/hydrology/discharge.py` — flow → discharge (m³/s) → channel width (local-map river widths)
- `worldgen/hydrology/channels.py` — `LAKE_MASK`, `RIVER_OBJECTS`, `RIVERS`, `STREAMS`

## `builder.py`

**`HydrologyParams`**:
- `river_flow_threshold` (500) — user dial for river density (higher = fewer, larger). Applied to the single-receiver **tree accumulation** (the drawn network), not D-inf, so its scale differs from a D-inf threshold. Units: upstream texels weighted by precipitation / world mean.
- `stream_flow_threshold` (100) — same accumulation, cut lower: the saved stream network (town bonus; region tributaries).
- `min_lake_pixels` (50) — min flooded component kept as a lake.
- `pet_baseline_mm_yr` / `pet_per_degree_mm_yr` / `pet_max_mm_yr` — `PET = clamp(baseline + slope·max(0,T), 0, max)` for lake shrinking.

**`HydrologyBuilder`** — after elevation **and** climate; reads `"precipitation"`,
`"temperature"`, optionally `"ocean_mask"`. Adds textures:
- `"filled_height"` (float32; saved to 0.1 m as a difference from elevation) — Priority-Flood DEM, rewritten over endorheic basins to the equilibrium lake surface / dry bed; region tier floods lakes to it
- `"lake_mask"` — 1.0 inside post-shrinking lake footprints

The delta stage ([delta.md](delta.md)) extends rivers and streams in place, so
`serialize` re-flattens the `River` objects at save time.

Attributes: `rivers` (`List[River]`; flattened tuples after `load`),
`lake_polygons`, `streams` (`StreamNetwork`).

Features: `"rivers"` (flattened `(positions, flows, in_lake)` polylines),
`"river_objects"` (rich `River`s for Resources/Cost; build only, not restored on
load), `"streams"`, `"lakes"`. Display: **Rivers** overlay.

**`StreamNetwork`** (`types.py`, `RiverGenerator.stream_network(threshold)`): the
single-receiver tree down to `stream_flow_threshold` — channel texels plus the
sea/endorheic texels they drain into, as sparse parallel arrays: `nodes` (flat
`face·H·W + row·W + col`), `positions`, `receiver` (-1 at an end), `flow` (tree
accumulation), `channel`, `stop` (sea/endorheic end). Rivers = its part above
`river_flow_threshold`. Saved (`stream_*`; ~27k nodes on seed 4). Used by the
town stream bonus and region rivers (`region/rivers.py`).

## `rivers.py` — RiverGenerator

1. Priority-Flood fill (Planchon-Darboux)
2. D-infinity direction (Tarboton)
3. Precipitation-weighted D-inf accumulation (primary receiver = larger-accumulation side)
4. Water-balance lake shrinking (endorheic basins sized by PET vs inflow)
5. Single-receiver **tree accumulation** along primary receivers — the drawn network
6. Channel threshold + main-stem (Hack) decomposition: headwater leaf → sea / endorheic sink / confluence with a larger stem
7. Receiver-graph routing + Laplacian smoothing (8 iterations); flow = running max of tree accumulation (never decreases)
8. Self-crossing removal

`extract_rivers_graph` is the extractor. `main_stem_paths(receiver, accum,
is_channel, is_stop, min_nodes)` is step 6 over any single-receiver tree (the
region tier runs it on the stream network). Rivers route through exorheic lakes
(`in_lake` hides those segments); endorheic basins are sinks.

API: `generate() → list[River]`, `get_lake_polygons()`,
`get_filled_height_texture()`, `get_lake_mask_texture()`.

## Design notes

- **Routing along the single-receiver tree** (not the bilinear gradient) keeps
  the drawn polyline on the accumulation channel and painted flow monotonic,
  stepping up only at drawn confluences. D-inf picks direction; single-receiver
  accumulation supplies magnitude and threshold. (The old gradient trace sampled
  off-channel — flat-then-leap staircases, flow jumping/dropping between
  overlapping traces; removed.)
- **Inter-river crossings are accepted** (~25% of rivers touch one): each river
  is smoothed independently, so arcs can intersect without sharing a node. The
  old inter-river truncation was dropped — it chopped long modest-flow main stems
  wherever a short high-flow river crossed. `resolve_self_crossings` still removes
  a river's own loops. `test_inter_river_crossings_within_baseline`
  (`worldgen/hydrology/test/test_rivers.py`) is a per-seed ratchet failing only
  if crossings get worse.
- `"river_objects"` is what downstream stages consume; `"rivers"` is the
  render/serialize form.
