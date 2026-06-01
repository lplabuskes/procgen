# River Generation — As-Built Design

**Modules:** `worldgen/hydrology/` (`types.py`, `routing.py`, `rivers.py`, `crossings.py`, `stitching.py`, `builder.py`)
**Purpose:** Produce flow-weighted river polylines on a globe whose terrain lives in per-face elevation textures.

This document describes the system as currently implemented. Earlier drafts of the spec (D4 routing, Catmull-Rom smoothing, grid-snapped traces) are preserved only in git history; the algorithms below superseded them after measuring directional-bias artifacts and runaway loops.

---

## Pipeline overview

```
elevation textures (+ optional precipitation, temperature, ocean_mask)
       │
       ▼
[ Priority-Flood + ε-bias ]         # priority_flood (routing.py) — numba JIT when available
       │  eff, was_raised
       ▼
[ Lake mask building ]              # _build_lake_mask
       │  _lake_mask_full (filled regions ≥ min_lake_pixels)
       ▼
[ D-infinity flow direction ]       # compute_dinf_flow + inter_face_fallback (routing.py)
       │  receiver1/2, frac1/2, flow_angle  (per node)
       ▼
[ Precipitation-weighted accumulation ] # accumulate_flow (routing.py) — numba JIT when available
       │  flow_accum (per node)
       ▼
[ Water-balance lake shrinking ]    # _shrink_endorheic_lakes
       │  _endorheic_mask, updated _lake_mask, updated _eff
       ▼
[ Lake outlet detection ]           # _build_lake_outlets
       │  _lake_outlet_pos (per-node outlet 3D pos), _flat_mask
       ▼
[ Headwater identification ]        # extract_rivers
       │  headwaters (above-threshold, not in lake, not post-lake)
       ▼
[ Sub-pixel gradient walk ]         # _trace_river_subpixel + _gradient_angle_at
       │  one River per headwater, with per-segment in_lake flags
       ▼
[ Self-crossing removal ]           # resolve_self_crossings (crossings.py)
       ▼
[ Inter-river crossing resolution ] # resolve_crossings (crossings.py)
       ▼
List[River]
```

---

## Data structures

```python
@dataclass
class RiverNode:
    position: np.ndarray          # 3D point on unit sphere, shape (3,)
    elevation: float
    face_id: int
    uv: tuple[float, float]       # face-local UV in [0, 1]²

@dataclass
class RiverSegment:
    start: RiverNode
    end:   RiverNode
    flow_rate: float              # upstream flow accumulation at the start node
    tributary_count: int = 0
    in_lake: bool = False         # True when this segment lies inside a lake mask

@dataclass
class River:
    segments: list[RiverSegment]
    source:   Optional[RiverNode]
    mouth:    Optional[RiverNode]
    def as_polyline()  -> list[np.ndarray]   # 3D positions, source → mouth
    def flow_rates()   -> list[float]        # one per segment
```

Trace-walk segments hold **off-grid sub-pixel positions** — the `face_id`/`uv` fields on intermediate `RiverNode`s are placeholders (filled with the segment's last known face and (0, 0)). The source and mouth nodes hold valid grid coordinates.

`laplacian_smooth_sphere` (in `types.py`) exists for external use but is not called by `generate()`.

---

## Indexing & adjacency

Every grid cell across the whole mesh has a flat index:

```
node_idx = face * resolution² + row * resolution + col
```

`row` corresponds to the v-axis (north in ~90% of faces), `col` to the u-axis (east).

- **Intra-face**: D8 (8-connected) neighbours derived on the fly in `_neighbors`. Used by connected-component searches and the inter-face fallback.
- **Inter-face**: sparse adjacency `self._inter_adj: dict[int, list[int]]` built in `_build_inter_adj_from_pairs`. This uses `mesh.boundary_pixel_pairs(resolution)` — the same seam cache used by the elevation/climate stages — so no separate stitch pass is needed.
- **CSR adjacency**: `_build_neighbor_csr()` combines all D8 intra-face edges (vectorised over all faces) with the inter-face edges into CSR arrays (`_nbr_indptr`, `_nbr_flat`). These are consumed by the numba-JIT Priority-Flood, giving an identical neighbour ordering to `_neighbors`.
- **Tangent-frame data**: `_u_axes`, `_v_axes`, `_u_ranges` and `_vertex_uvs` are cached at init from `mesh.face_tangent_frames()` (indices 0, 1, 4, 6 respectively) for the trace walk's UV ↔ 3D conversions. The walk calls `mesh.face_uv_to_3d` and `mesh.face_3d_to_uv` for position conversions rather than maintaining its own projection helpers.

---

## Stage 1 — Priority-Flood with imposed gradient

Standard Planchon-Darboux (2001): seed every node at or below `sea_level` (or where `ocean_mask` is True, if supplied), then BFS outward in a min-heap keyed on filled elevation `eff`. The filled value of a newly-popped neighbour is

```
eff[nb] = max(elev[nb], eff[parent] + ε)        # ε = 1e-6
```

The plain `max(elev[nb], eff[parent])` form would leave the interior of any filled depression perfectly flat. The bilinear gradient of a flat field is exactly zero → `_gradient_angle_at` returns NaN → the trace walk has nothing to follow. The ε term gives those filled regions a tiny consistent slope from spill point inward, so the gradient field is non-zero everywhere except at true pits (which don't exist after PF) and so walks route through filled lakes to their outlets.

`eff` and the boolean `was_raised` array are stored on the generator as `self._eff` and `self._was_raised`.

**Numba acceleration.** `priority_flood` dispatches to `_priority_flood_njit` when numba is available, else `_priority_flood_py`. Both are bit-identical: the JIT kernel implements a manual binary min-heap with `(priority, index)` lexicographic ordering matching Python `heapq` exactly. Input neighbours come from the precomputed CSR arrays rather than per-call Python lists.

---

## Stage 1b — Lake mask building

After Priority-Flood, `_build_lake_mask` marks flooded land cells. A node is "flooded" if it was raised by PF (`was_raised` True) and is not in the ocean mask. Small components (< `min_lake_pixels` pixels) are dropped. The result is stored as `self._lake_mask_full` (snapshot before water-balance shrinking — used to gate headwater detection) and `self._lake_mask` (the working mask, modified by lake shrinking).

---

## Stage 2 — D-infinity flow direction (Tarboton 1997)

For each grid node, consider the 8 triangular **facets** formed by adjacent cardinal/diagonal neighbour pairs:

```
(E, NE), (N, NE), (N, NW), (W, NW),
(W, SW), (S, SW), (S, SE), (E, SE)
```

For each facet, compute two perpendicular slopes in face-UV space:

```
s1 = (eff[src] - eff[cardinal]) / d        # cardinal-direction slope
s2 = (eff[cardinal] - eff[diagonal]) / d   # orthogonal slope toward diagonal
r  = atan2(s2, s1)                         # gradient direction angle within facet
```

Where `d = 1/(res-1)` is the UV grid spacing. The aspect angle `r` is clamped to `[0, π/4]` (the facet's angular extent):

- `r < 0`: steepest direction is "behind" the cardinal → snap to `r = 0`, slope = `s1`
- `r > π/4`: steepest direction is "past" the diagonal → snap to `r = π/4`, slope = `(eff[src] - eff[diag]) / (d·√2)`
- otherwise: slope = `√(s1² + s2²)`

The steepest-positive-slope facet wins. Flow is split between its cardinal and diagonal vertices with fractions

```
frac_to_diagonal = r / (π/4)
frac_to_cardinal = 1 - frac_to_diagonal
```

Stored per node: `receiver1` (cardinal), `receiver2` (diagonal), `frac1`, `frac2`. The global UV `flow_angle` (CCW from +U) is `base + sign * r` where `base` is the cardinal direction's global angle and `sign` is ±1 depending on which side of the cardinal the diagonal sits.

Vectorisation: the 8 facets are processed one at a time. For each, build flat-index arrays for `(src, cardinal, diagonal)` triples valid in that facet (clipping at face borders), compute slopes element-wise, and update `best_slope`/`best_nb` where the new facet's slope strictly improves.

**Boundary fallback.** Pixels on a face edge whose only descending neighbour lives in a different face have no intra-face facet with positive slope. `inter_face_fallback` scans `_inter_adj` for those nodes and picks the steepest single inter-face neighbour (D8-style), setting `frac1=1`, `frac2=0`. The flow angle is derived by projecting the 3D step direction `(receiver_pos - src_pos)` into the source face's UV basis: `atan2(dv, du)`.

The `flow_angle` field is retained for external inspection. The trace walk does **not** use it; it computes its own continuous flow direction from the bilinear gradient of `eff`.

---

## Stage 3 — Multi-receiver flow accumulation

If `precipitation_textures` are supplied (4 seasons, summed to annual), each node is initialised with a flow contribution proportional to its annual precipitation (normalised to the global mean, so flow units remain comparable regardless of absolute scale). Without precipitation textures each node starts at 1.0.

Process nodes in `eff`-descending order. Each node forwards its accumulated value to up to two receivers per the D-inf fractions:

```python
flow_accum[r1] += contrib * frac1[k]
flow_accum[r2] += contrib * frac2[k]    # only if r2 ≥ 0
```

Single-receiver nodes forward the full contribution to `r1`.

**Important consequence.** Unlike D8 single-receiver accumulation, with D-inf `flow_accum` can *decrease* downstream at a branch point — both receivers get less than the source. Code must not threshold on `flow_accum` while walking down a river; gates are applied only at headwater selection (`flow_accum ≥ min_flow`).

**Numba acceleration.** `accumulate_flow` dispatches to `_accumulate_flow_njit` when numba is available. The argsort order is computed in NumPy before the kernel runs, so the floating-point addition order — and therefore the float32 result — is identical between the JIT and pure-Python paths.

---

## Stage 3b — Water-balance lake shrinking

`_shrink_endorheic_lakes` sizes endorheic basins to their hydrological equilibrium. No-op without both `precipitation_textures` and `temperature_textures`.

For each connected flooded component, compare total inflow (peak `flow_accum` at the spill pixel) against the cumulative PET of the whole component. PET per node uses a clamped-linear temperature model:

```
PET = clamp(pet_baseline_mm_yr + pet_per_degree_mm_yr · max(0, T_annual), 0, pet_max_mm_yr)
```

PET is normalised to the same units as flow accumulation (divided by mean annual precipitation).

- **Exorheic** (inflow ≥ total PET): lake overflows, `_lake_mask` unchanged.
- **Endorheic** (inflow < total PET): pixels are sorted by elevation; the smallest prefix whose cumulative PET equals the inflow forms the equilibrium lake. Higher dry-bed pixels are removed from `_lake_mask`. The `_endorheic_mask` is set on kept pixels (trace walks terminate here). `_eff` is rewritten so the kept lake surface is flat at the equilibrium elevation and the dry bed reverts to raw elevation (so D-inf routing out of the old spill point is suppressed).

---

## Stage 3c — Lake outlet detection

`_build_lake_outlets` covers all Priority-Flood-raised land (any size, not just those large enough to render), recording the outlet position for each component. The outlet is the boundary node with the highest `flow_accum`.

The outlet 3D position is broadcast to every node in the component and stored in `_lake_outlet_pos`. `_flat_mask` marks all raised land pixels. During the trace walk, any node inside a flat region (and not in an endorheic sink) uses this outlet position to steer toward the exit.

---

## Stage 4 — Headwater detection & extraction

Above-threshold nodes are those with `flow_accum ≥ min_flow` **and** not in `_lake_mask_full` (the pre-shrinking snapshot). Additionally, nodes immediately downstream of a lake outlet are excluded from headwater candidacy to avoid duplicate river origins at every outflow point.

A node is a **headwater** if no above-threshold, non-lake node sends it flow with a non-zero fraction:

```python
has_inflow[r1] |= is_above[src] and is_above[r1] and frac1[src] > 0
has_inflow[r2] |= is_above[src] and is_above[r2] and frac2[src] > 0
headwaters = above_idx[~has_inflow[above_idx]]
```

Each headwater seeds one `_trace_river_subpixel` call. Rivers with fewer than 3 segments are dropped (trivial traces from noise). Rivers whose trace reason was `'loop'` are also dropped.

---

## Stage 5 — Sub-pixel trace walk

The trace operates in **face-local UV coordinates** so we never have to snap to a grid pixel mid-walk. Snapping only happens when reading scalar values from the grid (nearest-node lookup for elevation and `flow_accum`) and when switching face frames at a crossing.

State carried through the loop:
- `current_face`: integer face id
- `(u, v)`: position in current face's UV in `[0, 1]²`
- `(prev_du, prev_dv)`: previous step's unit direction in current face's UV (for momentum)
- `recent_positions`: ring buffer (length `stall_window = 40`) for stall detection
- `visited_cells`: set of nearest-grid nodes visited (for loop detection; seam nodes excluded)

Each iteration:

```
ang ← _gradient_angle_at(current_face, u, v)

# --- Outlet pull (flat lake regions) ---
if nearest node is in _flat_mask and not _endorheic_mask:
    compute outlet_dir from _lake_outlet_pos[nearest]
    has_outlet_dir = True

# --- Direction blending ---
if isfinite(ang):
    (du, dv) ← (cos ang, sin ang)
    if has_outlet_dir:
        (du, dv) ← 0.3·(du,dv) + 0.7·outlet_dir    # gradient unreliable on flat surface
    if has_prev:
        (du, dv) ← normalize(0.5·grad_blended + 0.5·prev)
elif has_outlet_dir:                                 # flat with no gradient: steer to outlet
    (du, dv) ← 0.15·prev + 0.85·outlet_dir          # or pure outlet_dir if no prev
elif has_prev:
    (du, dv) ← (prev_du, prev_dv)                   # coast on momentum
else:
    break                                            # genuine dead end at start

(new_u, new_v) ← (u + uv_step·du, v + uv_step·dv)  # uv_step = 1/(res-1)

if (new_u, new_v) ∈ [0, 1]²:
    move within current_face
else:
    # Face crossing — see below.
```

After moving, the nearest grid pixel is consulted to read `ocean_mask`, `_endorheic_mask`, `_flat_mask`, `_lake_mask`, and `flow_accum`.

**Termination conditions:**
- `ocean_mask[nearest]` — normal terminus
- `_endorheic_mask[nearest]` — endorheic lake sink
- Loop detected (`nearest` in `visited_cells`, after `loop_check_start = 4` steps; seam nodes skipped)
- Stall (displacement over last 40 steps < 0.5 face-cells)
- `max_steps = 4 · res²` hard cap

### Face crossing

When the intended UV step exits `[0, 1]²`:

1. Project the attempted 3D position (from `mesh.face_uv_to_3d`) into every adjacent face via `mesh.face_3d_to_uv`.
2. Sort candidates by L1 distance outside `[0, 1]²` (0 = inside).
3. Accept the best if L1 overflow ≤ 0.5 (well above the ~0.14 max for one UV-step).
4. **Triple-point tie-breaking:** if multiple candidates are within `1.5 · uv_step` of the best overflow, project the current momentum into each candidate's UV frame and pick the face the momentum vector points most into (dot product with direction toward face centre).
5. **Momentum reprojection:** on crossing, `_reproject_momentum` converts the prior `(prev_du, prev_dv)` from the old face's UV basis into the new face's basis via the shared 3D tangent axes, then re-normalises.

### Why negative bilinear gradient instead of bilinear D-inf angle?

Initial drafts walked along a **bilinear interpolation of the D-inf flow_angle vectors** (interpolating `(cos, sin)` to handle wraparound). That vector field has non-zero curl: closed integral curves form around grid cells, and walks got trapped in tiny vortices a fraction of a face wide.

Switching to the negative gradient of the (bilinearly interpolated) scalar field `eff` removed the trap entirely — gradient fields are curl-free by Stokes, so the walk is guaranteed to descend monotonically through `eff`. Combined with the ε-bias in Priority-Flood that gives even filled flats a non-zero gradient, the walk reaches an outlet from anywhere on land that isn't a true ocean pixel.

### Why momentum?

Per-face elevation textures are sampled independently of one another. At a shared face boundary, face A's bilinear sample of the elevation at point X can differ from face B's sample of the same point — the surfaces don't match. Right after crossing into the new face, the new face's gradient can point back into the old face (because its first interior pixel reads as lower than the boundary). Without momentum the walk would ping-pong across the seam for a step or two and `stall_window` would kill it.

Blending 50% prior direction + 50% new gradient lets the walk carry its existing velocity through the boundary band; by step 2 or 3 it has moved far enough into the new face that the local gradient settles down. Momentum also pushes the walk across short flats that escaped the ε-bias (e.g., genuinely identical adjacent pixel values).

### Stall detection

If displacement over the last `stall_window = 40` steps is less than half a face-cell, give up. With ε-bias, outlet pull, and momentum this rarely fires in normal terrain; the cap mainly catches cases where momentum is taking the walk in a near-circle.

A hard `max_steps = 4 · resolution²` cap also exists as a final safety. Across-globe rivers are nowhere near that long; hitting the cap means a runaway.

---

## Stage 6 — Crossing resolution

`resolve_self_crossings(rivers)` removes loops introduced by the trace walk. For each river it finds the earliest self-crossing (segment pair whose great-circle arcs intersect, excluding endpoint touches), splices the detour out at the crossing point, and repeats until none remain.

`resolve_crossings(rivers)` resolves inter-river crossings. Rivers are processed in descending flow order so high-flow terminus targets are finalised before lower-flow rivers are truncated. For each crossing pair, the lower-flow river is truncated at its crossing point. Already-truncated rivers are re-checked against current geometry before applying any further truncation.

Both use `crossing_pairs` in `crossings.py`, which vectorises the great-circle intersection test over batches of segment pairs using normal-vector cross products and excludes near-endpoint touches (`t ∈ (0.001, 0.999)`).

---

## in_lake flags

As the trace walk steps, each position is checked against `_lake_mask` (the post-shrinking mask). The resulting per-step `in_lake_flags` are OR'd into each segment's `in_lake` field. The renderer uses this flag to hide in-lake segments (rivers pass through exorheic lakes continuously without visual interruption). Endorheic basins are true sinks — rivers terminate there and no in-lake segment is rendered.

---

## Tuning knobs (constructor + private)

| Parameter | Default (HydrologyParams) | Effect |
|---|---|---|
| `min_flow` | `200.0` | Headwater threshold. Higher → fewer / larger rivers. With D-inf splitting, intermediate `flow_accum` is lower than D8 would produce; a value that gave dense rivers under D8 will give sparser ones here. |
| `sea_level` | `0.0` | Elevation at/below which nodes are PF seeds and trace stops. |
| `min_lake_pixels` | `50` | Minimum flooded-component size to render as a lake. Smaller components are dropped from the lake mask (rivers flow through them). |
| `pet_baseline_mm_yr` | `100.0` | Baseline PET in mm/yr at 0 °C; determines minimum evaporation. |
| `pet_per_degree_mm_yr` | `60.0` | PET slope per °C above zero. |
| `pet_max_mm_yr` | `2500.0` | PET cap in mm/yr. |
| `delta_min_flow` | `1000.0` | Minimum mouth flow for a river to deposit a delta marker. |
| `delta_min_length_km` | `500.0` | Minimum river length (km) for delta eligibility. |
| `eps` (in `priority_flood`) | `1e-6` | Imposed gradient on filled regions. Must be small relative to real terrain gradients. |
| `momentum` (in `_trace_river_subpixel`) | `0.5` | Blend weight of previous direction. Higher → straighter rivers, more runaway risk; lower → more dead-ends at face seams. |
| `lake_bias` | `0.7` (gradient present) / `0.85` (flat) | Pull weight toward lake outlet when traversing a flat region. Higher → rivers exit lakes more directly. |
| `stall_window` | `40` steps | Loop guard window. |
| `max_steps` | `4 · res²` | Hard runaway cap. |
| `loop_check_start` | `4` steps | Steps before loop detection activates (avoids false positives at the headwater). |

---

## Current limitations / known follow-ups

- **Per-face texture continuity.** The fundamental cause of momentum and stall handling is that the elevation field is C0-discontinuous at face boundaries. Reconciling boundary elevations (force matching values during PF, or share one global field) would let us drop momentum entirely and could improve trace quality.
- **Bias residual.** Geographic bearing-histogram bias is now ~1.2–1.5× peak/mean (uniform = 1.0), down from D8's ~1.7×. The residual comes from the D-inf clamping behaviour at facet edges (cardinal/diagonal snaps) and from face-boundary kinks; it's no longer visually obvious but still measurable.
- **Runaway rate.** A small fraction of rivers (single-digit % typically, more on rough terrain) still hit `max_steps`. Most appear to be momentum-driven loops in low-relief basins where the ε-gradient is too small to dominate the prior direction.
- **Flat fraction reaching the ocean.** Depending on seed, 50–80% of generated rivers reach `sea_level`. The rest terminate inland (stall detection, max-steps, endorheic sink, or NaN gradient at a true flat the ε-bias didn't catch).
- **Lake outlet walking.** The outlet-pull heuristic steers traces through lakes correctly in most cases, but highly elongated or multi-lobed lakes can still produce awkward exits if the outlet is in a reentrant bay relative to the entry point.
