# River Generation — As-Built Design

**Module:** `worldgen/hydrology/rivers.py`
**Purpose:** Produce flow-weighted river polylines on a globe whose terrain lives in per-face elevation textures.

This document describes the system as currently implemented. Earlier drafts of the spec (D4 routing, Catmull-Rom smoothing, grid-snapped traces) are preserved only in git history; the algorithms below superseded them after measuring directional-bias artifacts and runaway loops.

---

## Pipeline overview

```
elevation textures
       │
       ▼
[ Priority-Flood + ε-bias ]         # _priority_flood
       │  eff (filled DEM, slopes toward outlets)
       ▼
[ D-infinity flow direction ]       # _compute_dinf_flow + _inter_face_fallback
       │  receiver1/2, frac1/2, flow_angle  (per node)
       ▼
[ Multi-receiver flow accumulation ]# _accumulate_flow
       │  flow_accum (per node)
       ▼
[ Headwater identification ]        # extract_rivers (above-threshold, no above-threshold inflow)
       │
       ▼
[ Sub-pixel gradient walk ]         # _trace_river_subpixel + _gradient_angle_at
       │  one River per headwater
       ▼
[ Laplacian smoothing on sphere ]   # laplacian_smooth_sphere
       │
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

@dataclass
class River:
    segments: list[RiverSegment]
    source:   Optional[RiverNode]
    mouth:    Optional[RiverNode]
    def as_polyline()  -> list[np.ndarray]   # 3D positions, source → mouth
    def flow_rates()   -> list[float]        # one per segment
```

Trace-walk segments hold **off-grid sub-pixel positions** — the `face_id`/`uv` fields on intermediate `RiverNode`s are placeholders (filled with the segment's last known face and (0, 0)). The source and mouth nodes hold valid grid coordinates.

---

## Indexing & adjacency

Every grid cell across the whole mesh has a flat index:

```
node_idx = face * resolution² + row * resolution + col
```

`row` corresponds to the v-axis (north in ~90% of faces), `col` to the u-axis (east).

- **Intra-face**: D8 (8-connected) neighbours derived on the fly in `_neighbors`. Used by Priority-Flood and by the inter-face fallback.
- **Inter-face**: sparse adjacency `self._inter_adj: dict[int, list[int]]` built once in `_stitch_face_boundaries`. For each adjacent face pair the two shared mesh vertices define the polygon edge in 3D; each face's tangent frame projects those vertices into its own UV space, and pixels lying within ½ pixel of that UV segment are matched by parameter `t` along the edge. Matched pixel pairs become bidirectional edges.
- **Tangent-frame data**: `_u_axes`, `_v_axes`, `_u_offsets`, `_v_offsets`, `_u_ranges`, `_v_ranges`, `_tangent_centroids` are cached at init for the trace walk's UV ↔ 3D projections via `_uv_to_3d` and `_3d_to_uv`. The first six come from `mesh.face_tangent_frames()`; the last is read directly from `mesh._tangent_centroids` since the public API doesn't expose it.

---

## Stage 1 — Priority-Flood with imposed gradient

Standard Planchon-Darboux (2001): seed every node at or below `sea_level`, then BFS outward in a min-heap keyed on filled elevation `eff`. The filled value of a newly-popped neighbour is

```
eff[nb] = max(elev[nb], eff[parent] + ε)        # ε = 1e-6
```

The plain `max(elev[nb], eff[parent])` form would leave the interior of any filled depression perfectly flat. The bilinear gradient of a flat field is exactly zero → `_gradient_angle_at` returns NaN → the trace walk has nothing to follow. The ε term gives those filled regions a tiny consistent slope from spill point inward, so the gradient field is non-zero everywhere except at true pits (which don't exist after PF) and so walks routes through filled lakes to their outlets.

`eff` is stored on the generator as `self._eff` for the trace walk.

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

**Boundary fallback.** Pixels on a face edge whose only descending neighbour lives in a different face have no intra-face facet with positive slope. `_inter_face_fallback` scans `_inter_adj` for those nodes and picks the steepest single inter-face neighbour (D8-style), setting `frac1=1`, `frac2=0`. The flow angle is derived by projecting the 3D step direction `(receiver_pos - src_pos)` into the source face's UV basis: `atan2(dv, du)`.

The `flow_angle` field is retained for headwater detection and external inspection. The trace walk does **not** use it; it computes its own continuous flow direction from the bilinear gradient of `eff`.

---

## Stage 3 — Multi-receiver flow accumulation

Process nodes in `eff`-descending order. Each node starts with `flow_accum[k] = 1.0` and forwards its accumulated value to up to two receivers per the D-inf fractions:

```python
flow_accum[r1] += contrib * frac1[k]
flow_accum[r2] += contrib * frac2[k]    # only if r2 ≥ 0
```

Single-receiver nodes (from the inter-face fallback or where one fraction was exactly zero) forward the full contribution to `r1`.

**Important consequence.** Unlike D8 single-receiver accumulation, with D-inf `flow_accum` can *decrease* downstream at a branch point — both receivers get less than the source. Code must not threshold on `flow_accum` while walking down a river; gates are applied only at headwater selection (`flow_accum ≥ min_flow`).

---

## Stage 4 — Headwater detection & extraction

Above-threshold nodes are those with `flow_accum ≥ min_flow`. A node is a **headwater** if no above-threshold node sends it flow with a non-zero fraction:

```python
has_inflow[r1] |= is_above[src] and is_above[r1] and frac1[src] > 0
has_inflow[r2] |= is_above[src] and is_above[r2] and frac2[src] > 0
headwaters = above_idx[~has_inflow[above_idx]]
```

Each headwater seeds one `_trace_river_subpixel` call. Rivers with fewer than 3 segments are dropped (trivial traces from noise).

---

## Stage 5 — Sub-pixel trace walk

The trace operates in **face-local UV coordinates** so we never have to snap to a grid pixel mid-walk. Snapping only happens when the step crosses a face boundary and we need to switch UV frames.

State carried through the loop:
- `current_face`: integer face id
- `(u, v)`: position in current face's UV in `[0, 1]²`
- `(prev_du, prev_dv)`: previous step's unit direction in UV (for momentum)
- `recent_positions`: ring buffer (length `stall_window = 40`) for stall detection

Each iteration:

```
ang ← _gradient_angle_at(current_face, u, v)
if isfinite(ang):
    (du, dv) ← (cos ang, sin ang)
    if has_prev:
        (du, dv) ← normalize((1-α)·grad + α·prev)        # α = 0.5
elif has_prev:
    (du, dv) ← (prev_du, prev_dv)                        # coast on momentum
else:
    break                                                # genuine dead end at start

(new_u, new_v) ← (u + uv_step·du, v + uv_step·dv)       # uv_step = 1/(res-1)

if (new_u, new_v) ∈ [0, 1]²:
    move within current_face
else:
    project attempted 3D position into each adjacent face;
    pick the first whose UV ∈ [0, 1]² (with small slop), clamp, switch face
```

After moving, the closest grid pixel is consulted only to read `elevations[nearest]` for the sea-level stop condition and `flow_accum[nearest]` for the rendered segment width.

### Why negative bilinear gradient instead of bilinear D-inf angle?

Initial drafts walked along a **bilinear interpolation of the D-inf flow_angle vectors** (interpolating `(cos, sin)` to handle wraparound). That vector field has non-zero curl: closed integral curves form around grid cells, and walks got trapped in tiny vortices a fraction of a face wide.

Switching to the negative gradient of the (bilinearly interpolated) scalar field `eff` removed the trap entirely — gradient fields are curl-free by Stokes, so the walk is guaranteed to descend monotonically through `eff`. Combined with the ε-bias in Priority-Flood that gives even filled flats a non-zero gradient, the walk reaches an outlet from anywhere on land that isn't a true ocean pixel.

### Why momentum?

Per-face elevation textures are sampled independently of one another. At a shared face boundary, face A's bilinear sample of the elevation at point X can differ from face B's sample of the same point — the surfaces don't match. Right after crossing into the new face, the new face's gradient can point back into the old face (because its first interior pixel reads as lower than the boundary). Without momentum the walk would ping-pong across the seam for a step or two and `stall_window` would kill it.

Blending 50% prior direction + 50% new gradient lets the walk carry its existing velocity through the boundary band; by step 2 or 3 it has moved far enough into the new face that the local gradient settles down. Momentum also pushes the walk across short flats that escaped the ε-bias (e.g., genuinely identical adjacent pixel values).

### Stall detection

If displacement over the last `stall_window = 40` steps is less than half a face-cell, give up. With ε-bias and momentum this rarely fires in normal terrain; the cap mainly catches cases where momentum is taking the walk in a near-circle.

A hard `max_steps = 4 · resolution²` cap also exists as a final safety. Across-globe rivers are nowhere near that long; hitting the cap means a runaway.

---

## Stage 6 — Laplacian smoothing on the sphere

`laplacian_smooth_sphere(polyline, iterations=8, alpha=0.5)`:

```
for each iteration:
    interior[i] ← interior[i] + α · (mid(neighbours) - interior[i])
    interior[i] ← interior[i] / ‖interior[i]‖           # re-project to unit sphere
```

Endpoints (source, mouth) are pinned. This removes residual sub-pixel zig-zag in the trace, where consecutive steps may differ in angle by a few degrees because each step recomputes the gradient at the new sub-pixel position.

This is approximating (control points move), not interpolating (a la Catmull-Rom). Catmull-Rom was tried first; it preserved the zigzag because it just passed through the noisy control points. Laplacian flattens them.

---

## Tuning knobs (constructor + private)

| Parameter | Default | Effect |
|---|---|---|
| `min_flow` | `50.0` | Headwater threshold. Higher → fewer / larger rivers. With D-inf splitting, intermediate `flow_accum` is lower than D8 would produce; a value that gave dense rivers under D8 will give sparser ones here. |
| `sea_level` | `0.0` | Elevation at/below which nodes are PF seeds and trace stops. |
| `smoothing_iterations` | `8` | Laplacian passes. Higher → smoother, more shortening of meanders. |
| `smoothing_alpha` | `0.5` | Per-iteration pull strength toward neighbour midpoint. |
| `eps` (in `_priority_flood`) | `1e-6` | Imposed gradient on filled regions. Must be small relative to real terrain gradients. |
| `momentum` (in `_trace_river_subpixel`) | `0.5` | Blend weight of previous direction. Higher → straighter rivers, more runaway risk; lower → more dead-ends at face seams. |
| `stall_window` | `40` steps | Loop guard window. |
| `max_steps` | `4 · res²` | Hard runaway cap. |

---

## Current limitations / known follow-ups

- **Per-face texture continuity.** The fundamental cause of momentum and stall handling is that the elevation field is C0-discontinuous at face boundaries. Reconciling boundary elevations (force matching values during PF, or share one global field) would let us drop momentum entirely and could improve trace quality.
- **Bias residual.** Geographic bearing-histogram bias is now ~1.2–1.5× peak/mean (uniform = 1.0), down from D8's ~1.7×. The residual comes from the D-inf clamping behaviour at facet edges (cardinal/diagonal snaps) and from face-boundary kinks; it's no longer visually obvious but still measurable.
- **Runaway rate.** A small fraction of rivers (single-digit % typically, more on rough terrain) still hit `max_steps`. Most appear to be momentum-driven loops in low-relief basins where the ε-gradient is too small to dominate the prior direction.
- **Flat fraction reaching the ocean.** Depending on seed, 50–80% of generated rivers reach `sea_level`. The rest terminate inland (stall detection, max-steps, or NaN gradient at a true flat the ε-bias didn't catch). Whether this is acceptable depends on whether interior endorheic basins are desirable visually.
- **No rainfall weighting.** Each grid cell contributes the same 1.0 to flow. Latitude- or noise-based precipitation weights would multiply the initial `flow_accum` array before stage 3.
