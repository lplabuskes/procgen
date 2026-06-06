# Settlement Generation (`realmgen`) — Design Document

## Purpose & Scope

This module takes a fully-generated physical world and produces a coherent human geography layer: settlements of varying scale, trade networks, polities, and military positions. It is strictly downstream of physical world generation and treats the physical world as read-only — it may *attach* new features and metadata to the `Globe`, but never rewrites elevation, climate, hydrology, or resource data.

Settlements, routes, polities, and military positions are the primary outputs. Population is derived, not given — it emerges from carrying capacity and trade wealth.

The module name is `realmgen`, a top-level package sibling to `worldgen`. This document uses "realm layer" and "human geography layer" interchangeably for its output.

---

## Codebase Grounding

The original draft of this doc assumed a 2D raster tile grid. **There is no tile grid.** The world is a dual `PolygonMesh` (`worldgen/mesh/mymesh.py`) — 12 pentagons plus many hexagons on the unit sphere — and every existing stage (geology → climate → hydrology → resources) reasons over it in two registers:

- **Per-face data.** Each face carries a `face_metadata` dict and a unit-sphere `face_centroid`. Faces are the natural unit for graph algorithms. The precomputed `face_adjacency: List[List[int]]` is the movement graph; `face_areas` weights hinterland and carrying-capacity sums.
- **Per-pixel data.** Each face also carries `(N, H, W, C)` float32 `face_textures` — a small local grid sampled across the face's tangent plane. These hold continuous fields (elevation, climate, masks) and are how smooth gradients are rendered.

**The realm layer follows the same two-register pattern (confirmed design decision):** routing, flow accumulation, hinterland assignment, and polity growth run on the **face-adjacency graph**; per-face suitability / situational-value / danger scalars are **rasterized to per-pixel textures for display**, exactly the way `worldgen/resources/fertility.py` computes a per-face value and interpolates it across pixels via vertex averaging. There is direct precedent for every primitive realmgen needs:

| realmgen needs | Existing precedent |
| --- | --- |
| Least-cost paths over a face graph | Generalize `face_hops_from_seeds` (`worldgen/resources/terrain.py`, unweighted multi-source BFS) to weighted multi-source Dijkstra |
| Flow accumulation along least-cost paths | Hydrology's multi-receiver D∞ accumulation (`worldgen/hydrology/routing.py`); the "trade as flow accumulation" analogy is literal |
| Proximity falloff (danger edges, water bonus) | `face_hops_from_seeds` + `exp(-hops · decay)`, used throughout resources |
| Point outputs mapped onto faces | `minerals` / `rare_bio` features: `(positions (M,3), kinds, abundances)`, mapped to faces via a centroid lookup in `GlobeBuilder.apply_display` |
| Vector polyline outputs | `rivers` feature: list of `(positions (M,3), flows, in_lake)` |
| Ruggedness in place of slope | `face_elevation_range` — **see the slope caveat below** |

### Slope is unreliable here — use ruggedness

The physical pipeline deliberately avoids per-pixel slope. At the default texture size the pixel-spacing gradient is dominated by the noise overlay rather than real terrain (documented in `docs/resources_generation_spec.md` and `worldgen/resources/terrain.py`). Anywhere this design says "slope" — biome traversal cost, defensibility, local relief — it means **`face_elevation_range(elevation_tex)`**, the per-face pixel elevation span (km), and/or the centroid-elevation difference between adjacent faces. Do not reintroduce per-pixel slope.

### The Globe contract (concrete)

`Globe` (`worldgen/globe/globe.py`) holds `mesh: PolygonMesh` plus a generic feature registry (`add_feature` / `get_feature` / `has_feature`). Everything realmgen reads is already there:

**Per-face metadata** (`mesh.face_metadata[i]`): `plate_id`, `plate_type` (`'oceanic'|'continental'`), `boundary_type` (`'interior'|'convergent'|'divergent'|'transform'`), `surface_speed`, `volcanic_activity`, `biome` (uint8 index into the 9-class palette), `temperature_annual` / `temperature_warmest` / `temperature_coldest` (°C), `precipitation_annual` (mm/yr) / `precipitation_seasonality`, `humidity_annual`, `distance_to_coast`, `fertility_mean`, `forest_density_mean`, `dominant_crop`.

**Per-pixel textures** (`mesh.face_textures[name]`, each `(N, H, W, C)`): `elevation` (1ch, km), `temperature` / `precipitation` / `humidity` (K seasonal channels), `biome` (1ch uint8), `ocean_mask` (1ch), `lake_mask` (1ch), `filled_height` (1ch, Priority-Flood DEM), `fertility` (1ch), `forest_density` (1ch), `dominant_crop` (1ch uint8). Plus `*_rgba` display variants.

**Vector features** (`globe.get_feature(name)`): `rivers` (list of `(positions, per-segment flow, in_lake)`), `lakes` (list of polygons), `deltas`, `volcanoes`, `minerals`, `rare_bio` (both `(positions, kinds, abundances)` 3-tuples), `winds`, `currents`.

**Geometry helpers** (`PolygonMesh`): `face_adjacency`, `face_centroids`, `face_areas`, `face_texture_positions(h, w)` (per-pixel 3D positions — the starting point for any per-pixel geographic computation), `face_tangent_frames()` (project an arbitrary 3D point into a face's UV space — needed to drop a settlement/resource point onto its pixel).

**Notably absent:** there is no coastline vector feature and no port/harbor data. Coast is derived from `ocean_mask` (the land/ocean boundary is where an `ocean_mask`-false face is adjacent to an `ocean_mask`-true face) and the `distance_to_coast` metadata. Harbor quality must be computed by realmgen from shoreline geometry (see Static Situational Value).

---

## Module Boundary

**Inputs**
- `Globe` — the physical world contract above, treated as read-only.
- **Base cost surfaces** — the physical *land* and *sea* traversal-cost layers (confirmed decision: these are built in **`worldgen`**, not realmgen, because they are derived purely from physical Globe data and are reusable). They are attached to the Globe as face-texture / metadata channels (e.g. `land_cost`, `sea_cost`) plus a `strait` chokepoint tag. realmgen consumes them and never recomputes physical cost.
- `DangerField` — a per-face scalar `(N,)` array marking faces too dangerous to settle. realmgen-owned (RPG-specific). Has a sensible procedural default derived from physical data (deep wilderness, volcanic zones via `volcanic_activity`, extreme terrain via `face_elevation_range`) but is designed to be overridden per-face by a designer. This is the primary RPG hook and the only externally-supplied input beyond the Globe and its base cost surfaces.

**Outputs** (all registered as `Globe` features by `RealmBuilder`, mirroring how hydrology/resources register theirs; per-face summaries also written to `face_metadata` for color-scheme rendering):
- `settlements` — each with a position (3D + `face_id`), continuous economic/military scores, a display tier, and a hinterland assignment
- `trade_routes` — polylines tagged with the goods that drive them and their flow volume
- `polities` — each with a capital, a `polity_id` territory label per face, and a power score
- `forts` — border forts and frontier outposts at contested or dangerous edges

---

## Core Concepts

**Two cost surfaces, not one.** Land and sea are qualitatively different mediums used by different actors. A single cost surface cannot serve both. The *land surface* treats ocean faces as impassable and encodes overland difficulty — biome, ruggedness, river crossings, danger. The *sea surface* inverts this: ocean faces are the cheap medium, land is impassable, coastal-water faces (low `distance_to_coast` on the ocean side) are cheapest of all. Each subsystem uses whichever surface matches its movement model. Polities grow over land; maritime trade flows over sea. Straits appear as near-free crossings on the sea surface but are flagged as chokepoints, making them strategically equivalent to mountain passes on land.

The **physical base** of both surfaces lives in worldgen (reusable, purely geographic). realmgen adds only the danger-derived premiums on top, so the boundary stays clean: worldgen answers "how hard is this terrain to cross," realmgen answers "how dangerous is it to cross right now."

**Suitability vs. situational value.** A face's *suitability* is how good it is to live on locally — fresh-water access (river/lake proximity via hop-decay, `distance_to_coast`), fertility (`fertility_mean`), defensibility (local relief from `face_elevation_range` and centroid-elevation gaps), climate habitability (`temperature_*`, `precipitation_annual`). Its *situational value* is how strategically important it is in the network — a harbor, a river confluence, a mountain pass, a trade-artery junction. Great cities need both; forts need situational value but little suitability; farming villages need suitability but little situational value.

**Trade as flow accumulation.** Rare goods have sources (the `minerals` and `rare_bio` features, plus staple production from `fertility_mean` / `dominant_crop`) and diffuse demand across all settlements. Routes form along least-cost paths from sources to consumers on the appropriate surface. Path usage is accumulated onto face-graph edges, producing a flow network directly analogous to hydrology's flow accumulation — the same mechanism, driven by economic gravity instead of precipitation. High-flow junctions become hubs; hubs attract population; population increases demand; demand reinforces routes.

**Population as output.** Settlement population is not seeded — it emerges from carrying capacity (hinterland fertility summed with `face_areas` weighting) plus trade wealth (position on the flow network). The rank-size distribution is therefore a validation signal, not a parameter.

**Danger as the universal suppressor.** Whether a region is uninhabitable due to monster activity, cursed land, active volcanism, or a war zone, all of it flows through the `DangerField`. High danger disqualifies settlement, inflates travel cost on both surfaces, and generates frontier outposts at its edges. Sea-specific hazards (piracy, sea monsters) apply as a danger premium on the sea surface only. Adding a new kind of hazard requires no changes to settlement logic — only the field.

---

## Subsystem Decomposition

**1. Cost Surfaces** *(base built in worldgen; danger overlay in realmgen)*

*Physical base (worldgen).* A new worldgen step attaches two per-face cost layers to the Globe:
- *Land surface:* per-edge or per-face traversal cost combining a tunable biome cost table, ruggedness (`face_elevation_range`), and river-crossing cost (cheap along a river's flow direction, expensive across it — flow direction recovered from `rivers` segment ordering and `river_traversed_faces`). The along-valley discount is tapered by river navigability (ruggedness-driven), so rugged gorges give no discount while the ford penalty stays. All open water — **ocean ∪ lake** (`ocean_mask`, `lake_mask`) — is impassable.
- *Sea surface:* sailable water is the cheap medium; land is impassable. **Lakes are sailable like the ocean**, and **navigable rivers** (enough flow, flat enough — up to the *fall line*, where rising ruggedness turns the river to rapids) extend the sailable network inland so a river connects to the sea directly. Coastal-water faces are discounted further (lakes and navigable rivers read as coastal). Open-ocean cost scales with distance from coast. Narrow passages below a width threshold are tagged `strait` (chokepoints), near-free to cross.

The biome cost table is a worldgen-level tunable and must not require touching realmgen.

*Danger overlay (realmgen).* realmgen reads the two base surfaces and adds the `DangerField` premium — a multiplicative or additive inflation on land cost everywhere, plus a sea-only piracy/monster premium on the sea surface. Everything downstream consumes the combined surface. Each subsystem that does spatial reasoning declares which surface it uses — there is no default.

Ports and navigable-river faces are the connection points between the two surfaces: a navigable-river face is *amphibious* (passable on both surfaces), so a trade route can move between land and sea wherever a navigable river reaches — not only at coastal ports. The fall line (the head of navigation) is therefore a natural transshipment point and high-situational-value site. No other surface-switching is permitted.

**2. Suitability Scorer**
Scores each land face for settlement potential from Globe data: fresh-water proximity (hop-decay from `river_traversed_faces` ∪ `lake_containing_faces`, plus coastal access from `distance_to_coast`), `fertility_mean`, defensibility (local relief), climate habitability (`temperature_warmest` / `temperature_coldest` / `precipitation_annual`), and resource access (proximity to `minerals` / `rare_bio` deposits). The `DangerField` applies here as a hard disqualifier above a threshold. Produces a continuous per-face suitability scalar, rasterized to a `suitability` texture for inspection.

**3. Static Situational Value**
Scores faces for terrain-derived strategic importance, all computable before any settlement is placed:
- *Harbor quality* — shoreline concavity (fraction of a coastal face's neighbors that are ocean) and adjacent depth (elevation of neighboring ocean faces). Computed by realmgen since no harbor data exists upstream.
- *River confluences* — faces where multiple `rivers` polylines meet (via `face_id` co-occurrence).
- *Navigable river heads* — the highest face reachable along a river above a flow threshold (`rivers` carry per-segment flow).
- *Mountain passes* — low-cost gaps through high-cost terrain on the land surface (local minima of land cost between high-cost neighborhoods).
- *Strait adjacency* — faces bordering a `strait`-tagged ocean face on the sea surface.

Passes and straits are strategically equivalent — narrow low-cost corridors between larger regions — and downstream military/trade logic treats them symmetrically. Result is a per-face situational-value scalar, rasterized for display.

**4. Settlement Seeder**
Combines suitability and static situational value into a placement score and seeds settlements hierarchically — large first with wide exclusion radii (measured as cost-distance on the land surface, not Euclidean), then progressively smaller tiers. Exclusion radii enforce central-place spacing. Output is provisional: positions (3D + `face_id`, optionally a sub-face UV via `face_tangent_frames`) and rough tier only, no population.

**5. Trade Network**
For each rare good, accumulates flow from sources to provisional settlements along the least-cost path. Overland routes use the land surface; maritime routes use the sea surface, switching at port faces. Aggregates per-edge flow across all goods and both surfaces into a single trade-artery map (the flow-accumulation analogy to hydrology). Settlements at high-flow junctions — including major ports where sea lanes converge — are marked as commercial hubs. This stage also identifies the critical routes worth fortifying or contesting, including strait chokepoints.

Islands unreachable on the land surface are settled via maritime flow if a port can reach them cheaply enough on the sea surface. They contribute to and benefit from maritime trade but are otherwise governed by the same settlement and population rules as mainland faces.

**6. Population & Calibration**
Assigns each settlement a hinterland (weighted multi-source Dijkstra over the land surface — generalize `face_hops_from_seeds` to weighted cost-distance, with source weights by tier). Computes carrying capacity from hinterland `fertility_mean` summed with `face_areas`. Adds trade wealth from route position. Population emerges from the sum. Settlements whose hinterland cannot support them are culled or downgraded. Run trade and population together for 2–3 relaxation passes to let hub reinforcement stabilize.

**7. Polity Formation**
Seeds polities from major settlements and grows territory outward over the *land surface only* — ocean faces are impassable, so coastlines become natural borders and islands remain outside territorial control unless bridged by maritime reach. High-cost terrain naturally becomes contested border zones. Writes a `polity_id` per face. Scores inter-polity tension along shared borders from barrier strength (land cost across the border), trade-artery proximity, and power balance. Contested borders with low natural barriers and high strategic value are fortification candidates.

Maritime reach is a secondary calculation for coastal polities: a polity whose capital has strong port access may extend influence to nearby islands or across narrow straits proportional to its naval power score. Kept separate from territorial growth; does not affect hinterland or population — it governs only whether a polity can contest or fortify a maritime chokepoint.

**8. Military Layer**
Places border forts at contested chokepoints from the polity step — defensible faces (high local relief) on key routes at polity boundaries. Chokepoints include mountain passes (land surface) and strait adjacencies (sea surface); both are treated identically. Places frontier outposts at the habitable edges of high-danger zones — the face where `DangerField` drops below the settle-able threshold (a hop-distance-1 boundary on the danger mask). Both fort types are settlements with military character and low civilian population.

---

## Module Structure & Conventions

realmgen mirrors the existing stage builders so it slots into the project cleanly.

**Orchestration.** Add `RealmParams` and `RealmBuilder` (package `realmgen/`, with a `builder.py`), following `ResourcesBuilder` / `HydrologyBuilder`. Recommended integration: a **standalone `RealmBuilder.build(globe, progress=...)`** that consumes a finished `Globe` and returns it with realm features attached — this matches the doc's "strictly downstream, read-only" framing and keeps realmgen optional. (The alternative — a 5th stage inside `GlobeBuilder` — is viable but couples realmgen into every world build; prefer the standalone builder unless realm output becomes a default deliverable.) The physical base cost surfaces, by contrast, *are* a worldgen step and belong inside `GlobeBuilder`.

**Params.** Use the project's `@paramclass` / `param_field` helpers (`utils/params.py`, as in `worldgen/resources/builder.py`): nested sub-params dataclasses, each leaf field carrying `label` / `min` / `max` / `step` metadata for the params dialog, and an optional top-level `seed` that fans out to per-subsystem sub-seeds via fixed offsets (e.g. `seed + 99991`), as resources does. `None` seed means random each run. **Do not clamp outputs to expected counts/sizes** — that is exactly the "emergence over imposition" guideline, and it matches how the physical stages already behave.

**Outputs & display.** Register vector outputs as `Globe` features and write per-face summaries to `face_metadata`, then wire display in an `apply_display`-style method that registers color-scheme callables over `face_metadata` and texture overlays — see `GlobeBuilder.apply_display`. Likely new schemes: **Suitability**, **Situational Value**, **Danger**, **Land Cost**, **Sea Cost**, **Polities**; likely new overlays: **settlements** (point markers, rendered like `minerals`), **trade_routes** (polylines, rendered like `rivers`), **forts**. Resource/world-building schemes are registered unconditionally (not gated on `debug`); realm outputs are world-building outputs, so register them the same way and reserve `debug` for diagnostic intermediates (raw cost surfaces, flow-accumulation heatmaps). Inspect everything in the dev viewer (`python3 -m utils.mapview.main`) and the main entrypoint (`main.py`).

**Spec doc.** This file is realmgen's design doc, the counterpart to `docs/river_generation_spec.md` and `docs/resources_generation_spec.md`. Keep it in sync as subsystems land, and add a one-line entry to the architecture section of `CLAUDE.md` when the module is created.

**Testing.** Follow the pytest convention (`realmgen/test/...`, picked up by `testpaths` in `pyproject.toml`). Pin determinism (same seed → same settlements/routes/polities) and the vectorised-vs-scalar equivalence of any classifier, as `classify_textures` / `classify_crops` do.

---

## Implementation Order

**Stage 0 — Base cost surfaces (worldgen)** — *implemented in `worldgen/cost/` (`CostBuilder`, registered as the `"cost_surfaces"` globe feature).*
Physical land + sea traversal cost and `strait` tagging, attached to the Globe. Sailable water is ocean ∪ lake; navigable rivers (gated by flow and by a ruggedness-driven fall line) join the sea network and stay land-passable (amphibious). The land surface carries an anisotropic river term (cheap along a navigable valley with the discount tapered by navigability, expensive to ford) folded into `build_land_edge_costs`; realmgen calls the same function after adding its danger field to the per-face base cost.
*Done when:* the land-cost and sea-cost schemes render sensibly; open water (ocean and lakes) is impassable on land and vice-versa; navigable rivers extend the sea network up to the fall line; straits are tagged at narrow passages; the biome cost table is tunable without touching realmgen. ✅

**Stage 1 — Foundation (realmgen)** *(suitability scorer implemented)*
DangerField (procedural default), danger overlay on both surfaces, suitability scorer, static situational value. No placement yet.
*Done when:* the suitability heatmap places likely cities in recognizable regions; harbor quality and passes/straits read correctly; the danger overlay inflates cost without corrupting the physical base.

Suitability scorer (`realmgen/suitability.py`, orchestrated by `RealmBuilder`): five independent per-face factors (water proximity via BFS hop-decay from river+lake seeds + coastal exponential decay; climate habitability as piecewise T_warmest × T_coldest × P_annual; fertility direct from `fertility_mean`; defensibility as tanh of `face_elevation_range`; resource access via BFS hop-decay from mineral+rare_bio faces) combined as a tunable weighted sum; water faces forced to 0. Remaining items (DangerField, danger overlay, static situational value) not yet implemented.

**Stage 2 — Placement** *(settlement seeder implemented)*
Settlement seeder using the combined score, hierarchical with cost-distance exclusion radii. `placement_score` blends suitability + situational value; `build_cost_graph` converts per-face edge costs to a scipy CSR matrix; `seed_settlements` runs hierarchical greedy Dijkstra-exclusion placement (4 tiers, cumulative forbidden mask). Settlements registered as `"settlements"` globe feature (list of `Settlement` objects with `face_id`, `position`, `tier`, `score`); `settlement_tier` written to `face_metadata`; Settlements overlay checkbox added to the map viewer.
*Done when:* settlements appear at reasonable density, spacing feels varied, no clustering or desert-of-nothing regions.

**Stage 3 — Trade**
Rare-good flow accumulation and hub identification across both surfaces.
*Done when:* overland routes follow terrain intelligently; maritime routes connect major ports; routes detour around high-cost biomes and danger; hubs appear at chokepoints including strait adjacencies; coastal islands show activity.

**Stage 4 — Population & Calibration** *(implemented)*
Carrying capacity, trade wealth, relaxation passes, culling.
`realmgen/population/calibration.py` — pure functions: `compute_hinterland_shares` (soft, fractional hinterland — multi-source Dijkstra, then each settlement bids `relu(1 − dist/max_distance) · exp(−influence/influence_scale)` for every face within `max_distance`, and each face's capacity is split among bidders in proportion; returns the `(S,N)` share matrix plus a `(N,)` argmax-owner array for display), `compute_carrying_capacity` (`shares · (fertility × area)` per settlement), `compute_trade_wealth` (face flow at settlement), `compute_population` (carrying_cap × log-trade bonus), `cull_settlements` (settlements below `min_pop_fraction` of peak are culled; display tier is re-derived from population at render time). The soft sharing lets a low-pull village beside a high-pull city keep a nonzero capacity share — so it survives culling — even though the city is the displayed owner of those contested faces. Wired in `RealmStage.build`: computes hinterland shares once, then iterates 2–3 passes of (wealth → population → trade with population demands); culls after last pass; recomputes the owner array for the culled list; final trade pass with culled list. Writes `population` and `hinterland_id` per-face metadata; settlement dots scale by sqrt(pop/max_pop); Hinterland color scheme shows territory colored by population-derived display tier.
*Done when:* the population distribution roughly follows a rank-size curve without being enforced. Isolated fertile regions stay modest; trade-hub regions are dense.

**Stage 5 — Polities**
Territory growth (land surface), maritime reach, tension scoring.
*Done when:* land borders fall on natural barriers; coastlines form natural political edges; maritime empires emerge where port-heavy polities project reach; contested zones are identifiable.

**Stage 6 — Military**
Border forts and frontier outposts; designer-override DangerField.
*Done when:* forts appear at passes and strait chokepoints and route crossings; outposts ring dangerous zones; the danger override hook works with a manually painted test region; sea-specific danger elevates sea cost without affecting land routes.

---

## Design Guidelines

**Emergence over imposition.** Don't hardcode counts, sizes, or distributions. The system should produce plausible geography because the rules are right, not because outputs are clamped. Use observable distributions (rank-size, borders on natural barriers) as validation, not parameters. This is consistent with how the physical stages already work.

**Feedback loops through explicit relaxation, not live coupling.** Trade → population → demand → trade is a genuine loop. Handle it with a fixed number of relaxation passes, keeping the system deterministic and debuggable.

**Every spatial decision runs over the appropriate cost surface.** No straight-line distances, no raw centroid proximity. Each subsystem declares land, sea, or both-via-port-switching, and never mixes them implicitly. Exclusion radii, hinterlands, and spacing are all cost-distances on the face graph.

**Each stage must be independently inspectable.** Before moving on, the current stage's output should render in the map viewer as a color scheme or overlay and make intuitive sense on its own. A stage that can only be validated after later stages are built is too tightly coupled.

**Settlement character is a spectrum, not an enum.** Track continuous economic and military scores per settlement and bin to a display tier only at output time. Keeps the model flexible as RPG requirements evolve.

**Danger is the one designer-facing dial.** All RPG-specific world modifications go through the `DangerField`. Adding a new kind of hazard should require zero changes to settlement logic. Keep this boundary clean — and keep it in realmgen, not in the physical base cost surfaces.

**Polity formation is geographic, not genealogical.** At this layer, polities are purely territorial — regions of influence around a dominant settlement. Historical simulation (succession, conquest, cultural drift) is downstream. Don't let polity logic anticipate it.

**Faces, not pixels, for logic; pixels for display.** All graph reasoning (cost, flow, hinterland, polity growth) is per-face. Per-pixel textures exist only to render smooth heatmaps, computed by rasterizing per-face scalars the way `fertility.py` does. Don't run graph algorithms at pixel resolution, and remember the slope caveat: ruggedness via `face_elevation_range`, never per-pixel slope.
