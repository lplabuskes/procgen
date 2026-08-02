# Regional & Local Maps — Design Document

> **Implementation status (evolving):** The polytope choice below assumed a
> geodesic icosphere at depth 3. The implementation instead uses an **equal-area
> cubed-sphere** subdivision — `N_SUBDIV=3` cells per cube-face axis → **54
> regions** — because it gives near-equal-area cells that map cleanly to a flat
> `[0,1]²` local frame (ideal for a hex grid and for upsampling). The geometry
> lives in the Qt-free `region/` package (`region/geometry.py`): `ecef_to_cell`
> (sphere → region id), `cell_border_polylines` (overlay curves), and
> `RegionGeometry` (per-region local `(s,t)` ↔ sphere maps: `local_to_ecef`,
> `local_to_lonlat`, `sample_grid`). The map viewer's region overlay and the
> zoomed-in `RegionView` (`viz/mapview/regionview.py`) both build on it.
>
> A region is **not** meshed at depth 6/7 as sketched below. Its local square is
> tiled with hexagons (`region/grid.py`), each carrying a square texture that
> over-hangs it — the globe's face model one level down, and the footprint for a
> future local map. `region/upsample.py` fills those textures with elevation,
> climate and biome by the upsample + zero-mean-perturbation recipe in this
> document, enforced exactly (`conservation_report`). Hydrology, resources, cost
> and realmgen inside a region are still to do. Reference: [docs/region.md](../region.md).

## Purpose & Scope

The procedural world generator currently produces a single global view at icosphere depth 5 (10,242 hexagon + pentagon faces, ~50,000 km² per face). This design adds two additional zoom levels that maintain consistency with the global map while providing higher detail for localized exploration:

- **Regional maps** — subdivisions of the global sphere into equal-area polytope cells, each containing detailed elevation/climate/resource/population data at higher resolution than the global map.
- **Local maps** — further subdivision of each region into local-scale tiles, appropriate for gameplay at the D&D local/city scale.

Both levels are generated on-demand when a user selects a region, then cached to the world save. The generation process ensures seamless consistency: boundaries match the global map exactly, elevation/climate/hydrology are continuous across region edges, and population/settlement data nest hierarchically (a global city's population is consistent with the sum of its local settlements).

This document does NOT cover street-level maps (a separate future project).

---

## Codebase Grounding

### The Existing Global Model

**Mesh and faces.** The world is an icosphere dual (`PolygonMesh`) at depth 5: 10,242 hexagons plus 12 pentagons on the unit sphere. Every stage (geology → climate → hydrology → resources → realmgen) reasons over it in two registers:

- **Per-face** (`mesh.face_metadata[i]`): discrete graph-level data — scalars like `biome`, `temperature_annual`, `fertility_mean`, `population`, `polity_id`.
- **Per-pixel** (`mesh.face_textures[name]`): continuous `(N, H, W, C)` float32 grids sampled across each face's tangent plane — elevation, temperature, precipitation, masks, authority fill.

**The contract.** Realmgen reads the full face model (geometry, metadata, textures) as read-only and writes settlements, polities, and trade routes on top. The result is serialized as mesh metadata + world features, with larger objects (polity geometry, authority fill) stored as per-face textures.

### Regional & Local Extension

Regions and locals will follow the same two-register pattern:

- **Per-face/per-cell** — region metadata (boundary polygon, center, subdivision-depth), local metadata (analogous to face metadata, but scoped to the local).
- **Per-pixel** — region textures (elevation, climate, resources, authority fill), local textures (same).

A region is **not** a set of depth-5 faces. Instead, it is an independent geometric cell (defined by a polytope), with a boundary polygon on the sphere. When generated, it produces its own mesh at higher resolution (depth 6, 7, or 8 depending on performance tuning) whose faces/texels live within the region's geometric bounds.

---

## Mathematical Regions: Polytope Subdivision

### Polytope Choice

Regions are defined by equal-area spherical polytope cells. The polytope should:
1. Tessellate the sphere perfectly (no gaps or overlaps)
2. Produce roughly equal-area cells (minimize warping on planar projection)
3. Be hierarchical (regions can subdivide into locals)
4. Be deterministic (same polytope always, reproducible)

**Recommended: Geodesic polyhedra (icospheric subdivision) or equal-area Voronoi cells.**

- **Geodesic option**: Use icosphere at depth 3 (642 cells) for regions, depth 4 (2,562 cells) for locals. Simpler, inherits existing mesh code. Trade-off: icosphere cells are not perfectly equal-area (distortion ~±15% at depth 3), but projection distortion is mild.
- **Equal-area Voronoi option**: Generate seed points via equal-area Fibonacci sphere or similar, then compute Voronoi cells. Perfect equal-area, but requires new geometric code. Cell shapes vary (some hexagons, some irregular).

**Decision point for implementation: Geodesic is faster to build (reuse mesh code); equal-area Voronoi is more principled but requires a separate library (e.g., `sphericalvoronoi` from scipy).**

For this design, assume **geodesic icosphere at depth 3 for regions** (~642 regions globally, ~500k–1M km² per region — roughly the size of a medieval kingdom). Locals are depth 4 within each region (~2.5k cells per region, or can vary by region complexity).

### Region Definition

A region is defined by:

```python
class RegionDef:
    id: int                              # globally unique
    boundary: np.ndarray                 # (M, 3) spherical polygon vertices
    center: np.ndarray                   # (3,) sphere center
    polytope_depth: int                  # 3 for regions
    local_depth: int                     # 4 for locals (can override per region)
    name: Optional[str] = None           # user-assigned
    description: Optional[str] = None
```

Regions are **pre-computed and immutable** at startup. The list of all region definitions is either:
- Baked into code as a constant (smallest surface area) — generate once with `icosphere(depth=3)`, serialize to JSON, commit.
- Loaded from `data/regions.json` — allows designer tweaks without code changes.

### Hierarchical Locals

Each region contains a hierarchical subdivision of locals. Locals are defined similarly:

```python
class LocalDef:
    id: int                              # unique within region
    region_id: int                       # parent region
    boundary: np.ndarray                 # (M, 3) polygon within region
    center: np.ndarray                   # (3,)
    polytope_depth: int                  # 4 (or region-specific override)
```

Locals are pre-computed at region-definition time (not on-demand). When a region is generated, its locals are pre-defined but not yet rendered — they are geometries waiting to be filled in.

---

## Data Model & State

### Region & Local Objects

Regions and locals have two distinct phases:

1. **Defined but not-generated**: Geometry exists (boundary, center), but no detail data.
2. **Generated**: Full detail (elevation, climate, hydrology, resources, polities) is computed and cached.

```python
class Region:
    def_: RegionDef                      # geometry
    
    # Not-yet-generated:
    is_generated: bool = False
    
    # Generated (set when is_generated = True):
    mesh: Optional[PolygonMesh] = None   # depth-6/7 icosphere aligned to region bounds
    world: Optional[World] = None        # BuildContext result
    realm_pipeline: Optional[RealmPipeline] = None  # populated features
    
    def generate(self, global_world: World, realm_params: RealmParams) -> None:
        """Generate or load from cache."""
        ...
```

### Boundary Sampling & Multi-Scale Consistency

When a region is generated, two consistency constraints must be satisfied simultaneously:

**1. Boundary continuity** — the region's boundary values must match the global map exactly (Dirichlet boundary condition for elevation, temperature, etc.).

**2. Aggregate conservation** — **every cell in the global map that overlaps the region must remain consistent when refined.** If a global texel has average elevation X, the region's fine-grained texels within that area must average to X. This applies to:

- **Elevation** — average elevation per global texel must be preserved in the regional refinement
- **Temperature & Precipitation** — average climate values must match
- **Resources** — average fertility, forest density, crop suitability must match
- **Population** — total population per global face is fixed; regional detail subdivides it
- **Polities** — per-face polity assignment is fixed; regional borders subdivide existing territories

**Why this matters**: The global map is not an infinite texture; it is a discretized field at depth-5 resolution with per-texel values. When you zoom into a region, you are *refining* the global map, not replacing it. A user must see consistent aggregate properties at all zoom levels — water scarcity, biome distribution, total population, political alignment.

**Mechanism**: Regional detail is generated as **global coarse field + zero-mean perturbation**. For elevation:
1. Upsample the global elevation field to regional resolution (e.g., via cubic interpolation)
2. Compute the target average elevation for each overlapping global texel
3. Generate fine-grained noise (simplex, fractal Brownian motion) at regional resolution
4. Normalize the noise to have zero mean over each global texel's area
5. Add the normalized noise to the upsampled global field
6. Result: fine detail that preserves the global aggregate

This upsampling + zero-mean perturbation approach applies to all conserved properties. Climate, resources, and population follow the same pattern.

### Serialization Format

Regions are persisted as part of the world save. The serialization mirrors realmgen's strategy:

**Per-region metadata** (attached to `Region` object):
- `region_id`, `is_generated`
- `mesh` — serialized as a mesh (reuse existing `worldgen.mesh.save/load` code)
- Per-face metadata — same as global faces (`biome`, `temperature`, `population`, `polity_id`, etc.)

**Per-region features** (registered to `Region.world` just as they are to `Globe`):
- `settlements`, `polities`, `trade_routes` — same structure as global
- `polity_borders`, `authority_fill` — same geometry/textures

**Locals** — similarly scoped to their parent region.

A world save contains a dict of `{region_id: Region}` for all generated regions. On load, only generated regions are restored; the region defs remain pre-computed.

---

## Generation Pipeline

### On-Demand Generation

When a user selects a region in the UI:

1. Check if `region.is_generated`; if yes, load from world cache and return.
2. If no, acquire the global world (already built) and call `region.generate(global_world, realm_params)`.
3. Display the region map.

Generation should be **deterministic and repeatable**: the same region always generates the same detail (same seed, same params). This is achieved by seeding all per-region RNGs with `world.seed XOR region_id`.

### Generation Stages

Region generation runs a simplified pipeline. Each stage respects both boundary continuity and aggregate conservation (see above):

1. **Create mesh** — icosphere at depth 6/7, clipped/aligned to region boundary. Record which global texels overlap the region; these define the aggregation targets for all subsequent stages.

2. **Elevation detail** — Upsample global elevation to regional resolution (cubic interpolation). Compute target average elevation for each overlapping global texel. Generate fBm noise at regional resolution; **zero-mean normalize** over each global texel's area. Add to upsampled field. Boundary values match global exactly; interior has fine detail; aggregates match.

3. **Climate detail** — Upsample global temperature/precipitation to regional resolution (same upsampling + zero-mean perturbation as elevation). Apply local lapse rate based on regional elevation — this refines the climate field without violating aggregates. Boundary values fixed; interior detail follows elevation-driven variation.

4. **Hydrology** — Route water within region using local elevation. Identify global rivers that cross region boundary; these are inlet/outlet conditions (specified flow direction and volume). Local rivers must connect smoothly to boundary rivers. Ensure water conservation: inflow from boundary = outflow to boundary + lakes.

5. **Resources** — Upsample global fertility/forest/crop to regional resolution using zero-mean perturbation. Assign biome at texel resolution, but ensure texels within each global face average to the global biome's fertility/forest signature.

6. **Cost surfaces** — Build local land/sea cost layers using the refined elevation and resource data. Cost is derived, not seeded — use the same biome/ruggedness cost table as the global stage.

7. **Realmgen** — Run full settlement/polity/trade pipeline **within** the region. Seeded by `world.seed XOR region_id`. **Population constraint**: sum of all settlements within each global face must equal that face's global population (hard constraint). **Polity constraint**: per-face polity assignment is immutable; regional borders can be refined at texel resolution but must respect face-level ownership.

Each stage reads boundary data from the global world and enforces it exactly. Interior data is generated procedurally but constrained by aggregation targets.

### Consistency Enforcement Per Stage

**Elevation:**
- Upsample global elevation field to regional mesh resolution via cubic interpolation
- Identify which global texels overlap the region; compute target average for each
- Generate fine-grained noise (simplex, fBm) at regional resolution with high frequencies
- **Zero-mean normalize** noise over each global texel's area (subtract local average within each texel, so the mean stays zero)
- Add normalized noise to upsampled field
- Result: boundary matches global exactly; interior has fine detail; each global texel's average is preserved

**Climate (Temperature & Precipitation):**
- Same upsampling + zero-mean perturbation approach as elevation
- Additionally, apply local lapse rate based on regional elevation — this refines the climate but respects the global average constraint
- Boundary values are fixed from global; interior detail follows elevation-driven lapse rate (but preserves aggregate)

**Hydrology:**
- Global rivers that cross the region boundary are boundary conditions (inlet/outlet points with specified flow)
- Route local water within the region using the refined elevation
- Local rivers must connect smoothly to global boundary rivers
- Conservation: total water volume entering from global boundary must equal volume leaving + sum of lakes (water is conserved within the region)

**Resources (Fertility, Forest, Crops):**
- Upample global fertility/forest density to regional resolution; add zero-mean detail
- The detail adds spatial variation without changing the per-global-texel aggregate
- Biome assignment: the global biome label per face is the "primary" biome; regional texels can vary (finer biome patchiness) but must average back to the global label's characteristics

**Population:**
- Global face population is fixed and immutable
- Regional settlements subdivide existing faces
- A settlement in the regional map must be assigned to a face, and the sum of all settlements within a global face must equal that face's global population
- This is a hard constraint: regional population distribution is constrained by global totals

**Polities:**
- Global face polity assignment (which sovereign/vassal controls each face) is immutable
- Regional borders can be drawn at texel resolution (finer than faces) but must respect face-level ownership
- A global face entirely owned by polity A cannot contain texels claimed by polity B
- Regional polity detail refines existing territorial boundaries but never violates face-level assignments

**Critical invariant**: Downsampling a regional map by averaging must recover the global map exactly. This is the test for consistency.

### Consistency Validation

A region passes validation if and only if **downsampling it recovers the global map**. For each stage:

- **Elevation**: For each global texel overlapping the region, average the regional texels within that area. The result must equal the global texel's value (within floating-point tolerance, ~0.1%).
- **Climate**: Same averaging check for temperature and precipitation.
- **Resources**: Fertility, forest, crop must average back to global values.
- **Population**: Sum of regional settlements within each global face must equal that face's global population (exact).
- **Polities**: Per-face polity assignment must be identical (exact).

This validation should be run as a test suite during implementation:

```python
def validate_region_consistency(global_world: World, region: Region) -> bool:
    """Check that downsampling region recovers global map."""
    for texel in region.global_texels_overlapping():
        regional_texels = region.texels_within(texel)
        
        # Elevation
        global_elev = texel.elevation
        regional_avg_elev = mean(regional_texels.elevation)
        assert abs(regional_avg_elev - global_elev) < 0.001 * global_elev
        
        # Climate
        assert abs(mean(regional_texels.temperature) - texel.temperature) < 0.1  # °C
        assert abs(mean(regional_texels.precipitation) - texel.precipitation) < 10  # mm/yr
        
        # Resources
        assert abs(mean(regional_texels.fertility) - texel.fertility) < 0.01
        
    # Population
    for face in region.global_faces_overlapping():
        regional_pop = sum(s.population for s in region.settlements if s.face == face)
        assert regional_pop == face.population
    
    # Polities
    for face in region.global_faces_overlapping():
        for texel in region.texels_within_face(face):
            assert texel.polity == face.polity
    
    return True
```

Every region commit to a world save should be validated. Failed validation indicates a bug in the generation pipeline.

---

## User Experience

### Global Map → Region Selection

The global map displays region boundaries (a subtle overlay, turned on/off by toggle). When the user clicks a region:

1. If already generated, the view seamlessly zooms to the region map.
2. If not, a spinner appears while generation runs (background task, non-blocking).
3. Once ready, the region map is displayed.

Region names can be auto-generated (e.g., "Kingdom #47") or designer-assigned (stored in `RegionDef.name`).

### Region Map

Region map displays the same layers as the global map:
- Elevation/terrain texture
- Climate (temperature/precipitation contours, biome coloring)
- Water (rivers, lakes, ocean/coast)
- Polities (sovereign territories, settlement markers)
- Debug overlays (cost surface, authority field, etc.)

Same controls as global: pan/zoom, layer toggles, hover tooltips.

### Local Map within a Region

Selecting a local (or double-clicking a settlement) zooms to that local's map. Locals inherit the same visual language as regions.

Navigation is hierarchical: back button returns to parent (region or global).

### Caching & Persistence

Once a region is generated, it appears in the world save. When loading a world, all generated regions are restored; ungenerated regions remain as stubs. If a region is deleted from the save (user starts fresh), it can be regenerated on-demand using the same seed, producing identical results.

---

## Critical Design Decisions

### Decision 1: Region-to-Face Independence

**Chosen: Regions are independent geometric cells, not aligned to depth-5 faces.**

**Rationale:** Regions must tile perfectly with equal area. Icosphere depth 5 faces do not — they have ~±10% area variance due to the sphere geometry. Regions must be polytope cells (equal-area Voronoi or geodesic depth-3) to guarantee projection consistency.

**Consequence:** Region boundaries may cut across depth-5 faces. This requires interpolation and blending when sampling global data at region edges. The extra complexity is justified by the geometric guarantee.

### Decision 2: Cached vs. Deterministic Generation

**Chosen: Cached on first generation (deterministic, but results are saved).**

**Rationale:** Allows optional user edits later (if an interactive editor is added). Pure determinism (no cache) would prevent custom tweaks. Full caching (no determinism) wastes space and breaks reproducibility.

**Hybrid approach:** Regions are deterministically seeded; the computed result is cached. If deleted, they regenerate identically. If edited, the edit is stored and regeneration is blocked.

### Decision 3: Local Depth

**Chosen: Locals are pre-defined and hierarchical, not infinite zoom.**

**Rationale:** Two levels (region + local) match D&D kingdom/city scales. Three or more levels add complexity and diminishing returns. A local's locals (streets) are a future project.

### Decision 4: Realmgen Scope

**Chosen: Full realmgen pipeline runs within each region.**

**Rationale:** Ensures settlement distribution and polities within a region are internally consistent and unique. A region doesn't just have finer terrain — it has its own political actors (minor lords, frontier settlements). This adds richness and keeps the region self-contained.

**Consequence:** A region may contain many settlements that did not exist in the global view (sub-settlement-scale details). The global settlements are the "canonical" major cities; regional settlements are the minor ones that feed into them.

---

## Serialization Strategy

### What Gets Saved

For each generated region:
- Region metadata (`is_generated`, `region_id`, `local_depth` override if any)
- Mesh (faces, adjacency, centroids, tangent frames) — use existing `save_mesh` code
- Per-face metadata (`biome`, `temperature`, `population`, `polity_id`, etc.)
- Per-face textures (`elevation`, `climate`, `authority_fill`, etc.) — reuse `save_texture` code
- World features (`settlements`, `polities`, `trade_routes`, `polity_history`, `polity_borders`)

### Storage Location

```
saved_worlds/
  world_seed_4/
    globe.bin                # existing global world
    regions/
      region_0/
        mesh.bin
        metadata.json
        features.json
        textures/
          elevation.npy
          temperature.npy
          ...
      region_1/
        ...
```

### Save/Load Integration

Extend `worldgen.save_globe()` and `load_globe()` to handle regions:

```python
def save_globe(globe, regions, path):
    save_mesh(globe.mesh, ...)
    save_features(globe, ...)
    for region in regions.values():
        if region.is_generated:
            save_region(region, path / "regions" / f"region_{region.def_.id}")

def load_globe(path):
    globe = load_mesh(...)
    load_features(globe, ...)
    regions = {}
    for region_dir in (path / "regions").glob("region_*"):
        region = load_region(region_dir)
        regions[region.def_.id] = region
    return globe, regions
```

World save/load in `main.py` (`save_world` / `load_world`) already iterate over these and wire them to the UI. No changes needed there — regions are part of the `BuildContext.world` and flow through automatically.

---

## Implementation Phases

### Phase 0: Infrastructure (Prerequisite)

- [ ] Define polytope cells (geodesic depth-3 icosphere or equal-area Voronoi)
- [ ] Implement `RegionDef` / `LocalDef` classes
- [ ] Create `Region` / `Local` wrapper classes
- [ ] Add region/local save/load scaffolding to `worldgen.persistence`

### Phase 1: Region Mesh & Boundary Sampling

- [x] Create the region's working surface — a hex grid with per-hex textures (`region/grid.py`), not a clipped icosphere
- [x] Implement global-field sampling (`region/sampling.py`: nearest-face location, bilinear read, anchor assignment)
- [x] Test: `region/test/test_grid.py`, `region/test/test_sampling.py`

### Phase 2: Elevation & Climate Detail Generation

- [x] Implement elevation detail generation (band-limited fBm, slope-proportional, aggregate-conserving)
- [x] Implement climate detail generation — elevation is the only noise source; temperature follows it by lapse rate and precipitation by an orographic response, and biome is reclassified from an area-average of the two (an independent noise stream in either input frays the classifier's boundaries)
- [x] Test: `region/test/test_upsample.py` pins the conservation invariant; `scripts/debug_region.py` is the visual check

### Phase 3: Hydrology & Resources

- [ ] Extend hydrology stage to trace rivers within regions (boundary-rooted)
- [ ] Extend resources stage to compute fertility/forest at local resolution
- [ ] Test: rivers cross boundaries correctly, resource patterns match global

### Phase 4: Cost Surfaces & Realmgen

- [ ] Build land/sea cost surfaces for region mesh
- [ ] Run full realmgen pipeline within region (scoring, placement, population, polity)
- [ ] Test: settlements and polities within region are consistent with global assignments

### Phase 5: UX Integration

- [x] Add region boundary overlay to global map
- [x] Add "select region" interaction (click region → generate on demand)
- [x] Add region map display (`RegionView` + a Layer combo: Terrain / Elevation / Temperature / Precipitation)
- [ ] Add zoom-to-local interaction within region maps
- [ ] Test: selection flow, generation background task, cache loading

### Phase 6: Serialization

- [ ] Implement region save/load
- [ ] Extend `save_world` / `load_world` to persist regions
- [ ] Test: save world with regions, load, verify data is identical

### Phase 7: Locals (Repeat Phases 1–6 at Local Scope)

- [ ] Create local meshes within regions (depth 7 or 8)
- [ ] Implement boundary sampling (region → local)
- [ ] Elevation/climate/hydrology/resources for locals
- [ ] Realmgen for locals (settlement clustering, street-level polity detail)
- [ ] UX: zoom from region to local
- [ ] Serialization for locals

---

## Open Questions & Future Work

1. **Polytope choice**: Geodesic (faster) vs. equal-area Voronoi (more principled)? Recommend geodesic for v1, Voronoi for v2.
2. **Region count**: How many regions globally? 642 (depth 3) may be too many to display; consider depth 2 (~162 regions) for initial prototype.
3. **Interactive editing**: Should users be able to edit a region's terrain, settlements, polities after generation? Requires a different serialization strategy (original + deltas, not just result).
4. **Performance**: How long does regional generation take? If >30s, consider async generation, progress bar, cancellation.
5. **Locals per region**: Should all regions have the same number of locals, or vary by complexity (e.g., mountain regions get fewer, larger locals)?
6. **Narrative layer**: Should designers be able to pre-seed certain polities, settlements, or landmarks within a region? Requires parameter overrides at region generation time.

---

## References

- [docs/core.md](../core.md) — mesh and BuildContext contract
- [docs/worldgen.md](../worldgen.md) — pipeline stages and stage contract
- [docs/realm_*.md](../realm_refine.md) — realmgen details
