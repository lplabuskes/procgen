# Core — Infrastructure, Mesh, and Visualization

## `core/pipeline.py` — Stage contract and BuildContext

**`BuildContext`** (dataclass): Shared state threaded through every stage's `build`.
- `world: World` — the container all stages read/write
- `seed: int | None` — base seed for per-stage RNG fan-out (random if omitted)
- `progress: ProgressFn` — optional status callback; called via `ctx.report(msg)`
- `ctx.mesh` — shorthand property for `ctx.world.mesh`
- `ctx.rng(name) → np.random.Generator` — deterministic per-name RNG; stable for a given `(seed, name)` pair and independent of call order across stages

**`Stage`** (ABC): One unit of world generation. Subclasses must implement `build`; all other hooks default to no-ops.
- `name: str` — stage identifier
- `requires / provides: Tuple[Channel, ...]` — declared channel dependencies (for validation, deferred)
- `build(ctx)` — reads inputs from `ctx`, writes outputs back
- `register_display(world, reg, *, debug=False)` — adds color schemes/overlays to `DisplayRegistry`
- `serialize() → dict[str, np.ndarray]` / `load(arrays, ctx)` — persistence hooks

**`Pipeline`** (subclass of `Stage`): Ordered list of stages run against a shared `BuildContext`. Iterable; `build(ctx)` runs each stage in order and caches `self.world`. Because a `Pipeline` *is* a `Stage`, pipelines nest — its `register_display` / `serialize` / `load` hooks delegate to its children, so an app-level pipeline can hold sub-pipelines (e.g. `GlobePipeline` + `RealmPipeline`) as ordinary stages and build/display/serialize the whole tree in one uniform walk.
- `apply_display(window, *, debug=False)` — builds **one** `DisplayRegistry` from the whole (recursive) stage tree and wires it into `window` via `viz.mapview.registry.apply_registry` in a single pass (imported lazily so `core` stays Qt-free).

## `core/channels.py` — Inter-stage data contract

Stages read and write shared world state through typed channel descriptors instead of magic strings. Each stage exports the channels it produces as module-level constants; downstream stages import those constants, making dependencies explicit and type-checked.

- **`MetaChannel(name, dtype, default)`**: Per-face scalar in `mesh.face_metadata`. `read(mesh) → (N,) ndarray`; `write(mesh, values)` bulk-sets the key; `present(mesh) → bool`.
- **`TextureChannel(name, n_channels)`**: `(N, H, W, C)` array in `mesh.face_textures`. Same `present/read/write` API operating on the mesh.
- **`FeatureChannel(name)`**: Arbitrary Python object in the `World` feature registry (rivers, deposits, settlements, etc.). `present/read/write` operate on a `World` rather than a mesh.

`Channel = Union[MetaChannel, TextureChannel, FeatureChannel]`

## `core/world.py` — World container

**`World`**: Container for a generated world: a mesh plus a named feature registry. The constructor loads a mesh from the shared cache (`core/mesh_cache.py`), building it on first run. Use `World.from_mesh(mesh)` to wrap an already-built mesh (e.g. when loading a saved world).

Feature registry: `add_feature(name, data)`, `get_feature(name)`, `has_feature(name)`, `remove_feature(name)`, `feature_names()`.

## `core/mesh_cache.py` — Mesh cache

**`MeshParams`** (`@paramclass`): `icosphere_depth`, `n_swaps`, `n_relax`, `stiffness`, `outer_iters` — same parameters as `Mesh.icosphere` + `deregularize` + `mesh_dual`. Exposed in `GlobeParams.mesh`.

**`build(params, cache_dir, n, progress)`**: Generates `n` random deregularized dual meshes and writes them to `mesh_cache/`. Run once; subsequent runs load from cache. **`load_random(seed, cache_dir, n)`**: Loads one mesh deterministically by `seed % n`.

`World.__init__` calls `mesh_cache.build` on first run, then `mesh_cache.load_random(seed)`.

## `core/params.py` — Params utilities

**`paramclass`**: Alias for `@dataclass` — used to mark params dataclasses so the UI dialog can reflect over them.

**`param_field(default, *, label, min, max, step?, hidden?)`**: Declares a leaf param field with UI dialog metadata (`label`, range, optional step, optional hidden flag). Import alias `_f` is convention throughout the codebase.

**`field`**: Re-exported `dataclasses.field` for declaring nested sub-param fields with `metadata={"label": ...}`.

## `core/display.py` — DisplayRegistry

Qt-free surface that stages contribute visualization to. A viz adapter (`viz/mapview/registry.py`) later walks it and wires into a concrete `MapWindow`, keeping Qt/OpenGL out of `core`.

- **`DisplayRegistry`**: `add_scheme(name, fn, *, texture?, legend?, recipe?)`, `add_overlay(feature, label, *, checked?, legend?)`, `add_hover(fn, *, refined=())`. `hover(face_idx, uv, world, local=None)` concatenates all registered contributor fragments; `refined` names the face textures a contributor reports on, and a regional view's `local(keys)` replaces that contributor's fragments in place when it has refined all of them.
- **`SchemeSpec`**: `fn: SchemeFn` (callable over face metadata dict → RGBA tuple), optional `texture` (associated per-face RGBA texture name), optional `legend`, optional `recipe`.
- **`TextureRecipe`**: how a scheme's RGBA texture is made — `fn(textures) -> (N, H, W, 4)` over a `{key: (N, H, W, C)}` dict, the `sources` keys it needs (extras such as water masks are read with `.get`), and `relief` (a regional view may hillshade it). The owning stage builds its world texture with `mesh.set_face_textures(key, recipe(mesh.face_textures))` and registers the same object; the region view reruns it on its refined textures, so both views colour a layer with one function. `available(textures)` checks the sources. The reserved source `POSITIONS_KEY` (`"positions"`) is the `(N, H, W, 3)` unit-sphere texel positions, for layers that are a function of position rather than stored data (the polity fill); the region view supplies it.
- **`OverlaySpec`**: `feature: str`, `label: str`, `checked: bool`, optional `legend`.

## `core/mesh.py` — Mesh data structures

- **`PolygonMesh`**: Base class for any polygon mesh. Pre-computes `face_adjacency`, `face_centroids`, `face_areas`, `face_metadata`, and fan-triangulated render arrays at construction. Key API:
  - `set_face_colors(colors)`
  - `set_face_metadata(face_idx, **kw)`
  - `apply_metadata_channel(key, values)` — bulk-write one metadata key across all faces
  - `set_face_textures(name, textures, n_channels=4)` — attach `(N, H, W, C)` float32; use `n_channels=1` for data layers like elevation, 4 for RGBA display
  - `face_texture_positions(h, w)` — unit-sphere 3D position of every texture pixel; starting point for any per-pixel geographic computation
  - `face_tangent_frames()` — raw per-face tangent-plane data for projecting arbitrary 3D points into a face's UV space

- **`Mesh`** (subclass of `PolygonMesh`): Triangular mesh.
  - `icosphere(depth)` — generates a subdivided icosahedron; vertices sorted along the Z-axis for spatial locality
  - `.deregularize(n_swaps, n_relax, stiffness, outer_iters)` — breaks regularity via random edge-swaps and spring relaxation on the unit sphere
  - `.mesh_dual()` — returns a `PolygonMesh` by placing new vertices at face centroids (normalized to unit sphere), then for each original vertex builds a CCW-wound face from the centroids of its adjacent faces

**Key invariant:** `Mesh` stores only triangular faces; `PolygonMesh` handles arbitrary polygon faces (e.g., the dual has 12 pentagons + many hexagons). The dual normalizes centroids to the unit sphere.

## Tangent frame convention

Per-face tangent-plane axes are `u_axis = cross(z, n)`, `v_axis = cross(n, u_axis)`, so the texture U-axis is geographic east and V-axis is geographic north for ~90% of faces. Any D4 grid-walk algorithm will produce N-S/E-W-biased output; `rivers.py` avoids this with D-infinity accumulation and Laplacian-smoothed polylines rather than grid-snapped steps.
