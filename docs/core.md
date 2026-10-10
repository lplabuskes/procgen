# Core — Infrastructure and Mesh

Qt-free shared infrastructure. Qt viewers: [viz.md](viz.md).

## `core/pipeline.py` — Stage contract and BuildContext

**`BuildContext`** (dataclass), threaded through every `build`:
- `world: World`; `ctx.mesh` = `ctx.world.mesh`
- `seed` — base for per-stage RNG fan-out (defaults to `world.seed`; random if both `None`)
- `progress` — status callback via `ctx.report(msg)`
- `ctx.rng(name)` — deterministic per-name `Generator`, stable per `(seed, name)`, independent of call order

**`Stage`** (ABC; only `build` required): `name`, `requires` / `provides`
(channel tuples; validation deferred), `build(ctx)`, `register_display(world,
reg, *, debug=False)`, `serialize() → dict[str, ndarray]` / `load(arrays, ctx)`.

**`Pipeline`** (a `Stage`): ordered stages over one `BuildContext`; `build` runs
them and caches `self.world`. Pipelines nest — display/serialize/load delegate to
children, so an app pipeline holds `GlobePipeline` + `RealmPipeline` as ordinary
stages. `apply_display(window, *, debug=False)` builds **one** `DisplayRegistry`
from the whole tree and wires it via `viz.mapview.registry.apply_registry`
(imported lazily so `core` stays Qt-free).

## `core/channels.py` — Inter-stage data contract

Typed descriptors instead of magic strings; each stage exports the channels it
produces as module constants and downstream stages import them.

- `MetaChannel(name, dtype, default)` — per-face scalar in `mesh.face_metadata`; `read(mesh) → (N,)`, `write`, `present`.
- `TextureChannel(name, n_channels, codec=None)` — `(N, H, W, C)` in `mesh.face_textures`; same API. A `codec` (`core.codec.TextureCodec`) registers the texture's quantized save encoding by name.
- `FeatureChannel(name)` — arbitrary object in the `World` feature registry; API on a `World`.

`Channel = Union[MetaChannel, TextureChannel, FeatureChannel]`

## `core/world.py` — World container

`World(mesh_params, *, seed, progress)` loads a mesh from `core/mesh_cache.py`
(building on first run). `World.from_mesh(mesh)` wraps a built mesh (`seed=None`;
`load_world` restores it from `world_seed`).

Attributes: `mesh`, `seed` (for per-world-deterministic draws — polity scalars,
region noise, …), `antimeridian` (map cut longitude, set by `ElevationBuilder`).
Features: `add_feature`, `get_feature(name, default=None)`, `has_feature`,
`remove_feature`, `feature_names`.

## `core/mesh_cache.py` — Mesh cache

- `MeshParams` (`@paramclass`, `GlobeParams.mesh`): `icosphere_depth`, `n_swaps`, `n_relax`, `stiffness`, `outer_iters` (as `Mesh.icosphere` + `deregularize` + `mesh_dual`).
- `build(params, cache_dir, n, progress)` — generates `n` (5) deregularized duals into `mesh_cache/` *beside* the repo (`core/../../mesh_cache`). `load_random(seed, cache_dir, n)` picks `seed % n` (random if `None`). `is_populated()`.
- `World.__init__` builds if unpopulated, then `load_random(seed)`. **Changing `MeshParams` does not rebuild an existing cache.**

**Derived geometry.** A loaded mesh's `geometry_store` (`_SidecarStore`) persists
`face_texture_positions`, `face_texture_in_polygon` (full set, default `eps`) and
`boundary_pixel_pairs` as `mesh_<i>.<key>.npz` beside it (~40 MB at `tex_size` 16;
first build ~1 s). Stamped with `GEOMETRY_VERSION` and the mesh file's size/mtime —
**bump `GEOMETRY_VERSION` when any of these computations changes**. Other meshes (`geometry_store is None`) cache in
memory only.

## `core/persistence.py` — Save / load

`.npz` with `params_json`, mesh geometry (`mesh_vertices` / `mesh_face_sizes` /
`mesh_face_data`), every face texture except display-only `*_rgba`, the face
metadata, plus each stage's `serialize()`. ~23 MB for a physical world.

- **Textures** with a registered codec are saved as `texq_<name>` with their codec
  parameters in `tex_codecs_json` (so loading needs no registry); the rest verbatim
  as `tex_<name>`. Loads return each texture in its original dtype.
- **Face metadata** is columnar: a typed `meta_<key>` array per key every face holds
  as a uniform scalar (bool, narrowest int, float32, or string codes into a vocab),
  the rest (ragged, partial) as sparse JSON, all described by `face_metadata_header`.

## `core/codec.py` — Save encodings

`TextureCodec(step, log=False, log_offset=0.0, relative_to=None)`: values are
quantized to multiples of `step` (error ≤ `step/2`) — in `log(value + log_offset)`
space when `log` (a relative step), as a difference from the decoded
`relative_to` texture when set — then differenced along each texel row and
narrowed to the smallest int dtype, which zlib compresses far better than float
bits. Non-finite textures fall back to verbatim. Encoding is idempotent
(re-saving a loaded world reproduces it exactly). Current steps: elevation and
filled height (relative to elevation) 0.1 m; temperature 0.05 °C; precipitation
0.1% / humidity 0.25% (log); coast smoothness 1/255; delta, wetland, fertility,
forest density 0.001.

- `save_world(world, params, stages, path)`
- `load_world(path, params_cls) → (params, world, data)` — caller then runs each stage's `load(data, ctx)`

Used by `main.py` and `worldgen.save_globe` / `load_globe`. No backward
compatibility.

## `core/antimeridian.py`

`find_ideal_antimeridian(mesh, n_samples=360) → float` — cut longitude
minimising land faces crossing it (vertex longitude span > π after shifting);
ties → farthest from any land centroid; no `is_ocean` → π. Stored as
`World.antimeridian`; map views and region cells cut there.

## `core/params.py` — Params utilities

- `paramclass` — alias for `@dataclass`, marks params for the UI dialog.
- `param_field(default, *, label, min, max, step?, hidden?)` — leaf field with dialog metadata; conventionally imported as `_f`.
- `field` — re-exported `dataclasses.field` for nested sub-params (`metadata={"label": ...}`).
- A paramclass may declare `DERIVED: ClassVar[frozenset[str]]`: fields its owner fills in from other params (the dialog skips them).

## `core/display.py` — DisplayRegistry

Qt-free surface stages contribute visuals to; `viz/mapview/registry.py` wires it
to a `MapWindow`.

- **`DisplayRegistry`**: `add_scheme(name, fn, *, texture?, legend?, recipe?)`,
  `add_overlay(feature, label, *, checked?, legend?)`, `add_hover(fn, *,
  refined=(), world_only=False)`. `hover(face_idx, uv, world, local=None)`
  concatenates fragments; with `local`, a contributor whose `refined` texture
  keys are all refined is replaced in place by `local(keys)`, and `world_only`
  contributors are skipped.
- **`SchemeSpec`**: `fn` (face metadata → RGBA), optional `texture`, `legend`,
  `recipe`.
- **`TextureRecipe`**: `fn(textures) -> (N, H, W, 4)` over `{key: (N, H, W, C)}`,
  `sources` (extras like water masks read with `.get`), `relief` (zoomed view may
  hillshade). The stage builds its world texture with
  `mesh.set_face_textures(key, recipe(mesh.face_textures))` and registers the
  same object; the region view reruns it on refined textures — one colouring
  function for both. `available(textures)` checks sources. Reserved source
  `POSITIONS_KEY` (`"positions"`): `(N, H, W, 3)` unit-sphere texel positions for
  layers that are functions of position (polity fill); supplied by the region
  view.
- **`OverlaySpec`**: `feature`, `label`, `checked`, optional `legend`.

## `core/mesh.py` — Mesh data structures

- **`PolygonMesh`** — precomputes `face_adjacency`, `face_centroids`,
  `face_areas`, `face_metadata`, fan-triangulated render arrays.
  - `set_face_colors`, `set_face_metadata(face_idx, **kw)`, `apply_metadata_channel(key, values)`
  - `set_face_textures(name, textures, n_channels=4)` — `(N, H, W, C)` float32 (1 for data, 4 for RGBA)
  - `face_texture_positions(h, w)` — unit-sphere position per texel (grid endpoints, `linspace(0, 1, T)`; over-hanging corners extrapolated from the nearest fan triangle)
  - `face_texture_in_polygon(h, w)` — `(N, H, W)` bool: inside polygon vs over-hang (full set cached read-only)
  - `face_tangent_frames()`; `face_uv_to_3d` / `face_3d_to_uv`
  - `boundary_pixel_pairs(tex_size)` — matched texel pairs across shared edges, cached per size (seam smoothing, ocean flood, seam metrics); `measure_boundary_texture_jumps(texture)`
- **`Mesh`** (triangles only):
  - `icosphere(depth)` — vertices sorted along Z for locality
  - `.deregularize(n_swaps, n_relax, stiffness, outer_iters)` — random edge swaps + spring relaxation
  - `.mesh_dual()` → `PolygonMesh`: vertices at face centroids (normalised to the sphere), one CCW face per original vertex (12 pentagons + hexagons)

## Tangent frame convention

`u_axis = cross(z, n)`, `v_axis = cross(n, u_axis)` (`x` in place of `z` when
`|n_z| ≥ 0.9`, above ~64°), so U is east and V north for ~90% of faces. D4 grid
walks produce N-S/E-W bias; `rivers.py` avoids it with D-infinity accumulation
and smoothed polylines.
