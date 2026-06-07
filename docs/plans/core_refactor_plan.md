# Core Refactor Plan — Stage/Pipeline Architecture

## Motivation

The per-stage compute layer is clean, but four **centralized chokepoints** must be
edited for every new stage, and adding `realmgen` (and soon city/name generators)
makes this worse:

| Chokepoint | Today | Problem |
|---|---|---|
| Orchestration | `GlobeBuilder.build` (`worldgen/globe/builder.py:121-199`) | Manually threads `hydrology_builder.rivers` into later stages and hand-registers every feature |
| Display wiring | `GlobeBuilder.apply_display` (`builder.py:201-496`) | ~300-line god method hardcoding every stage's textures, schemes, legends, overlays, hover string |
| Persistence | `save_globe`/`load_globe` (`worldgen/globe/serialization.py`) | Names every builder attribute by hand, twice |
| Cross-stage contract | `mesh.face_metadata[...]["key"]` + `face_textures["key"]` + `globe.features` | Stringly-typed; back-compat fallbacks (`m.get("is_water", m.get("is_ocean", False))`) smeared across `worldgen/globe/builder.py`, `realmgen/builder.py`, display callables |

`realmgen` also reaches **back up** into worldgen internals (`worldgen.geology.display.OCEAN_COLOR`,
`worldgen.resources.terrain.face_elevation_range`), so the downstream boundary already leaks.

**Goal:** make adding a generation stage cost *one new class appended to a list* — zero
edits to orchestration, display, or persistence god-functions.

## Design Principles

1. **`core/` imports numpy only — no Qt, no OpenGL, no domain knowledge.** This is the
   discipline that makes the package boundary real. The moment core imports PySide6 it
   stops being core.
2. **Stages contribute; the core iterates.** Orchestration, display, and persistence
   become `for stage in pipeline: stage.<hook>(...)`.
3. **The inter-stage contract is typed.** Magic strings become `Channel` descriptors
   exported by the stage that owns them and imported by downstream consumers — the import
   *is* the dependency edge.
4. **Don't gold-plate.** Extract into `core/` only what converting the first stage forces.
   `requires`/`provides` validation is optional and lands last (or never).

## Target Package Layout

```
core/                  # numpy-only substrate + pipeline vocabulary
  mesh.py              # PolygonMesh, Mesh        (was worldgen/mesh/mymesh.py)
  mesh_cache.py        # (was worldgen/mesh/cache.py)
  world.py             # World container (was Globe) + FeatureLayer convention
  pipeline.py          # Stage (ABC), Pipeline, BuildContext (+ seed fan-out)
  channels.py          # MetaChannel, TextureChannel descriptors
  display.py           # DisplayRegistry (NO Qt)
  params.py            # paramclass framework (was utils/params.py)
  persistence.py       # numpy encoder, deep_from_dict, stage-iteration save/load

worldgen/              # physical stages (geology … cost) — internals unchanged
humangen/  (or realmgen expanded)   # realm -> settlements -> cities -> names sub-stages
viz/                   # mapview, projections, params_dialog, mesh/gui  (consumes DisplayRegistry)
```

Dependency DAG: `worldgen -> core`, `humangen -> core`, `viz -> core`. `humangen` may
import worldgen's exported `Channel` constants (an explicit domain dependency).

## Key Signatures

### core/world.py
```python
class World:                      # was Globe
    mesh: PolygonMesh | None      # None until the mesh-producing stage runs
    # feature registry unchanged: add_feature / get_feature / has_feature / remove_feature
```

### core/pipeline.py
```python
ProgressFn = Callable[[str], None]

@dataclass
class BuildContext:
    world: World
    _seeds: np.random.SeedSequence
    progress: ProgressFn = lambda _msg: None
    def rng(self, stage_name: str) -> np.random.Generator: ...   # deterministic per-stage fan-out
    def report(self, msg: str) -> None: ...
    @property
    def mesh(self) -> PolygonMesh: ...                            # asserts non-None

class Stage(ABC):
    name: str
    requires: tuple[Channel, ...] = ()      # optional ordering validation
    provides: tuple[Channel, ...] = ()
    @abstractmethod
    def build(self, ctx: BuildContext) -> None: ...
    def register_display(self, reg: DisplayRegistry, *, debug: bool) -> None: ...  # no-op default
    def serialize(self) -> dict[str, np.ndarray]: return {}
    def load(self, arrays: Mapping[str, np.ndarray], ctx: BuildContext) -> None: ...

class Pipeline:
    def __init__(self, stages: list[Stage]): ...
    def build(self, ctx: BuildContext) -> World: ...   # for s in stages: s.build(ctx)
    def __iter__(self) -> Iterator[Stage]: ...
```

### core/channels.py
```python
@dataclass(frozen=True)
class MetaChannel:                # per-face scalar in mesh.face_metadata
    name: str
    dtype: np.dtype
    fallback: "MetaChannel | None" = None   # is_water -> is_ocean chain lives HERE, once
    default: float | bool = 0.0
    def read(self, world) -> np.ndarray: ...     # (N,)
    def write(self, world, values) -> None: ...
    def present(self, world) -> bool: ...

@dataclass(frozen=True)
class TextureChannel:              # (N,H,W,C) in mesh.face_textures
    name: str
    n_channels: int = 1
    def read(self, world) -> np.ndarray: ...
    def write(self, world, tex) -> None: ...
    def present(self, world) -> bool: ...
```
Stages export channels as module constants; downstream stages import them:
```python
# worldgen/cost/__init__.py
IS_WATER = MetaChannel("is_water", np.bool_, fallback=IS_OCEAN)
# realmgen
from worldgen.cost import IS_WATER
is_water = IS_WATER.read(ctx.world)
```

### core/display.py  (NO Qt)
```python
HoverFn = Callable[[int, np.ndarray | None, World], list[str]]   # status fragments

class DisplayRegistry:
    def add_scheme(self, name, fn, *, texture=None, legend=None) -> None: ...
    def add_overlay(self, feature, label, *, checked=True) -> None: ...
    def add_hover(self, fn: HoverFn) -> None: ...   # registry concatenates all stages' fragments
```
`viz` provides `apply_registry(reg, window)` doing the Qt wiring.

## How the Three Awkward Realities Are Handled

- **Elevation creates the mesh.** `World.mesh` is `Optional`; the elevation stage sets
  `ctx.world.mesh` in its `build`. `BuildContext.mesh` asserts non-None for later stages.
- **Climate runs twice.** Model as two pipeline nodes sharing one builder. The lake-feedback
  pass becomes a thin `Stage` (`climate.lake_feedback_stage()`) placed *after* hydrology, so
  the ordering is visible in the pipeline list instead of an `if "lake_mask" in textures` branch.
- **Redundant threaded args.** `elevation_textures` and `rivers=` are already on the
  mesh/world; stages read them from `ctx.world`, so all `build()` signatures collapse to
  `build(ctx)`.

## Migration Order (vertical slice first)

1. **[DONE] Move the mesh into core** — `worldgen/mesh/mymesh.py -> core/mesh.py`,
   `worldgen/mesh/cache.py -> core/mesh_cache.py`, `test_mymesh.py -> core/test/`. 30 importers
   rewritten (`from core.mesh import ...`, `from core import mesh_cache`); `pyproject.toml`
   testpath updated. The dev 3-D viewer (`worldgen/mesh/gui.py`, `main.py`) stays put for now
   (moves to `viz/` in step 7), repointed at `core.mesh`. All 281 tests green.
2. **[DONE] Stand up `core/` skeleton** — added `core/world.py` (`World`; mesh `Optional`),
   `core/channels.py` (`MetaChannel`/`TextureChannel` + `Channel` union, with the fallback
   chain), `core/pipeline.py` (`Stage` ABC, `Pipeline`, `BuildContext` with deterministic
   per-stage `rng(name)` fan-out + mesh guard), `core/display.py` (Qt-free `DisplayRegistry`
   with scheme/overlay/decomposed-hover accumulation). Moved `utils/params.py -> core/params.py`
   (6 importers rewired). `worldgen/globe/globe.py` is now a one-line shim: `Globe = World`, so
   every existing `Globe` import resolves to the one canonical container. Added
   `core/test/test_skeleton.py` (12 tests). Nothing is wired onto the pipeline yet — pure
   addition. All 293 tests green.
3. **[DONE] Convert the proof slice: `realmgen`.** `RealmStage(Stage)` reads every upstream
   layer through typed channels (killing the `m.get("is_water", m.get("is_ocean", …))` smear)
   and contributes its schemes via `register_display`. Channel constants now live in the
   owning packages: `worldgen/{geology,climate,hydrology,resources,cost}/channels.py` and
   `realmgen/channels.py`. `RealmBuilder` is a back-compat façade over `RealmStage` (delegates
   `build`/`apply_display`, exposes `.suitability`/`.situational_value`), so the standalone API
   and its 61 tests are unchanged. Added `realmgen/test/test_stage.py`. All green.
4. **[DONE] Convert the worldgen stages + run `GlobeBuilder` on a `Pipeline`.** Added thin
   `Stage` wrappers (`ElevationStage`, `ClimateStage` + `ClimateLakeFeedbackStage`,
   `HydrologyStage`, `ResourcesStage`, `CostStage`) that adapt the existing builders and
   register each stage's features. `GlobeBuilder.__init__` composes the pipeline; `build()` is
   now `world = World(); self.pipeline.build(ctx); return world` — the ~80-line manual
   orchestration + feature-registration block is gone, the climate two-pass is an explicit
   node. Verified: full suite **295 green**, end-to-end pipeline build, save/load round-trip,
   `apply_display` (debug + non-debug), and realm-on-top.
   - **Deviation from plan:** to keep churn low, the worldgen wrappers live together in
     `worldgen/globe/stages.py` rather than each in its package (a trivial later move), and the
     builders are kept intact + wrapped rather than rewritten. Consequently `apply_display` and
     `serialization.py` are **still centralized** (they read the preserved builder attributes
     and the world features) — their per-stage decomposition is deferred to steps 5/6 below.
5. **[DONE] Decompose `apply_display`** — each stage's slice of the ~300-line god-method now
   lives in its `Stage.register_display` (in `worldgen/globe/stages.py`: elevation, climate,
   hydrology, resources, cost; plus `realmgen`'s Suitability/Situational Value). Added the viz
   adapter `apply_registry(reg, window, world)` (`utils/mapview/registry.py`) — the one place
   that knows both the `DisplayRegistry` shape and the Qt window API. `GlobeBuilder.apply_display`
   is now `reg = DisplayRegistry(); for stage in pipeline: stage.register_display(world, reg,
   debug=debug); apply_registry(reg, window, world)`, importing the adapter lazily so `builder.py`
   stays Qt-free. All 295 tests green.
   - **Deviation from plan:** the hook signature is `register_display(world, reg, *, debug)` (not
     `register_display(reg, *, debug)`) — stages need `world` to read textures/features.
6. **[DONE] Rewrite `serialization.py`** as the core framework + per-stage `serialize`/`load`.
   Added `core/persistence.py` (`NumpyEncoder`, `deep_from_dict`, `save_world`, `load_world`) —
   it owns only the generic slice (params JSON, mesh geometry, `tex_*` textures, face-metadata
   JSON) and stays free of any `worldgen` import. Each worldgen `Stage` now implements
   `serialize() -> dict` / `load(arrays, ctx)`, keeping the **existing array key names**
   (`eb_*`, `cb_*`, `river_*`, `mineral_*`, `cost_*`) so old `.npz` worlds still load; the
   navigability-defaults-to-zeros back-compat lives in `CostStage.load`, and the cost stage's
   absent-key early-return handles pre-cost saves. `worldgen/globe/serialization.py` is now a
   ~40-line shim: `save_globe` → `save_world(builder.globe, params, builder.pipeline, path)`;
   `load_globe` → `load_world(...)` then `for stage in pipeline: stage.load(data, ctx)`. Added
   `worldgen/globe/test/test_serialization.py` (legacy-fixture load + re-save round-trip);
   verified fresh full-build round-trip too. **297 tests green.**
   - **Note:** `HydrologyStage` caches the flattened `(positions, flows, in_lake)` river tuples
     it registers (in both `build` and `load`) because `serialize` has no world access and a
     *loaded* world's `hb.rivers` holds tuples, not `River` objects.
6b. **[DONE] Collapse the wrappers — builders *are* the stages.** Resolved the step-4 deviation:
   `worldgen/globe/stages.py` is deleted and each `*Builder` now **inherits `Stage` directly**
   (`ElevationBuilder`/`ClimateBuilder`/`HydrologyBuilder`/`ResourcesBuilder`/`CostBuilder`),
   owning its own `build(ctx)` + `register_display` + `serialize`/`load`. All `build()` signatures
   collapsed to `build(ctx)` — inputs (mesh, elevation/lake textures) are read from `ctx.world`,
   and the hydrology→resources/cost `River` handoff goes through a new `"river_objects"` world
   feature (the rich objects) alongside the flattened `"rivers"` feature (rendering/serialization),
   so the `hydrology_builder` back-reference is gone. `GlobeBuilder` composes the builders straight
   into the `Pipeline`. The two display-texture helpers moved to their owning display modules
   (`build_elevation_display_texture` → `geology/display.py`, `build_biome_display_texture` →
   `climate/display.py`), re-exported from `worldgen.globe.builder`. The lake-feedback pass is
   `ClimateBuilder.lake_feedback_stage()` → a `ClimateLakeFeedbackStage`. Standalone callers
   (scripts `measure_*`, tests `test_elevation`/`test_cost`) converted to the `build(ctx)` idiom.
   **297 tests green**; verified end-to-end build, save/load round-trip, `apply_display`
   (debug + non-debug), and realm-on-top.
7. **[DONE] `viz/` extraction** — moved `mapview`, `projections`, `params_dialog`, `mesh/gui`
   under `viz/`; `apply_registry` lives in `viz/mapview/registry.py`. Back-compat shims left
   in `utils/mapview/mapview.py`, `utils/params_dialog.py`, and `worldgen/mesh/gui.py`.
   Callers updated: `main.py`, `worldgen/globe/builder.py`, `worldgen/mesh/main.py`,
   `worldgen/main_mesh.py`. `pyproject.toml` testpath updated. **297 tests green.**
8. **`humangen/`** — rename/expand `realmgen` once city/name stages exist; not required for
   the refactor itself.

Each step keeps `python3 -m pytest` and `python3 main.py` green before the next.

## Stays Out of Core

- Params **dialog** (Qt) → `viz`. Only the `paramclass`/`param_field` framework is core.
- `mesh/gui.py`, `mapview`, `projections` → `viz`.
- Anything that knows what a biome, plate, or river *is* — domain stays in `worldgen`/`humangen`.

## Risks / Watch-Items

- **Import churn** when mesh moves (step 2) — many files import `worldgen.mesh.mymesh`. Do it
  as one sweep with tests as the gate.
- **Serialization compatibility** — saved `.npz` worlds in `saved_worlds/` must still load.
  The per-stage `load` must preserve the existing array key names and back-compat defaults.
- **Params dialog introspection** — it walks the nested `GlobeParams` dataclass; keep that
  shape stable so the dialog needs no changes during the refactor.
- **Over-abstraction** — `requires`/`provides` validation and `FeatureLayer` typing are
  deferrable; add them only when a concrete stage needs them.

## Definition of Done (for the proof slice, step 3)

- `RealmStage` runs via `Pipeline` and produces identical `suitability` /
  `situational_value` metadata to today.
- Its display schemes register via `register_display` (no edits to a central god-method).
- It reads `is_water` etc. via `MetaChannel`, with zero `m.get(..., m.get(...))` fallbacks.
- `pytest` green; `main.py` renders the realm schemes as before.
