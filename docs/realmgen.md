# Realmgen — Human Geography Layer

See also: `docs/plans/settlement_generation_design.md` for the original design.

## Overview

Standalone module downstream of worldgen. Takes a fully-built `World` (read-only physical data) and attaches human geography features. Integrated via `RealmPipeline(RealmParams()).build(BuildContext(world=globe))` — not wired into `GlobePipeline`; keeps realmgen optional.

Stage detail docs: [realm_scoring.md](realm_scoring.md) · [realm_placement.md](realm_placement.md) · [realm_population.md](realm_population.md) · [realm_polity.md](realm_polity.md) · [realm_refine.md](realm_refine.md) · [realm_towns.md](realm_towns.md)

## Files

- `realmgen/builder.py` — `RealmParams`, `DangerInfluenceParams`, `RealmPipeline`
- `realmgen/channels.py` — MetaChannel definitions for realm outputs
- `realmgen/persistence.py` — pack/unpack helpers for the realm features (`settle_*`, `pol_*`, `route_*`, `border_*` keys)

Stage files are covered in their respective docs above.

## `builder.py`

**`DangerInfluenceParams`**: Cross-stage influence of the worldgen danger field ([danger.md](danger.md)) — `suitability_weight` (0.5; suitability × `(1 − w · danger)`), `block_threshold` (0.65; no settlement is placed on a face this dangerous), `movement_scale` (2.0; × danger added to the land base cost, making dangerous terrain harder to traverse for the placement, trade, hinterland and polity graphs; the region roads charge it too).

**`RealmParams`**: Nested sub-params — `suitability: SuitabilityParams`, `situational: SituationalValueParams`, `seeder: SeederParams`, `trade: TradeParams`, `population: PopulationParams`, `polity: PolityParams`, `refine: RefineParams`, `towns: TownParams`, `danger: DangerInfluenceParams`.

**`RealmPipeline`** (extends `Pipeline`): Six-stage pipeline:
1. `ScoringStage` — suitability + situational value (danger penalises suitability)
2. `PlacementStage` — settlement seeding (danger blocks high-danger faces)
3. `PopulationStage` — trade network, relaxation loop, population calibration, defensibility (danger adds traversal cost)
4. `PolityStage` — territory growth + border tension (danger adds traversal cost to authority projection)
5. `RefineStage` — texel-level refinement (seeded from `World.seed`): sites each city on the best texel of its face (with a minimum city spacing), remaps trade-route endpoints, smooths polity borders into continuous curves, and builds a texel-accurate authority-weighted polity fill texture. See [realm_refine.md](realm_refine.md).
6. `TownStage` — display-only towns (1k people up to the city floor) carved out of the rural population, plus a per-texel rural density texture. See [realm_towns.md](realm_towns.md).

After `build(ctx)`, each stage stores its results as attributes and contributes its display through `Stage.register_display` (color schemes, overlays, and per-stage hover fragments), exactly like the worldgen stages. There is no bespoke realm display path: `Pipeline.apply_display` (inherited) builds **one** `DisplayRegistry` across the whole nested pipeline and wires it via `apply_registry` in a single pass.

- Scheme: **Polities** (the texel-level fill registers as a per-scheme texture, with a recipe the region view reruns). Under `--debug` also Suitability, Situational Value, Trade Flow, Hinterland, Border Tension, Sovereigns, Control Class, Authority Field, Rural Density.
- Overlays (all checked): Settlements, Trade Routes, Polity Borders, Towns (region and local views only).
- Hover: settlement tier/population, the full sovereign-to-leaf chain for multi-level polities (e.g. `Pol: 3 › 7`), control class, towns on the face; under `--debug` also suitability and border tension.

**Serialization.** Each realm stage implements `serialize()`/`load()` so a saved world round-trips its human geography (no realm rebuild on load). Per-face fields persist generically via mesh metadata; the realm *features* are packed by `realmgen/persistence.py`: `PopulationStage` owns `settlements` (final, refined positions), `PolityStage` owns the `polities` forest, `RefineStage` owns the final `trade_routes` + `polity_borders` geometry (captured post-refine), and `TownStage` owns `towns`. The `polity_fill` RGBA texture persists generically as a mesh texture (it is not regenerated on load). `load()` re-derives per-face stage attrs from the restored mesh and re-adds the world features. The app-level save/load is `main.py` calling `core.persistence.save_world` / `load_world` over the nested `AppPipeline` + `AppParams`; `worldgen.save_globe`/`load_globe` remain physical-world-only for the measurement scripts that cache the globe and re-run realm. The refined town and city positions (region and local tiers) are saved with the `settlements` and `towns` features.

Writes per-face metadata: `"suitability"`, `"situational_value"`, `"settlement_tier"` (display tier from population), `"trade_flow"`, `"is_trade_hub"`, `"population"` (urban headcount in people), `"rural_population"` (per-face countryside people, land only — net of towns once `TownStage` has run), `"town_population"` (per-face people in towns), `"hinterland_id"`, `"polity_id"` (directly-governing), `"sovereign_id"` (root), `"control_class"`, `"border_tension"`, `"authority"` (winning authority value per face — control strength; consumed by `RefineStage` to bias the texel-level polity fill).

Adds face textures: `"polity_fill"` (RGBA, refine stage) and `"rural_density"` (people/km², town stage).

Registers features: `"settlements"` (list of `Settlement` with `.population`, `.defensibility`, `.has_walls`, `.wall_strength` set), `"trade_routes"`, `"trade_attribution"` (`(S, S)` inter-settlement trade matrix), `"polities"` (list of `Polity`), `"polity_history"` (list of `FragmentationEvent` — the Stage 5 secession chronicle for later history generation), `"polity_tensions"` (list of `(a, b, score)`), `"polity_borders"` (`dict[depth, np.ndarray]` with border segment endpoints keyed by hierarchy depth), `"towns"` (`Towns` parallel arrays), and the in-memory `"polity_label_field"` (not saved; rebuilt on load).

## Key design constraint

The land cost surface is land-only (`land_passable = ~is_water`), so land authority and claiming use a land-only graph — coastlines become natural borders. The sea authority channel reuses the hybrid land/sea graph (`realmgen/trade/network.py::build_base_hybrid_graph`): a high-`maritime_reach` coastal capital projects authority along coastlines and onto reachable islands; claiming stays restricted to land faces. Maritime reach scales with capital trade wealth, so a landlocked capital contributes none.
