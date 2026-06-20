# Realmgen — Human Geography Layer

See also: `docs/plans/settlement_generation_design.md` for the original design.

## Overview

Standalone module downstream of worldgen. Takes a fully-built `World` (read-only physical data) and attaches human geography features. Integrated via `RealmPipeline(RealmParams()).build(BuildContext(world=globe))` — not wired into `GlobePipeline`; keeps realmgen optional.

Stage detail docs: [realm_scoring.md](realm_scoring.md) · [realm_placement.md](realm_placement.md) · [realm_population.md](realm_population.md) · [realm_polity.md](realm_polity.md) · [realm_refine.md](realm_refine.md)

## Files

- `realmgen/builder.py` — `RealmParams`, `DangerInfluenceParams`, `RealmPipeline`
- `realmgen/channels.py` — MetaChannel definitions for realm outputs

Stage files are covered in their respective docs above.

## `builder.py`

**`DangerInfluenceParams`**: Cross-stage danger influence — `suitability_weight` (penalty applied to suitability score on high-danger faces), `block_threshold` (placement is blocked above this combined score), `movement_scale` (added to land base cost, making dangerous terrain harder to traverse for trade/hinterland/polity graphs).

**`RealmParams`**: Nested sub-params — `suitability: SuitabilityParams`, `situational: SituationalValueParams`, `seeder: SeederParams`, `trade: TradeParams`, `population: PopulationParams`, `polity: PolityParams`, `refine: RefineParams`, `danger: DangerInfluenceParams`.

**`RealmPipeline`** (extends `Pipeline`): Five-stage pipeline:
1. `ScoringStage` — suitability + situational value (danger penalises suitability)
2. `PlacementStage` — settlement seeding (danger blocks high-danger faces)
3. `PopulationStage` — trade network, relaxation loop, population calibration, defensibility (danger adds traversal cost)
4. `PolityStage` — territory growth + border tension (danger adds traversal cost to authority projection)
5. `RefineStage` — texel-level refinement of settlement positions, trade-route endpoints, and polity borders (geometric only; mutates render features in place). See [realm_refine.md](realm_refine.md).

After `build(ctx)`, each stage stores its results as attributes. `apply_display(window, debug=False)` merges realm color schemes (Suitability, Situational Value, Trade Flow, Hinterland, Polities, Border Tension; plus Sovereigns, Control Class, Authority Field under `--debug`), overlays (Settlements, Trade Routes, Polity Borders), and chains hover with population/polity display. Hover shows the full sovereign-to-leaf chain for multi-level polities (e.g. `Pol: 3 › 7`).

Writes per-face metadata: `"suitability"`, `"situational_value"`, `"settlement_tier"`, `"trade_flow"`, `"is_trade_hub"`, `"population"` (urban headcount in people), `"rural_population"` (per-face people, land only), `"hinterland_id"`, `"polity_id"` (directly-governing), `"sovereign_id"` (root), `"control_class"`, `"border_tension"`, `"authority"` (winning authority value per face — control strength; consumed by `RefineStage` to bias border refinement).

Registers features: `"settlements"` (list of `Settlement` with `.population`, `.defensibility`, `.has_walls`, `.wall_strength` set), `"trade_routes"`, `"polities"` (list of `Polity`), `"polity_history"` (list of `FragmentationEvent` — the Stage 5 secession chronicle for later history generation), `"polity_tensions"` (list of `(a, b, score)`), `"polity_borders"` (`dict[depth, np.ndarray]` with border segment endpoints keyed by hierarchy depth).

## Key design constraint

The land cost surface is land-only (`land_passable = ~is_water`), so land authority and claiming use a land-only graph — coastlines become natural borders. The sea authority channel reuses the hybrid land/sea graph (`realmgen/trade/network.py::build_base_hybrid_graph`): a high-`maritime_reach` coastal capital projects authority along coastlines and onto reachable islands; claiming stays restricted to land faces. Maritime reach scales with capital trade wealth, so a landlocked capital contributes none.
