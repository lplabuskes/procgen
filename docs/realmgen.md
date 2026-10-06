# Realmgen — Human Geography Layer

Original design: `docs/plans/settlement_generation_design.md`.

## Overview

Optional module downstream of worldgen: takes a built `World` (physical data
read-only) and attaches human geography. `RealmPipeline(RealmParams()).build(BuildContext(world=globe))`
— not part of `GlobePipeline`.

Stage docs: [realm_scoring.md](realm_scoring.md) · [realm_placement.md](realm_placement.md) · [realm_population.md](realm_population.md) · [realm_polity.md](realm_polity.md) · [realm_refine.md](realm_refine.md) · [realm_towns.md](realm_towns.md)

## Files

- `realmgen/builder.py` — `RealmParams`, `DangerInfluenceParams`, `RealmPipeline`
- `realmgen/channels.py` — MetaChannels for realm outputs
- `realmgen/persistence.py` — pack/unpack for realm features (`settle_*`, `pol_*`, `route_*`, `border_*` keys)

## `builder.py`

**`DangerInfluenceParams`** — cross-stage use of the danger field ([danger.md](danger.md)):
- `suitability_weight` (0.5) — suitability × `(1 − w · danger)`
- `block_threshold` (0.65) — no settlement on faces this dangerous
- `movement_scale` (2.0) — × danger added to land base cost (placement, trade, hinterland, polity graphs; region roads too)

**`RealmParams`**: `suitability`, `situational`, `seeder`, `trade`, `population`,
`polity`, `refine`, `towns`, `danger`.

**`RealmPipeline`**:
1. `ScoringStage` — suitability + situational value
2. `PlacementStage` — settlement seeding
3. `PopulationStage` — trade, relaxation loop, population, defensibility
4. `PolityStage` — territory, hierarchy, forts (then territory again), border tension
5. `RefineStage` — texel-level city sites, route endpoints, smooth borders, polity fill
6. `TownStage` — display-only towns carved from rural population; rural density texture

Display goes through the ordinary `Stage.register_display` / inherited
`Pipeline.apply_display` (one `DisplayRegistry` across the nested pipeline; no
bespoke realm path).
- Scheme **Polities** (texel fill as texture + recipe the region view reruns). `--debug`: Suitability, Situational Value, Trade Flow, Hinterland, Border Tension, Sovereigns, Control Class, Authority Field, Rural Density.
- Overlays (checked): Settlements, Trade Routes, Polity Borders, Forts, Towns (region/local only).
- Hover: settlement tier/population, sovereign-to-leaf chain (`Pol: 3 › 7`), control class, towns; `--debug` adds suitability and border tension.

**Serialization.** Each stage implements `serialize()`/`load()` (no realm rebuild
on load). Per-face fields persist via mesh metadata; features via
`realmgen/persistence.py`: `PopulationStage` owns `settlements` (final, refined
positions incl. region/local tiers), `PolityStage` `polities`, `RefineStage` final
`trade_routes` + `polity_borders`, `TownStage` `towns` (incl. refined positions).
`polity_fill` persists as a mesh texture (not regenerated). `load()` re-derives
stage attrs and re-adds features. App save/load: `main.py` →
`core.persistence.save_world` / `load_world` over `AppPipeline` + `AppParams`;
`worldgen.save_globe`/`load_globe` stay physical-only (scripts cache the globe and
re-run realm).

Per-face metadata: `suitability`, `situational_value`, `settlement_tier` (display
tier), `trade_flow`, `is_trade_hub`, `population` (urban people),
`rural_population` (land only; net of towns after `TownStage`),
`town_population`, `hinterland_id`, `polity_id` (direct), `sovereign_id` (root),
`control_class`, `border_tension`, `authority` (winning value; biases the polity
fill).

Face textures: `polity_fill` (RGBA, refine), `rural_density` (people/km², towns).

Features: `settlements`, `trade_routes`, `trade_attribution` (`(S, S)`),
`polities`, `polity_history` (`FragmentationEvent` list), `polity_tensions`
(`(a, b, score)`), `polity_borders` (`{depth: segments}`), `towns`, and in-memory
`polity_label_field` (rebuilt on load).

## Key design constraint

Land cost is land-only (`land_passable = ~is_water`), so land authority and
claiming use a land-only graph — coastlines are natural borders. The sea channel
reuses the hybrid graph (`realmgen/trade/network.py::build_base_hybrid_graph`): a
high-`maritime_reach` coastal capital projects along coasts and onto islands, but
only land faces are claimed. Maritime reach scales with capital trade wealth
(landlocked → none).
