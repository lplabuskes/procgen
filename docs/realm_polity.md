# Realm Polity Stage

Part of `realmgen/`. [Back to realmgen overview](realmgen.md)

## Overview

A polity is a **tree, not a flat partition** (`docs/plans/polity_generation_brief.md`): sovereigns are roots, vassals children.
- Seed sovereign capitals; project an authority field from each over land + sea cost surfaces; each face goes to the strongest projector.
- Classify each settlement direct / vassal / independent (projected authority vs local power), recursing to carve vassal sub-territories.
- Per-polity scalars (`cohesion`, `infrastructure`, `legitimacy`, `maritime_reach`), rolled per capital from `world.seed` + capital face, modulate the field; their cross-product is the main source of variety (Song ↔ Mongol ↔ HRE ↔ city-state belt).
- **Stage 5 fragmentation epochs**: over-extended low-cohesion sovereigns shed peripheral seats as new sovereigns, re-resolved → successor states.

## Files

- `realmgen/polity/builder.py` — `PolityStage`, `PolityParams` (IO, cost graphs, tension, display)
- `realmgen/polity/forest.py` — `ForestBuilder` + `_Ctx` (tree algorithm)
- `realmgen/polity/epochs.py` — `FragmentationParams`, `run_fragmentation_epochs` (Stage 5)
- `realmgen/polity/seeding.py` — defensibility, capital seeding, scalar rolls (Stage 0)
- `realmgen/polity/authority.py` — authority field, claiming, classification (Stages 1-3)
- `realmgen/polity/loyalty.py` — `LoyaltyParams`, economic grip / peripheral wealth (Stage 4)
- `realmgen/polity/tension.py` — border tension
- `realmgen/polity/display.py` — colour schemes
- `realmgen/polity/types.py` — `Polity` / `Scalars` / `ControlClass`, `POLITY_COLORS`

## `PolityParams`

- **`seeding`** (`CapitalSeedingParams`) — `min_separation` (land cost-distance between capitals), `min_pop_fraction` (of largest city), `trade_weight`.
- **`defensibility`** (`DefensibilityParams`; consumed by `PopulationStage` via `compute_settlement_defensibility`) — terrain/river weights and wall probability/strength knobs; see [realm_population.md](realm_population.md#defensibility).
- **`projection`** (`ProjectionParams`) — `base_range` (master consolidation knob; land range = `base_range · (range_floor + infrastructure)`), `range_floor`, `sea_range_factor` (× land range for the sea channel; 1.4, down from 1.5 when trade was restricted to settlement ports, which concentrates `maritime_reach` at port capitals), `sea_weight`, `legitimacy_spread` (± magnitude jitter), `maritime_scale` (`maritime_reach ← scale × trade_norm`), `cohesion_min`, `infrastructure_min`, `infrastructure_emergent` (weight of pop+trade in the infrastructure roll).
- **`claiming`** (`ClaimingParams`) — `hysteresis` (incumbent bonus), `max_iters` (claim loop), `outer_max_iters` (independence-relaxation loop; promotes independents to sovereigns until none remain — **no cap on sovereign count**).
- **`hierarchy`** (`HierarchyParams`) — `direct_threshold` (0.8; the infra-correlation knob: lower → more negative `corr(infrastructure, vassal_frac)`), `vassal_floor` (vassal ratio floor), `power_exponent` (0.5; sqrt of raw population dampens size), `cohesion_gain_floor` / `_span`, `max_recursion_depth` (2), `vassal_cluster_distance` (15; candidates within this land cost-distance merge, largest becomes the seat; 0 = off).
- **`loyalty`** (`LoyaltyParams`, Stage 4; both **off (0.0) by default**) — `economic_grip` (secession resistance ∝ intra-polity trade fraction), `peripheral_wealth_ratio` (non-capital ≥ this × its capital's trade wealth breaks away), `peripheral_external_floor` (min external trade fraction). Monotonic: more grip → fewer, bigger domains; more aggressive peripheral thresholds → more sovereigns.
- **`fragmentation`** (`FragmentationParams`) — `n_epochs` (0 = off), `shock_scale` (max cohesion drop per epoch), `size_reference` (domain face count demanding full cohesion).
- **`tension`** (`BorderTensionParams`) — `barrier_scale`, `flow_ref`, `artery_weight`.

## Types (`polity/types.py`)

- **`Polity`**: `polity_id`, `capital_face`, `capital_position` (xyz), the four scalars, `liege` (None ⇒ sovereign), `autonomy`, `depth` (0 = sovereign), `power` (own-territory member population), `domain_power` (own + descendant vassals), `n_faces`, `territory_faces`, `member_settlement_faces`, `vassals`.
- **`ControlClass`** (IntEnum): `DIRECT` / `VASSAL` / `INDEPENDENT`.
- **`Scalars`** (NamedTuple): four per-polity arrays, `.take(idx)`.
- **`POLITY_COLORS`**: 15 hues (yellow-green deliberately skipped); `UNCLAIMED_COLOR`.

## Stage 0 (`polity/seeding.py`)

- `compute_defensibility(...)` — terrain-only; kept for its tests (stages use the per-settlement one).
- `compute_settlement_defensibility(...)` — terrain + river barriers + probabilistic walls; called from `PopulationStage`.
- `seed_capitals(settlement_faces, populations, cost_graph, *, weights, min_separation, min_pop_fraction)` — descending `population × (1 + trade_weight · trade_norm) × defensibility`, skipping those < `min_pop_fraction` of peak or within `min_separation` (land Dijkstra) of an accepted capital.
- `roll_polity_scalars(...)` — per-capital RNG keyed `(base_seed, face)` → `Scalars`.
- `cohesion_to_gain(cohesion)` — [0,1] → gain in `[floor, floor+span]`.

## Stages 1-3 (`polity/authority.py`)

- `compute_authority_fields` → `(C, N)`: `legitimacy · exp(−land_dist / land_range)`, max-combined with sea channel `sea_weight · legitimacy · maritime_reach · exp(−hybrid_dist / (sea_range_factor · land_range))` over the hybrid land/sea graph.
- `claim_faces` — argmax authority with incumbent hysteresis, iterated to a bounded fixed point.
- `classify_settlements` — `r = capital_pop^e · field[claimant, face] · cohesion_gain / (pop^e · defensibility)`: `r ≥ direct_threshold` DIRECT, `≥ vassal_floor` VASSAL, else INDEPENDENT (unclaimed → INDEPENDENT). Stage-4 hooks: `loyalty_mult` (per-settlement ratio scale) and `force_independent`. Capitals pinned DIRECT last, so no lever dislodges a seat.

## Stage 4 loyalty levers (`polity/loyalty.py`)

Opposite-direction modifiers from `trade_attribution` + the live claim partition
("who does this city's economy depend on?"); they change *secession resistance*,
not the field. Applied in `ForestBuilder._loyalty_modifiers` during sovereign
resolution.

- `compute_internal_trade_fraction(...)` — share of bilateral trade (`attr + attr.T`) with partners in the same claim cell (0 if untraded/unclaimed). Economic grip: `1 + strength·frac` × ratio.
- `peripheral_wealth_breakaways(...)` — claimed non-capital, outward-facing (`1 − internal ≥ external_floor`) and ≥ `wealth_ratio` × its capital's trade wealth → forced INDEPENDENT. `wealth_ratio ≤ 0` disables.

The partition is recomputed each relaxation pass, so grip weakens as a realm
fragments (deliberate feedback).

## Forest algorithm (`polity/forest.py`)

`ForestBuilder.build`:
1. `_resolve_sovereigns` — roll, project, claim, classify; every INDEPENDENT becomes a capital; repeat (≤ `outer_max_iters`).
2. `_build_subtree` per sovereign to `max_recursion_depth`: members below `direct_threshold` against this polity's field are vassal candidates (clustered by `vassal_cluster_distance`); territory re-claimed between capital and vassal seats (parent keeps its scalars); each vassal's share recurses. `autonomy = 1 − r / direct_threshold`.
3. Renumber sovereigns first; `authority_of_face` = max field over sovereign capitals.

`_Ctx`: read-only shared inputs (centroids, settlements, graphs, params, `base_seed = world.seed`).

## Fragmentation epochs (`polity/epochs.py`, Stage 5)

`run_fragmentation_epochs` builds the base forest, then `n_epochs` shocks
(a modifier over the base mechanism, never a rewrite):
- `select_fragmentations`: per (seed, capital, epoch) shock `U[0, shock_scale]`; fragment if `cohesion − shock < demand`, `demand = min(domain_face_count / size_reference, 1)`.
- A fragmenting sovereign sheds **one** seat: its highest-`autonomy` vassal, else its lowest-`authority_of_face` direct settlement (interior **saddle**).
- New (non-sovereign) shed seats join the capital set; Stages 2-4 re-run via a fresh `ForestBuilder` — the empire retreats to its core, breakaways hold the periphery. Domains shrink so it converges (early-out when stable). `n_epochs == 0` → plain forest.
- Each shed → `FragmentationEvent` (epoch, parent + breakaway capital faces, `kind` ∈ {`vassal_secession`, `saddle_split`}, parent cohesion/shock/demand/domain size). Returns `(ForestBuilder, events)`; `PolityStage` exposes them as `self.polity_history` and the `polity_history` feature — raw material for later history generation (faces are stable ids mapping to final sovereigns).

## Tension (`polity/tension.py`)

`compute_border_tension` on the final sovereign partition: high cross-border
trade flow (normalised to the 90th-percentile face flow) and large polities on
both sides (`domain_power`) → high tension; costly crossings (`barrier_scale`)
damp it.

## PolityStage

Fourth of six in `RealmPipeline`. Reads `IS_WATER`, `COST_SURFACES`,
`TRADE_FLOW`, `settlements` (populations, `Settlement.defensibility` from
`PopulationStage`; non-settlement faces 1.0), `trade_attribution`, `danger`.
Danger × `DangerInfluenceParams.movement_scale` is added to land base cost before
building land edge costs (`build_land_edge_costs`) and the hybrid graph
(`build_base_hybrid_graph`, wind-directional sea edges). Seeds capitals, runs
`run_fragmentation_epochs`, computes tension. No settlements / no qualifying
capital → everything written empty.

## Outputs

- Per face: `POLITY_ID` (direct governor), `SOVEREIGN_ID`, `CONTROL_CLASS`, `BORDER_TENSION`, `AUTHORITY` (winning field value; weights `RefineStage`'s label field).
- Features: `polities` (forest), `polity_history` (`FragmentationEvent` list), `polity_tensions` (`(a, b, score)`), `polity_borders` (`{depth: (M, 6)}` raw hexagon-edge segments; 0 = sovereign, d > 0 = internal within a depth-d parent; replaced by smooth polylines in `RefineStage`).

Display: **Polities** scheme (refine's `polity_fill` + `polity_fill_recipe` when
present, else `make_polity_tree_color`: sovereign hue, vassals lighter tints),
**Polity Borders** overlay; hover `Pol: 3 › 7` + control class. Debug: Border
Tension, Sovereigns, Control Class, Authority Field schemes, tension hover.

Validation: `scripts/measure_polities.py`.
