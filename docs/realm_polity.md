# Realm Polity Stage

Part of `realmgen/`. [Back to realmgen overview](realmgen.md)

## Overview

A polity is a **tree, not a flat partition** (`docs/plans/polity_generation_brief.md`): sovereigns are roots, vassals are children. The stage seeds sovereign capitals, projects an authority field from each over the land + sea cost surfaces, lets every face be claimed by the strongest projector, then classifies each settlement direct / vassal / independent by comparing projected authority against local power — recursing to carve vassal sub-territories. Per-polity scalars (`cohesion`, `infrastructure`, `legitimacy`, `maritime_reach`) — rolled from a per-capital RNG substream keyed on `world.seed` + capital face — modulate the field; their cross-product is the main generator of variety (Song ↔ Mongol ↔ HRE ↔ city-state belt). **Stage 5 fragmentation epochs** then run a few historical shocks on top: over-extended low-cohesion sovereigns shed peripheral seats as new sovereigns and the partition is re-resolved, producing successor-state texture.

## Files

- `realmgen/polity/builder.py` — `PolityStage`, `PolityParams` (IO, cost graphs, tension, display)
- `realmgen/polity/forest.py` — `ForestBuilder` + `_Ctx` (the sovereign/vassal tree algorithm)
- `realmgen/polity/epochs.py` — `FragmentationParams` + `run_fragmentation_epochs` (Stage 5)
- `realmgen/polity/seeding.py` — defensibility, capital seeding, scalar rolls (Stage 0 pure functions)
- `realmgen/polity/authority.py` — authority field, claiming, classification (Stages 1-3 pure functions)
- `realmgen/polity/loyalty.py` — `LoyaltyParams` + economic-grip / peripheral-wealth pure functions (Stage 4 levers)
- `realmgen/polity/tension.py` — border-crossing tension
- `realmgen/polity/display.py` — color schemes
- `realmgen/polity/types.py` — `Polity` / `Scalars` / `ControlClass` + `POLITY_COLORS` palette

## `PolityParams`

Sub-paramclasses (consumed location noted):

- **`seeding: CapitalSeedingParams`** (builder.py) — `min_separation` (land cost-distance between capitals), `min_pop_fraction` (of the largest city), `trade_weight`
- **`defensibility: DefensibilityParams`** (builder.py, but consumed by `PopulationStage`) — `rugged_weight`, `situational_weight`, `river_weight`, `wall_base_prob`, `wall_danger_boost`, `wall_wealth_boost`, `wall_min_strength`, `wall_max_strength`. Governs `compute_settlement_defensibility` called from `PopulationStage`; see [realm_population.md](realm_population.md#defensibility).
- **`projection: ProjectionParams`** (forest.py) — `base_range` (master consolidation knob: larger → fewer, bigger polities; land range = `base_range · (range_floor + infrastructure)`), `range_floor`, `sea_range_factor` (× land range for the sea channel; 1.4 — dropped from 1.5 when trade was restricted to settlement ports, which concentrates trade, and so `maritime_reach`, at port capitals), `sea_weight` (sea-channel magnitude vs land), `legitimacy_spread` (±, field magnitude jitter), `maritime_scale` (maritime_reach ← scale × trade_norm), `cohesion_min`, `infrastructure_min`, `infrastructure_emergent` (weight of pop+trade in infrastructure roll)
- **`claiming: ClaimingParams`** (forest.py) — `hysteresis` (incumbent face-claim bonus), `max_iters` (claim inner loop), `outer_max_iters` (independence-relaxation outer loop — the loop promotes independents to new sovereigns until none remain or this bound is hit; there is no cap on the sovereign count)
- **`hierarchy: HierarchyParams`** (forest.py) — `direct_threshold` (authority ratio to hold directly; 0.8 — the infra-correlation knob, lower → more negative `corr(infrastructure, vassal_frac)`), `vassal_floor` (authority ratio floor for vassal), `power_exponent` (0.5 = sqrt of raw population, dampens size dominance), `cohesion_gain_floor`, `cohesion_gain_span`, `max_recursion_depth` (2), `vassal_cluster_distance` (15; vassal candidates within this land cost-distance of each other are merged, the largest becoming the one vassal seat; 0 = off)
- **`loyalty: LoyaltyParams`** (loyalty.py) — Stage-4 trade-attribution levers, both **off (`0.0`) by default**: `economic_grip` (scales secession-resistance up by intra-polity trade fraction → trade blocs hold), `peripheral_wealth_ratio` (a non-capital settlement ≥ this × its claiming capital's trade wealth breaks away; `0` = off), `peripheral_external_floor` (min external trade fraction to qualify). With no sovereign cap the levers are cleanly monotonic — more `economic_grip` → fewer sovereigns / bigger domains; more aggressive peripheral thresholds → more sovereigns.
- **`fragmentation: FragmentationParams`** (epochs.py) — `n_epochs` (historical shocks; 0 = off), `shock_scale` (max downward cohesion roll per epoch), `size_reference` (domain face count demanding full cohesion)
- **`tension: BorderTensionParams`** (builder.py) — `barrier_scale`, `flow_ref`, `artery_weight`

## Types (`polity/types.py`)

**`Polity`**: `polity_id`, `capital_face`, `capital_position` (unit-sphere xyz), the four scalars (`cohesion`, `infrastructure`, `legitimacy`, `maritime_reach`), `liege` (parent polity id; None ⇒ sovereign root), `autonomy`, `depth` (0 = sovereign root), `power` (own-territory member-settlement population), `domain_power` (own + all descendant-vassal member population), `n_faces` (own-territory face count), `territory_faces`, `member_settlement_faces`, `vassals` (child polity ids).

**`ControlClass`** (IntEnum): `DIRECT` / `VASSAL` / `INDEPENDENT` — how a settlement relates to the polity that claims it.

**`Scalars`** (NamedTuple): four per-polity arrays with `.take(idx)` sub-select.

**`POLITY_COLORS`**: 15-entry palette (crimson → orange → gold → greens → blues → purples); yellow-green band intentionally skipped. `UNCLAIMED_COLOR` for land faces no polity reached.

## Stage 0 pure functions (`polity/seeding.py`)

- `compute_defensibility(land_base_cost, situational_value)` — simple terrain-only formula (ruggedness + situational value). Kept for its tests; the stages use the per-settlement version below.
- `compute_settlement_defensibility(settlement_faces, land_base_cost, situational_value, river_flow_mag, trade_wealth, danger, land_mask, base_seed, ...)` — per-settlement defensibility combining terrain, river barriers, and probabilistic walls; called from `PopulationStage`.
- `seed_capitals(settlement_faces, populations, cost_graph, *, weights, min_separation, min_pop_fraction)` — picks which settlements found sovereign polities: iterates in descending `weights` order (`population × (1 + trade_weight · trade_norm) × defensibility`), skipping any below `min_pop_fraction` of the peak population or within `min_separation` (land Dijkstra) of an accepted capital
- `roll_polity_scalars(capital_faces, capital_pop_norm, capital_trade_norm, base_seed, ...)` — per-capital RNG keyed on `(base_seed, face)` → `Scalars`
- `cohesion_to_gain(cohesion)` — maps cohesion in [0,1] → authority-gain multiplier in `[floor, floor+span]`

## Stages 1-3 (`polity/authority.py`)

- `compute_authority_fields` — `(C, N)`: `legitimacy · exp(−land_dist / land_range)`, `land_range = base_range · (range_floor + infrastructure)`, max-combined with a sea channel `sea_weight · legitimacy · maritime_reach · exp(−hybrid_dist / (sea_range_factor · land_range))` over the hybrid land/sea graph
- `claim_faces` — argmax-authority face assignment with hysteresis bonus for incumbents, iterated to bounded fixed point
- `classify_settlements` — direct/vassal/independent from the ratio `r = capital_pop^e · field[claimant, face] · cohesion_gain` / `pop^e · defensibility` (`e = power_exponent`): `r ≥ direct_threshold` DIRECT, `r ≥ vassal_floor` VASSAL, else INDEPENDENT; unclaimed settlements are INDEPENDENT. Two optional Stage-4 hooks: `loyalty_mult` (scales the ratio per settlement — economic grip) and `force_independent` (overrides a non-capital settlement to INDEPENDENT — peripheral-wealth secession). Capitals are pinned DIRECT last, so neither lever can dislodge a seat.

## Stage 4 loyalty levers (`polity/loyalty.py`)

Two **opposite-direction** modifiers keyed on the `trade_attribution` matrix + the live claim partition; both answer "who does this city's economy depend on?" and adjust *secession resistance* without touching the authority field. Applied in `ForestBuilder._loyalty_modifiers` during sovereign resolution, so they shape which settlements become independent sovereigns.

- `compute_internal_trade_fraction(settlement_faces, partition, trade_attribution)` — per-settlement fraction of bilateral trade (`attr + attr.T`) with partners in the same claim cell; `0` for an untraded or unclaimed settlement. **Economic grip:** `economic_grip_multiplier(frac, strength) = 1 + strength·frac` scales the classification ratio up, so a trade-bloc-bound city resists secession even under a weak field.
- `peripheral_wealth_breakaways(...)` — flags a non-capital settlement that is claimed, outward-facing (`1 − internal ≥ external_floor`), and ≥ `wealth_ratio` × its claiming capital's trade wealth → forced INDEPENDENT (a rich, externally-trading edge breaks off as a new sovereign). `wealth_ratio ≤ 0` disables it.

Note the partition is recomputed each relaxation pass, so the internal fraction (hence grip) is measured against the *current* partition — as a realm fragments the grip holding its periphery weakens (a deliberate feedback).

## Forest algorithm (`polity/forest.py`)

`ForestBuilder.build` threads the pure functions into the tree:

1. `_resolve_sovereigns` — roll scalars, project, claim, classify; every INDEPENDENT settlement becomes a new capital and the loop repeats (up to `outer_max_iters`).
2. `_build_subtree` per sovereign, recursing to `max_recursion_depth` — members whose ratio against this polity's own field falls below `direct_threshold` are vassal candidates (clustered by `vassal_cluster_distance`); the territory is re-claimed between the capital and its vassal seats (the parent keeps its scalars), and each vassal's share recurses. `autonomy = 1 − r / direct_threshold`.
3. Polities are renumbered sovereigns first; `authority_of_face` is the max field over the sovereign capitals.

`_Ctx` holds read-only shared inputs (centroids, settlement arrays, graphs, params, `base_seed = world.seed`).

## Fragmentation epochs (`polity/epochs.py`, Stage 5)

`run_fragmentation_epochs` builds the base forest, then runs `n_epochs` historical shocks on top. Each epoch, `select_fragmentations` rolls a per-(seed, capital, epoch) cohesion shock `U[0, shock_scale]` for every sovereign and fragments those whose `cohesion - shock < demand`, where `demand = min(domain_face_count / size_reference, 1)` (bigger domains demand more cohesion). A fragmenting sovereign sheds **one** peripheral seat — its weakest-held vassal (highest `autonomy`) if it has vassals, else the lowest-`authority_of_face` directly-held settlement (the interior **saddle**). The shed seats (only those not already sovereign) are added to the capital set and Stages 2-4 are re-run via a fresh `ForestBuilder`, so the empire retreats to its strong core while breakaways hold the periphery beyond the saddle. Domains shrink each shed, so demand drops and the process converges (early-out once an epoch is stable). With `n_epochs == 0` the result is the plain Stage 0-3 forest. This is a modifier over the base mechanism, never a rewrite.

Each shed is recorded as a `FragmentationEvent` (epoch, parent + breakaway capital faces, `kind` ∈ {`vassal_secession`, `saddle_split`}, parent cohesion / shock / demand / domain face count). `run_fragmentation_epochs` returns `(ForestBuilder, list[FragmentationEvent])`; `PolityStage` exposes the chronicle as `self.polity_history` and the `polity_history` world feature. Faces are stable identifiers, so both endpoints map to final sovereigns by capital face — the chronicle is intended as raw material for a later history-generation pass (who broke away from whom, in which epoch).

## Tension (`polity/tension.py`)

`compute_border_tension` — scored on the final sovereign partition. Border edges with high cross-border trade flow (normalised to the 90th-percentile face flow) and large polities on both sides (sovereigns weigh in with `domain_power`) produce the highest tension; costly crossings (`barrier_scale`) damp it. Scored once after the tree is fully resolved.

## PolityStage

Fourth of six in `RealmPipeline` (after population, before refine). Reads `IS_WATER`, `COST_SURFACES`, `TRADE_FLOW`, the `settlements` feature (populations and the `Settlement.defensibility` stamped by `PopulationStage`), `trade_attribution` (loyalty levers) and `danger`. Danger × `DangerInfluenceParams.movement_scale` is added to the land base cost before the land edge costs (`build_land_edge_costs`) and the hybrid land/sea graph (`build_base_hybrid_graph`, with the wind-directional sea edges) are built.

Builds per-face defensibility array from `Settlement.defensibility` (non-settlement faces get 1.0). Seeds capitals, builds `_Ctx` and runs `run_fragmentation_epochs` (which builds the `ForestBuilder`). Computes border tension on the sovereign partition. With no settlements or no qualifying capital everything is written empty.

## Outputs

Per-face: `POLITY_ID` (directly-governing polity), `SOVEREIGN_ID` (root), `CONTROL_CLASS`, `BORDER_TENSION`, `AUTHORITY` (the winning field value — `RefineStage` weights the polity label field by it).
Features: `polities` (the full forest), `polity_history` (ordered list of `FragmentationEvent` — the Stage 5 secession chronicle, kept for later history generation; faces are stable ids that map to final sovereigns), `polity_tensions` (list of `(a, b, score)`), `polity_borders` (`dict[int, np.ndarray shape (M, 6)]` — border segment endpoints as `(xyz, xyz)`, keyed by hierarchy depth; depth 0 = sovereign borders, depth d > 0 = internal subtree borders within the same depth-d parent). These are raw hexagon edges; `RefineStage` replaces them with smooth polylines.

Display: the **Polities** scheme (the refine stage's `polity_fill` texture and `polity_fill_recipe` when present, else `make_polity_tree_color`: each sovereign a `POLITY_COLORS` hue, vassals lighter tints of it) and the **Polity Borders** overlay; hover shows the sovereign-to-leaf chain (`Pol: 3 › 7`) and control class. Debug: Border Tension, Sovereigns, Control Class, Authority Field schemes and a tension hover fragment.

Validation: `scripts/measure_polities.py` (sovereign/vassal counts, domain consolidation, max depth, `corr(infrastructure, vassal_fraction)`).
