# Polity Generation Module — Implementation Brief

## Module boundary

Polity generation is a **human-geography** module. It reads the physical world and the
settlement layer as **read-only input** and writes a polity layer on top. No back-coupling
into the physical simulation. Downstream consumers (trade, border forts) read the polity
layer; they do not feed into it during generation.

**Inputs (read-only) — actual availability (see per-stage flags for detail):**
- **Settlements** (`world.get_feature("settlements")` → list of `Settlement`):
  `face_id`, `position`, `tier`, `score`, `population`, `defensibility`, `has_walls`,
  `wall_strength`. `tier` is a *seeding label* nothing downstream reads — key off `population`.
  `defensibility` is the composite multiplier (terrain + river + walls) stamped by
  `PopulationStage`. **No** `archetype` or water-adjacency fields exist.
- **Per-settlement trade wealth**: *not* on the record. It is `population_stage.trade_wealth`
  (an `(S,)` array aligned with `settlements`) and per-face via the `TRADE_FLOW` channel
  (= flow sampled at the settlement face).
- **Per-settlement trade attribution**: `trade_attribution` world feature — an `(S, S)`
  directional matrix aligned with `settlements`, where `[i, j]` is the trade volume exporter
  *i* sends to importer *j*. Column-sum ≈ `trade_wealth` (imports); row = exports. Lets the
  loyalty levers attribute a settlement's wealth to specific partners (intra- vs extra-polity).
- **Land + sea cost surfaces** (`COST_SURFACES` feature → `CostSurfaces`): separate per-face
  `land_base_cost`/`land_passable` and `sea_base_cost`/`sea_passable`/`strait_mask`. Ports
  are *edges*, not tiles.
- **River network**: *not* a separate graph — anisotropic land edge-cost + amphibious sea nodes.
- **Danger field**: `DANGER_ARRAY` world feature (`FeatureChannel`, `(N,)` float32 in [0, 1]) — written by `DangerBuilder` (`worldgen/danger/builder.py`), which runs after resources and before cost in the globe pipeline. Sources span 6 categories (Volcanic/Tectonic, Coastal/Ocean, Highland/Pass, Forest/Wilderness, Planar Intrusions, Undead/Necromantic) across 19 named threat types. Already consumed by `PolityStage`: `land_base_cost + danger_scale * danger` raises movement cost through high-danger terrain, attenuating authority projection accordingly (`DangerInfluenceParams.movement_scale`).
- **Border-tension field**: ⚠️ currently an *output* of this stage, not an input (lifecycle conflict).

**Output:**
- A set of polity records forming a *forest* (sovereigns are roots; vassals are children),
  plus a per-settlement assignment of `(polity, control_class, liege)`.

The central design commitment: **a polity is a tree, not a flat partition of the map.**
Everything below exists to decide, per settlement, whether it is directly ruled, a vassal,
or independent — and to let those decisions recurse.

---

## Current state of the codebase

**Stages 0–3 are fully implemented.** `PolityStage` runs last in `RealmPipeline`, and
`ForestBuilder` (`realmgen/polity/forest.py`) drives the complete sovereign/vassal tree:

- **Stage 0**: capitals seeded by `seed_capitals` (weighted by population × trade × defensibility),
  per-polity scalars rolled by `roll_polity_scalars` (deterministic, keyed on `world.seed` + face).
  `Settlement` now carries `defensibility`, `has_walls`, and `wall_strength` — computed in
  `PopulationStage` by `compute_settlement_defensibility` (terrain + river + walls), so trade
  wealth is available when walls are rolled.
- **Stage 1**: `compute_authority_fields` projects land + sea authority fields (`(C, N)`) from
  each capital via exponential decay over cost-distance, with range modulated by `infrastructure`
  and a sea channel scaled by `maritime_reach` and `sea_range_factor`.
- **Stage 2**: `claim_faces` argmax-assigns every land face with an incumbent-hysteresis bonus,
  iterated to a fixed point. An outer loop promotes INDEPENDENT settlements to new sovereigns.
- **Stage 3**: `classify_settlements` labels each settlement DIRECT/VASSAL/INDEPENDENT by
  comparing `capital_power^e × field × gain` vs `population^e × defensibility`; `_build_subtree`
  recurses for vassals (capped by `max_recursion_depth`).

**Stage 5 fragmentation epochs are now implemented** (see below). Outstanding work is the remaining
Stage 4 tuning levers. The "Starting point" context in earlier drafts of this brief is now
superseded by the above.

### Every settlement here is significant — tune accordingly

By the time this stage runs, the population stage has **culled every settlement below
`city_min` (~7k people)**; the survivors are all city-class (a face is province-scale, and
the tiers are rank *within* the set of major cities). There is **no** background of villages
or insignificant nodes. No structural change — but it shapes parameter choices:

- The settlement pool is **small and uniformly weighty**, so capital-seeding thresholds
  (`min_pop_fraction`, exclusion/separation radii) operate over a thin, high-value set —
  don't tune them as if filtering a dense field of minor places.
- "Local power" comparisons are between peers; a single settlement defecting is a meaningful
  event, so hysteresis and the vassal/independent thresholds should expect **few,
  consequential** reassignments rather than mass churn.
- Recursion (Stage 3) operates on sparse sub-territories — a vassal's hinterland may hold
  only a handful of settlements. Cap depth low and expect small subtrees.

---

## Core concepts

**Authority projection field.** Each polity projects authority outward from its capital
along the cost surfaces, decaying with accumulated travel cost. The decay rate is modulated
by per-polity scalars. Geography does much of the work for free: cheap terrain (plains, navigable
rivers) yields a far-reaching field; fractured terrain attenuates it quickly.

**Local power.** Each settlement has a resistance score (population, walls, garrison potential,
wealth). The comparison of *projected authority* vs *local power* at a settlement is the single
decision that produces the direct/vassal/independent texture.

**Per-polity scalars:**
| Scalar | Meaning | Source |
|---|---|---|
| `cohesion` | willingness to hold together | random stability roll (the requested dial) |
| `infrastructure` | ability to administer at range | partly emergent from capital tier + trade wealth, partly random |
| `legitimacy` | optional global multiplier on projection | random / era |
| `maritime_reach` | quality of authority projection over the sea surface | scales with capital trade wealth |

`cohesion` and `infrastructure` are **independent axes**. Their cross-product is the main
generator of polity variety:

| | High infrastructure | Low infrastructure |
|---|---|---|
| **High cohesion** | Song (tight control over vast area) | Mongol empire (vast claim, fast fragmentation) |
| **Low cohesion** | late Rome (machinery runs, politics pull apart) | city-state belt |

---

## Subsystem decomposition

### Stage 0 — Data model + capital seeding
Define the polity record: `capital`, `cohesion`, `infrastructure`, `legitimacy`,
`maritime_reach`, `liege` (null for sovereigns), `autonomy`, `territory` (settlement set).
Seed candidate capitals from top-tier settlements, weighted by trade wealth and defensibility.

**Validation:** N capitals seeded, no duplicates, weighting visibly favors high-tier/defensible
settlements. Inspect by dumping seed list with weights.

> ✅ **Resolved — settlement records.** `Settlement` (`realmgen/placement/types.py`) has
> five core fields (`face_id, position, tier, score, population`) plus three defensibility
> fields added in 2026-06: `defensibility: float = 1.0` (composite multiplier consumed by
> `classify_settlements`), `has_walls: bool = False`, `wall_strength: float = 0.0`. `tier`
> is a seeding label nothing downstream reads — weight capitals by `population`. Trade wealth
> for trade-weighted seeding is the per-face `TRADE_FLOW` channel sampled at settlement faces;
> it is **not** on the record. No `archetype` or water-adjacency fields exist.

> ✅ **Resolved — defensibility.** Now stored on `Settlement` and computed in `PopulationStage`
> (after final trade wealth is known) via `compute_settlement_defensibility`
> (`realmgen/polity/seeding.py`). Formula: `terrain(ruggedness + situational_value) + river_bonus
> + wall_strength`. Walls are randomly assigned per settlement using a per-(seed, face) RNG;
> wall probability is boosted by local danger and trade wealth normalized over the settlement
> pool. `PolityStage.build()` reads `s.defensibility` directly from each `Settlement` —
> no re-derivation needed.

### Stage 1 — Authority projection field (single polity)
Implement projection from one capital: accumulate travel cost outward over the **land** cost
surface, convert accumulated cost to authority via a decay function modulated by
`infrastructure` (range) and `legitimacy` (magnitude). Then add the **sea** channel: project
through the capital's port(s) along the sea surface with a gentler decay scaled by
`maritime_reach`. Take the max (or a soft combine) of land and sea authority per tile.

**Validation:** render a single polity's authority field as a heatmap. Confirm cheap terrain
extends it and ridges/forests pinch it. Confirm a high-`maritime_reach` coastal capital shows
authority beads along coastlines, not just a land blob.

> ✅ **Resolved — cost surfaces.** Accumulated cost is a one-liner: convert per-face edge
> costs to a CSR graph with `build_cost_graph` (`realmgen/placement/seeder.py`) and call
> `scipy.sparse.csgraph.dijkstra(graph, directed=False, indices=source)`. `grow_territory`
> already does this. Land and sea are separate surfaces on `CostSurfaces` (`land_*` vs
> `sea_*`). **Ports are edges, not tiles**: the land↔sea bridge is built in
> `_collect_hybrid_edges` (`realmgen/trade/network.py`) as a `(2N,2N)` graph — land nodes
> `0..N-1`, sea nodes `N..2N-1`, port edges at coastal land↔adjacent-sea, amphibious switches
> at navigable-river faces. PolityStage is land-only today and never touches this; the sea
> channel means reusing that hybrid-graph pattern. (Maritime reach is noted "deferred — not
> yet implemented" in `docs/realmgen.md`.)

> ✅ **Resolved — river network.** Rivers are **not** a separate graph; they're an anisotropic
> modifier already baked into the land edge cost (along-valley discount / ford penalty,
> `build_land_edge_costs`), and navigable reaches become amphibious sea nodes. So the
> river-projection bonus is a **surface modifier already present** in the cost graph — no
> second pass. (Raw `River` geometry is a world feature if you ever need finer adjacency.)

### Stage 2 — Competition / claiming pass ✅
Every settlement is claimed by whichever polity projects the most authority over it. Add a
**hysteresis bonus** for the incumbent claimant so borders settle rather than flicker. Iterate
until quiescent (no reassignments, or below a churn threshold).

**Validation:** iteration count to convergence is bounded; borders are stable across reruns with
the same seed; lowering hysteresis visibly increases border churn. Render the partition.

**Implementation:** `claim_faces` (`realmgen/polity/authority.py`) — argmax with incumbent
hysteresis, iterated to a fixed point. The outer `_resolve_sovereigns` loop promotes INDEPENDENT
settlements to new capitals and re-runs claiming until convergence.

### Stage 3 — Hierarchy classification + recursion ✅
Within each polity, classify each settlement by `projected_authority` vs `local_power`:
- authority ≫ local power → **direct rule** (province)
- authority ≈ local power → **vassal** (gets `liege` pointer, an `autonomy` float, its own
  sub-territory; recurse Stage 1–2 one level down on its hinterland)
- authority < local power everywhere nearby → **independent** (new sovereign root)

Recurse one level for vassals (cap recursion depth — 2–3 levels is plenty).

**Validation:** a low-cohesion realm on fractured terrain produces mostly vassals (HRE texture);
a high-cohesion realm on cheap terrain produces mostly direct rule (Song); an in-between realm
shows a strong core + semi-independent periphery (France c.1100). Dump the tree per polity.

**Implementation:** `classify_settlements` (`realmgen/polity/authority.py`) — ratio
`capital_power^e × field × gain / population^e × defensibility`, thresholded by
`direct_threshold` and `vassal_floor`. `ForestBuilder._build_subtree` recursion carves vassal
sub-territories, capped by `max_recursion_depth`. Validation harness: `scripts/measure_polities.py`.

### Stage 4 — Tuning levers
Apply as modifiers on top of the working base. Group by what they touch so each is independently
toggleable and inspectable.

**Projection modifiers (change the field):**
- **Capital archetype bonus.** ❌ *Not implemented.* River-confluence capital → cheaper
  projection along river network. Coastal → the maritime bonus from Stage 1. Mountain-pass →
  strong projection in two directions, pinched hinterland. Drive from the settlement's functional
  archetype. (Archetype fields don't exist yet; would need to be derived or stored.)
- **Maritime reach.** ✅ *Implemented in Stage 1.* `maritime_reach` scales with capital trade
  wealth and modulates the sea authority channel via `sea_range_factor` and `sea_weight`.

**Loyalty modifiers (change secession resistance without changing the field):**
- **Economic grip.** ✅ *Implemented* (`realmgen/polity/loyalty.py`). `compute_internal_trade_fraction`
  reads the `trade_attribution` matrix against the live claim partition; `economic_grip_multiplier`
  scales each settlement's classification ratio by `1 + economic_grip · internal_fraction`, so a
  city whose trade is dominated by intra-polity partners resists secession. Off by default
  (`economic_grip = 0.0`). Validated monotonic on seeds 4/31/35: raising grip reduces sovereign
  count and grows the largest domain (trade blocs hold).
- **External pressure consolidation.** ❌ *Not implemented.* A polity adjacent to a strong rival
  field or a high-danger-field zone gets a temporary `cohesion` bonus (rally-around-the-flag).
  The danger field (`DANGER_ARRAY`) is available and already read by `PolityStage` — the lever
  logic itself is the remaining work.

**Fragmentation triggers (override classification):**
- **Marcher autonomy.** ⚠️ *Conflict not resolved.* Border tension is an output of this stage
  (computed post-classification), so using it as input is chicken-and-egg. Resolution paths:
  recompute tension inside the classification loop, or switch to the authority-field saddle
  depth as a proxy for "contested fringe."
- **Peripheral wealth secession.** ✅ *Implemented* (`realmgen/polity/loyalty.py`).
  `peripheral_wealth_breakaways` flags a non-capital settlement that is claimed, outward-facing
  (external trade fraction ≥ `peripheral_external_floor`), and whose trade wealth is ≥
  `peripheral_wealth_ratio` × its claiming capital's — forcing it INDEPENDENT (a rich, externally-
  trading edge breaks off as a new sovereign). Off by default (`peripheral_wealth_ratio = 0.0`).
  Validated on seeds 4/31/35: more aggressive thresholds raise sovereign count as prosperous edges
  secede.

**Validation:** toggle each lever individually and confirm the predicted archetype appears —
a militarily-weak but trade-dependent coastal cluster survives as one polity (economic grip);
a squeezed peripheral realm tightens then loosens (external pressure); rich vassals break off
(peripheral wealth); frontier vassals are more autonomous than interior ones (marcher).

> ✅ **Unblocked — trade route attribution implemented.** `compute_trade_network`
> (`realmgen/trade/network.py`) now returns `pair_flow` alongside `face_flow`, `hub_mask`, and
> `routes`: an `(S, S)` directional matrix where `pair_flow[i, j]` is the inter-settlement trade
> volume exporter *i* sends to importer *j* (the Phase-2 `inter_flow`, previously computed and
> discarded). `PopulationStage` publishes the final-pass matrix as the `trade_attribution` world
> feature, index-aligned with the `settlements` feature; column *j* answers "which partners feed
> settlement *j*'s trade wealth" and row *i* "where do *i*'s exports go." Economic-grip and
> peripheral-wealth secession can read it directly — only the lever logic itself remains. (The
> aggregate scalar `trade_wealth` is still available and equals each settlement's import-column
> sum to within transit flow.)

> ⚠️ **Conflict — border tension is an *output*, not an input.** It's produced by this stage
> *after* assignment (`compute_border_tension`, run on `polity_of_face`) and stored two ways:
> per-face (`face_tension` / `BORDER_TENSION` channel) **and** per-pair (`pair_tension` dict →
> `"polity_tensions"` feature). So marcher-autonomy is chicken-and-egg: tension derives from the
> partition the lever perturbs. Resolution: recompute tension inside the relaxation loop, or move
> it pre-classification onto the authority fields.

> ✅ **Unblocked — danger field exists.** `DangerBuilder` (`worldgen/danger/builder.py`) writes
> `DANGER_ARRAY` (a `(N,)` float32 world feature in [0, 1]); `PolityStage` already reads it and
> folds it into land movement cost via `DangerInfluenceParams.movement_scale`. The
> external-pressure-consolidation lever can now read `DANGER_ARRAY` directly — no field build
> required. Remaining work is the lever's logic itself.

### Stage 5 — Fragmentation epochs (optional, cheap, high-value) ✅
*Implemented* in `realmgen/polity/epochs.py` (`run_fragmentation_epochs`, `select_breakaway_seats`,
`FragmentationParams`; wired into `PolityStage` as `PolityParams.fragmentation`). Runs `n_epochs`
"historical epochs" (default 2). Each epoch rolls a per-(seed, capital, epoch) cohesion shock; any
sovereign whose `cohesion - shock` falls below what its territory size demands
(`demand = min(domain_face_count / size_reference, 1)`) sheds **one** peripheral seat — its
weakest-held vassal (highest `autonomy`) if it has vassals, else the lowest-`authority_of_face`
directly-held settlement (the interior **saddle**). The shed seat is added to the capital set and
Stages 2–4 are re-run (fresh `ForestBuilder` build); the loop early-outs once an epoch is stable and
converges because each shed shrinks the domain (lower demand). With `n_epochs == 0` the result is
the plain Stage 0–3 forest — a modifier, never a rewrite. Every shed is recorded as a
`FragmentationEvent` (epoch, parent + breakaway capital faces, `kind`, cohesion/shock/demand/domain)
and exposed as the `polity_history` world feature — a secession chronicle kept as raw material for a
later history-generation pass.

**Validation:** `scripts/measure_polities.py` shows successor-state texture — e.g. seed 31's largest
domain drops 41% → 30% of claimed and sovereign count rises (14 → 25) while breakaways are
substantial realms (80/61/53/30 faces around a strong 539-face core), not 1-face statelets;
`corr(infrastructure, vassal_fraction)` stays negative (Song ↔ Mongol axis preserved).

---

## Implementation order

**Stages 0–3 are complete.** The base partition-to-hierarchy pipeline is in place. Remaining
work, in recommended order:

1. **Stage 4 unblocked levers** (implement and validate one at a time):
   - ✅ Economic grip (implemented — `realmgen/polity/loyalty.py`)
   - ✅ Peripheral wealth secession (implemented — `realmgen/polity/loyalty.py`)
   - External pressure consolidation (danger field is available, no new infra needed)
   - Capital archetype bonus (requires deriving or storing settlement archetypes first)
   - Marcher autonomy (resolve border-tension chicken-and-egg; consider authority saddle depth as proxy)
2. **Stage 5** — Fragmentation epochs: ✅ implemented (`realmgen/polity/epochs.py`).

## Design guidelines
- Each stage independently inspectable (heatmap / tree dump / partition render) before integration.
- Levers are modifiers, never rewrites of the base mechanism — keep them toggleable.
- Relaxation passes over live coupling, consistent with the rest of the codebase.
- No writes back into the physical or settlement layers.

## Single highest-risk assumption to validate first
That the cost-surface decay produces **saddles** — interior low-authority corridors — rather than
smooth radial falloff. Saddles are where vassalage and splits concentrate; if the field is too
smooth, vassal/split behavior won't appear. If so, add terrain-class multipliers to the decay
(Stage 1) before building anything downstream. Check this immediately after Stage 1's heatmap.
