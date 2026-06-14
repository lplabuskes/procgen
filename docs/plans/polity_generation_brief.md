# Polity Generation Module — Implementation Brief

## Module boundary

Polity generation is a **human-geography** module. It reads the physical world and the
settlement layer as **read-only input** and writes a polity layer on top. No back-coupling
into the physical simulation. Downstream consumers (trade, border forts) read the polity
layer; they do not feed into it during generation.

**Inputs (read-only) — actual availability (see per-stage flags for detail):**
- **Settlements** (`world.get_feature("settlements")` → list of `Settlement`): only
  `face_id`, `position`, `tier`, `score`, `population`. `tier` is a *seeding label* that
  nothing downstream reads — key off `population`. **No** `archetype`, `walls`,
  defensibility, or water-adjacency fields exist; derive them.
- **Per-settlement trade wealth**: *not* on the record. It is `population_stage.trade_wealth`
  (an `(S,)` array aligned with `settlements`) and per-face via the `TRADE_FLOW` channel
  (= flow sampled at the settlement face).
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

## Starting point in the codebase

`PolityStage` (`realmgen/polity/builder.py`) already exists and runs **last** in
`RealmPipeline` (scoring → placement → population → polity), so settlements carry their
final populations and trade flow. What it does today is **one flat pass**: a power-weighted
Voronoi partition (`grow_territory`) keyed on population, plus a post-hoc border-tension
score. No hierarchy, no per-polity scalar, no sea projection. Treat the brief as a rewrite
of Stages 1–3 that *reuses* the existing pieces, not a greenfield build:

- **Authority field ≈ existing cost-distance.** `grow_territory` already runs a multi-source
  Dijkstra for `cost_dist[capital, face]`, then takes `argmin`. The brief's projection field
  is the same `cost_dist` with `exp(-cost/range)` applied instead — a small change, not new
  infrastructure. (This also makes the Stage-1 saddle check trivial to prototype.)
- **`Polity` record is flat** (`polity_id, capital_face, capital_position, power, n_faces,
  member_settlement_faces`). The forest output needs `liege`, `autonomy`, `control_class`,
  and sub-territory added (Stage 0).
- **No RNG.** The stage is fully deterministic with no seeded generator. Every per-polity
  scalar (`cohesion`, `infrastructure`, `legitimacy`, `maritime_reach`) needs one — thread a
  `Generator` off `ctx.world.seed`.
- **`power` = population.** Both "projected authority" and "local power" collapse to
  population today; the direct/vassal/independent comparison is entirely new.

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

> ✅ **Resolved — settlement records.** `Settlement` (`realmgen/placement/types.py`) =
> `face_id, position, tier, score, population` only. `tier` is a seeding label nothing
> downstream reads — weight capitals by `population`. Trade wealth for trade-weighted seeding
> is `population_stage.trade_wealth` (`(S,)`, aligned with `settlements`) or the per-face
> `TRADE_FLOW` channel; it is **not** on the record. No `archetype`/`walls`/water-adjacency
> fields exist — don't reference them.

> ✅ **Resolved — defensibility.** Not a stored field, and there are **no walls/garrison**
> anywhere. Derive it: per-face ruggedness already feeds `cs.land_base_cost`
> (`worldgen/cost/surfaces.py`), and `situational_value` is per-face metadata. Until wall/
> garrison fields exist, "local power" = `population` + terrain-derived defensibility.

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

### Stage 2 — Competition / claiming pass
Every settlement is claimed by whichever polity projects the most authority over it. Add a
**hysteresis bonus** for the incumbent claimant so borders settle rather than flicker. Iterate
until quiescent (no reassignments, or below a churn threshold).

**Validation:** iteration count to convergence is bounded; borders are stable across reruns with
the same seed; lowering hysteresis visibly increases border churn. Render the partition.

### Stage 3 — Hierarchy classification + recursion
Within each polity, classify each settlement by `projected_authority` vs `local_power`:
- authority ≫ local power → **direct rule** (province)
- authority ≈ local power → **vassal** (gets `liege` pointer, an `autonomy` float, its own
  sub-territory; recurse Stage 1–2 one level down on its hinterland)
- authority < local power everywhere nearby → **independent** (new sovereign root)

Recurse one level for vassals (cap recursion depth — 2–3 levels is plenty).

**Validation:** a low-cohesion realm on fractured terrain produces mostly vassals (HRE texture);
a high-cohesion realm on cheap terrain produces mostly direct rule (Song); an in-between realm
shows a strong core + semi-independent periphery (France c.1100). Dump the tree per polity.

### Stage 4 — Tuning levers
Apply as modifiers on top of the working base. Group by what they touch so each is independently
toggleable and inspectable.

**Projection modifiers (change the field):**
- **Capital archetype bonus.** River-confluence capital → cheaper projection along river network.
  Coastal → the maritime bonus from Stage 1. Mountain-pass → strong projection in two directions,
  pinched hinterland. Drive from the settlement's functional archetype.
- **Maritime reach.** Already in Stage 1; expose `maritime_reach` as trade-wealth-scaled.

**Loyalty modifiers (change secession resistance without changing the field):**
- **Economic grip.** Compute a dependency score per settlement: fraction of its imported rare
  goods routed through the capital vs alternate routes. High dependency dampens secession even
  when military authority is low (Venice / Hanse archetype). Vulnerability falls out for free:
  cut the route, dissolve the loyalty.
- **External pressure consolidation.** A polity adjacent to a strong rival field or a
  high-danger-field zone gets a temporary `cohesion` bonus (rally-around-the-flag). Recedes when
  the pressure does. Feedback with the danger field with no module coupling — danger field stays
  read-only.

**Fragmentation triggers (override classification):**
- **Marcher autonomy.** Settlements on the boundary between two strong authority fields get an
  `autonomy` bonus. Reuse the existing border-tension field as the boundary signal.
- **Peripheral wealth secession.** If a vassal's trade wealth substantially exceeds its liege
  capital's, flag it as a secession candidate regardless of authority (Lombard-cities pattern).

**Validation:** toggle each lever individually and confirm the predicted archetype appears —
a militarily-weak but trade-dependent coastal cluster survives as one polity (economic grip);
a squeezed peripheral realm tightens then loosens (external pressure); rich vassals break off
(peripheral wealth); frontier vassals are more autonomous than interior ones (marcher).

> ⛔ **Blocked — trade route attribution.** `compute_trade_network` (`realmgen/trade/network.py`)
> returns only `face_flow`, `hub_mask`, and anonymous `routes` polylines. The per-settlement
> provenance (`edge_flow`, `deposit_goods`) is **computed and discarded** — there is no "which
> capital does this settlement's rare goods flow through." Economic-grip and peripheral-wealth
> **block on extending the trade module** to return import/route attribution. Only the aggregate
> scalar `trade_wealth` is available today.

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

### Stage 5 — Fragmentation epochs (optional, cheap, high-value)
Run 2–3 "historical epochs." Each epoch: roll cohesion shocks; any polity whose `cohesion` falls
below what its territory size demands sheds its weakest-authority vassals as independents, or
splits along the lowest-authority **saddle** in its field. Re-run Stages 2–4 after each shock.

**Validation:** successor-state texture appears (one large realm fragments into a strong core +
several breakaways along interior low-authority corridors) without simulating real history.

---

## Implementation order

0 → 1 → 2 → 3 establishes the base partition-to-hierarchy pipeline; validate the three historical
textures (Song / HRE / France) at the end of Stage 3 **before** touching levers. Stage 4 levers
are additive and independently toggleable — implement and validate one at a time. Stage 5 is
optional and should come last.

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
