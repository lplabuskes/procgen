# City Population Rework — Grounding the Realm Layer in Real People

**Status:** Proposed (planning). Not yet implemented.
**Scope:** `realmgen` scoring → placement → population. Physical worldgen unchanged.
**Motivation owner:** investigation in `scripts/investigate_pop_grounding.py` (seeds 4/31/35).

---

## 1. Why this change

The realm layer currently emits abstract, unitless settlement "populations" tuned only
for *distribution shape* (rank-size slope, Gini). Two questions exposed that the
abstraction is hiding a category error:

1. *How many real people does a settlement represent?*
2. *What spatial scale is a "settlement"?*

### The scale problem

At `icosphere_depth=5` the mesh is coarse (measured, averaged over seeds 4/31/35):

| Quantity | Value |
|---|---|
| Faces total | 10,242 |
| Mean face area | **~49,800 km²** (~250 km across — bigger than Belgium, ≈ Bosnia/Slovakia) |
| Land | ~137 M km² (27% — Earth-like) |
| Land faces | ~2,770 |
| Settlements seeded | ~200–270 → **~1 per 10 land faces**, each commanding ~500,000 km² |

Reality check on what fits in one face:

- **Villages** (Domesday England: ~13k settlements / 130k km², ~3–4 km apart) → **~5,000 villages per face.**
- **Market towns** at a 10-mile (16 km) "walk to market and back" spacing → **~225 towns per face.**

So a face is *province-scale* and a modeled settlement is the **major city anchoring a
multi-province region** — not a village, town, or even a single city in the literal
sense. ~200–400 settlements across a medieval Earth ≈ the real count of genuine cities
(>10k people) on the planet c.1300. **The settlement layer is, and should be, a map of
cities.** Villages and towns are a separate, finer concern (see §6).

### The semantic problem

Today `settlement.population = carrying_cap × (1 + trade_weight·log1p(wealth/median))`.
That number is the **whole region's** population (rural millions), not the city. Worse,
trade enters *multiplicatively*, so a settlement can only modestly exceed what its own
land grows. The iconic trade cities — Constantinople on its rock, Venice on its lagoon,
Bruges — had poor hinterlands and were fed by *imported* grain. Imported food **adds**
carrying capacity; it does not multiply the local harvest. The current model
structurally cannot produce a great city on a poor-but-strategic site.

---

## 2. Grounding constants (measured)

Conversion from the model's raw carrying-capacity units to people:

```
area_km²       = face_area_steradian · R²,   R² = 6371² ≈ 40.6 M km²/steradian
people(region) = D_max · R² · (Σ shares · fertility · face_area_sr)
```

`D_max` = people/km² on perfectly fertile land (fertility = 1), the single density knob.

Anchoring the **whole-globe** total to a medieval-tech Earth (~400 M people c.1300) —
the world is a globe, not just Europe, so 70 M is the wrong anchor:

| World target | Implied `D_max` (people/km² at fert=1) |
|---|---|
| ~300 M (world c.1000) | ~4.4 |
| **~400 M (world c.1300)** | **~6** (model already folds coastal-fishing fertility in) |

Measured per-seed totals: `Σ carrying_cap ≈ 1.67` raw units; trade bonus currently
inflates the global headcount **~2.6×** above carrying capacity — fine as *relative*
concentration, unphysical as an absolute world total (the planet can't import food from
off-world). **Anchor absolute numbers on carrying capacity, not the trade-inflated pop.**

---

## 3. Proposed model

Three distinct quantities, cleanly separated by where they live:

| Quantity | Lives on | Formula |
|---|---|---|
| **Rural population** | each **face** | `D_max · fertility^γ · area_km²`  (= carrying capacity) |
| **Regional population** | **settlement** | Σ rural pop over its claimed hinterland |
| **Urban population** | **settlement** | `s · (hinterland carrying cap) + β · (trade flow)` |

Rural fills the countryside to its Malthusian ceiling; the city is the surplus that
ceiling throws off, *plus* whatever trade imports on top.

All three live in **realmgen** — rural population is derived human geography (carrying
capacity × density calibration), not a physical-world field, so `D_max` and γ belong to
the realm params, and rural population is computed in the population stage even though it
is keyed per-face. It can still be exposed as a per-face debug texture.

### 3a. Rural population — fertility exponent γ (keep this!)

`carrying_capacity(face) = D_max · fertility^γ · area_km²`, with **γ ≈ 2**.

A linear law (γ=1) is too flat: the fertility field spans ~1 order of magnitude, real
population density spans ~3, so a single linear constant cannot give both a realistic
*world total* and a *dense European core*. The exponent restores contrast. Measured
sweep (land-only, target world 400 M):

| γ | `D_max` | density at fert 0.9 | density at fert 0.3 |
|---|---|---|---|
| 1.0 | ~9 | ~8 /km² | ~2.8 /km² |
| **2.0** | **~17** | **~13 /km²** | **~1.5 /km²** |
| 3.0 | ~30 | ~20 /km² | ~0.8 /km² |

γ≈2 yields dense river-valley cores (~13/km², ≈ real intensive medieval farmland) and
near-empty margins (<1.5/km²) while keeping the world total fixed. Rural population of a
face is then **equal to** its carrying capacity (linearly — it *is* the people the land
feeds). Note: the quadratic is in *fertility*, not in carrying capacity.

### 3b. Urban population — additive surplus + trade (the core rewrite)

Replace the multiplicative `cap × (1 + log trade)` in `compute_population` with:

```
urban_pop = s · (claimed hinterland carrying capacity)   # local agricultural surplus
          + β · (trade flow at the city)                  # imported food
```

- **`s`** — agricultural surplus rate / urbanization ceiling, ~0.10–0.20. New
  first-class parameter. Converts "food the region grows" into "city people the surplus
  feeds." (The old model implicitly assumed s = 1, i.e. 100% urbanization.)
- **`β`** — converts trade flow into fed urban population. **Additive**, so a
  thin-hinterland chokepoint can still spike a large urban core on imported grain. This
  is the Constantinople/Venice unlock and is now *load-bearing*, not optional.
- Carrying capacity stops *being* the population and becomes the *food base* feeding it.
  The hinterland-shares machinery is unchanged; it just feeds the surplus term.

The reported `settlement.population` becomes the **urban headcount** (hundreds of
thousands), not the regional millions. Regional/rural population belongs to the face
layer (§3a) and the village pipeline (§6).

### 3c. City existence is conditional

Today every fertile face yields nonzero population; the cull is just "below a fraction of
peak." For a cities layer, a city should **exist only if** `urban_pop > city_min` — a
genuine urban-formation floor. A rich farming region with no trade node and no political
center has *villages but no city*, which the current `cap > 0` model cannot express.
Cities become sparse and conditional.

### 3d. Placement tilts toward situational value

Cities form at **nodes** — harbors, river confluences, passes, straits — not merely at
fertile spots (fertile spots get villages). The seeder already blends suitability +
situational value; weight situational up for a cities layer. Mostly tuning, but it
follows directly from the reframe.

---

## 4. Why this is not cosmetic

Urban pop is **not** a fixed fraction of regional pop — the additive trade term breaks
proportionality. A thin-hinterland chokepoint leaps up the city rankings while a
fertile-but-isolated region drops off entirely. The ordering genuinely reshuffles; that
reshuffle is the point. Plus the existence threshold (§3c) is new structure, not a
relabel.

**Freebie:** rank-size/Zipf validation now legitimately targets a **city-size
distribution**, exactly where the rank-size rule is empirically strongest. The existing
`measure_population.py` shape check becomes *more* defensible, not less.

---

## 5. Consistency trap — don't double-count surplus

`s · hinterland_cap` is the *total* urban headroom of a region, and that same surplus
also feeds the smaller towns the village/town pipeline (§6) will model. The two pipelines
must not both spend it. Either:

- the cities layer claims only its **share** of regional carrying capacity via the
  existing hinterland-shares split (nearby smaller centers claim less, naturally dividing
  the surplus), and the town pipeline gets the remainder; or
- partition `s` explicitly between the two layers.

Otherwise urban population is counted twice.

---

## 6. Out of scope (separate pipeline)

Villages and towns are **not** representable at this mesh resolution — even depth-8
(~650k faces) leaves each face ~10× too coarse. They must come from a **sub-face
procedural scatter**: given a face's rural population and terrain, drop N villages at
~3 km spacing on zoom-in, decoupled from the global mesh. Deferred. The current task is
the cities layer only.

Also note: the tier labels (capital/city/town/village) are a **misnomer** under this
reframe — every settlement is city-class; the tiers are rank *within* the set of major
cities. Worth renaming if surfaced in UI.

---

## 7. New / changed parameters

| Param | Where | Proposed default | Meaning |
|---|---|---|---|
| `D_max` | new (rural/calibration) | ~6 people/km² | density at fertility=1 |
| `fertility_exponent` γ | new (rural) | ~2.0 | fertility→density contrast |
| `surplus_rate` s | new (`PopulationParams`) | ~0.15 | urbanization ceiling |
| `trade_food_weight` β | new (`PopulationParams`) | TBD — tune | trade flow → fed urban pop |
| `city_min` | new (`PopulationParams`) | TBD — tune | urban-formation floor (city existence) |
| `trade_weight` | **removed** | — | replaced by additive β |
| `min_pop_fraction` | repurposed | — | becomes/feeds `city_min` |

`compute_population` rewrite is the central code change; `compute_carrying_capacity`,
`compute_hinterland_shares`, and the relaxation loop stay structurally intact.

---

## 8. Validation plan

1. **World total:** `Σ urban_pop` + rural total should land at a medieval-Earth scale
   (~300–400 M people) on seeds 4/31/35.
2. **Top city sizes:** the largest cities should fall in a plausible medieval band
   (~50k–500k). This is a *new* absolute check the old shape-only harness lacked.
3. **Trade-city test:** confirm a high-trade, low-fertility chokepoint now ranks as a
   major city (it cannot today).
4. **Shape:** existing `measure_population.py` (Zipf slope ≈ −1, R² ≳ 0.9, Gini ≈ 0.55)
   — now interpreted as a city-size distribution.
5. Extend `scripts/investigate_pop_grounding.py` into a proper
   `measure_population_absolute.py` harness alongside the shape one.

---

## 9. Open questions

- Calibrate `β` and `city_min` against the top-city-size band (§8.2) — needs a sweep.
- Does `s` get partitioned with the future town pipeline now, or is that deferred until
  the town pipeline exists (§5)?
- Per-face rural population as a displayable debug texture (overlay), keyed per-face but
  owned by realmgen.

**Resolved:** `D_max` / γ / rural population live in **realmgen**, not worldgen — rural
population is derived human geography, not a physical field.
