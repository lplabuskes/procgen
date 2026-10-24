# Households, Lots and Buildings — Plan

Three follow-ups to the settlement-map layout ([settlement_maps.md](settlement_maps.md),
reference [../settlement_maps.md](../settlement_maps.md)), in this order:

1. **Households and trades** — not every household farms. Village craftsmen
   (smith, baker, carpenter …) live on small plots near the middle; in towns and
   cities the everyday trades are spread through the houses instead of all sitting
   in a craft quarter.
2. **Lots as boundaries** — a block (or a village's street band) plus the lines
   dividing it, not free-standing lot polygons. Lots tile their ground, so they
   cannot overlap (the village arm-junction crowding), and neighbours share edges.
3. **Buildings** on the lots, in the **main view**: a house on the street front
   plus outbuildings to start, built so a fuller typology can slot in later.

## User decisions (2026-10-24)

- Crafts: villages gain craft households; towns and cities **disperse the
  everyday trades** through residential lots. Noxious/water trades (tanning,
  dyeing, butchery …) keep their quarters.
- The boundary model replaces lot packing at **every tier**, villages first.
- Buildings are in the **main view**.
- Buildings start as **house + outbuildings**, designed for a full typology later.

## Phase 1 — Households and trades

New `settlegen/layout/households.py`, Qt- and geometry-free:

```python
@dataclass
class Household:
    use: str        # "farm", "fisher", "miner", or a trade: "smith", "baker", …
    scale: float    # plot area relative to a farm toft

def households(profile, params, rng) -> list[Household]
```

- **Everyday trades** (`TRADES`): people per practitioner, a minimum population
  and a plot scale. Expected count `P / people_per`, Poisson-jittered, floored by
  `min_people`. First cut (to tune against ~10–20% non-farming households in a
  300-person village and ~25–35% at 1,000):

  | Trade | People per | Min people | Notes |
  |---|---|---|---|
  | smith | 250 | 150 | |
  | carpenter | 300 | 200 | |
  | alehouse | 250 | 150 | |
  | baker | 400 | 300 | |
  | miller | 400 | 200 | to the stream if there is one, else a windmill site at the edge |
  | weaver | 300 | 200 | |
  | cobbler | 500 | 400 | |
  | butcher | 600 | 400 | |
  | potter / cooper / thatcher | 800 | 500 | one of them, random |

  The profile's industries scale these (`weaving` raises weavers, `brewing`
  alehouses and so on).
- **Non-farming villagers**: the village's `1 − farming_share` (fishers,
  miners) get cottage plots (`cottage_scale`, ~0.35) instead of farm tofts.
- Plot sizes keep today's total. The farm toft is solved so that
  `Σ scale × toft_m2 = toft_share × footprint`, and craft plots are
  `craft_scale` (~0.4) of it.
- **Villages** (`village.py`): trades take the slots nearest the middle and the
  main street, before the farms. The miller takes the slot nearest running water.
- **Towns and cities**: `Lot.use` (below) tags a share of residential, suburb and
  burgage lots with trades. They are counted per-capita from the same table, so
  the noxious and water trades stay out because their quarters cover them. They
  prefer lots fronting an arterial or a street corner and are spread by Poisson-disc
  spacing per trade (each baker's own catchment).
- `Block.uses` (per lot, parallel to `Block.lots`) holds the use; phase 2's
  `LotRun` carries it on. Hover shows the trade ("Smithy · plot 9 × 18 m").
  `summary` counts households by use.

Tests: household counts and shares by population; area conservation; trades
spread out in a city (min nearest-neighbour distance per trade); villages place
trades nearer the middle than farms.

## Phase 2 — Lots as boundaries

The stored form is the ground a run of lots shares and the lines dividing it.
Lots are derived from that (polygonize, cached):

```python
@dataclass
class LotRun:
    ground: Polygon                 # the block, or one street's band
    front: LineString               # the street edge the lots face
    dividers: list[LineString]      # side lines between neighbours (+ the back line between rows)
    uses: list[str]                 # per lot, ordered along ``front``
```

- **Villages**: the toft zone is the union of every street's band (street
  buffered by `half_street + depth`, inside `allowed`). Each piece of it goes to its
  **nearest street**: a Voronoi split of the band by street, which is a bisector
  where two arms meet, so nothing can overlap. Dividers are cut perpendicular to
  the street at each household's width, from the front to the back line. Gaps
  (`_GAP_CHANCE`) are runs left undivided (open ground). Back lanes and the
  church/manor use the same mechanism with wider slots.
- **Towns and cities**: `strip_lots` (was `lots_of`) emits dividers across the block (frontage cuts
  along each long side, the back line between rows) instead of rectangles
  intersected with the block. `lane_lots` adds its lane-fronting runs to the same
  block, and polygonizing the block's union of lines replaces its first-come
  first-served overlap fix. The burgage band works the same way.
- **Drawing**: lot lines drawn once as a line path (block outlines + dividers),
  not each lot's outline, so shared edges aren't stroked twice. That's also cheaper
  for the ~64k garden plots of a big city.
- `describe_at` and `n_lots` read the derived lots. `measure_settlements.py`
  reports overlap area (must be 0) and the share of the toft zone left open.

Tests: lots tile their ground (pairwise intersections have zero area, union =
ground less gaps), the village arm-junction case (two arms at 60°) and the
existing layout tests.

## Phase 3 — Buildings

New `settlegen/layout/buildings.py`, with `SettlementLayout.buildings: list[Building]`:

```python
@dataclass
class Building:
    polygon: Polygon
    kind: str          # "house", "barn", "workshop", "church", "manor", "row" …
    lot: int           # index into the derived lots (-1: on open ground)

BUILDING_TYPES: dict[str, Callable[[LotGeom, rng, params], list[Building]]]
```

- A lot's geometry for placement (`LotGeom`): its polygon, front edge,
  depth direction and frontage width. These come from the `LotRun`, so phase 2 has
  to land first.
- **Typology hook**: a registry keyed by the lot's use (falling back to
  district) maps to a placer. New types (L-shaped farmhouse, courtyard house, hall,
  church with tower) are just new entries. Phase 3 ships:
  - **house**: on the front edge, detached (side gap) or terraced (full width) by
    density. Depth 6–10 m, gable along the street in the core, sideways on
    larger plots.
  - **farm toft**: house at the front plus a barn behind or beside it.
  - **craft**: house plus a workshop at the back (smithy, bakehouse, mill by the
    water).
  - **church** (precinct): nave + chancel by size. **manor**: hall + range
    round a yard.
  - **open districts** (graveyard, square, gardens): none, or a shed in gardens.
- **Drawing (main view)**: building fill + thin outline above the lots, shown
  once a building spans a few pixels (like lots now). Cached painter paths per
  kind.
- Hover names the building ("Farmhouse", "Barn", "Smithy").
- `measure_settlements.py`: building counts, build time, coverage share of lots.

Tests: buildings inside their lots, not overlapping each other, every built lot
with a house, the registry falls back to district.

## Status

- 2026-10-24: plan written; reviewed (start).
- 2026-10-24: **Phase 1 done**, awaiting review. `layout/households.py` (`TRADES`
  table — people per practitioner 200–700, minimum 120–400 people), village
  groups placed trades → cottages → farms with `tofts_along`, `disperse_trades`
  in towns/cities. Seed 4 city 13 (110k): 2,695 shops of 16,555 plots (16%).
  Village 536582844 50/115/3 (982 people): 35 craftsmen, 161 farmsteads, all placed.
- 2026-10-24: review — deep quay blocks were left with empty middles (city 13 quay
  cover median 0.62); user chose **deep merchant plots**: quay/market/elite lots run
  back to fill the block up to `merchant_lot_depth_m` (45), lanes beyond. Quay cover
  → 0.98 (p10 0.91). Phase 1 accepted.
- 2026-10-24: **Phase 2 built**, awaiting review. `layout/lots.py` (derive lots by
  polygonizing ground + dividers, snap-rounded 1 µm), `Block` stores ground /
  dividers / seeds (`of_lots` keeps the lots it was built with; `strip_lots` emits
  its dividers directly), `layout/frontage.py` village packer (runs per street side,
  cursors out from the middle, shared dividers), viewer draws lot lines once.
  Junctions: a Voronoi bisector split made slanted trapezoid corner plots; user
  steered to **ranked streets** instead (merged end to end, longest claims its band
  first, the shorter's plots start square behind) plus no plot across a bend > 25°.
  measure_settlements: lot overlap 0 m² at every tier (seed 4). Layout time on par
  with or below the pre-phase-1 commit (city 7 3.2 s vs 3.7 s, same machine).
