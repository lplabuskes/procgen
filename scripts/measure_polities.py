"""Measure polity-forest structure — sovereign/vassal mix and consolidation.

The physical world does not depend on realm params, so we build it once per seed
and cache it to ``saved_worlds/world_<seed>.npz`` (gitignored), then re-run only
the realm stage. Reports the shape of the polity *forest* produced by the
rewritten PolityStage (Stages 0-3): how many sovereigns/vassals, how consolidated
the map is (largest-domain face fraction), and whether the infrastructure /
cohesion axes drive per-realm fragmentation as intended (Song ↔ Mongol).

Usage:
    python3 scripts/measure_polities.py                 # seeds 4, 31, 35
    python3 scripts/measure_polities.py 1 2 3           # specific seeds
    python3 scripts/measure_polities.py --random 5      # 5 random seeds
    python3 scripts/measure_polities.py --detail        # per-sovereign dump
    python3 scripts/measure_polities.py --rebuild-cache # force rebuild worlds

Tuning workflow: edit PolityParams (nested sub-paramclasses in
``realmgen/polity/builder.py`` and ``realmgen/polity/forest.py``) —
``projection.base_range`` is the master consolidation knob; ``hierarchy.vassal_floor`` /
``hierarchy.direct_threshold`` set the secession/vassalage cutoffs — re-run, compare.

Texture targets (qualitative, from docs/plans/polity_generation_brief.md):
    - a believable mix: a dominant empire + competing kingdoms + city-states;
    - high-infrastructure realms hold directly, low-infrastructure ones shed
      vassals — i.e. corr(infrastructure, vassal_fraction) should be negative.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from core.pipeline import BuildContext
from core.world import World
from realmgen.builder import RealmParams, RealmPipeline
from worldgen import GlobeParams, GlobePipeline, load_globe, save_globe

CACHE_DIR = Path(__file__).parent.parent / "saved_worlds"
DEFAULT_SEEDS = [4, 31, 35]


def _get_physical_world(seed: int, rebuild: bool) -> World:
    path = CACHE_DIR / f"world_{seed}.npz"
    if not (path.exists() and not rebuild):
        params = GlobeParams()
        params.seed = seed
        world = World(params.mesh, seed=params.seed)
        gp = GlobePipeline(params)
        gp.build(BuildContext(world=world))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        save_globe(gp, path)
    _, world = load_globe(path)
    world.seed = seed  # round-trip drops it; restore for deterministic scalars
    return world


def measure_seed(seed: int, rebuild: bool, detail: bool) -> None:
    world = _get_physical_world(seed, rebuild)
    realm = RealmPipeline(RealmParams())
    realm.build(BuildContext(world=world, seed=seed))
    ps = realm.polity_stage
    pols = ps.polities or []
    sov = [p for p in pols if p.liege is None]
    vas = [p for p in pols if p.liege is not None]
    assert ps.polity_of_face is not None
    claimed = int((ps.polity_of_face >= 0).sum())

    n_cities = len(world.get_feature("settlements") or [])
    n_vassal_seats = len(vas)
    largest = max((p.n_faces for p in sov), default=0)
    largest_frac = largest / claimed if claimed else 0.0
    max_depth = max((p.depth for p in pols), default=0)

    # Does infrastructure drive fragmentation? (over realms big enough to split)
    multi = [p for p in sov if len(p.member_settlement_faces) + len(p.vassals) >= 4]
    corr = float("nan")
    if len(multi) > 2:
        infra = np.array([p.infrastructure for p in multi])
        vfrac = np.array([
            len(p.vassals) / (len(p.member_settlement_faces) + len(p.vassals))
            for p in multi
        ])
        if infra.std() > 0 and vfrac.std() > 0:
            corr = float(np.corrcoef(infra, vfrac)[0, 1])

    print(
        f"seed {seed:>4}: {len(sov):>2} sovereigns, {n_vassal_seats:>2} vassals "
        f"| {n_cities} cities | largest realm {largest_frac * 100:4.1f}% of claimed "
        f"| max depth {max_depth} | corr(infra, vassal_frac) {corr:+.2f}"
    )
    if detail:
        for p in sorted(sov, key=lambda q: -q.n_faces)[:12]:
            print(
                f"    sov {p.polity_id:>3}: {p.n_faces:>4} faces, "
                f"{len(p.member_settlement_faces):>3} cities, {len(p.vassals)} vassals "
                f"| coh {p.cohesion:.2f} infra {p.infrastructure:.2f} "
                f"legit {p.legitimacy:.2f} mar {p.maritime_reach:.2f} "
                f"| domain {p.domain_power / 1e3:.0f}k people"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("seeds", nargs="*", type=int)
    ap.add_argument("--random", type=int, metavar="N", help="use N random seeds")
    ap.add_argument("--detail", action="store_true", help="per-sovereign dump")
    ap.add_argument("--rebuild-cache", action="store_true", help="force rebuild worlds")
    args = ap.parse_args()

    if args.random:
        seeds = [random.randint(0, 2**31 - 1) for _ in range(args.random)]
    elif args.seeds:
        seeds = args.seeds
    else:
        seeds = DEFAULT_SEEDS

    for seed in seeds:
        measure_seed(seed, args.rebuild_cache, args.detail)


if __name__ == "__main__":
    main()
