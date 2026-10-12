"""Shared utilities for the measure_*.py diagnostic scripts."""
from __future__ import annotations

import json
import random
from pathlib import Path

from core.pipeline import BuildContext
from core.world import World
from worldgen import GlobeParams, GlobePipeline, load_globe, save_globe

CACHE_DIR = Path(__file__).parent.parent / "saved_worlds"


def load_history(path: Path) -> list:
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return []


def save_history(history: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(history, f, indent=2)


def add_seed_args(parser) -> None:  # type: ignore[no-untyped-def]
    parser.add_argument("seeds", nargs="*", type=int,
                        help="Explicit seed values")
    parser.add_argument("--random", metavar="N", type=int,
                        help="Use N random seeds instead of explicit ones")


def resolve_seeds(args, defaults: list[int]) -> list[int]:  # type: ignore[no-untyped-def]
    if args.random:
        return [random.randint(0, 2**31 - 1) for _ in range(args.random)]
    return list(args.seeds) if args.seeds else defaults


def get_physical_world(seed: int, rebuild: bool, cache_dir: Path = CACHE_DIR) -> World:
    """Load the cached physical world for seed, building and caching on miss.

    Always returns the round-tripped (saved→loaded) world. Restores world.seed
    since the serialize/load cycle drops it.
    """
    path = cache_dir / f"world_{seed}.npz"
    if not (path.exists() and not rebuild):
        params = GlobeParams()
        params.seed = seed
        world = World(params.mesh, seed=params.seed)
        gp = GlobePipeline(params)
        gp.build(BuildContext(world=world, seed=seed))
        cache_dir.mkdir(parents=True, exist_ok=True)
        save_globe(gp, path)
    _, world = load_globe(path)
    world.seed = seed
    return world


def get_realm_world(seed: int, rebuild: bool, cache_dir: Path = CACHE_DIR) -> World:
    """:func:`get_physical_world` with the realm stage run on it (default
    ``RealmParams``): settlements, towns, rural density, trade routes, danger —
    everything the region and local tiers read."""
    from realmgen.builder import RealmParams, RealmPipeline
    world = get_physical_world(seed, rebuild, cache_dir)
    RealmPipeline(RealmParams()).build(BuildContext(world=world, seed=seed))
    return world


def get_map_store(seed: int, rebuild: bool = False, cache_dir: Path = CACHE_DIR):  # type: ignore[no-untyped-def]
    """A :class:`localmap.store.MapStore` on :func:`get_realm_world`, its local maps
    carrying the region tier's shared detail as the viewer's do — what the settlement
    chain (:mod:`settlegen.chain`) builds through."""
    import dataclasses
    from localmap.store import MapStore
    from localmap.upsample import with_region_detail
    store = MapStore(get_realm_world(seed, rebuild, cache_dir), region_capacity=4, local_capacity=8)
    local = store.local_params
    store.set_local_params(dataclasses.replace(
        local, upsample=with_region_detail(local.upsample, store.region_params.upsample)))
    return store


def settlement_map_params(store):  # type: ignore[no-untyped-def]
    """Default ``SettlementMapParams`` with ``store``'s local tier's shared detail, as the viewer passes them."""
    import dataclasses
    from settlegen.params import SettlementMapParams
    from settlegen.upsample import with_local_detail
    params = SettlementMapParams()
    return dataclasses.replace(params, upsample=with_local_detail(params.upsample, store.local_params.upsample))


def realm_params(overrides: list[str]):  # type: ignore[no-untyped-def]
    """``RealmParams`` with ``a.b.c=value`` overrides (value parsed as the field's type)."""
    from realmgen.builder import RealmParams
    params = RealmParams()
    for item in overrides:
        path, value = item.split("=", 1)
        *parents, leaf = path.split(".")
        obj = params
        for name in parents:
            obj = getattr(obj, name)
        setattr(obj, leaf, type(getattr(obj, leaf))(value))
    return params


def add_set_arg(parser) -> None:  # type: ignore[no-untyped-def]
    parser.add_argument("--set", action="append", default=[], metavar="PATH=VALUE",
                        help="Override a RealmParams field, e.g. refine.border_ridge_reach_km=0")


def avg(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and r[key] == r[key]]
    return sum(vals) / len(vals) if vals else float("nan")
