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
        gp.build(BuildContext(world=world))
        cache_dir.mkdir(parents=True, exist_ok=True)
        save_globe(gp, path)
    _, world = load_globe(path)
    world.seed = seed
    return world


def avg(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and r[key] == r[key]]
    return sum(vals) / len(vals) if vals else float("nan")
