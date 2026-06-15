"""Profile where time goes in the full worldgen + realmgen pipeline.

Every stage builder threads a ``progress(msg)`` callback that is called once at
the start of each internal phase. By wrapping that callback with a timer we get
a per-phase wall-clock breakdown of the entire pipeline without touching any
builder code — the duration of a phase is simply the gap between its progress
message and the next one (the final phase is closed when ``build()`` returns).

Phases are rolled up into top-level stages by recognising each stage's first
message: five worldgen stages (elevation, climate, hydrology, lake_feedback,
resources) and four realmgen stages (scoring, placement, population, polity).

Usage:
    python3 scripts/profile_pipeline.py              # seed 42, 1 timed run
    python3 scripts/profile_pipeline.py 1 2 3        # average over seeds 1,2,3
    python3 scripts/profile_pipeline.py --repeat 3   # seed 42, average of 3 runs
    python3 scripts/profile_pipeline.py --top 15     # show only 15 slowest phases
    python3 scripts/profile_pipeline.py --cprofile   # cProfile drill-down by function
    python3 scripts/profile_pipeline.py --cprofile --module hydrology
    python3 scripts/profile_pipeline.py --cprofile --module realmgen/polity
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import time
from collections import OrderedDict

from core.pipeline import BuildContext
from core.world import World
from worldgen import GlobePipeline, GlobeParams
from realmgen.builder import RealmPipeline, RealmParams
from core import mesh_cache


# Substring → stage name. The first phase whose message contains the key marks
# the start of that stage; everything before the first match is "elevation".
STAGE_MARKERS = [
    # worldgen
    ("insolation", "climate"),
    ("rivers and lakes", "hydrology"),
    ("lake humidity feedback", "lake_feedback"),
    ("terrain ruggedness", "resources"),
    # realmgen
    ("Reading physical layers", "scoring"),
    ("Building land cost graph", "placement"),
    ("Building trade network", "population"),
    ("Seeding sovereign capitals", "polity"),
]
STAGE_ORDER = [
    "elevation", "climate", "hydrology", "lake_feedback", "resources",
    "scoring", "placement", "population", "polity",
]


class PhaseTimer:
    """Records the wall-clock duration of each progress phase.

    Each call marks the boundary between the previous phase and the next one.
    Call ``finish()`` after the pipeline returns to close the final phase.
    """

    def __init__(self) -> None:
        self._marks: list[tuple[str, float]] = []

    def __call__(self, msg: str) -> None:
        self._marks.append((msg, time.perf_counter()))

    def finish(self) -> None:
        self._marks.append(("<<done>>", time.perf_counter()))

    def phases(self) -> list[tuple[str, float]]:
        """Return [(msg, duration_s)] for every phase, in order."""
        out = []
        for (msg, t0), (_, t1) in zip(self._marks, self._marks[1:]):
            out.append((msg, t1 - t0))
        return out


def _stage_of(msg: str, current: str) -> str:
    for key, stage in STAGE_MARKERS:
        if key in msg:
            return stage
    return current


def profile_once(seed: int) -> list[tuple[str, str, float]]:
    """Run the full worldgen + realmgen pipeline; return [(stage, phase_msg, duration_s)]."""
    params = GlobeParams()
    params.seed = seed

    timer = PhaseTimer()
    ctx = BuildContext(world=World(params.mesh, seed=params.seed), progress=timer)
    GlobePipeline(params).build(ctx)
    RealmPipeline(RealmParams()).build(ctx)
    timer.finish()

    rows: list[tuple[str, str, float]] = []
    stage = "elevation"
    for msg, dur in timer.phases():
        stage = _stage_of(msg, stage)
        rows.append((stage, msg, dur))
    return rows


def aggregate(runs: list[list[tuple[str, str, float]]]) -> tuple[OrderedDict, OrderedDict]:
    """Average phase and stage durations across runs.

    Returns (phase_totals, stage_totals) as OrderedDicts preserving first-seen
    phase order and STAGE_ORDER respectively.
    """
    n = len(runs)
    phase_sum: "OrderedDict[str, list]" = OrderedDict()  # msg -> [stage, total]
    stage_sum: dict[str, float] = {s: 0.0 for s in STAGE_ORDER}

    for run in runs:
        for stage, msg, dur in run:
            if msg not in phase_sum:
                phase_sum[msg] = [stage, 0.0]
            phase_sum[msg][1] += dur
            stage_sum[stage] += dur

    phase_avg = OrderedDict(
        (msg, (stage, total / n)) for msg, (stage, total) in phase_sum.items()
    )
    stage_avg = OrderedDict((s, stage_sum[s] / n) for s in STAGE_ORDER)
    return phase_avg, stage_avg


def run_cprofile(seed: int, module: str, top: int) -> None:
    """Run one build under cProfile and print the hottest functions.

    Reports both by total time (time inside the function itself) and cumulative
    time (function + everything it calls). If ``module`` is given, the listing is
    filtered to functions defined under worldgen/<module>/.
    """
    params = GlobeParams()
    params.seed = seed

    ctx = BuildContext(world=World(params.mesh, seed=params.seed), progress=lambda _msg: None)
    profiler = cProfile.Profile()
    print(f"Running build under cProfile (seed={seed})…", flush=True)
    profiler.enable()
    GlobePipeline(params).build(ctx)
    RealmPipeline(RealmParams()).build(ctx)
    profiler.disable()

    restriction: tuple = (top or 25,)
    if module:
        # Accept bare name (e.g. "hydrology" → "worldgen/hydrology/")
        # or a full sub-path (e.g. "realmgen/polity" used as-is).
        path_filter = module if "/" in module else f"worldgen/{module}/"
        restriction = (path_filter, top or 25)

    stats = pstats.Stats(profiler)
    stats.strip_dirs = lambda: stats  # keep paths so the module filter works
    full = pstats.Stats(profiler)

    print("\n" + "=" * 70)
    print(f"TOP BY TOTAL TIME (tottime){' — ' + module if module else ''}")
    print("=" * 70)
    full.sort_stats("tottime").print_stats(*restriction)

    print("\n" + "=" * 70)
    print(f"TOP BY CUMULATIVE TIME (cumtime){' — ' + module if module else ''}")
    print("=" * 70)
    full.sort_stats("cumtime").print_stats(*restriction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("seeds", nargs="*", type=int,
                        help="Seeds to average over (default: 42)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Timed runs per seed (default: 1)")
    parser.add_argument("--top", type=int, default=0,
                        help="Show only the N slowest phases / functions (default: all / 25)")
    parser.add_argument("--cprofile", action="store_true",
                        help="cProfile drill-down by function instead of phase timing")
    parser.add_argument("--module", type=str, default="",
                        help="With --cprofile, filter to worldgen/<module>/ (e.g. hydrology)")
    args = parser.parse_args()

    seeds = args.seeds or [42]

    if args.cprofile:
        run_cprofile(seeds[0], args.module, args.top)
        return

    if not mesh_cache.is_populated():
        print("Mesh cache not populated — building it once (untimed)…")
        mesh_cache.build(GlobeParams().mesh)

    # One throwaway run to warm imports / numpy / scipy paths.
    print("Warming up…", flush=True)
    profile_once(seeds[0])

    runs: list[list[tuple[str, str, float]]] = []
    total_runs = len(seeds) * args.repeat
    print(f"Profiling {total_runs} run(s) over seed(s) {seeds}…\n", flush=True)
    for seed in seeds:
        for r in range(args.repeat):
            t0 = time.perf_counter()
            runs.append(profile_once(seed))
            print(f"  seed={seed} run={r + 1}/{args.repeat}  "
                  f"{time.perf_counter() - t0:.2f}s", flush=True)

    phase_avg, stage_avg = aggregate(runs)
    grand_total = sum(stage_avg.values())

    print("\n" + "=" * 60)
    print("STAGE ROLLUP (avg per run)")
    print("=" * 60)
    print(f"{'stage':<16}{'seconds':>10}{'%':>9}")
    print("-" * 35)
    for stage, dur in sorted(stage_avg.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * dur / grand_total if grand_total else 0.0
        print(f"{stage:<16}{dur:>10.3f}{pct:>8.1f}%")
    print("-" * 35)
    print(f"{'TOTAL':<16}{grand_total:>10.3f}{100.0:>8.1f}%")

    print("\n" + "=" * 60)
    print("PHASE BREAKDOWN (avg per run, slowest first)")
    print("=" * 60)
    print(f"{'stage':<14}{'seconds':>9}{'%':>7}  phase")
    print("-" * 60)
    items = sorted(phase_avg.items(), key=lambda kv: -kv[1][1])
    if args.top:
        items = items[:args.top]
    for msg, (stage, dur) in items:
        pct = 100.0 * dur / grand_total if grand_total else 0.0
        label = msg if len(msg) <= 38 else msg[:37] + "…"
        print(f"{stage:<14}{dur:>9.3f}{pct:>6.1f}%  {label}")
    print()


if __name__ == "__main__":
    main()
