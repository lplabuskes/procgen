"""Measure realm settlement-population *distribution shape* vs high-medieval Europe.

The physical world (elevation → climate → hydrology → resources → cost) does not
depend on the realm parameters, so we build it once per seed and cache it to
``saved_worlds/world_<seed>.npz`` (gitignored). Every run loads that cached world
and re-runs only the cheap realm stage (suitability → seeding → trade →
population), then reports the *shape* of the resulting settlement-population
distribution. Population is in arbitrary units, so all metrics are dimensionless.

The design spec (docs/settlement_generation_design.md) names the rank-size curve
as the validation signal — "done when the population distribution roughly follows
a rank-size curve without being enforced". So the headline numbers are the
log-log rank-size **slope** (Zipf ≈ −1) and its **R²** (how straight the line is).

Usage:
    python3 scripts/measure_population.py              # seeds 42, 137, 999
    python3 scripts/measure_population.py 1 2 3        # specific seeds
    python3 scripts/measure_population.py --random 5   # 5 random seeds
    python3 scripts/measure_population.py --detail     # per-seed tier populations
    python3 scripts/measure_population.py --rebuild-cache   # force rebuild worlds

Tuning workflow (mirrors measure_resources.py): edit the population knobs in
``realmgen/builder.py`` (PopulationParams / SeederParams), re-run this script, and
compare the new row in scripts/results/population_results.md against the anchors.

High-medieval Europe reference (shape only, rough — see notes in the .md):
    Zipf slope (log pop vs log rank):   ≈ -1.0   (rank-size rule)
    Rank-size fit R²:                   ≳ 0.90   (line is reasonably straight)
    Gini of settlement sizes:           ≈ 0.55
    Tier hierarchy:                     steep pyramid — villages ≫ towns > cities > capitals
    Primacy P1/P2:                      informational only at world scale. A whole-
                                        globe map is poly-centric (several comparable
                                        regional capitals), so P1/P2 runs ~1.3-1.5, not
                                        the ~2.0 of a single integrated urban system.
                                        Measure primacy per-realm once polities exist.
"""
from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402

from core.pipeline import BuildContext  # noqa: E402
from core.world import World  # noqa: E402
from realmgen.builder import RealmParams, RealmPipeline  # noqa: E402
from realmgen.placement.types import population_display_tier  # noqa: E402
from worldgen import GlobeParams, GlobePipeline, load_globe, save_globe  # noqa: E402


RESULTS_DIR = Path(__file__).parent / "results"
JSON_PATH = RESULTS_DIR / "population_results.json"
MD_PATH = RESULTS_DIR / "population_results.md"
# Cached physical worlds live in the gitignored saved_worlds/ dir.
CACHE_DIR = Path(__file__).parent.parent / "saved_worlds"


# Rough shape anchors for a high-medieval European city/region system. These are
# ballpark targets, not hard bounds — the point is to watch the trend across
# parameter-tuning sessions, not to hit a single number.
# primacy_p1_p2 is intentionally absent: at whole-globe (multi-realm) scale the
# system is poly-centric, so P1/P2 is reported as informational, not a target.
MEDIEVAL_REF = {
    "zipf_slope": -1.0,
    "zipf_r2": 0.90,
    "gini": 0.55,
}


# ── Metrics ───────────────────────────────────────────────────────────────────


def _gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative 1-D array (0 = equal, →1 = concentrated)."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    total = x.sum()
    if n == 0 or total <= 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * x).sum()) / (n * total) - (n + 1.0) / n)


def _distribution_metrics(populations: np.ndarray) -> dict:
    """Dimensionless shape metrics for a settlement-population array."""
    pop = np.sort(np.asarray(populations, dtype=np.float64))[::-1]
    pop = pop[pop > 0.0]
    s = int(pop.size)

    nan = float("nan")
    out: dict = {
        "n_settlements": s,
        "tier_counts":   [0, 0, 0, 0],
        "zipf_slope":    nan,
        "zipf_r2":       nan,
        "primacy_p1_p2": nan,
        "top4_share":    nan,
        "gini":          nan,
        "pop_max_over_median": nan,
    }
    if s == 0:
        return out

    max_pop = float(pop[0])
    # Display tiers (capital/city/town/village) exactly as the viewer derives them.
    tiers = [population_display_tier(float(p), max_pop) for p in pop]
    out["tier_counts"] = [int(tiers.count(t)) for t in range(4)]

    if s >= 2:
        out["primacy_p1_p2"] = float(pop[0] / pop[1]) if pop[1] > 0 else nan
    out["top4_share"] = float(pop[:4].sum() / pop.sum())
    out["gini"] = _gini(pop)
    med = float(np.median(pop))
    out["pop_max_over_median"] = float(max_pop / med) if med > 0 else nan

    # Rank-size (Zipf) fit: log(pop) = slope·log(rank) + b. Needs ≥3 points.
    if s >= 3:
        ranks = np.arange(1, s + 1, dtype=np.float64)
        lx = np.log(ranks)
        ly = np.log(pop)
        slope, intercept = np.polyfit(lx, ly, 1)
        pred = slope * lx + intercept
        ss_res = float(((ly - pred) ** 2).sum())
        ss_tot = float(((ly - ly.mean()) ** 2).sum())
        out["zipf_slope"] = float(slope)
        out["zipf_r2"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else nan

    return out


# ── Physical-world caching ────────────────────────────────────────────────────


def _get_physical_world(seed: int, rebuild: bool) -> World:
    """Load the cached physical world for ``seed`` (building + caching on miss).

    Always returns the *round-tripped* (saved → loaded) world so every seed runs
    the realm stage under identical conditions and the serialize/load path is
    exercised on each call.
    """
    path = CACHE_DIR / f"world_{seed}.npz"
    if not (path.exists() and not rebuild):
        params = GlobeParams()
        params.seed = seed
        world = World(params.mesh, seed=params.seed)
        ctx = BuildContext(world=world)
        gp = GlobePipeline(params)
        gp.build(ctx)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        save_globe(gp, path)

    _, world = load_globe(path)
    return world


def measure_seed(seed: int, rebuild: bool) -> dict:
    """Run the realm stage on the (cached) physical world; return shape metrics."""
    world = _get_physical_world(seed, rebuild)
    ctx = BuildContext(world=world, seed=seed)
    realm = RealmPipeline(RealmParams())
    realm.build(ctx)

    settlements = realm.population_stage.settlements or []
    pops = np.array([s.population for s in settlements], dtype=np.float64)
    metrics = _distribution_metrics(pops)
    metrics["seed"] = seed
    # Keep the raw sorted populations (rounded) for the --detail view.
    metrics["top_populations"] = [
        round(float(p), 4) for p in np.sort(pops)[::-1][:12]
    ]
    return metrics


# ── Reporting ─────────────────────────────────────────────────────────────────


def _fmt(v: float, prec: int = 2) -> str:
    return "   nan" if v != v else f"{v:>6.{prec}f}"


def print_table(rows: list[dict]) -> None:
    print("\nSettlement-population distribution shape:")
    header = f"{'metric':<22}  " + "  ".join(f"{'s=' + str(r['seed']):>8}" for r in rows) + f"  {'medieval':>9}"
    print(header)
    print("-" * len(header))

    def line(label: str, key: str, ref_key: str | None, prec: int = 2) -> None:
        cells = "  ".join(f"{_fmt(r[key], prec):>8}" for r in rows)
        ref = MEDIEVAL_REF.get(ref_key) if ref_key else None
        ref_s = f"{ref:>9.2f}" if ref is not None else f"{'—':>9}"
        print(f"{label:<22}  {cells}  {ref_s}")

    line("n_settlements",   "n_settlements",        None, 0)
    line("Zipf slope",      "zipf_slope",           "zipf_slope")
    line("Zipf fit R²",     "zipf_r2",              "zipf_r2")
    line("Gini",            "gini",                 "gini")
    line("top-4 share",     "top4_share",           None)
    line("primacy P1/P2*",  "primacy_p1_p2",        None)
    line("pop max/median",  "pop_max_over_median",  None)
    print("  * primacy is informational at world scale (poly-centric, multi-realm).")

    # Tier-count pyramid (Capital / City / Town / Village).
    print(f"\n{'tier counts (cap/city/town/village)':<22}")
    for r in rows:
        tc = r["tier_counts"]
        print(f"  s={r['seed']:<10} " + " / ".join(f"{n}" for n in tc))
    print()


def print_detail(rows: list[dict]) -> None:
    for r in rows:
        print(f"\n  Top settlement populations (seed={r['seed']}):")
        for i, p in enumerate(r["top_populations"], start=1):
            print(f"    rank {i:>2}: {p:>12.4g}")


# ── JSON history + markdown ──────────────────────────────────────────────────────


def _load_history() -> list:
    if JSON_PATH.exists():
        with JSON_PATH.open() as f:
            return json.load(f)
    return []


def _save_history(history: list) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with JSON_PATH.open("w") as f:
        json.dump(history, f, indent=2)


def _params_summary(rp: RealmParams) -> str:
    pp = rp.population
    sd = rp.seeder
    return (
        f"vil_thr={sd.tier3.score_threshold:.2f} vil_excl={sd.tier3.exclusion_radius:.0f} "
        f"town_excl={sd.tier2.exclusion_radius:.0f} "
        f"trade_w={pp.trade_weight:.2f} cull={pp.min_pop_fraction:.3f} "
        f"hinter_max={pp.hinterland_max_distance:.0f} "
        f"hinter_infl={pp.hinterland_influence_scale:.1f} "
        f"hinter_reach={pp.hinterland_reach_scale:.0f}"
    )


def _avg(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and r[key] == r[key]]
    return sum(vals) / len(vals) if vals else float("nan")


def _avg_tiers(rows: list[dict]) -> list[float]:
    return [sum(r["tier_counts"][t] for r in rows) / max(len(rows), 1) for t in range(4)]


def _update_markdown(history: list) -> None:
    lines = [
        "# Population — Settlement Distribution Shape Tracking",
        "",
        "Auto-generated by `scripts/measure_population.py`. Do not edit by hand.",
        "",
        "Population is in arbitrary units, so every metric here is dimensionless",
        "shape. The physical world is cached per seed in `saved_worlds/`; only the",
        "realm stage is re-run, so iterating on population params is fast.",
        "",
        "## High-Medieval Europe Reference (shape only)",
        "",
        "| Metric | Target | Meaning |",
        "|--------|-------:|---------|",
        "| Zipf slope | -1.0 | log(pop) vs log(rank); the rank-size rule |",
        "| Rank-size fit R² | ≳ 0.90 | how straight the log-log line is (the spec's validation signal) |",
        "| Gini | ~0.55 | settlement-size inequality |",
        "| Tier pyramid | villages ≫ towns > cities > capitals | central-place hierarchy |",
        "| Primacy P1/P2 | *informational* | n/a at world scale (see note) |",
        "",
        "These are ballpark anchors, not hard bounds — medieval urban systems",
        "varied by region and were often somewhat flatter than a pure Zipf line.",
        "Use the *trend* across tuning sessions, not a single number.",
        "",
        "**Primacy is informational, not a target.** P1/P2 ≈ 2.0 assumes a single",
        "integrated urban system. A whole-globe map is poly-centric — several",
        "comparable regional capitals (~4-7) — so P1/P2 naturally runs ~1.3-1.5.",
        "Measure primacy *per realm* once polities group hinterlands into kingdoms.",
        "",
        "## Run History (averaged across seeds in run)",
        "",
        "| Date | Seeds | n | Zipf slope | R² | P1/P2 | top4 | Gini | cap/city/town/vil | Params |",
        "|------|-------|--:|-----------:|---:|------:|-----:|-----:|-------------------|--------|",
    ]
    for run in history:
        rows = run["results"]
        seeds_str = " ".join(str(r["seed"]) for r in rows)
        date = run["timestamp"][:10]
        tiers = _avg_tiers(rows)
        tier_str = " / ".join(f"{t:.0f}" for t in tiers)
        lines.append(
            f"| {date} | {seeds_str} "
            f"| {_avg(rows, 'n_settlements'):.0f} "
            f"| {_avg(rows, 'zipf_slope'):.2f} "
            f"| {_avg(rows, 'zipf_r2'):.2f} "
            f"| {_avg(rows, 'primacy_p1_p2'):.2f} "
            f"| {_avg(rows, 'top4_share'):.2f} "
            f"| {_avg(rows, 'gini'):.2f} "
            f"| {tier_str} "
            f"| {run.get('params_summary', '—')} |"
        )

    if history:
        last = history[-1]["results"]
        lines += [
            "",
            "## Per-Seed Detail (latest run)",
            "",
            "| Seed | n | Zipf slope | R² | P1/P2 | top4 | Gini | cap/city/town/vil |",
            "|------|--:|-----------:|---:|------:|-----:|-----:|-------------------|",
        ]
        for r in last:
            tier_str = " / ".join(str(n) for n in r["tier_counts"])
            lines.append(
                f"| {r['seed']} | {r['n_settlements']} "
                f"| {r['zipf_slope']:.2f} | {r['zipf_r2']:.2f} "
                f"| {r['primacy_p1_p2']:.2f} | {r['top4_share']:.2f} "
                f"| {r['gini']:.2f} | {tier_str} |"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text("\n".join(lines) + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("seeds", nargs="*", type=int,
                        help="Explicit seed values (default: 42 137 999)")
    parser.add_argument("--random", metavar="N", type=int,
                        help="Use N random seeds instead of explicit ones")
    parser.add_argument("--detail", action="store_true",
                        help="Print the top settlement populations per seed")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force-rebuild the cached physical worlds")
    args = parser.parse_args()

    seeds = (
        [random.randint(0, 2**31 - 1) for _ in range(args.random)]
        if args.random
        else (args.seeds or [42, 137, 999])
    )

    print(f"\nPopulation distribution shape — {len(seeds)} seed(s)\n")

    rows: list[dict] = []
    for seed in seeds:
        cached = (CACHE_DIR / f"world_{seed}.npz").exists() and not args.rebuild_cache
        tag = "load" if cached else "build"
        print(f"  seed={seed} ({tag})…", end="", flush=True)
        row = measure_seed(seed, rebuild=args.rebuild_cache)
        rows.append(row)
        print(
            f"  n={row['n_settlements']:>4}  "
            f"slope={_fmt(row['zipf_slope'])}  "
            f"R²={_fmt(row['zipf_r2'])}  "
            f"P1/P2={_fmt(row['primacy_p1_p2'])}  "
            f"Gini={_fmt(row['gini'])}"
        )

    print_table(rows)
    if args.detail:
        print_detail(rows)

    run_entry = {
        "timestamp":      datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(),
        "params_summary": _params_summary(RealmParams()),
        "results":        rows,
    }
    history = _load_history()
    history.append(run_entry)
    _save_history(history)
    _update_markdown(history)

    def rel(p):
        return p.relative_to(Path(__file__).parent.parent)
    print(f"Results appended to {rel(JSON_PATH)}")
    print(f"Markdown updated at  {rel(MD_PATH)}")
    print()


if __name__ == "__main__":
    main()
