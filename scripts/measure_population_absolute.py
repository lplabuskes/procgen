"""Measure *absolute* population grounding vs a medieval-tech Earth.

Companion to ``measure_population.py`` (shape): checks that people-denominated
quantities land at plausible medieval-Earth scales.

Anchors (whole globe, c.1300):
    World total (rural + urban):   ~300-400 M people
    Largest city:                  ~50k-500k (Constantinople ~400k, Paris ~200k)
    Urban fraction:                ~2-6 % in major cities (smaller towns are not
                                   in this layer)
    Trade-fed cities:              at least one top-10 city should draw most of
                                   its food from trade (the Constantinople test)

The physical world is cached per seed in ``saved_worlds/``; only the realm stage is re-run.

Usage:
    python3 scripts/measure_population_absolute.py            # seeds 4, 31, 35
    python3 scripts/measure_population_absolute.py 1 2 3      # specific seeds
    python3 scripts/measure_population_absolute.py --random 5
    python3 scripts/measure_population_absolute.py --detail   # top-10 city table
    python3 scripts/measure_population_absolute.py --rebuild-cache
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np

from core.pipeline import BuildContext
from realmgen.builder import RealmParams, RealmPipeline
from realmgen.channels import FISHING_POPULATION, MINING_POPULATION
from realmgen.population.calibration import KM2_PER_STERADIAN, split_fertility
from worldgen.cost.channels import IS_WATER
from worldgen.geology.channels import OCEAN_MASK
from worldgen.hydrology.channels import LAKE_MASK
from _common import (
    CACHE_DIR, load_history, save_history, add_seed_args, resolve_seeds,
    avg, get_physical_world,
)

RESULTS_DIR = Path(__file__).parent / "results"
JSON_PATH = RESULTS_DIR / "population_absolute_results.json"
MD_PATH = RESULTS_DIR / "population_absolute_results.md"

# Vetted Earth-like seeds (land ~29%, continent-scale landmasses).
DEFAULT_SEEDS = [4, 31, 35]

MEDIEVAL_REF = {
    "world_total_M": (300.0, 400.0),   # range, not single target
    "top_city": (50_000.0, 500_000.0),
    "urban_fraction": (0.02, 0.06),  # major cities only; towns are unmodeled
}


# ── Physical-world caching ────────────────────────────────────────────────────


# ── Measurement ───────────────────────────────────────────────────────────────


def measure_seed(seed: int, rebuild: bool) -> dict:
    """Run the realm stage on the (cached) physical world; return absolute metrics."""
    world = get_physical_world(seed, rebuild)
    ctx = BuildContext(world=world, seed=seed)
    mesh = ctx.mesh

    realm = RealmPipeline(RealmParams())
    realm.build(ctx)
    ps = realm.population_stage
    pp = realm.params.population

    settlements = ps.settlements or []
    pops = np.asarray(ps.population, dtype=np.float64)
    cap = np.asarray(ps.carrying_cap, dtype=np.float64)
    wealth = np.asarray(ps.trade_wealth, dtype=np.float64)
    rural = np.asarray(ps.rural_population, dtype=np.float64)

    surplus_term = pp.surplus_rate * pp.city_surplus_share * np.maximum(cap, 0.0)
    trade_term = pp.trade_food_weight * np.maximum(wealth, 0.0)

    rural_total = float(rural.sum())
    urban_total = float(pops.sum())
    world_total = rural_total + urban_total

    order = np.argsort(-pops)
    top10 = [
        {
            "pop": float(pops[k]),
            "surplus": float(surplus_term[k]),
            "trade": float(trade_term[k]),
            # Share of the city's food that is imported (its miners, fed the
            # same way, have left for the deposits).
            "trade_share": (float(trade_term[k] / (trade_term[k] + surplus_term[k]))
                            if trade_term[k] + surplus_term[k] > 0 else 0.0),
        }
        for k in order[:10]
    ]
    # Trade-city test: cities whose imported food exceeds local surplus.
    trade_fed = trade_term > surplus_term
    trade_fed_top10 = int(trade_fed[order[:10]].sum()) if pops.size else 0
    band_lo, band_hi = MEDIEVAL_REF["top_city"]

    # Implied D_max (``rural_density_max``) for exactly 400 M rural at the current γ,
    # fishers and miners held fixed.
    land = ~IS_WATER.read(mesh)
    fishing_total = float(FISHING_POPULATION.read(mesh)[land].sum()) if FISHING_POPULATION.present(mesh) else 0.0
    mining_total = float(MINING_POPULATION.read(mesh).sum()) if MINING_POPULATION.present(mesh) else 0.0
    water_tex = OCEAN_MASK.read(mesh)[..., 0] > 0.5
    if LAKE_MASK.present(mesh):
        water_tex |= LAKE_MASK.read(mesh)[..., 0] > 0.5
    land_fert, land_frac, _, _ = split_fertility(np.asarray(mesh.face_textures["fertility"][..., 0]), water_tex)
    areas_km2 = np.asarray(mesh.face_areas, dtype=np.float64) * KM2_PER_STERADIAN
    eff_area = float((land_fert[land] ** pp.fertility_exponent * (areas_km2 * land_frac)[land]).sum())
    implied_d_max = (400e6 - fishing_total - mining_total) / eff_area if eff_area > 0 else float("nan")

    return {
        "seed": seed,
        "n_cities": int(len(settlements)),
        "rural_total_M": rural_total / 1e6,
        "fishing_total_M": fishing_total / 1e6,
        "mining_total_M": mining_total / 1e6,
        "urban_total_M": urban_total / 1e6,
        "world_total_M": world_total / 1e6,
        "urban_fraction": urban_total / world_total if world_total > 0 else float("nan"),
        "top_city": float(pops[order[0]]) if pops.size else float("nan"),
        "n_top_band": int(((pops >= band_lo) & (pops <= band_hi)).sum()),
        "trade_fed_total": int(trade_fed.sum()),
        "trade_fed_top10": trade_fed_top10,
        "max_trade_share_top10": max((c["trade_share"] for c in top10), default=float("nan")),
        "implied_d_max_400M": implied_d_max,
        "top10": top10,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────


def _fmt_people(p: float) -> str:
    if p != p:
        return "nan"
    if p >= 1e6:
        return f"{p / 1e6:.2f}M"
    if p >= 1e3:
        return f"{p / 1e3:.0f}k"
    return f"{p:.0f}"


def print_table(rows: list[dict]) -> None:
    print("\nAbsolute population grounding:")
    header = f"{'metric':<26}  " + "  ".join(f"{'s=' + str(r['seed']):>9}" for r in rows) + f"  {'medieval':>13}"
    print(header)
    print("-" * len(header))

    def line(label: str, fmt, ref: str) -> None:
        cells = "  ".join(f"{fmt(r):>9}" for r in rows)
        print(f"{label:<26}  {cells}  {ref:>13}")

    lo, hi = MEDIEVAL_REF["world_total_M"]
    line("world total (M)", lambda r: f"{r['world_total_M']:.0f}", f"{lo:.0f}-{hi:.0f}")
    line("  rural (M)", lambda r: f"{r['rural_total_M']:.0f}", "—")
    line("    of which fishing (M)", lambda r: f"{r.get('fishing_total_M', 0.0):.1f}", "—")
    line("    of which mining (M)", lambda r: f"{r.get('mining_total_M', 0.0):.2f}", "—")
    line("  urban (M)", lambda r: f"{r['urban_total_M']:.1f}", "—")
    ulo, uhi = MEDIEVAL_REF["urban_fraction"]
    line("urban fraction", lambda r: f"{r['urban_fraction']:.3f}", f"{ulo:.2f}-{uhi:.2f}")
    line("n cities", lambda r: f"{r['n_cities']}", "~200-400")
    blo, bhi = MEDIEVAL_REF["top_city"]
    line("top city", lambda r: _fmt_people(r["top_city"]), f"{_fmt_people(blo)}-{_fmt_people(bhi)}")
    line("cities in 50k-500k", lambda r: f"{r['n_top_band']}", ">0")
    line("trade-fed cities", lambda r: f"{r['trade_fed_total']}", ">0")
    line("  of top 10", lambda r: f"{r['trade_fed_top10']}", "≥1")
    line("max trade share (top10)", lambda r: f"{r['max_trade_share_top10']:.2f}", ">0.5")
    line("implied D_max (400M)", lambda r: f"{r['implied_d_max_400M']:.1f}", "—")
    print()


def print_detail(rows: list[dict]) -> None:
    for r in rows:
        print(f"\n  Top cities (seed={r['seed']}):  pop = surplus + trade  (trade share)")
        for i, c in enumerate(r["top10"], start=1):
            print(
                f"    rank {i:>2}: {_fmt_people(c['pop']):>7} = "
                f"{_fmt_people(c['surplus']):>7} + {_fmt_people(c['trade']):>7}  "
                f"({c['trade_share']:.2f})"
            )
    print()


# ── JSON history + markdown ───────────────────────────────────────────────────


def _params_summary(rp: RealmParams) -> str:
    pp = rp.population
    return (
        f"D_max={pp.rural_density_max:.0f} D_fish={pp.fishing_density_max:.0f} gamma={pp.fertility_exponent:.1f} "
        f"s={pp.surplus_rate:.2f} share={pp.city_surplus_share:.3f} "
        f"beta={pp.trade_food_weight:.0f} "
        f"city_min={pp.city_min:.0f} sit_w={rp.seeder.situational_weight:.2f}"
    )


def _update_markdown(history: list) -> None:
    lines = [
        "# Population — Absolute Grounding Tracking",
        "",
        "Auto-generated by `scripts/measure_population_absolute.py`. Do not edit by hand.",
        "",
        "All quantities are real people. Anchors are a medieval-tech Earth c.1300:",
        "world total ~300-400 M, largest city 50k-500k, urban fraction ~3-10 %,",
        "and at least one top-10 city fed mostly by trade (the Constantinople test).",
        "",
        "## Run History (averaged across seeds in run)",
        "",
        "| Date | Seeds | cities | world M | rural M | urban M | urb frac | top city "
        "| 50-500k | trade-fed top10 | Params |",
        "|------|-------|-------:|--------:|--------:|--------:|---------:|---------:"
        "|--------:|----------------:|--------|",
    ]
    for run in history:
        rows = run["results"]
        seeds_str = " ".join(str(r["seed"]) for r in rows)
        date = run["timestamp"][:10]
        lines.append(
            f"| {date} | {seeds_str} "
            f"| {avg(rows, 'n_cities'):.0f} "
            f"| {avg(rows, 'world_total_M'):.0f} "
            f"| {avg(rows, 'rural_total_M'):.0f} "
            f"| {avg(rows, 'urban_total_M'):.1f} "
            f"| {avg(rows, 'urban_fraction'):.3f} "
            f"| {_fmt_people(avg(rows, 'top_city'))} "
            f"| {avg(rows, 'n_top_band'):.0f} "
            f"| {avg(rows, 'trade_fed_top10'):.1f} "
            f"| {run.get('params_summary', '—')} |"
        )

    if history:
        last = history[-1]["results"]
        lines += [
            "",
            "## Per-Seed Detail (latest run)",
            "",
            "| Seed | cities | world M | rural M | urban M | top city | 50-500k | trade-fed | max trade share top10 |",
            "|------|-------:|--------:|--------:|--------:|---------:|--------:|----------:|----------------------:|",
        ]
        for r in last:
            lines.append(
                f"| {r['seed']} | {r['n_cities']} "
                f"| {r['world_total_M']:.0f} | {r['rural_total_M']:.0f} "
                f"| {r['urban_total_M']:.1f} "
                f"| {_fmt_people(r['top_city'])} | {r['n_top_band']} "
                f"| {r['trade_fed_total']} | {r['max_trade_share_top10']:.2f} |"
            )

    MD_PATH.write_text("\n".join(lines) + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_seed_args(parser)
    parser.add_argument("--detail", action="store_true",
                        help="Print the top-10 city decomposition per seed")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force-rebuild the cached physical worlds")
    args = parser.parse_args()

    seeds = resolve_seeds(args, DEFAULT_SEEDS)

    print(f"\nAbsolute population grounding — {len(seeds)} seed(s)\n")

    rows: list[dict] = []
    for seed in seeds:
        cached = (CACHE_DIR / f"world_{seed}.npz").exists() and not args.rebuild_cache
        tag = "load" if cached else "build"
        print(f"  seed={seed} ({tag})…", end="", flush=True)
        row = measure_seed(seed, rebuild=args.rebuild_cache)
        rows.append(row)
        print(
            f"  cities={row['n_cities']:>4}  world={row['world_total_M']:.0f}M  "
            f"top={_fmt_people(row['top_city'])}  "
            f"trade-fed(top10)={row['trade_fed_top10']}"
        )

    print_table(rows)
    if args.detail:
        print_detail(rows)

    run_entry = {
        "timestamp":      datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(),
        "params_summary": _params_summary(RealmParams()),
        "results":        rows,
    }
    history = load_history(JSON_PATH)
    history.append(run_entry)
    save_history(history, JSON_PATH)
    _update_markdown(history)

    def rel(p):
        return p.relative_to(Path(__file__).parent.parent)
    print(f"Results appended to {rel(JSON_PATH)}")
    print(f"Markdown updated at  {rel(MD_PATH)}")
    print()


if __name__ == "__main__":
    main()
