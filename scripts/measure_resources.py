"""Measure per-biome fertility and forest-density vs Earth-ish reference values.

Runs the full pipeline (elevation + climate + hydrology + lake feedback +
resources) for each seed and reports the mean fertility and forest density of
*land* pixels grouped by their pixel-level biome assignment. Results are
appended to scripts/results/resources_results.json and a human-readable summary
is regenerated at scripts/results/resources_results.md.

Usage:
    python3 scripts/measure_resources.py              # seeds 42, 137, 999
    python3 scripts/measure_resources.py 1 2 3        # specific seeds
    python3 scripts/measure_resources.py --random 5   # 5 random seeds
    python3 scripts/measure_resources.py --detail     # full per-biome breakdown

Earth-ish reference anchors (rough):
    Tropical forest density:        ~0.85
    Temperate forest density:       ~0.75
    Taiga forest density:           ~0.60
    Steppe forest density:          ~0.05
    Tropical fertility (mean):      ~0.70
    Temperate fertility (mean):     ~0.75
    Tundra fertility (mean):        ~0.10
    Desert fertility (mean):        ~0.03
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

from worldgen.climate.classification import BIOME_NAMES, N_BIOMES  # noqa: E402
from core.pipeline import BuildContext  # noqa: E402
from core.world import World  # noqa: E402
from worldgen import GlobePipeline, GlobeParams  # noqa: E402


RESULTS_DIR = Path(__file__).parent / "results"
JSON_PATH = RESULTS_DIR / "resources_results.json"
MD_PATH = RESULTS_DIR / "resources_results.md"


EARTH_REF_FERTILITY = {
    "Tropical":  0.70,
    "Temperate": 0.75,
    "Savannah":  0.45,
    "Steppe":    0.30,
    "Taiga":     0.40,
    "Tundra":    0.10,
    "Semi-Arid": 0.15,
    "Desert":    0.03,
    "Icecap":    0.00,
}
EARTH_REF_FOREST = {
    "Tropical":  0.85,
    "Temperate": 0.75,
    "Savannah":  0.30,
    "Steppe":    0.05,
    "Taiga":     0.60,
    "Tundra":    0.05,
    "Semi-Arid": 0.02,
    "Desert":    0.00,
    "Icecap":    0.00,
}

# Biomes we care about reporting first.
PRIMARY_BIOMES = ["Tropical", "Temperate", "Taiga", "Savannah", "Steppe",
                  "Tundra", "Semi-Arid", "Desert"]


def measure_seed(seed: int) -> dict:
    """Build full pipeline for one seed; return per-biome aggregate dict."""
    params = GlobeParams()
    params.elevation.seed = seed
    ctx = BuildContext(world=World(), progress=lambda _: None)
    builder = GlobePipeline(params)
    builder.build(ctx)

    mesh = builder.globe.mesh
    fertility = mesh.face_textures["fertility"][..., 0]               # (N, H, W)
    forest = mesh.face_textures["forest_density"][..., 0]             # (N, H, W)
    biome_tex = mesh.face_textures["biome"][..., 0].astype(np.int32)  # (N, H, W)
    ocean_mask = mesh.face_textures["ocean_mask"][..., 0] > 0.5       # (N, H, W) bool

    land_mask = ~ocean_mask
    if "lake_mask" in mesh.face_textures:
        lake_mask = mesh.face_textures["lake_mask"][..., 0] > 0.5
        land_mask &= ~lake_mask

    per_biome: dict[str, dict] = {}
    for idx in range(N_BIOMES):
        name = BIOME_NAMES[idx]
        sel = land_mask & (biome_tex == idx)
        count = int(sel.sum())
        if count == 0:
            per_biome[name] = {
                "fertility_mean": float("nan"),
                "forest_mean":    float("nan"),
                "land_pixels":    0,
            }
            continue
        per_biome[name] = {
            "fertility_mean": round(float(fertility[sel].mean()), 3),
            "forest_mean":    round(float(forest[sel].mean()), 3),
            "land_pixels":    count,
        }

    overall = {
        "fertility_mean": round(float(fertility[land_mask].mean()), 3) if land_mask.any() else float("nan"),
        "forest_mean":    round(float(forest[land_mask].mean()), 3) if land_mask.any() else float("nan"),
        "land_pixels":    int(land_mask.sum()),
    }

    return {
        "seed":       seed,
        "overall":    overall,
        "per_biome":  per_biome,
    }


def print_table(rows: list[dict]) -> None:
    header = f"{'biome':<11}  " + "  ".join(f"{'s='+str(r['seed']):>10}" for r in rows) + f"  {'Earth':>8}"
    print("\nFertility (mean across biome's land pixels):")
    print(header)
    print("-" * len(header))
    for name in PRIMARY_BIOMES:
        cells = []
        for r in rows:
            v = r["per_biome"].get(name, {}).get("fertility_mean", float("nan"))
            cells.append(f"{v:>10.3f}")
        earth = EARTH_REF_FERTILITY.get(name, float("nan"))
        print(f"{name:<11}  " + "  ".join(cells) + f"  {earth:>8.2f}")

    print("\nForest density (mean across biome's land pixels):")
    print(header)
    print("-" * len(header))
    for name in PRIMARY_BIOMES:
        cells = []
        for r in rows:
            v = r["per_biome"].get(name, {}).get("forest_mean", float("nan"))
            cells.append(f"{v:>10.3f}")
        earth = EARTH_REF_FOREST.get(name, float("nan"))
        print(f"{name:<11}  " + "  ".join(cells) + f"  {earth:>8.2f}")
    print()


def print_detail(rows: list[dict]) -> None:
    for r in rows:
        print(f"\n  Per-biome (seed={r['seed']}) — land_pixels|fertility|forest:")
        items = sorted(
            r["per_biome"].items(),
            key=lambda kv: -kv[1].get("land_pixels", 0),
        )
        for name, vals in items:
            if vals["land_pixels"] == 0:
                continue
            print(
                f"    {name:<12} "
                f"px={vals['land_pixels']:>8} "
                f"fert={vals['fertility_mean']:.3f} "
                f"forest={vals['forest_mean']:.3f}"
            )


def _load_history() -> list:
    if JSON_PATH.exists():
        with JSON_PATH.open() as f:
            return json.load(f)
    return []


def _save_history(history: list) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with JSON_PATH.open("w") as f:
        json.dump(history, f, indent=2)


def _params_summary(gp: GlobeParams) -> str:
    fp = gp.resources.fertility
    forest = gp.resources.forest_density
    return (
        f"opt_T={fp.optimal_temperature_c:.0f}°C "
        f"opt_P={fp.optimal_precipitation_mm:.0f}mm "
        f"rug_thr={fp.elevation_range_threshold_km:.1f}km "
        f"flood={fp.floodplain_bonus:.2f} "
        f"volc={fp.volcanic_bonus:.2f} "
        f"treeline={forest.treeline_elevation_km:.1f}km"
    )


def _avg(rows: list[dict], biome: str, key: str) -> float:
    vals = []
    for r in rows:
        v = r["per_biome"].get(biome, {}).get(key, float("nan"))
        if v == v:  # not NaN
            vals.append(v)
    return sum(vals) / max(len(vals), 1) if vals else float("nan")


def _update_markdown(history: list) -> None:
    lines = [
        "# Resources — Fertility & Forest Density Tracking",
        "",
        "Auto-generated by `scripts/measure_resources.py`. Do not edit by hand.",
        "",
        "## Earth-ish Reference Anchors",
        "",
        "| Biome | Fertility | Forest density |",
        "|-------|----------:|--------------:|",
    ]
    for name in PRIMARY_BIOMES:
        lines.append(
            f"| {name} "
            f"| {EARTH_REF_FERTILITY.get(name, float('nan')):.2f} "
            f"| {EARTH_REF_FOREST.get(name, float('nan')):.2f} |"
        )
    lines += [
        "",
        "Rough literature anchors — exact numbers depend on definitions of",
        "\"forest\" and soil quality classes. Use as ballpark targets.",
        "",
        "## Run History (averaged across seeds in run)",
        "",
        "### Fertility",
        "",
        "| Date | Seeds | Tropical | Temperate | Taiga | Savannah | Steppe | Tundra | SemiArid | Desert | Params |",
        "|------|-------|---------:|----------:|------:|---------:|-------:|-------:|---------:|-------:|--------|",
    ]
    for run in history:
        rows = run["results"]
        seeds_str = " ".join(str(r["seed"]) for r in rows)
        date = run["timestamp"][:10]
        cells = [f"{_avg(rows, b, 'fertility_mean'):.2f}" for b in PRIMARY_BIOMES]
        lines.append(
            f"| {date} | {seeds_str} | " + " | ".join(cells) +
            f" | {run.get('params_summary', '—')} |"
        )

    lines += [
        "",
        "### Forest density",
        "",
        "| Date | Seeds | Tropical | Temperate | Taiga | Savannah | Steppe | Tundra | SemiArid | Desert | Params |",
        "|------|-------|---------:|----------:|------:|---------:|-------:|-------:|---------:|-------:|--------|",
    ]
    for run in history:
        rows = run["results"]
        seeds_str = " ".join(str(r["seed"]) for r in rows)
        date = run["timestamp"][:10]
        cells = [f"{_avg(rows, b, 'forest_mean'):.2f}" for b in PRIMARY_BIOMES]
        lines.append(
            f"| {date} | {seeds_str} | " + " | ".join(cells) +
            f" | {run.get('params_summary', '—')} |"
        )

    if history:
        last = history[-1]["results"]
        lines += [
            "",
            "## Per-Seed Detail (latest run)",
            "",
            "| Seed | Biome | land_pixels | fertility | forest |",
            "|------|-------|------------:|----------:|-------:|",
        ]
        for r in last:
            for name in PRIMARY_BIOMES:
                vals = r["per_biome"].get(name, {})
                if vals.get("land_pixels", 0) == 0:
                    continue
                lines.append(
                    f"| {r['seed']} | {name} "
                    f"| {vals['land_pixels']} "
                    f"| {vals['fertility_mean']:.3f} "
                    f"| {vals['forest_mean']:.3f} |"
                )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text("\n".join(lines) + "\n")


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
                        help="Print full per-biome breakdown per seed")
    args = parser.parse_args()

    seeds = (
        [random.randint(0, 2**31 - 1) for _ in range(args.random)]
        if args.random
        else (args.seeds or [42, 137, 999])
    )

    print(f"\nFertility & forest density — {len(seeds)} seed(s)\n")

    rows: list[dict] = []
    for seed in seeds:
        print(f"  building seed={seed}…", end="", flush=True)
        row = measure_seed(seed)
        rows.append(row)
        overall = row["overall"]
        print(
            f"  land_px={overall['land_pixels']:>8}  "
            f"fert={overall['fertility_mean']:.3f}  "
            f"forest={overall['forest_mean']:.3f}"
        )

    print_table(rows)
    if args.detail:
        print_detail(rows)

    gp = GlobeParams()
    run_entry = {
        "timestamp":      datetime.datetime.utcnow().isoformat(),
        "params_summary": _params_summary(gp),
        "results":        rows,
    }
    history = _load_history()
    history.append(run_entry)
    _save_history(history)
    _update_markdown(history)

    rel = lambda p: p.relative_to(Path(__file__).parent.parent)
    print(f"Results appended to {rel(JSON_PATH)}")
    print(f"Markdown updated at  {rel(MD_PATH)}")
    print()


if __name__ == "__main__":
    main()
