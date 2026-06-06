"""Measure tropical-zone biome coverage vs. Earth reference values.

Tracks Tropical, Savannah, and Desert biome fractions within tropical
latitudes across one or more seeds, comparing against Earth reference values.
Runs the elevation + climate pipeline (no hydrology). Results are appended
to scripts/results/zone_biome_results.json and a human-readable summary is
regenerated at scripts/results/zone_biome_results.md.

Usage:
    python3 scripts/measure_zone_biomes.py              # seeds 42, 137, 999
    python3 scripts/measure_zone_biomes.py 1 2 3        # specific seeds
    python3 scripts/measure_zone_biomes.py --random 5   # 5 random seeds
    python3 scripts/measure_zone_biomes.py --detail     # full biome breakdown

Earth reference values (approximate, ±30° land):
    Tropical rainforest as % of global land:       ~9 %
    Tropical rainforest as % of ±23.5° land:      ~30 %
    Tropical rainforest as % of ±30.0° land:      ~22 %
    Savannah as % of ±30.0° land:                 ~20 %
    Hot desert as % of ±30.0° land:               ~18 %
    Mean annual precipitation in equatorial band: ~2000 mm/yr
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
from worldgen.climate.builder import ClimateBuilder, ClimateParams  # noqa: E402
from worldgen.climate.classification import BIOME_NAMES  # noqa: E402
from worldgen.geology.elevation_builder import ElevationBuilder, ElevationParams  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
JSON_PATH = RESULTS_DIR / "zone_biome_results.json"
MD_PATH = RESULTS_DIR / "zone_biome_results.md"

TROPICAL_IDX = BIOME_NAMES.index("Tropical")   # 6
SAVANNAH_IDX = BIOME_NAMES.index("Savannah")   # 5
DESERT_IDX   = BIOME_NAMES.index("Desert")     # 8

# Earth reference values — WWF Terrestrial Ecoregions / GlobCover estimates.
# Savannah and Desert figures are rough literature estimates.
EARTH_REF = {
    "pct_global":   9.0,   # tropical RF as % of all land
    "pct_23_5":    30.0,   # tropical RF as % of ±23.5° land
    "pct_30_0":    22.0,   # tropical RF as % of ±30.0° land
    "sav_30_0":    20.0,   # savannah as % of ±30.0° land
    "des_30_0":    18.0,   # hot desert as % of ±30.0° land
    "eq_precip":  2000,    # mean annual P in equatorial (±5°) land, mm/yr
}


# ---------------------------------------------------------------------------
# Per-seed measurement
# ---------------------------------------------------------------------------

def measure_seed(seed: int) -> dict:
    """Build elevation + climate for one seed; return biome statistics dict."""
    ctx = BuildContext(world=World(seed=seed))
    elev_builder = ElevationBuilder(ElevationParams(), seed=seed)
    elev_builder.build(ctx)

    climate_params = ClimateParams()
    climate_builder = ClimateBuilder(climate_params)
    climate_builder.build(ctx)

    mesh = elev_builder.mesh
    elevations = elev_builder.elevations          # (N,) km
    n_faces = len(mesh.faces)

    # Face latitudes in degrees.
    z = np.clip(mesh.face_centroids[:, 2], -1.0, 1.0)
    lat_deg = np.degrees(np.arcsin(z))            # (N,)

    # Land mask: elevation above sea level.
    is_land = elevations > 0.0                    # (N,) bool

    # Per-face biome index and annual precipitation from metadata.
    biomes = np.empty(n_faces, dtype=np.int8)
    precip_annual = np.empty(n_faces, dtype=np.float64)
    for i, meta in enumerate(mesh.face_metadata):
        biomes[i] = meta.get("biome", -1)
        precip_annual[i] = meta.get("precipitation_annual", 0.0)

    # Helper: % of a given biome among land faces within ±lat_thresh degrees.
    def _biome_pct(biome_idx: int, lat_thresh: float) -> float:
        in_band = np.abs(lat_deg) < lat_thresh
        land = is_land & in_band
        n_land = int(land.sum())
        n_biome = int(((biomes == biome_idx) & land).sum())
        return round(100.0 * n_biome / max(n_land, 1), 1)

    pct_global = _biome_pct(TROPICAL_IDX, 90.0)
    pct_23_5   = _biome_pct(TROPICAL_IDX, 23.5)
    pct_30_0   = _biome_pct(TROPICAL_IDX, 30.0)
    sav_30_0   = _biome_pct(SAVANNAH_IDX, 30.0)
    des_30_0   = _biome_pct(DESERT_IDX,   30.0)

    n_land_global = int(is_land.sum())

    # Mean annual precipitation on equatorial land (±5°).
    eq_mask = is_land & (np.abs(lat_deg) < 5.0)
    mean_eq_precip = float(precip_annual[eq_mask].mean()) if eq_mask.any() else 0.0

    # Biome distribution (as %) for ±30° land — useful for diagnosing
    # which biome is "consuming" what share.
    in_30 = is_land & (np.abs(lat_deg) < 30.0)
    n_in_30 = max(int(in_30.sum()), 1)
    biome_dist_30 = {
        name: round(100.0 * int(((biomes == idx) & in_30).sum()) / n_in_30, 1)
        for idx, name in enumerate(BIOME_NAMES)
    }

    return {
        "seed":               seed,
        "n_land_faces":       n_land_global,
        "pct_global":         pct_global,
        "pct_23_5":           pct_23_5,
        "pct_30_0":           pct_30_0,
        "sav_30_0":           sav_30_0,
        "des_30_0":           des_30_0,
        "mean_eq_precip_mm":  round(mean_eq_precip, 0),
        "biome_dist_30":      biome_dist_30,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _fmt_row(label: str, r: dict) -> str:
    return (
        f"{label:>8}  {r['pct_23_5']:>8.1f}%  {r['pct_30_0']:>8.1f}%  "
        f"{r.get('sav_30_0', float('nan')):>8.1f}%  "
        f"{r.get('des_30_0', float('nan')):>8.1f}%  "
        f"{r['mean_eq_precip_mm']:>10.0f}"
    )


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'seed':>8}  {'T±23.5%':>9}  {'T±30.0%':>9}  "
        f"{'Sav±30%':>9}  {'Des±30%':>9}  {'eq_P(mm)':>10}"
    )
    sep = "-" * len(header)
    earth_row = {
        "pct_23_5":          EARTH_REF["pct_23_5"],
        "pct_30_0":          EARTH_REF["pct_30_0"],
        "sav_30_0":          EARTH_REF["sav_30_0"],
        "des_30_0":          EARTH_REF["des_30_0"],
        "mean_eq_precip_mm": EARTH_REF["eq_precip"],
    }
    print(header)
    print(sep)
    print(_fmt_row("Earth*", earth_row))
    print()
    for r in rows:
        print(_fmt_row(str(r["seed"]), r))
    print()
    print(
        "* Earth: Tropical RF from WWF Terrestrial Ecoregions (~Af/Am Köppen);\n"
        "  Savannah and Desert are rough estimates from biome area literature."
    )


def print_detail(rows: list[dict]) -> None:
    for r in rows:
        print(f"\n  Biome distribution in ±30° land (seed={r['seed']}):")
        # Sort by descending share for readability.
        sorted_items = sorted(r["biome_dist_30"].items(), key=lambda x: -x[1])
        for name, pct in sorted_items:
            bar = "█" * int(pct / 2)
            print(f"    {name:<12} {pct:>5.1f}%  {bar}")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _load_history() -> list:
    if JSON_PATH.exists():
        with JSON_PATH.open() as f:
            return json.load(f)
    return []


def _save_history(history: list) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with JSON_PATH.open("w") as f:
        json.dump(history, f, indent=2)


def _params_summary(cp: ClimateParams) -> str:
    p = cp.precipitation
    return (
        f"scale={p.precipitation_scale:.0f}mm "
        f"decay={p.moisture_decay_rate:.3f} "
        f"evap={p.evaporation_scale:.2f} "
        f"oro_k={p.orographic_k:.2f}"
    )


def _update_markdown(history: list) -> None:
    lines = [
        "# Tropical-Zone Biome Coverage — Tracking Results",
        "",
        "Auto-generated by `scripts/measure_zone_biomes.py`. Do not edit by hand.",
        "",
        "## Earth Reference Values",
        "",
        "| Metric | Earth (approx.) |",
        "|--------|----------------|",
        f"| Tropical RF as % of global land       | ~{EARTH_REF['pct_global']:.0f}% |",
        f"| Tropical RF as % of ±23.5° land       | ~{EARTH_REF['pct_23_5']:.0f}% |",
        f"| Tropical RF as % of ±30.0° land       | ~{EARTH_REF['pct_30_0']:.0f}% |",
        f"| Savannah as % of ±30.0° land          | ~{EARTH_REF['sav_30_0']:.0f}% (rough) |",
        f"| Hot desert as % of ±30.0° land        | ~{EARTH_REF['des_30_0']:.0f}% (rough) |",
        f"| Mean annual precip in equatorial land | ~{EARTH_REF['eq_precip']} mm/yr |",
        "",
        "> Source: WWF Terrestrial Ecoregions of the World; GlobCover land-cover datasets.",
        "",
        "## Run History",
        "",
        "Each row is the average across all seeds in that run.",
        "",
        "| Date | Seeds | T±23.5% | T±30.0% | Sav±30% | Des±30% | eq_P(mm) | Key params |",
        "|------|-------|---------|---------|---------|---------|----------|------------|",
    ]

    for run in history:
        results = run["results"]

        def avg(key: str, default: float = float("nan")) -> float:
            vals = [r.get(key, default) for r in results]
            valid = [v for v in vals if v == v]  # exclude nan
            return sum(valid) / max(len(valid), 1)

        seeds_str = " ".join(str(r["seed"]) for r in results)
        date = run["timestamp"][:10]
        lines.append(
            f"| {date} | {seeds_str} "
            f"| {avg('pct_23_5'):.1f}% "
            f"| {avg('pct_30_0'):.1f}% "
            f"| {avg('sav_30_0'):.1f}% "
            f"| {avg('des_30_0'):.1f}% "
            f"| {avg('mean_eq_precip_mm'):.0f} "
            f"| {run.get('params_summary', '—')} |"
        )

    lines += [
        "",
        "## Per-Seed Detail (latest run)",
        "",
    ]

    if history:
        last = history[-1]["results"]
        lines += [
            "| Seed | T±23.5% | T±30.0% | Sav±30% | Des±30% | eq_P(mm) |",
            "|------|---------|---------|---------|---------|----------|",
        ]
        for r in last:
            lines.append(
                f"| {r['seed']} "
                f"| {r['pct_23_5']:.1f}% "
                f"| {r['pct_30_0']:.1f}% "
                f"| {r.get('sav_30_0', float('nan')):.1f}% "
                f"| {r.get('des_30_0', float('nan')):.1f}% "
                f"| {r['mean_eq_precip_mm']:.0f} |"
            )

        lines += ["", "### Biome distribution in ±30° land (latest run)", ""]
        for r in last:
            lines.append(f"**seed={r['seed']}**")
            lines.append("")
            lines.append("| Biome | % of ±30° land |")
            lines.append("|-------|---------------|")
            for name, pct in sorted(r["biome_dist_30"].items(), key=lambda x: -x[1]):
                lines.append(f"| {name} | {pct:.1f}% |")
            lines.append("")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "seeds", nargs="*", type=int,
        help="Explicit seed values (default: 42 137 999)",
    )
    parser.add_argument(
        "--random", metavar="N", type=int,
        help="Use N random seeds instead of explicit ones",
    )
    parser.add_argument(
        "--detail", action="store_true",
        help="Print full biome breakdown per seed after summary table",
    )
    args = parser.parse_args()

    seeds = (
        [random.randint(0, 2**31 - 1) for _ in range(args.random)]
        if args.random
        else (args.seeds or [42, 137, 999])
    )

    print(f"\nTropical-zone biome coverage — {len(seeds)} seed(s)\n")

    rows: list[dict] = []
    for seed in seeds:
        print(f"  building seed={seed}…", end="", flush=True)
        row = measure_seed(seed)
        rows.append(row)
        print(
            f"  T±23.5°={row['pct_23_5']:.1f}%  "
            f"T±30°={row['pct_30_0']:.1f}%  "
            f"Sav±30°={row['sav_30_0']:.1f}%  "
            f"Des±30°={row['des_30_0']:.1f}%  "
            f"eq_P={row['mean_eq_precip_mm']:.0f}mm"
        )

    print()
    print_table(rows)

    if args.detail:
        print_detail(rows)

    cp = ClimateParams()
    run_entry = {
        "timestamp":      datetime.datetime.utcnow().isoformat(),
        "params_summary": _params_summary(cp),
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
