"""Measure the river *length distribution* vs Earth's great rivers.

Motivation: the generated worlds feel short on "great rivers of the world" —
the Rhine / Nile / Mississippi class of multi-thousand-km channels. This harness
quantifies that. It runs elevation → climate → hydrology (the lake-feedback,
resources, and cost stages do not change rivers, so they are skipped) and reports
the source-to-mouth length distribution of the generated rivers.

Each ``River`` object is one main stem or tributary from the Hack decomposition:
a chain from a headwater down to the sea, an endorheic sink, or the confluence
where it joins a larger stem. So a river's polyline length is its source-to-mouth
(or source-to-confluence) length — comparable to how real-world river lengths are
quoted.

Usage:
    python3 scripts/measure_rivers.py              # vetted seeds 4, 31, 35
    python3 scripts/measure_rivers.py 1 2 3        # specific seeds
    python3 scripts/measure_rivers.py --random 5   # 5 random seeds
    python3 scripts/measure_rivers.py --detail     # per-seed longest rivers + histogram

Earth reference — notable main-stem lengths (km):
    Nile                  ~6650      Mississippi (alone)  ~3766
    Amazon                ~6400      Danube               ~2850
    Yangtze               ~6300      Rhine                ~1233
    Mississippi-Missouri  ~6275      Seine                 ~777

Rough global counts of rivers exceeding a length (land-area dependent, so these
are order-of-magnitude anchors, not hard targets):
    > 1000 km: ~150     > 2000 km: ~60     > 3000 km: ~30     > 4000 km: ~17
"""
from __future__ import annotations

import argparse
import datetime
import json
import random
from pathlib import Path

import numpy as np

from core.pipeline import BuildContext
from core.world import World
from worldgen import GlobeParams
from worldgen.climate.builder import ClimateBuilder
from worldgen.geology.channels import OCEAN_MASK
from worldgen.geology.elevation_builder import ElevationBuilder
from worldgen.hydrology.builder import HydrologyBuilder


RESULTS_DIR = Path(__file__).parent / "results"
JSON_PATH = RESULTS_DIR / "rivers_results.json"
MD_PATH = RESULTS_DIR / "rivers_results.md"

EARTH_RADIUS_KM = 6371.0

# Length thresholds (km) we count rivers above, with rough global anchor counts.
LENGTH_THRESHOLDS = [1000, 2000, 3000, 4000]
EARTH_THRESHOLD_COUNTS = {1000: 150, 2000: 60, 3000: 30, 4000: 17}

# Notable real-world main-stem lengths (km) for context in the markdown.
EARTH_NOTABLE = {
    "Nile": 6650,
    "Amazon": 6400,
    "Yangtze": 6300,
    "Mississippi-Missouri": 6275,
    "Mississippi": 3766,
    "Danube": 2850,
    "Rhine": 1233,
    "Seine": 777,
}
# Headline anchor: the longest single channel on Earth.
EARTH_LONGEST_KM = 6650

# Histogram buckets (km) for the --detail / markdown distribution view.
HIST_EDGES = [0, 250, 500, 1000, 2000, 3000, 4000, float("inf")]


def _hist_labels() -> list[str]:
    labels = []
    for lo, hi in zip(HIST_EDGES[:-1], HIST_EDGES[1:]):
        if hi == float("inf"):
            labels.append(f"{lo}+")
        else:
            labels.append(f"{lo}-{hi}")
    return labels


# ── Per-seed measurement ──────────────────────────────────────────────────────


def _river_length_km(positions: np.ndarray) -> float:
    """Great-circle length (km) of a polyline of unit-sphere positions.

    Matches the delta-length formula in hydrology/builder.py for consistency.
    """
    if len(positions) < 2:
        return 0.0
    p = positions.astype(np.float64)
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    p = p / np.maximum(norms, 1e-12)
    dots = np.clip(np.einsum("ij,ij->i", p[:-1], p[1:]), -1.0, 1.0)
    return float(np.arccos(dots).sum()) * EARTH_RADIUS_KM


def measure_seed(seed: int) -> dict:
    """Build elevation → climate → hydrology for one seed; return length stats."""
    params = GlobeParams()
    params.seed = seed
    world = World(params.mesh, seed=seed)
    ctx = BuildContext(world=world, seed=seed)

    ElevationBuilder(params.elevation, seed=seed).build(ctx)
    ClimateBuilder(params.climate).build(ctx)
    hydro = HydrologyBuilder(params.hydrology)
    hydro.build(ctx)

    mesh = ctx.mesh
    ocean_tex = OCEAN_MASK.read(mesh)[..., 0]  # (N, H, W)
    res = ocean_tex.shape[1]

    lengths: list[float] = []
    mouth_flows: list[float] = []
    reaches_ocean: list[bool] = []
    for river in hydro.rivers or []:
        poly = river.as_polyline()
        if len(poly) < 2:
            continue
        positions = np.stack(poly)
        lengths.append(_river_length_km(positions))
        mouth_flows.append(float(river.segments[-1].flow_rate))
        # Classify the mouth as sea-reaching by sampling the ocean mask at the
        # mouth pixel (same mapping the delta-tagging logic uses).
        if river.mouth is None:
            continue
        face = int(river.mouth.face_id)
        u, v = river.mouth.uv
        row = min(max(int(round(v * (res - 1))), 0), res - 1)
        col = min(max(int(round(u * (res - 1))), 0), res - 1)
        reaches_ocean.append(bool(ocean_tex[face, row, col] > 0.5))

    return _summarize(seed, lengths, mouth_flows, reaches_ocean)


def _summarize(
    seed: int,
    lengths: list[float],
    mouth_flows: list[float],
    reaches_ocean: list[bool],
) -> dict:
    nan = float("nan")
    L = np.array(lengths, dtype=np.float64)
    F = np.array(mouth_flows, dtype=np.float64)
    ocean = np.array(reaches_ocean, dtype=bool)
    n = int(L.size)

    out: dict = {
        "seed": seed,
        "n_rivers": n,
        "longest_km": nan,
        "p95_km": nan,
        "p75_km": nan,
        "median_km": nan,
        "mean_km": nan,
        "total_length_kkm": nan,
        "pct_reaches_ocean": nan,
        "threshold_counts": {str(t): 0 for t in LENGTH_THRESHOLDS},
        "histogram": [0] * (len(HIST_EDGES) - 1),
        "top_rivers": [],
    }
    if n == 0:
        return out

    out["longest_km"] = round(float(L.max()), 1)
    out["p95_km"] = round(float(np.percentile(L, 95)), 1)
    out["p75_km"] = round(float(np.percentile(L, 75)), 1)
    out["median_km"] = round(float(np.median(L)), 1)
    out["mean_km"] = round(float(L.mean()), 1)
    out["total_length_kkm"] = round(float(L.sum()) / 1000.0, 1)
    out["pct_reaches_ocean"] = round(100.0 * float(ocean.mean()), 1)
    out["threshold_counts"] = {str(t): int((L >= t).sum()) for t in LENGTH_THRESHOLDS}
    out["histogram"] = [
        int(((L >= lo) & (L < hi)).sum())
        for lo, hi in zip(HIST_EDGES[:-1], HIST_EDGES[1:])
    ]

    # Top rivers by length (with mouth flow + sea-reaching flag) for --detail.
    order = np.argsort(L)[::-1][:12]
    out["top_rivers"] = [
        {
            "length_km": round(float(L[i]), 1),
            "mouth_flow": round(float(F[i]), 1),
            "reaches_ocean": bool(ocean[i]),
        }
        for i in order
    ]
    return out


# ── Reporting ─────────────────────────────────────────────────────────────────


def _fmt(v: float, prec: int = 1) -> str:
    return "   nan" if v != v else f"{v:>8.{prec}f}"


def print_table(rows: list[dict]) -> None:
    print("\nRiver length distribution (km):")
    header = (
        f"{'metric':<22}  "
        + "  ".join(f"{'s=' + str(r['seed']):>10}" for r in rows)
        + f"  {'Earth':>10}"
    )
    print(header)
    print("-" * len(header))

    def line(label: str, key: str, earth: str = "—", prec: int = 1) -> None:
        cells = "  ".join(f"{_fmt(r[key], prec):>10}" for r in rows)
        print(f"{label:<22}  {cells}  {earth:>10}")

    line("n_rivers", "n_rivers", "—", 0)
    line("longest", "longest_km", f"{EARTH_LONGEST_KM}")
    line("p95", "p95_km")
    line("p75", "p75_km")
    line("median", "median_km")
    line("mean", "mean_km")
    line("total (×1000 km)", "total_length_kkm", "—")
    line("% reaches ocean", "pct_reaches_ocean", "—")

    print()
    for t in LENGTH_THRESHOLDS:
        cells = "  ".join(
            f"{r['threshold_counts'][str(t)]:>10d}" for r in rows
        )
        earth = f"~{EARTH_THRESHOLD_COUNTS[t]}"
        print(f"{'count > ' + str(t) + ' km':<22}  {cells}  {earth:>10}")
    print()


def print_detail(rows: list[dict]) -> None:
    labels = _hist_labels()
    for r in rows:
        print(f"\n  Length histogram (seed={r['seed']}) — river count per km bucket:")
        peak = max(r["histogram"]) or 1
        for label, count in zip(labels, r["histogram"]):
            bar = "█" * int(round(40 * count / peak))
            print(f"    {label:>10} km | {count:>5d} {bar}")

        print(f"\n  Longest rivers (seed={r['seed']}) — length | mouth flow | sea?:")
        for i, rv in enumerate(r["top_rivers"], start=1):
            sea = "sea" if rv["reaches_ocean"] else "inland"
            print(
                f"    {i:>2}. {rv['length_km']:>8.1f} km "
                f"| flow {rv['mouth_flow']:>10.1f} | {sea}"
            )
    print()


# ── JSON history + markdown ──────────────────────────────────────────────────


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
    hp = gp.hydrology
    return (
        f"flow_thr={hp.river_flow_threshold:.0f} "
        f"min_lake={hp.min_lake_pixels} "
        f"ico_depth={gp.mesh.icosphere_depth}"
    )


def _avg(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and r[key] == r[key]]
    return sum(vals) / len(vals) if vals else float("nan")


def _avg_threshold(rows: list[dict], t: int) -> float:
    vals = [r["threshold_counts"][str(t)] for r in rows]
    return sum(vals) / max(len(vals), 1)


def _update_markdown(history: list) -> None:
    lines = [
        "# Rivers — Length Distribution Tracking",
        "",
        "Auto-generated by `scripts/measure_rivers.py`. Do not edit by hand.",
        "",
        "Each `River` is one main stem or tributary from the Hack decomposition (a",
        "chain from a headwater to the sea, an endorheic sink, or its confluence with",
        "a larger stem), so a river's polyline length is its source-to-mouth length —",
        "comparable to quoted real-world river lengths.",
        "",
        "## Earth Reference — notable main-stem lengths (km)",
        "",
        "| River | Length (km) |",
        "|-------|------------:|",
    ]
    for name, km in EARTH_NOTABLE.items():
        lines.append(f"| {name} | {km} |")
    lines += [
        "",
        "Rough global counts of rivers exceeding a length (land-area dependent —",
        "order-of-magnitude anchors, not hard targets):",
        "",
        "| Threshold | ~Global count |",
        "|-----------|--------------:|",
    ]
    for t in LENGTH_THRESHOLDS:
        lines.append(f"| > {t} km | ~{EARTH_THRESHOLD_COUNTS[t]} |")

    lines += [
        "",
        "## Run History (averaged across seeds in run)",
        "",
        "| Date | Seeds | n | longest | p95 | median | mean | "
        + " | ".join(f">{t}" for t in LENGTH_THRESHOLDS)
        + " | %sea | Params |",
        "|------|-------|--:|--------:|----:|-------:|-----:|"
        + "".join("---:|" for _ in LENGTH_THRESHOLDS)
        + "-----:|--------|",
    ]
    for run in history:
        rows = run["results"]
        seeds_str = " ".join(str(r["seed"]) for r in rows)
        date = run["timestamp"][:10]
        thr = " | ".join(f"{_avg_threshold(rows, t):.0f}" for t in LENGTH_THRESHOLDS)
        lines.append(
            f"| {date} | {seeds_str} "
            f"| {_avg(rows, 'n_rivers'):.0f} "
            f"| {_avg(rows, 'longest_km'):.0f} "
            f"| {_avg(rows, 'p95_km'):.0f} "
            f"| {_avg(rows, 'median_km'):.0f} "
            f"| {_avg(rows, 'mean_km'):.0f} "
            f"| {thr} "
            f"| {_avg(rows, 'pct_reaches_ocean'):.0f} "
            f"| {run.get('params_summary', '—')} |"
        )

    if history:
        last = history[-1]["results"]
        lines += [
            "",
            "## Per-Seed Detail (latest run)",
            "",
            "| Seed | n | longest | p95 | median | mean | "
            + " | ".join(f">{t}" for t in LENGTH_THRESHOLDS)
            + " | %sea |",
            "|------|--:|--------:|----:|-------:|-----:|"
            + "".join("---:|" for _ in LENGTH_THRESHOLDS)
            + "-----:|",
        ]
        for r in last:
            thr = " | ".join(
                str(r["threshold_counts"][str(t)]) for t in LENGTH_THRESHOLDS
            )
            lines.append(
                f"| {r['seed']} | {r['n_rivers']} "
                f"| {r['longest_km']:.0f} | {r['p95_km']:.0f} "
                f"| {r['median_km']:.0f} | {r['mean_km']:.0f} "
                f"| {thr} | {r['pct_reaches_ocean']:.0f} |"
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
                        help="Print per-seed length histogram and longest rivers")
    args = parser.parse_args()

    seeds = (
        [random.randint(0, 2**31 - 1) for _ in range(args.random)]
        if args.random
        # Default tuning set: chosen for Earth-like land fraction (~29%) and a
        # continent-scale landmass, so river length isn't capped by geography.
        else (args.seeds or [4, 31, 35])
    )

    print(f"\nRiver length distribution — {len(seeds)} seed(s)\n")

    rows: list[dict] = []
    for seed in seeds:
        print(f"  building seed={seed}…", end="", flush=True)
        row = measure_seed(seed)
        rows.append(row)
        print(
            f"  n={row['n_rivers']:>5}  "
            f"longest={_fmt(row['longest_km'], 0).strip():>6} km  "
            f"median={_fmt(row['median_km'], 0).strip():>5} km  "
            f">2000km={row['threshold_counts']['2000']:>3}"
        )

    print_table(rows)
    if args.detail:
        print_detail(rows)

    run_entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(),
        "params_summary": _params_summary(GlobeParams()),
        "results": rows,
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
