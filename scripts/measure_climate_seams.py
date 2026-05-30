"""Measure temperature and precipitation discontinuities at face-to-face boundaries.

Usage:
    python3 scripts/measure_climate_seams.py                  # seeds 42, 137, 999
    python3 scripts/measure_climate_seams.py 1 2 3            # specific seeds
    python3 scripts/measure_climate_seams.py --random 5       # 5 random seeds

For each seed, runs the full elevation + climate pipeline (no hydrology) and
reports mean/median/RMS/max boundary jump for both temperature (°C) and
precipitation (mm/season).  The jump at each matched seam-pixel pair is the
mean absolute difference averaged across all 4 seasons.  Use this to quantify
how much of a hard per-face edge remains after the IDW smoothing step.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from worldgen.climate.builder import ClimateBuilder, ClimateParams  # noqa: E402
from worldgen.geology.elevation_builder import ElevationBuilder, ElevationParams  # noqa: E402
from worldgen.mesh import cache as mesh_cache  # noqa: E402


def measure_seed(seed: int) -> dict:
    e_params = ElevationParams(seed=seed)
    if not mesh_cache.is_populated():
        mesh_cache.build(
            icosphere_depth=e_params.mesh.icosphere_depth,
            n_swaps=e_params.mesh.n_swaps,
            n_relax=e_params.mesh.n_relax,
            stiffness=e_params.mesh.stiffness,
            outer_iters=e_params.mesh.outer_iters,
        )

    e_builder = ElevationBuilder(e_params)
    e_builder.build(progress=lambda _: None)
    mesh = e_builder.mesh
    elev_tex = mesh.face_textures["elevation"]

    c_builder = ClimateBuilder(ClimateParams())
    c_builder.build(mesh, elev_tex, progress=lambda _: None)

    t_tex = mesh.face_textures["temperature"]    # (N, H, W, K) °C
    p_tex = mesh.face_textures["precipitation"]  # (N, H, W, K) mm/season
    h_tex = mesh.face_textures["humidity"]       # (N, H, W, K) dimensionless q_air

    t_stats = mesh.measure_boundary_texture_jumps(t_tex)
    p_stats = mesh.measure_boundary_texture_jumps(p_tex)
    h_stats = mesh.measure_boundary_texture_jumps(h_tex)

    return {
        "seed": seed,
        "temp_mean":   t_stats["mean"],
        "temp_median": t_stats["median"],
        "temp_rms":    t_stats["rms"],
        "temp_max":    t_stats["max"],
        "prec_mean":   p_stats["mean"],
        "prec_median": p_stats["median"],
        "prec_rms":    p_stats["rms"],
        "prec_max":    p_stats["max"],
        "hum_mean":    h_stats["mean"],
        "hum_median":  h_stats["median"],
        "hum_rms":     h_stats["rms"],
        "hum_max":     h_stats["max"],
        "n_pairs":     t_stats["n_pairs"],
    }


def print_table(rows: list[dict]) -> None:
    h = (
        f"{'seed':>8}  "
        f"{'T mean':>8}  {'T med':>8}  {'T RMS':>8}  {'T max':>8}  "
        f"{'P mean':>9}  {'P med':>9}  {'P RMS':>9}  {'P max':>9}  "
        f"{'H mean':>8}  {'H med':>8}  {'H RMS':>8}  {'H max':>8}  "
        f"{'n_pairs':>8}"
    )
    units = (
        f"{'':>8}  "
        f"{'°C':>8}  {'°C':>8}  {'°C':>8}  {'°C':>8}  "
        f"{'mm/seas':>9}  {'mm/seas':>9}  {'mm/seas':>9}  {'mm/seas':>9}  "
        f"{'q_air':>8}  {'q_air':>8}  {'q_air':>8}  {'q_air':>8}  "
        f"{'':>8}"
    )
    print(h)
    print(units)
    print("-" * len(h))
    for r in rows:
        print(
            f"{r['seed']:>8}  "
            f"{r['temp_mean']:>8.2f}  {r['temp_median']:>8.2f}  "
            f"{r['temp_rms']:>8.2f}  {r['temp_max']:>8.2f}  "
            f"{r['prec_mean']:>9.2f}  {r['prec_median']:>9.2f}  "
            f"{r['prec_rms']:>9.2f}  {r['prec_max']:>9.2f}  "
            f"{r['hum_mean']:>8.4f}  {r['hum_median']:>8.4f}  "
            f"{r['hum_rms']:>8.4f}  {r['hum_max']:>8.4f}  "
            f"{r['n_pairs']:>8}"
        )


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
    args = parser.parse_args()

    seeds = (
        [random.randint(0, 2**31 - 1) for _ in range(args.random)]
        if args.random
        else (args.seeds or [42, 137, 999])
    )

    print("\nClimate boundary seam statistics (mean across 4 seasons per pair)\n")

    rows = []
    for seed in seeds:
        print(f"  building seed={seed}…", end="", flush=True)
        row = measure_seed(seed)
        rows.append(row)
        print(f"  T_max={row['temp_max']:.2f}°C  P_max={row['prec_max']:.1f}mm/s  H_max={row['hum_max']:.4f}")

    print()
    print_table(rows)
    print()


if __name__ == "__main__":
    main()
