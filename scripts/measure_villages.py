"""Measure local-map villages — counts, sizes, spacing and the share of rural people.

For each seed, builds the realm on the cached physical world, samples regional
hexes across the rural-density range (the hexes of the land faces at the
``QUANTILES`` of face density), builds each hex's local map as the viewer does,
and reports per hex:

  • rural density (people/km²) and villages per 100 km² — against the count the
    budget predicts, ``village_fraction · ρ · 100 / mean size``: a shortfall
    means siting ran out of room (spacing) rather than people;
  • village size (mean, median) and nearest-neighbour spacing (median, p10, p90);
  • the share of rural people living in villages (target ``village_fraction``),
    and the share left in *no-village* budgets — global texel ∩ map pieces too
    small to afford one village (slivers at face seams, and thin land);
  • village kinds, and the budget conservation residual (must be ~1e-8).

Earth-ish anchors (rough, high-medieval England/northern France c.1300): rural
density ~30–40/km², villages of a few hundred people ~2–3 km apart, most people
in villages rather than dispersed farmsteads.  This world's rural density is
calibrated lower (``measure_population_absolute.py``), so its villages sit
further apart at the same fraction; compare spacing against density, not alone.

Usage:
    python3 scripts/measure_villages.py                 # seeds 4, 31, 35
    python3 scripts/measure_villages.py 1 2 3           # specific seeds
    python3 scripts/measure_villages.py --random 3      # random seeds
    python3 scripts/measure_villages.py --hexes 8       # more hexes per seed
    python3 scripts/measure_villages.py --rebuild-cache # rebuild the physical worlds

Results appended to ``scripts/results/village_results.json``.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import time
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from localmap import LocalParams, village_budget_report
from localmap.villages import VILLAGE_KIND_NAMES
from region.rivers import build_region_rivers
from region.roads import build_region_roads
from region.upsample import EARTH_RADIUS_KM, RegionUpsampleParams, build_region_fields
from localmap.builder import build_local_map
from debug_local import hex_of_face, land_face_density
from _common import add_seed_args, avg, get_realm_world, load_history, resolve_seeds, save_history

DEFAULT_SEEDS = [4, 31, 35]
# Face-density quantiles the sampled hexes are taken at (1.0 = the densest land).
QUANTILES = (0.5, 0.75, 0.9, 0.97, 1.0)
JSON_PATH = Path(__file__).parent / "results" / "village_results.json"


def sample_hexes(world, n: int, n_cols: int) -> list[tuple[int, int, float]]:  # type: ignore[no-untyped-def]
    """``(region_id, hex_index, quantile)`` for ``n`` land hexes across the density range."""
    faces, rho = land_face_density(world)
    order = np.argsort(rho)
    qs = np.linspace(QUANTILES[0], 1.0, n) if n != len(QUANTILES) else np.array(QUANTILES)
    out: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    for q in qs:
        face = int(faces[order[min(int(q * (len(order) - 1)), len(order) - 1)]])
        key = hex_of_face(world, face, n_cols)
        if key not in seen:
            seen.add(key)
            out.append((*key, float(q)))
    return out


def measure_hex(  # type: ignore[no-untyped-def]
    world, region_cache: dict, region_id: int, hex_index: int, params: LocalParams,
) -> dict:
    if region_id not in region_cache:
        region = build_region_fields(world, region_id, RegionUpsampleParams())
        rivers = build_region_rivers(region, world)
        region_cache[region_id] = (region, rivers, build_region_roads(region, world, rivers))
    region, rivers, roads = region_cache[region_id]
    t0 = time.perf_counter()
    local = build_local_map(world, region, hex_index, params, region_rivers=rivers, region_roads=roads)
    seconds = time.perf_counter() - t0

    v = local.villages
    budget = village_budget_report(local.fields, v)
    area = float(local.fields.domain.sum()) * local.fields.texel_area_km2
    rho = budget["people"] / max(area, 1e-9)
    vp = params.villages
    # α = 1 truncated Pareto on [min, max]: mean = min·max·ln(max/min) / (max − min).
    lo, hi = float(vp.village_min), float(vp.village_max)
    mean_size = lo * hi * np.log(hi / lo) / (hi - lo) if float(vp.size_exponent) == 1.0 else float("nan")
    nn = (KDTree(v.positions).query(v.positions, k=2)[0][:, 1] * EARTH_RADIUS_KM
          if len(v) > 1 else np.array([float("nan")]))
    kinds = np.bincount(v.kind, minlength=len(VILLAGE_KIND_NAMES)) / max(len(v), 1)
    empty = v.budget_placed == 0
    return {
        "region": region_id,
        "hex": hex_index,
        "density": rho,
        "villages": len(v),
        "per_100km2": len(v) / max(area, 1e-9) * 100.0,
        "predicted_per_100km2": float(vp.village_fraction) * rho * 100.0 / mean_size,
        "mean_size": float(v.population.mean()) if len(v) else float("nan"),
        "median_size": float(np.median(v.population)) if len(v) else float("nan"),
        "nn_median_km": float(np.median(nn)),
        "nn_p10_km": float(np.percentile(nn, 10)),
        "nn_p90_km": float(np.percentile(nn, 90)),
        "village_share": budget["village_share"],
        "no_village_share": float(v.budget_people[empty].sum()) / max(budget["people"], 1e-9),
        "kinds": {name: float(k) for name, k in zip(VILLAGE_KIND_NAMES, kinds)},
        "max_relative_residual": budget["max_relative_residual"],
        "seconds": seconds,
    }


def print_rows(seed: int, rows: list[dict]) -> None:
    print(f"\n  seed {seed}")
    print(f"  {'region/hex':>11} {'ρ/km²':>6} {'vill':>5} {'/100km²':>8} {'pred':>6} {'mean':>5} "
          f"{'nn med':>7} {'p10–p90 km':>11} {'share':>6} {'no-vill':>8} {'farm':>5} {'river':>6} "
          f"{'resid':>8} {'s':>5}")
    for r in rows:
        k = r["kinds"]
        print(f"  {r['region']:>5}/{r['hex']:<5} {r['density']:>6.1f} {r['villages']:>5} "
              f"{r['per_100km2']:>8.2f} "
              f"{r['predicted_per_100km2']:>6.2f} {r['mean_size']:>5.0f} {r['nn_median_km']:>7.1f} "
              f"{r['nn_p10_km']:>5.1f}–{r['nn_p90_km']:<5.1f} {r['village_share']:>6.1%} {r['no_village_share']:>8.1%} "
              f"{k['Farming village']:>5.0%} {k['River village']:>6.0%} {r['max_relative_residual']:>8.0e} "
              f"{r['seconds']:>5.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_seed_args(ap)
    ap.add_argument("--hexes", type=int, default=len(QUANTILES), help="hexes sampled per seed")
    ap.add_argument("--rebuild-cache", action="store_true", help="rebuild the physical worlds")
    args = ap.parse_args()
    seeds = resolve_seeds(args, DEFAULT_SEEDS)
    params = LocalParams()

    print(f"\nLocal-map villages — {len(seeds)} seed(s), {args.hexes} hexes each")
    results = []
    for seed in seeds:
        world = get_realm_world(seed, args.rebuild_cache)
        region_cache: dict = {}
        rows = [
            {**measure_hex(world, region_cache, region_id, hex_index, params), "quantile": q}
            for region_id, hex_index, q in sample_hexes(world, args.hexes, int(RegionUpsampleParams().n_cols))
        ]
        print_rows(seed, rows)
        results.append({"seed": seed, "hexes": rows})

    all_rows = [r for res in results for r in res["hexes"]]
    print(f"\n  all: share {avg(all_rows, 'village_share'):.1%} (target {params.villages.village_fraction:.0%}), "
          f"no-village {avg(all_rows, 'no_village_share'):.1%}, "
          f"villages/100 km² {avg(all_rows, 'per_100km2'):.2f} "
          f"vs predicted {avg(all_rows, 'predicted_per_100km2'):.2f}, "
          f"max residual {max(r['max_relative_residual'] for r in all_rows):.0e}")

    history = load_history(JSON_PATH)
    history.append({
        "timestamp": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(),
        "params": dataclasses.asdict(params.villages),
        "results": results,
    })
    save_history(history, JSON_PATH)
    print(f"  results appended to {JSON_PATH.relative_to(Path(__file__).parent.parent)}\n")


if __name__ == "__main__":
    main()
