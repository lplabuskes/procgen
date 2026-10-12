"""Measure settlement maps — build time, conservation, footprints, roads, anchors, layouts.

Builds settlement maps through the chain (``docs/settlement_maps.md``), as the
viewer does, for each seed's ``--cities`` largest cities, ``--towns`` towns at
population quantiles 0.5–1.0, and ``--villages`` villages of the largest city's
local map (spread over its list), and reports per settlement and per kind:

  • wall-clock per step (chain — the region and local maps it had to build, so
    the first settlement of a region pays for them — terrain, site, profile, layout);
  • the conservation residual: max per local texel of the frame's elevation
    against the local maps' (must stay at float32 noise, ~1e-9 km);
  • footprint error: the densely built area against ``population / density``
    (the sparse ground holds the rest, so a few percent under is expected),
    pieces, sparse share, density;
  • gates, bridges and fords, landings; anchor candidates per kind;
  • the layout: blocks or farmsteads, plots, piers, wall length and gates, and
    the ground the walls took in beyond the site's footprint.

Usage:
    python3 scripts/measure_settlements.py                     # seeds 4, 31, 35
    python3 scripts/measure_settlements.py 1 2 3               # specific seeds
    python3 scripts/measure_settlements.py --random 2          # random seeds
    python3 scripts/measure_settlements.py --cities 10 --towns 5 --villages 0
    python3 scripts/measure_settlements.py --detail            # one line per settlement
    python3 scripts/measure_settlements.py --rebuild-cache     # rebuild the physical worlds

Results appended to ``scripts/results/settlement_results.json``.
"""
from __future__ import annotations

import argparse
import datetime
import math
import time
from pathlib import Path

import numpy as np

from settlegen.builder import build_settlement_map
from settlegen.chain import SettlementRef, population
from settlegen.upsample import settlement_conservation_report
from _common import (
    add_seed_args, avg, get_map_store, load_history, resolve_seeds, save_history, settlement_map_params,
)

DEFAULT_SEEDS = [4, 31, 35]
RESULTS_PATH = Path(__file__).parent / "results" / "settlement_results.json"
STEPS = ("chain", "terrain", "site", "profile", "layout")


def pick(store, n_cities: int, n_towns: int, n_villages: int) -> list[SettlementRef]:  # type: ignore[no-untyped-def]
    world = store.globe
    n = len(world.get_feature("settlements"))
    city_pop = np.array([population(world, SettlementRef("city", i)) for i in range(n)])
    cities = [SettlementRef("city", int(i)) for i in np.argsort(-city_pop)[:n_cities]]
    refs = list(cities)
    if n_towns > 0:
        town_pop = np.asarray(world.get_feature("towns").population, dtype=np.float64)
        order = np.argsort(town_pop)
        at = np.unique(np.round(np.linspace(0.5, 1.0, n_towns) * (len(order) - 1)).astype(int))
        refs += [SettlementRef("town", int(order[k])) for k in at]
    if n_villages > 0 and cities:
        # The largest city's chain gives its owning local map: its villages, spread over the list.
        from settlegen.chain import ensure_local_maps
        params = settlement_map_params(store)
        chain = ensure_local_maps(store, cities[0], float(params.upsample.margin_km), params.footprint)
        region, hex_index = chain.keys[0]
        k = len(store.local(region, hex_index).villages)
        at = np.unique(np.round(np.linspace(0.0, 1.0, n_villages) * max(k - 1, 0)).astype(int)) if k else []
        refs += [SettlementRef("village", int(i), map=(region, hex_index)) for i in at]
    return refs


def measure(store, ref: SettlementRef, params) -> dict:  # type: ignore[no-untyped-def]
    timings: dict[str, float] = {}
    m = build_settlement_map(store, ref, params, timings)
    site, profile, layout = m.site, m.profile, m.layout
    report = settlement_conservation_report(m.fields, [mp.fields for mp in m.chain.maps], store.globe.antimeridian)
    pop = float(profile.population)
    want_m2 = pop / max(site.density_per_ha, 1e-9) * 1e4
    dense = site.footprint.difference(site.sparse) if not site.sparse.is_empty else site.footprint
    bridges = sum(c.bridge for c in site.roads.crossings)
    row = {
        "kind": ref.kind, "index": ref.index, "map": list(ref.map) if ref.map is not None else None,
        "tier": profile.tier_name, "site_kind": profile.kind, "population": pop,
        "maps": len(m.chain.keys), "frame_km": site.frame.side_m / 1000.0,
        **{f"t_{k}": timings.get(k, float("nan")) for k in STEPS},
        "t_total": sum(timings.values()),
        "conservation_max": float(report["elevation"]["max"]) if "elevation" in report else float("nan"),
        "density_per_ha": float(site.density_per_ha),
        "footprint_ha": site.footprint.area / 1e4,
        "footprint_error": dense.area / want_m2 - 1.0 if want_m2 > 0 else float("nan"),
        "pieces": len(getattr(site.footprint, "geoms", [])),
        "sparse_share": site.sparse.area / max(site.footprint.area, 1e-9),
        "gates": len(site.roads.gates), "bridges": bridges, "fords": len(site.roads.crossings) - bridges,
        "landings": len(site.roads.landings),
        **{f"anchors_{k}": len(v) for k, v in site.anchors.by_kind.items()},
        "blocks": len(layout.blocks), "lots": layout.n_lots, "piers": len(layout.piers),
        "wall_km": layout.walls.line.length / 1000.0 if layout.walls is not None else 0.0,
        "wall_gates": len(layout.walls.gates) if layout.walls is not None else 0,
        "walls_took_ha": (layout.footprint.difference(site.footprint).area / 1e4
                          if layout.footprint is not None else 0.0),
    }
    return row


def _fmt(row: dict) -> str:
    where = f" @{row['map'][0]}/{row['map'][1]}" if row["map"] else ""
    return (f"    {row['kind']:<7} {row['index']:>5}{where:<9} {row['population']:>9,.0f} {row['site_kind']:<9} "
            f"{row['t_total']:>5.1f} s (chain {row['t_chain']:>4.1f}) · cons {row['conservation_max']:.1e} · "
            f"fp {row['footprint_error']:+5.1%} {row['pieces']}p · gates {row['gates']} br {row['bridges']} "
            f"fd {row['fords']} land {row['landings']} · blocks {row['blocks']:,} plots {row['lots']:,} "
            f"piers {row['piers']}" + (f" · walls {row['wall_km']:.1f} km" if row["wall_km"] else ""))


def summarise(rows: list[dict]) -> dict:
    out: dict = {}
    for kind in ("city", "town", "village"):
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            continue
        out[kind] = {
            "n": len(sub),
            **{f"t_{k}": avg(sub, f"t_{k}") for k in (*STEPS, "total")},
            "t_after_chain": avg([{"v": r["t_total"] - r["t_chain"]} for r in sub], "v"),
            "conservation_max": max(r["conservation_max"] for r in sub),
            "footprint_abs_error": avg([{"v": abs(r["footprint_error"])} for r in sub], "v"),
            **{k: avg(sub, k) for k in ("pieces", "sparse_share", "gates", "bridges", "fords", "landings",
                                        "anchors_citadel", "anchors_market", "anchors_waterfront",
                                        "anchors_sacred", "blocks", "lots", "piers", "wall_km", "walls_took_ha")},
        }
    return out


def print_summary(summary: dict) -> None:
    for kind, s in summary.items():
        print(f"  {kind:<7} n={s['n']} · time {s['t_total']:.1f} s (after the chain {s['t_after_chain']:.2f} s: "
              f"terrain {s['t_terrain']:.2f}, site {s['t_site']:.2f}, layout {s['t_layout']:.2f}) · "
              f"conservation max {s['conservation_max']:.1e}")
        print(f"          footprint |error| {s['footprint_abs_error']:.1%}, {s['pieces']:.1f} pieces, "
              f"sparse {s['sparse_share']:.0%} · gates {s['gates']:.1f} · bridges {s['bridges']:.1f}, "
              f"fords {s['fords']:.1f}, landings {s['landings']:.1f}")
        print(f"          anchors citadel {s['anchors_citadel']:.1f} market {s['anchors_market']:.1f} "
              f"waterfront {s['anchors_waterfront']:.1f} sacred {s['anchors_sacred']:.1f} · blocks {s['blocks']:,.0f}, "
              f"plots {s['lots']:,.0f}, piers {s['piers']:.1f}, walls {s['wall_km']:.1f} km "
              f"(took in {s['walls_took_ha']:.1f} ha)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_seed_args(parser)
    parser.add_argument("--cities", type=int, default=6, help="largest cities per seed")
    parser.add_argument("--towns", type=int, default=3, help="towns per seed, at population quantiles 0.5–1")
    parser.add_argument("--villages", type=int, default=3, help="villages of the largest city's local map")
    parser.add_argument("--detail", action="store_true", help="one line per settlement")
    parser.add_argument("--rebuild-cache", action="store_true", help="rebuild the cached physical worlds")
    args = parser.parse_args()
    seeds = resolve_seeds(args, DEFAULT_SEEDS)

    history = load_history(RESULTS_PATH)
    all_rows: list[dict] = []
    for seed in seeds:
        t0 = time.perf_counter()
        store = get_map_store(seed, args.rebuild_cache)
        params = settlement_map_params(store)
        refs = pick(store, args.cities, args.towns, args.villages)
        rows = []
        for ref in refs:
            try:
                row = measure(store, ref, params)
            except Exception as exc:  # noqa: BLE001  (a measurement run reports, it doesn't stop)
                print(f"  seed {seed} {ref.kind} {ref.index}: FAILED {type(exc).__name__}: {exc}")
                continue
            row["seed"] = seed
            rows.append(row)
        print(f"seed {seed}: {len(rows)} settlements in {time.perf_counter() - t0:.0f} s")
        if args.detail:
            for row in rows:
                print(_fmt(row))
        print_summary(summarise(rows))
        all_rows += rows
    if len(seeds) > 1:
        print("all seeds:")
        print_summary(summarise(all_rows))

    def clean(v):  # type: ignore[no-untyped-def]
        return None if isinstance(v, float) and math.isnan(v) else v
    history.append({
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "seeds": seeds,
        "summary": {k: {kk: clean(vv) for kk, vv in v.items()} for k, v in summarise(all_rows).items()},
        "rows": [{k: clean(v) for k, v in r.items()} for r in all_rows],
    })
    save_history(history, RESULTS_PATH)
    print(f"appended to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
