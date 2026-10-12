"""Measure sub-grid relief at the region, local and settlement tiers, by terrain class.

The ruggedness work (``docs/plans/ruggedness.md``) is judged by how much relief
each tier draws on plains, hills and mountains.  Per seed this builds, as the
viewer does:

  • the local map (and so its region) of ``--hexes`` land faces per terrain class,
    spread over the class;
  • the settlement-map terrain (no site or layout) of ``--towns`` towns per class,
    at population quantiles 0.5–1;

and reports, per tier and class, over the dry texels each map draws:

  • **window relief** — max − min elevation (m) in a square window around each
    texel, water ignored: ~1 km on local and settlement maps (also 250 m on
    settlement maps), three texels (~13 km) on regions;
  • slope (km/km) p50 / p90;
  • the conservation max (km) of each map against its parent (must stay at float noise).

Terrain classes are terciles over land faces of the globe's per-face elevation
span (``face_elevation_range``) — the globe doesn't change in this work, so the
classes stay put across runs — and, once the world carries a ``ruggedness``
field, terciles of that too; and fixed bins of each texel's own ruggedness
(``rho``: <0.2, 0.2–0.4, 0.4–0.6, ≥0.6), which is what the spectrum reads.

Usage:
    python3 scripts/measure_relief.py                 # seeds 4, 31, 35
    python3 scripts/measure_relief.py 1 2 --hexes 1   # fewer maps
    python3 scripts/measure_relief.py --towns 0       # region and local tiers only
    python3 scripts/measure_relief.py --rebuild-cache

Results appended to ``scripts/results/relief_results.json``.
"""
from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

from localmap.upsample import local_conservation_report
from region.geometry import RegionGeometry, ecef_to_cell
from region.grid import RegionHexGrid
from region.resources import refined_slope
from region.upsample import RegionFields, conservation_report
from settlegen.chain import SettlementRef, ensure_local_maps
from settlegen.upsample import build_settlement_fields, settlement_conservation_report
from worldgen.geology.channels import ELEVATION, IS_OCEAN, RUGGEDNESS_FACE
from worldgen.resources.terrain import face_elevation_range
from _common import add_seed_args, get_map_store, load_history, resolve_seeds, save_history, settlement_map_params

DEFAULT_SEEDS = [4, 31, 35]
RESULTS_PATH = Path(__file__).parent / "results" / "relief_results.json"
CLASSES = ("flat", "hilly", "rugged")
# Fixed bins of each texel's own (refined copy of the) ruggedness, for calibrating the spectrum.
RHO_EDGES = (0.2, 0.4, 0.6)
RHO_CLASSES = ("<0.2", "0.2-0.4", "0.4-0.6", ">=0.6")
NAMES = {"span": CLASSES, "ruggedness": CLASSES, "rho": RHO_CLASSES}
# (tier, window km) pairs reported; a region's window is three of its texels whatever this says.
WINDOWS = (("region", None), ("local", 1.0), ("settlement", 1.0), ("settlement", 0.25))


def terciles(values: np.ndarray, land: np.ndarray) -> np.ndarray:
    """(N,) class index 0..2 by terciles of ``values`` over ``land`` faces (−1 off land)."""
    cuts = np.quantile(values[land], [1.0 / 3.0, 2.0 / 3.0])
    out = np.digitize(values, cuts).astype(np.int64)
    out[~land] = -1
    return out


def face_classes(world) -> dict[str, np.ndarray]:  # type: ignore[no-untyped-def]
    """``{scheme: (N,) class index}`` — by elevation span, and by ruggedness when the world has it."""
    land = ~np.asarray(IS_OCEAN.read(world.mesh), dtype=bool)
    schemes = {"span": terciles(face_elevation_range(ELEVATION.read(world.mesh)), land)}
    if "ruggedness" in world.mesh.face_textures:
        schemes["ruggedness"] = terciles(np.asarray(RUGGEDNESS_FACE.read(world.mesh), dtype=np.float64), land)
    return schemes


def hex_of_face(world, face: int, n_cols: int) -> tuple[int, int]:  # type: ignore[no-untyped-def]
    """``(region_id, hex_index)`` of the regional hex a global face's centroid lies in."""
    c = np.asarray(world.mesh.face_centroids[face], dtype=np.float64)
    c = c / np.linalg.norm(c)
    region_id = int(ecef_to_cell(c[None], world.antimeridian)[0])
    geometry = RegionGeometry(region_id, world.antimeridian)
    grid = RegionHexGrid(geometry, n_cols)
    s, t, _ = geometry.ecef_to_local(c[None])
    return region_id, int(np.argmin((grid.centers[:, 0] - s[0]) ** 2 + (grid.centers[:, 1] - t[0]) ** 2))


def spread(items: np.ndarray, n: int, lo: float = 0.0, hi: float = 1.0) -> list[int]:
    """Up to ``n`` entries of ``items`` at evenly spaced quantile positions in [lo, hi]."""
    if n <= 0 or len(items) == 0:
        return []
    at = np.unique(np.round(np.linspace(lo, hi, n + 2)[1:-1] * (len(items) - 1)).astype(int))
    return [int(items[k]) for k in at]


def dry(fields: RegionFields) -> np.ndarray:
    """(M, T, T) bool: texels in the map's domain that no sea or lake covers."""
    wet = np.zeros(fields.domain.shape, dtype=bool)
    for key in ("ocean_mask", "lake_mask"):
        if key in fields.textures:
            wet |= np.asarray(fields.textures[key][..., 0]) > 0.5
    return fields.domain & ~wet


def texel_km(fields: RegionFields) -> float:
    return float(np.sqrt(fields.texel_area_km2))


def window_relief_m(fields: RegionFields, land: np.ndarray, window_texels: int) -> np.ndarray:
    """(M, T, T) max − min land elevation (m) in a ``window_texels`` square, per hex texture."""
    elev = np.asarray(fields.textures["elevation"][..., 0], dtype=np.float64) * 1000.0
    size = (1, window_texels, window_texels)
    hi = maximum_filter(np.where(land, elev, -np.inf), size=size, mode="nearest")
    lo = minimum_filter(np.where(land, elev, np.inf), size=size, mode="nearest")
    return hi - lo


def window_texels(fields: RegionFields, window_km: Optional[float]) -> int:
    if window_km is None:
        return 3
    return max(3, int(round(window_km / texel_km(fields))) | 1)


def sample(fields: RegionFields, classes: dict[str, np.ndarray], window_km: Optional[float]) -> dict:
    """Per-texel arrays over the map's dry texels: window relief, slope, and each scheme's class."""
    land = dry(fields)
    relief = window_relief_m(fields, land, window_texels(fields, window_km))
    slope = refined_slope(fields.textures["elevation"][..., 0], texel_km(fields))
    faces = fields.faces[land]
    out = {
        "relief": relief[land], "slope": slope[land],
        **{scheme: cls[faces] for scheme, cls in classes.items()},
    }
    if "ruggedness" in fields.textures:
        out["rho"] = np.digitize(np.asarray(fields.textures["ruggedness"][..., 0])[land], RHO_EDGES)
    return out


def summarise(chunks: list[dict], scheme: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not chunks:
        return out
    pooled = {k: np.concatenate([c[k] for c in chunks]) for k in ("relief", "slope", scheme)}
    for k, name in enumerate(NAMES[scheme]):
        sel = pooled[scheme] == k
        if not sel.any():
            continue
        r, s = pooled["relief"][sel], pooled["slope"][sel]
        out[name] = {
            "n": int(sel.sum()),
            "relief_p10": float(np.percentile(r, 10)), "relief_p50": float(np.percentile(r, 50)),
            "relief_p90": float(np.percentile(r, 90)),
            "slope_p50": float(np.percentile(s, 50)), "slope_p90": float(np.percentile(s, 90)),
        }
    return out


def elevation_max(report: dict) -> float:
    return float(report.get("elevation", {}).get("max", float("nan")))


def measure_seed(seed: int, n_hexes: int, n_towns: int, rebuild: bool) -> dict:
    t0 = time.perf_counter()
    store = get_map_store(seed, rebuild)
    world = store.globe
    assert world is not None
    classes = face_classes(world)
    span = classes["span"]
    schemes = [*classes, "rho"]
    n_cols = int(store.region_params.upsample.n_cols)

    # Local maps (and their regions) of land faces spread over each span class.
    hexes: list[tuple[int, int]] = []
    for k in range(len(CLASSES)):
        faces = np.flatnonzero(span == k)
        order = faces[np.argsort(face_elevation_range(ELEVATION.read(world.mesh))[faces])]
        for f in spread(order, n_hexes):
            key = hex_of_face(world, f, n_cols)
            if key not in hexes:
                hexes.append(key)
    chunks: dict[tuple[str, Optional[float]], list[dict]] = {w: [] for w in WINDOWS}
    conservation: dict[str, list[float]] = {"region": [], "local": [], "settlement": []}
    regions_done: set[int] = set()
    for region_id, hex_index in hexes:
        local = store.local(region_id, hex_index).fields
        region = store.region(region_id).fields
        if region_id not in regions_done:
            regions_done.add(region_id)
            chunks[("region", None)].append(sample(region, classes, None))
            conservation["region"].append(elevation_max(conservation_report(region, world)))
        chunks[("local", 1.0)].append(sample(local, classes, 1.0))
        conservation["local"].append(elevation_max(local_conservation_report(local, region)))

    # Settlement terrain of towns spread over each span class (population quantiles 0.5–1).
    towns = world.get_feature("towns")
    params = settlement_map_params(store)
    n_frames = 0
    if n_towns > 0 and towns is not None:
        pop = np.asarray(towns.population, dtype=np.float64)
        town_class = span[np.asarray(towns.face_id)]
        for k in range(len(CLASSES)):
            members = np.flatnonzero(town_class == k)
            for i in spread(members[np.argsort(pop[members])], n_towns, 0.5, 1.0):
                ref = SettlementRef("town", i)
                chain = ensure_local_maps(store, ref, float(params.upsample.margin_km), params.footprint)
                maps = [m.fields for m in chain.maps]
                fields = build_settlement_fields(world, maps, chain.center, chain.side_km, params.upsample)
                for w in WINDOWS:
                    if w[0] == "settlement":
                        chunks[w].append(sample(fields, classes, w[1]))
                conservation["settlement"].append(
                    elevation_max(settlement_conservation_report(fields, maps, world.antimeridian)))
                n_frames += 1

    tiers = {
        f"{tier}" + (f"@{w}km" if tier == "settlement" else ""): {
            scheme: summarise(chunks[(tier, w)], scheme) for scheme in schemes
            if all(scheme in c for c in chunks[(tier, w)])
        }
        for tier, w in WINDOWS
    }
    return {
        "seed": seed, "regions": len(regions_done), "local_maps": len(hexes), "settlement_frames": n_frames,
        "tiers": tiers,
        "conservation_max_km": {k: (max(v) if v else float("nan")) for k, v in conservation.items()},
        "seconds": time.perf_counter() - t0,
    }


def print_seed(row: dict) -> None:
    print(f"\nseed {row['seed']}: {row['regions']} regions, {row['local_maps']} local maps, "
          f"{row['settlement_frames']} settlement frames ({row['seconds']:.0f} s)")
    cons = row["conservation_max_km"]
    print("  conservation max (km): " + ", ".join(f"{k} {v:.1e}" for k, v in cons.items()))
    for tier, schemes in row["tiers"].items():
        for scheme, by_class in schemes.items():
            for name, s in by_class.items():
                print(f"  {tier:<17} {scheme:<10} {name:<7} n={s['n']:>8}  relief m "
                      f"p10 {s['relief_p10']:7.1f}  p50 {s['relief_p50']:7.1f}  p90 {s['relief_p90']:7.1f}   "
                      f"slope p50 {s['slope_p50']:.4f}  p90 {s['slope_p90']:.4f}")


def print_pooled(rows: list[dict]) -> None:
    """Median over seeds of each tier/scheme/class's p50 relief and p90 slope."""
    print("\nacross seeds (median of per-seed values):")
    keys = sorted({(t, sc, c) for r in rows for t, d in r["tiers"].items() for sc, bc in d.items() for c in bc},
                  key=lambda k: (k[0], k[1], NAMES[k[1]].index(k[2])))
    for tier, scheme, name in keys:
        vals = [r["tiers"][tier][scheme][name] for r in rows if name in r["tiers"].get(tier, {}).get(scheme, {})]
        stats = ("relief_p10", "relief_p50", "relief_p90", "slope_p90")
        med = {k: float(np.median([v[k] for v in vals])) for k in stats}
        print(f"  {tier:<17} {scheme:<10} {name:<7} relief m p10 {med['relief_p10']:7.1f}  "
              f"p50 {med['relief_p50']:7.1f}  p90 {med['relief_p90']:7.1f}   slope p90 {med['slope_p90']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_seed_args(parser)
    parser.add_argument("--hexes", type=int, default=2, help="Local maps per terrain class (default 2)")
    parser.add_argument("--towns", type=int, default=2, help="Settlement frames per terrain class (default 2)")
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild the physical worlds")
    args = parser.parse_args()
    seeds = resolve_seeds(args, DEFAULT_SEEDS)

    rows = []
    for seed in seeds:
        row = measure_seed(seed, args.hexes, args.towns, args.rebuild_cache)
        print_seed(row)
        rows.append(row)
    if len(rows) > 1:
        print_pooled(rows)

    history = load_history(RESULTS_PATH)
    history.append({
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "hexes": args.hexes, "towns": args.towns, "rows": rows,
    })
    save_history(history, RESULTS_PATH)
    print(f"\nappended to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
