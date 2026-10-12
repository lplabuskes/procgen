"""Measure lakes and rivers on region and local maps — the shapes rough relief can spoil.

Rougher relief has twice turned lakes "lacy" (ragged shores, swarms of islands)
and can send a re-routed river over a rise its parent channel never crossed.
Per seed this builds, as the viewer does, the regions holding the most global
lake, the most river and the roughest ground (elevation span: unaffected by
the sub-grid work, so every run picks the same maps), and on each the local
maps of its most-lake and most-river hexes, and reports per tier:

  • **lakes** (the refined ``lake_mask`` resampled onto a square raster at the
    map's texel size): share of the dry-or-lake domain, count (≥ ``MIN_LAKE_PX``
    texels), islands (land wholly inside a lake) per lake and per 1,000 km² of
    lake, and the area-weighted shoreline development index (perimeter over the
    circumference of an equal-area circle: 1 a disc; lacy lakes run high);
    the same for the tier's own small lakes alone;
  • **rivers** (re-routed river paths, lake vertices skipped, elevation read at
    each vertex's texel): uphill climb per 100 km of river, the share of river
    length on uphill steps over ``UPHILL_M``, and the worst single path's climb.

Usage:
    python3 scripts/measure_lakes_rivers.py                  # seeds 4, 31, 35
    python3 scripts/measure_lakes_rivers.py 4 --regions 2
    python3 scripts/measure_lakes_rivers.py --cache-dir /tmp/worlds --rebuild-cache

Results appended to ``scripts/results/lakes_rivers_results.json``.
"""
from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path

import numpy as np
from scipy import ndimage


from region.geometry import ecef_to_cell
from region.upsample import SMALL_LAKE_KEY, RegionFields
from worldgen.geology.channels import ELEVATION, IS_OCEAN
from worldgen.resources.terrain import face_elevation_range
from _common import CACHE_DIR, add_seed_args, get_map_store, load_history, resolve_seeds, save_history

DEFAULT_SEEDS = [4, 31, 35]
RESULTS_PATH = Path(__file__).parent / "results" / "lakes_rivers_results.json"
MIN_LAKE_PX = 3
UPHILL_M = 2.0
EARTH_RADIUS_KM = 6371.0


def _label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """``ndimage.label`` with its types pinned."""
    out = ndimage.label(mask)
    return np.asarray(out[0], dtype=np.int64), int(out[1])  # type: ignore[index]


def raster(fields: RegionFields) -> dict[str, np.ndarray]:
    """The map's masks and elevation resampled onto a square (s, t) raster at its texel size."""
    grid, tex = fields.grid, fields.tex_size
    step = 2.0 * grid.radius / tex
    lo = -grid.radius
    n = int(np.ceil((1.0 - 2.0 * lo) / step)) + 1
    cols, rows = np.meshgrid(np.arange(n), np.arange(n))
    hexes, hr, hc, covered = grid.nearest_texels(lo + step * cols.ravel(), lo + step * rows.ravel(), tex)
    at = (hexes, hr, hc)
    domain = covered & fields.domain[at]
    out = {"domain": domain.reshape(n, n)}
    for key in ("ocean_mask", "lake_mask", SMALL_LAKE_KEY):
        if key in fields.textures:
            out[key] = ((fields.textures[key][..., 0][at] > 0.5) & domain).reshape(n, n)
        else:
            out[key] = np.zeros((n, n), dtype=bool)
    return out


def lake_stats(lake: np.ndarray, domain: np.ndarray, ocean: np.ndarray, px_km: float) -> dict:
    labels, n = _label(lake)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    big = np.flatnonzero(sizes >= MIN_LAKE_PX)
    big = big[big > 0]
    filled = ndimage.binary_fill_holes(lake)
    islands = filled & ~lake & domain & ~ocean
    isl_labels, n_isl = _label(islands)
    # Each island to the lake that surrounds it: the lake label just outside it.
    owner = np.zeros(n_isl + 1, dtype=np.int64)
    if n_isl:
        ring = ndimage.binary_dilation(islands) & lake
        grown = np.asarray(ndimage.grey_dilation(isl_labels, size=3), dtype=np.int64)
        pairs = np.unique(np.stack([grown[ring], labels[ring]], axis=1), axis=0)
        for i, lab in pairs:
            if i > 0:
                owner[i] = lab
    keep = np.isin(owner[1:], big)
    n_islands = int(keep.sum())
    area = float(sizes[big].sum()) * px_km ** 2
    # Perimeter: lake/non-lake texel edges, inside the domain.
    edges = np.zeros(n + 1, dtype=np.float64)
    for a, b in ((labels[1:, :], labels[:-1, :]), (labels[:, 1:], labels[:, :-1])):
        diff = a != b
        np.add.at(edges, a[diff], 1.0)
        np.add.at(edges, b[diff], 1.0)
    sdi = edges[big] * px_km / (2.0 * np.sqrt(np.pi * sizes[big] * px_km ** 2))
    land = float((domain & ~ocean).sum())
    return {
        "share": float(lake.sum()) / max(land, 1.0),
        "lakes": int(big.size),
        "islands": n_islands,
        "islands_per_lake": n_islands / max(big.size, 1),
        "islands_per_1000km2": 1000.0 * n_islands / max(area, 1e-9) if area > 0 else float("nan"),
        "sdi": float(np.average(sdi, weights=sizes[big])) if big.size else float("nan"),
        "lake_km2": area,
    }


def river_stats(fields: RegionFields, paths) -> dict:  # type: ignore[no-untyped-def]
    elev = np.asarray(fields.textures["elevation"][..., 0], dtype=np.float64) * 1000.0
    climb = length = uphill_len = 0.0
    worst = 0.0
    for pos, _flow, in_lake in paths:
        pos = np.asarray(pos, dtype=np.float64)
        if len(pos) < 2:
            continue
        s, t, valid = fields.geometry.ecef_to_local(pos)
        hexes, hr, hc, covered = fields.grid.nearest_texels(s, t, fields.tex_size)
        ok = valid & covered & fields.domain[hexes, hr, hc]
        e = elev[hexes, hr, hc]
        wet = np.asarray(in_lake, dtype=bool).reshape(-1)
        seg_ok = ok[1:] & ok[:-1]
        if wet.size == seg_ok.size:            # per segment
            seg_ok &= ~wet
        elif wet.size == ok.size:              # per vertex
            seg_ok &= ~wet[1:] & ~wet[:-1]
        seg_km = np.linalg.norm(pos[1:] - pos[:-1], axis=1) * EARTH_RADIUS_KM
        rise = np.diff(e)
        up = np.where(seg_ok, np.maximum(rise, 0.0), 0.0)
        climb += float(up.sum())
        length += float(seg_km[seg_ok].sum())
        uphill_len += float(seg_km[seg_ok & (rise > UPHILL_M)].sum())
        worst = max(worst, float(up.sum()))
    return {
        "river_km": length,
        "climb_m_per_100km": 100.0 * climb / max(length, 1e-9),
        "uphill_share": uphill_len / max(length, 1e-9),
        "worst_path_climb_m": worst,
    }


def pick_regions(world, n: int) -> list[int]:  # type: ignore[no-untyped-def]
    mesh = world.mesh
    c = np.asarray(mesh.face_centroids, dtype=np.float64)
    cell = ecef_to_cell(c / np.linalg.norm(c, axis=1, keepdims=True), world.antimeridian)
    land = ~np.asarray(IS_OCEAN.read(mesh), dtype=bool)
    lake = np.zeros(len(land))
    if "lake_mask" in mesh.face_textures:
        lake = np.asarray(mesh.face_textures["lake_mask"][..., 0]).mean(axis=(1, 2))
    span = face_elevation_range(ELEVATION.read(mesh))
    streams = world.get_feature("streams")
    river = np.zeros(len(land))
    if streams is not None:
        ch = np.asarray(streams.channel) & (np.asarray(streams.flow) >= 500)
        hw = mesh.face_textures["elevation"].shape[1] ** 2
        np.add.at(river, np.asarray(streams.nodes)[ch] // hw, 1.0)
    cells = np.unique(cell)
    picks: list[int] = []
    for score in (lake, river, np.where(land, span, 0.0)):
        totals = np.array([score[cell == k].sum() for k in cells])
        for k in cells[np.argsort(-totals)]:
            if int(k) not in picks:
                picks.append(int(k))
                break
        if len(picks) >= n:
            break
    return picks[:n]


def best_hexes(region_fields: RegionFields, paths) -> list[int]:  # type: ignore[no-untyped-def]
    lake = (region_fields.textures["lake_mask"][..., 0] > 0.5) & region_fields.domain
    by_lake = lake.sum(axis=(1, 2))
    counts = np.zeros(region_fields.n_hexes)
    for pos, _flow, _ in paths:
        s, t, valid = region_fields.geometry.ecef_to_local(np.asarray(pos, dtype=np.float64))
        hexes, _, _, covered = region_fields.grid.nearest_texels(s, t, region_fields.tex_size)
        np.add.at(counts, hexes[valid & covered], 1.0)
    out = []
    for score in (by_lake, counts):
        if score.max() > 0:
            k = int(np.argmax(score))
            if k not in out:
                out.append(k)
    return out


def measure_seed(seed: int, n_regions: int, rebuild: bool, cache_dir: Path) -> dict:
    t0 = time.perf_counter()
    store = get_map_store(seed, rebuild, cache_dir)
    world = store.globe
    assert world is not None
    rows: dict[str, list[dict]] = {"region": [], "local": []}
    for region_id in pick_regions(world, n_regions):
        rmap = store.region(region_id)
        maps = [("region", rmap.fields, rmap.rivers.paths)]
        for h in best_hexes(rmap.fields, rmap.rivers.paths):
            lmap = store.local(region_id, h)
            maps.append(("local", lmap.fields, lmap.rivers.paths))
        for tier, fields, paths in maps:
            r = raster(fields)
            px_km = float(np.sqrt(fields.texel_area_km2))
            all_lakes = lake_stats(r["lake_mask"], r["domain"], r["ocean_mask"], px_km)
            small = lake_stats(r[SMALL_LAKE_KEY], r["domain"], r["ocean_mask"], px_km)
            rows[tier].append({"region": region_id, **{f"lake_{k}": v for k, v in all_lakes.items()},
                               **{f"small_{k}": v for k, v in small.items()}, **river_stats(fields, paths)})
    return {"seed": seed, "rows": rows, "seconds": time.perf_counter() - t0}


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {}
    km2 = sum(r["lake_lake_km2"] for r in rows)
    river_km = sum(r["river_km"] for r in rows)
    return {
        "maps": len(rows),
        "lake_share": float(np.mean([r["lake_share"] for r in rows])),
        "lakes": int(sum(r["lake_lakes"] for r in rows)),
        "islands_per_lake": sum(r["lake_islands"] for r in rows) / max(sum(r["lake_lakes"] for r in rows), 1),
        "islands_per_1000km2": 1000.0 * sum(r["lake_islands"] for r in rows) / max(km2, 1e-9),
        "sdi": float(np.average([r["lake_sdi"] for r in rows if r["lake_lakes"]],
                                weights=[r["lake_lake_km2"] for r in rows if r["lake_lakes"]]))
        if any(r["lake_lakes"] for r in rows) else float("nan"),
        "small_share": float(np.mean([r["small_share"] for r in rows])),
        "small_lakes": int(sum(r["small_lakes"] for r in rows)),
        "climb_m_per_100km": sum(r["climb_m_per_100km"] * r["river_km"] for r in rows) / max(river_km, 1e-9),
        "uphill_share": sum(r["uphill_share"] * r["river_km"] for r in rows) / max(river_km, 1e-9),
        "worst_path_climb_m": max(r["worst_path_climb_m"] for r in rows),
    }


def print_summary(label: str, s: dict) -> None:
    for tier, v in s.items():
        if not v:
            continue
        print(f"  {label} {tier:<6} {v['maps']} maps · lakes {v['lake_share']:.1%} of land, {v['lakes']} lakes, "
              f"islands {v['islands_per_lake']:.2f}/lake ({v['islands_per_1000km2']:.1f}/1000 km²), "
              f"shore dev {v['sdi']:.2f} · small {v['small_share']:.2%} ({v['small_lakes']}) · rivers climb "
              f"{v['climb_m_per_100km']:.0f} m/100 km, uphill {v['uphill_share']:.1%}, "
              f"worst {v['worst_path_climb_m']:.0f} m")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_seed_args(parser)
    parser.add_argument("--regions", type=int, default=3, help="regions per seed (most lake, river, roughest)")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR, help="world cache directory")
    parser.add_argument("--rebuild-cache", action="store_true", help="rebuild the cached worlds")
    args = parser.parse_args()
    seeds = resolve_seeds(args, DEFAULT_SEEDS)
    results = []
    for seed in seeds:
        res = measure_seed(seed, args.regions, args.rebuild_cache, args.cache_dir)
        res["summary"] = {tier: summarise(rows) for tier, rows in res["rows"].items()}
        print(f"seed {seed} ({res['seconds']:.0f} s)")
        print_summary(f"s{seed}", res["summary"])
        results.append(res)
    if len(results) > 1:
        print("all seeds:")
        pooled = {tier: summarise([r for res in results for r in res["rows"][tier]]) for tier in ("region", "local")}
        print_summary("all", pooled)
    history = load_history(RESULTS_PATH)
    history.append({"timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "results": results})
    save_history(history, RESULTS_PATH)
    print(f"appended to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
