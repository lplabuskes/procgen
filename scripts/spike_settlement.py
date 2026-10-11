"""Phase 0 spike for settlement maps (docs/plans/settlement_maps.md): does a
street-scale frame refine cleanly from its local map(s)?

For the largest river cities and towns of a seed, plus settlements whose frame
crosses its local map's edge, it takes each to its local maps through
:func:`settlegen.chain.ensure_local_maps` (one shared :class:`MapStore`, so no
map is built twice), refines the frame at ``texel_m`` with :mod:`settlegen`,
and prints:

- the **conservation report** per local texel of every contributing map (must
  stay at float32 noise, as the local tier's does);
- the slope distribution against the local map's around the settlement;
- for frames crossing a seam, the elevation step across it against the
  typical step between neighbouring texels;
- wall-clock per step;
- with ``--order-check``, whether opening the neighbouring local maps *first*
  moves the settlement (A9), on a freshly built world.

Usage:
    python3 scripts/spike_settlement.py                 # seed 4: 6 river cities, 3 towns, 3 crossing
    python3 scripts/spike_settlement.py 31 --cities 3 --towns 2 --crossing 2
    python3 scripts/spike_settlement.py --order-check
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from localmap.store import MapStore
from localmap.upsample import with_region_detail
from realmgen.placement.types import SITE_KIND_CONFLUENCE, SITE_KIND_RIVER
from region.geometry import RegionGeometry, ecef_to_cell
from region.grid import RegionHexGrid
from region.resources import refined_slope
from region.upsample import RegionFields
from settlegen import (
    LocalMapsSource, SettlementUpsampleParams, build_settlement_fields, settlement_conservation_report,
    with_local_detail,
)
from settlegen.chain import SettlementRef, position, ensure_local_maps, population
from _common import get_realm_world

if TYPE_CHECKING:
    from core.world import World


def make_store(seed: int) -> MapStore:
    store = MapStore(get_realm_world(seed, False), region_capacity=4, local_capacity=8)
    local = store.local_params
    store.set_local_params(dataclasses.replace(
        local, upsample=with_region_detail(local.upsample, store.region_params.upsample)))
    return store


def seam_report(fields: RegionFields, source: LocalMapsSource) -> Optional[tuple[float, float, int, float, float]]:
    """``(mean |Δe| across a map seam, mean |Δe| elsewhere, seam pairs, p99 and max across)`` in metres, or None."""
    which = source.map_of(fields.positions.reshape(-1, 3).astype(np.float64)).reshape(fields.tex_size, -1)
    if len(np.unique(which)) < 2:
        return None
    e = fields.textures["elevation"][0, ..., 0].astype(np.float64) * 1000.0
    steps, seam = [], []
    for axis in (0, 1):
        de = np.abs(np.diff(e, axis=axis))
        cross = np.diff(which, axis=axis) != 0
        steps.append(de[~cross])
        seam.append(de[cross])
    inside, across = np.concatenate(steps), np.concatenate(seam)
    return (float(across.mean()), float(inside.mean()), int(across.size),
            float(np.percentile(across, 99)), float(across.max()))


def describe(world: "World", ref: SettlementRef) -> str:
    pop = population(world, ref)
    if ref.kind == "city":
        flags = world.get_feature("settlements")[ref.index].site_kind
        tag = "confluence" if flags & SITE_KIND_CONFLUENCE else "river" if flags & SITE_KIND_RIVER else "site"
        return f"city {ref.index} ({pop / 1000:.0f}k, {tag})"
    return f"town {ref.index} ({pop:.0f})"


def run_one(store: MapStore, ref: SettlementRef, sp: SettlementUpsampleParams) -> dict:
    world = store.globe
    assert world is not None
    t0 = time.perf_counter()
    chain = ensure_local_maps(store, ref, float(sp.margin_km))
    chain_s = time.perf_counter() - t0
    maps = [m.fields for m in chain.maps]
    t0 = time.perf_counter()
    fields = build_settlement_fields(world, maps, chain.center, chain.side_km, sp)
    seconds = time.perf_counter() - t0
    report = settlement_conservation_report(fields, maps, world.antimeridian)
    source = LocalMapsSource(maps, world.antimeridian)

    fine = refined_slope(fields.textures["elevation"][..., 0], math.sqrt(fields.texel_area_km2)).reshape(-1)
    lf = chain.owner.fields
    near = (np.linalg.norm(lf.positions.reshape(-1, 3) - chain.center, axis=1) * 6371.0 < 0.5 * chain.side_km) \
        & lf.domain.reshape(-1)
    coarse = refined_slope(lf.textures["elevation"][..., 0], math.sqrt(lf.texel_area_km2)).reshape(-1)[near]
    water = np.zeros(fields.tex_size ** 2, dtype=bool)
    for key in ("ocean_mask", "lake_mask"):
        if key in fields.textures:
            water |= fields.textures[key].reshape(-1) > 0.5
    relief = fields.textures["elevation"][0, ..., 0]
    return dict(
        name=describe(world, ref), keys=chain.keys, built=len(chain.built), chain_s=chain_s,
        side_km=chain.side_km, tex_size=fields.tex_size, seconds=seconds,
        octaves=fields.detail_octaves, report=report, fine_slope=np.percentile(fine[~water], [50, 90, 99])
        if (~water).any() else np.zeros(3),
        local_slope=np.percentile(coarse, [50, 90, 99]) if coarse.size else np.zeros(3),
        relief_m=float(np.ptp(relief[~water.reshape(relief.shape)]) * 1000.0) if (~water).any() else 0.0,
        water=float(water.mean()), seam=seam_report(fields, source),
        position=position(world, ref).copy(),
    )


# ── subjects ─────────────────────────────────────────────────────────────────


def pick_subjects(world: "World", n_cities: int, n_towns: int) -> list[SettlementRef]:
    cities = world.get_feature("settlements") or []
    river = [k for k, c in enumerate(cities) if c.site_kind & (SITE_KIND_RIVER | SITE_KIND_CONFLUENCE)]
    river.sort(key=lambda k: -cities[k].population)
    towns = world.get_feature("towns")
    town_order = list(np.argsort(-towns.population)[:n_towns]) if towns is not None else []
    return [SettlementRef("city", k) for k in river[:n_cities]] + [SettlementRef("town", int(k)) for k in town_order]


def pick_crossing(world: "World", n_cols: int, n: int, exclude: list[SettlementRef]) -> list[SettlementRef]:
    """Cities whose stage site sits nearest their regional hex's edge (likely to cross it)."""
    cities = world.get_feature("settlements") or []
    taken = {(s.kind, s.index) for s in exclude}
    scored = []
    for k, c in enumerate(cities):
        if ("city", k) in taken:
            continue
        region_id = int(ecef_to_cell(np.asarray(c.site_position)[None], world.antimeridian)[0])
        geometry = RegionGeometry(region_id, world.antimeridian)
        grid = RegionHexGrid(geometry, n_cols)
        s, t, _ = geometry.ecef_to_local(np.asarray(c.site_position)[None])
        d = np.sort(np.hypot(grid.centers[:, 0] - s[0], grid.centers[:, 1] - t[0]))
        # Near an edge the two nearest centres are almost equidistant.
        scored.append((float(d[1] - d[0]), k))
    scored.sort()
    return [SettlementRef("city", k) for _, k in scored[:n]]


# ── main ─────────────────────────────────────────────────────────────────────


def print_result(r: dict) -> None:
    maps = ", ".join(f"{a}/{b}" for a, b in r["keys"])
    print(f"\n{r['name']} — maps [{maps}] ({r['built']} built, {r['chain_s']:.1f} s) · frame {r['side_km']:.2f} km, "
          f"T={r['tex_size']}, {r['octaves']} octaves, {r['seconds']:.1f} s, water {r['water']:.0%}")
    for key, row in r["report"].items():
        print(f"    {key:<16} max {row['max']:.2e}  mean {row['mean']:.2e}  ({row['n_anchors']} local texels)")
    fs, ls = r["fine_slope"], r["local_slope"]
    print(f"    slope p50/p90/p99 km/km: settlement {fs[0]:.3f}/{fs[1]:.3f}/{fs[2]:.3f} · "
          f"local {ls[0]:.3f}/{ls[1]:.3f}/{ls[2]:.3f} · land relief {r['relief_m']:.0f} m")
    if r["seam"] is not None:
        across, inside, n, p99, top = r["seam"]
        print(f"    seam: mean step {across:.2f} m across ({n} pairs; p99 {p99:.2f}, max {top:.2f}) "
              f"vs {inside:.2f} m elsewhere")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seed", nargs="?", type=int, default=4)
    parser.add_argument("--cities", type=int, default=6)
    parser.add_argument("--towns", type=int, default=3)
    parser.add_argument("--crossing", type=int, default=3)
    parser.add_argument("--texel-m", type=float, default=None)
    parser.add_argument("--order-check", action="store_true",
                        help="rebuild the world and open each frame's neighbour maps first (A9)")
    args = parser.parse_args()

    t0 = time.perf_counter()
    store = make_store(args.seed)
    assert store.globe is not None
    print(f"world {args.seed}: {time.perf_counter() - t0:.1f} s")
    sp = with_local_detail(SettlementUpsampleParams(), store.local_params.upsample)
    if args.texel_m is not None:
        sp.texel_m = args.texel_m

    subjects = pick_subjects(store.globe, args.cities, args.towns)
    subjects += pick_crossing(store.globe, int(store.region_params.upsample.n_cols), args.crossing, subjects)
    results = []
    for ref in subjects:
        r = run_one(store, ref, sp)
        results.append((ref, r))
        print_result(r)

    worst = max((row["max"] for _, r in results for row in r["report"].values()), default=0.0)
    crossing = sum(len(r["keys"]) > 1 for _, r in results)
    print(f"\n{len(results)} frames · {crossing} cross a local map's edge · worst residual {worst:.2e}")
    print(f"time: chain {sum(r['chain_s'] for _, r in results):.1f} s ({len(store.builds)} maps built), "
          f"settlement {sum(r['seconds'] for _, r in results):.1f} s")

    if args.order_check:
        print("\norder check (A9): neighbours first, on a fresh world")
        fresh = make_store(args.seed)
        assert fresh.globe is not None
        for ref, r in results:
            if len(r["keys"]) < 2:
                continue
            for key in r["keys"][1:]:
                fresh.local(*key)
            ensure_local_maps(fresh, ref, float(sp.margin_km))
            moved_km = float(np.linalg.norm(position(fresh.globe, ref) - r["position"])) * 6371.0
            print(f"  {r['name']}: moved {moved_km * 1000:.0f} m")


if __name__ == "__main__":
    main()
