"""Phase 0 spike for settlement maps (docs/plans/settlement_maps.md): does a
street-scale frame refine cleanly from its local map(s)?

For the largest river cities and towns of a seed, plus settlements whose frame
crosses its local map's edge, it builds the region and local maps the way the
viewer does (the owning local map first, then any neighbour the frame
touches), refines the frame at ``texel_m`` with :mod:`settlegen`, and prints:

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
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from localmap import LocalMap, LocalParams, build_local_map
from localmap.cities import CityParams
from localmap.farmland import FarmlandParams
from localmap.upsample import with_region_detail
from realmgen.placement.types import SITE_KIND_CONFLUENCE, SITE_KIND_RIVER
from region.builder import RegionMap, build_region_map
from region.geometry import RegionGeometry, ecef_to_cell
from region.grid import RegionHexGrid
from region.params import RegionParams
from region.resources import refined_slope
from region.upsample import RegionFields
from settlegen import (
    LocalMapsSource, SettlementUpsampleParams, build_settlement_fields, settlement_conservation_report,
    with_local_detail,
)
from settlegen.upsample import km_per_local_unit
from _common import get_realm_world

if TYPE_CHECKING:
    from core.world import World

_KM2_PER_HA = 0.01


@dataclass
class Chain:
    """Regions and local maps built so far for one world, in build order."""

    world: "World"
    region_params: RegionParams
    local_params: LocalParams
    regions: dict[int, RegionMap] = field(default_factory=dict)
    locals: dict[tuple[int, int], LocalMap] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)

    def _time(self, key: str, t0: float) -> None:
        self.seconds[key] = self.seconds.get(key, 0.0) + time.perf_counter() - t0

    def region(self, region_id: int) -> RegionMap:
        if region_id not in self.regions:
            t0 = time.perf_counter()
            self.regions[region_id] = build_region_map(self.world, region_id, self.region_params)
            self._time("region", t0)
        return self.regions[region_id]

    def local(self, region_id: int, hex_index: int) -> LocalMap:
        key = (region_id, hex_index)
        if key in self.locals:
            return self.locals[key]
        r = self.region(region_id)
        t0 = time.perf_counter()
        self.locals[key] = build_local_map(
            self.world, r.fields, hex_index, self.local_params,
            region_rivers=r.rivers, region_roads=r.roads, region_deltas=r.deltas,
            region_streams=r.streams(self.world, self.local_params.streams),
        )
        self._time("local", t0)
        return self.locals[key]


def nearest_hex(geometry: RegionGeometry, grid: RegionHexGrid, points: np.ndarray) -> np.ndarray:
    s, t, _ = geometry.ecef_to_local(np.asarray(points, dtype=np.float64).reshape(-1, 3))
    return np.argmin((grid.centers[None, :, 0] - s[:, None]) ** 2
                     + (grid.centers[None, :, 1] - t[:, None]) ** 2, axis=1)


# ── one settlement ───────────────────────────────────────────────────────────


@dataclass
class Subject:
    kind: str            # "city" | "town"
    index: int


def record(world, subj: Subject) -> tuple[np.ndarray, np.ndarray, float]:  # type: ignore[no-untyped-def]
    """``(site_position, region_position, population)``."""
    if subj.kind == "city":
        c = world.get_feature("settlements")[subj.index]
        return c.site_position, c.region_position, float(c.population)
    towns = world.get_feature("towns")
    return towns.site_positions[subj.index], towns.region_positions[subj.index], float(towns.population[subj.index])


def current_position(world, subj: Subject) -> np.ndarray:  # type: ignore[no-untyped-def]
    if subj.kind == "city":
        return np.asarray(world.get_feature("settlements")[subj.index].position, dtype=np.float64)
    return np.asarray(world.get_feature("towns").positions[subj.index], dtype=np.float64)


def owning_map(chain: Chain, subj: Subject) -> tuple[int, int, LocalMap]:
    site, _, _ = record(chain.world, subj)
    region_id = int(ecef_to_cell(np.asarray(site)[None], chain.world.antimeridian)[0])
    r = chain.region(region_id)
    _, region_pos, _ = record(chain.world, subj)          # after the region refined it
    hex_index = int(nearest_hex(r.fields.geometry, r.fields.grid, region_pos)[0])
    return region_id, hex_index, chain.local(region_id, hex_index)


def drawn_position_and_radius(chain: Chain, subj: Subject, local: LocalMap) -> tuple[np.ndarray, float]:
    """The settlement's drawn (bank-shifted) position on its local map and footprint radius (km)."""
    if subj.kind == "city":
        k = int(np.flatnonzero(local.city_index == subj.index)[0])
        area = float(local.city_footprints.area_km2[k]) if local.city_footprints is not None else 0.0
        return local.city_positions[k], math.sqrt(max(area, 0.0) / math.pi)
    k = int(np.flatnonzero(local.town_index == subj.index)[0])
    _, _, pop = record(chain.world, subj)
    area = pop / (CityParams().density_per_ha / _KM2_PER_HA)
    return local.town_positions[k], math.sqrt(area / math.pi)


def frame_points(local: RegionFields, center: np.ndarray, side_km: float, n: int = 33) -> np.ndarray:
    """An ``n × n`` lattice over the frame (unit vectors), for finding the maps it touches."""
    s, t, _ = local.geometry.ecef_to_local(center[None])
    half = 0.5 * side_km / km_per_local_unit(local)
    g = np.linspace(-half, half, n)
    ss, tt = np.meshgrid(s[0] + g, t[0] + g)
    return local.geometry.local_to_ecef(ss.ravel(), tt.ravel())


def maps_for_frame(chain: Chain, owner_key: tuple[int, int], points: np.ndarray) -> list[tuple[int, int]]:
    """Owning map first, then every other (region, hex) the frame's points fall in."""
    keys = [owner_key]
    cells = ecef_to_cell(points, chain.world.antimeridian)
    for region_id in np.unique(cells):
        r = chain.region(int(region_id))
        pts = points[cells == region_id]
        for h in np.unique(nearest_hex(r.fields.geometry, r.fields.grid, pts)):
            key = (int(region_id), int(h))
            if key not in keys:
                keys.append(key)
    return keys


def frame_side_km(radius_km: float, params: SettlementUpsampleParams) -> float:
    return 2.0 * (radius_km + float(FarmlandParams().city_garden_ring_km) + float(params.margin_km))


def seam_report(fields: RegionFields, source: LocalMapsSource) -> Optional[tuple[float, float, int]]:
    """``(mean |Δe| across a map seam, mean |Δe| elsewhere, seam pairs)`` in metres, or None."""
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
    return float(across.mean()), float(inside.mean()), int(across.size)


def describe(world, subj: Subject) -> str:  # type: ignore[no-untyped-def]
    _, _, pop = record(world, subj)
    if subj.kind == "city":
        flags = world.get_feature("settlements")[subj.index].site_kind
        tag = "confluence" if flags & SITE_KIND_CONFLUENCE else "river" if flags & SITE_KIND_RIVER else "site"
        return f"city {subj.index} ({pop / 1000:.0f}k, {tag})"
    return f"town {subj.index} ({pop:.0f})"


def run_one(chain: Chain, subj: Subject, sp: SettlementUpsampleParams) -> dict:
    world = chain.world
    region_id, hex_index, local = owning_map(chain, subj)
    center, radius = drawn_position_and_radius(chain, subj, local)
    side_km = frame_side_km(radius, sp)
    keys = maps_for_frame(chain, (region_id, hex_index), frame_points(local.fields, center, side_km))
    maps = [chain.local(*k).fields for k in keys]
    t0 = time.perf_counter()
    fields = build_settlement_fields(world, maps, center, side_km, sp)
    seconds = time.perf_counter() - t0
    report = settlement_conservation_report(fields, maps, world.antimeridian)
    source = LocalMapsSource(maps, world.antimeridian)

    texel_km = math.sqrt(fields.texel_area_km2)
    fine = refined_slope(fields.textures["elevation"][..., 0], texel_km).reshape(-1)
    lf = local.fields
    near = (np.linalg.norm(lf.positions.reshape(-1, 3) - center, axis=1) * 6371.0 < 0.5 * side_km) \
        & lf.domain.reshape(-1)
    coarse = refined_slope(lf.textures["elevation"][..., 0], math.sqrt(lf.texel_area_km2)).reshape(-1)[near]
    water = np.zeros(fields.tex_size ** 2, dtype=bool)
    for key in ("ocean_mask", "lake_mask"):
        if key in fields.textures:
            water |= fields.textures[key].reshape(-1) > 0.5
    relief = fields.textures["elevation"][0, ..., 0]
    return dict(
        name=describe(world, subj), keys=keys, side_km=side_km, tex_size=fields.tex_size, seconds=seconds,
        octaves=fields.detail_octaves, report=report, fine_slope=np.percentile(fine[~water], [50, 90, 99])
        if (~water).any() else np.zeros(3),
        local_slope=np.percentile(coarse, [50, 90, 99]) if coarse.size else np.zeros(3),
        relief_m=float(np.ptp(relief[~water.reshape(relief.shape)]) * 1000.0) if (~water).any() else 0.0,
        water=float(water.mean()), seam=seam_report(fields, source),
        position=current_position(world, subj).copy(),
    )


# ── subjects ─────────────────────────────────────────────────────────────────


def pick_subjects(world, n_cities: int, n_towns: int) -> list[Subject]:  # type: ignore[no-untyped-def]
    cities = world.get_feature("settlements") or []
    river = [k for k, c in enumerate(cities) if c.site_kind & (SITE_KIND_RIVER | SITE_KIND_CONFLUENCE)]
    river.sort(key=lambda k: -cities[k].population)
    towns = world.get_feature("towns")
    town_order = list(np.argsort(-towns.population)[:n_towns]) if towns is not None else []
    return [Subject("city", k) for k in river[:n_cities]] + [Subject("town", int(k)) for k in town_order]


def pick_crossing(world, chain: Chain, n: int, exclude: list[Subject],  # type: ignore[no-untyped-def]
                  sp: SettlementUpsampleParams) -> list[Subject]:
    """Cities whose region position sits nearest their regional hex's edge (a frame-width estimate)."""
    cities = world.get_feature("settlements") or []
    taken = {(s.kind, s.index) for s in exclude}
    scored = []
    for k, c in enumerate(cities):
        if ("city", k) in taken:
            continue
        region_id = int(ecef_to_cell(np.asarray(c.site_position)[None], world.antimeridian)[0])
        geometry = RegionGeometry(region_id, world.antimeridian)
        grid = RegionHexGrid(geometry, int(chain.region_params.upsample.n_cols))
        s, t, _ = geometry.ecef_to_local(np.asarray(c.site_position)[None])
        d = np.sort(np.hypot(grid.centers[:, 0] - s[0], grid.centers[:, 1] - t[0]))
        # Near an edge the two nearest centres are almost equidistant.
        scored.append((float(d[1] - d[0]), k))
    scored.sort()
    return [Subject("city", k) for _, k in scored[:n]]


# ── main ─────────────────────────────────────────────────────────────────────


def make_chain(seed: int) -> Chain:
    region_params = RegionParams()
    local_params = LocalParams()
    local_params.upsample = with_region_detail(local_params.upsample, region_params.upsample)
    return Chain(get_realm_world(seed, False), region_params, local_params)


def print_result(r: dict) -> None:
    maps = ", ".join(f"{a}/{b}" for a, b in r["keys"])
    print(f"\n{r['name']} — maps [{maps}] · frame {r['side_km']:.2f} km, T={r['tex_size']}, "
          f"{r['octaves']} octaves, {r['seconds']:.1f} s, water {r['water']:.0%}")
    for key, row in r["report"].items():
        print(f"    {key:<16} max {row['max']:.2e}  mean {row['mean']:.2e}  ({row['n_anchors']} local texels)")
    fs, ls = r["fine_slope"], r["local_slope"]
    print(f"    slope p50/p90/p99 km/km: settlement {fs[0]:.3f}/{fs[1]:.3f}/{fs[2]:.3f} · "
          f"local {ls[0]:.3f}/{ls[1]:.3f}/{ls[2]:.3f} · land relief {r['relief_m']:.0f} m")
    if r["seam"] is not None:
        across, inside, n = r["seam"]
        print(f"    seam: mean step {across:.2f} m across ({n} pairs) vs {inside:.2f} m elsewhere")


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
    chain = make_chain(args.seed)
    print(f"world {args.seed}: {time.perf_counter() - t0:.1f} s")
    sp = with_local_detail(SettlementUpsampleParams(), chain.local_params.upsample)
    if args.texel_m is not None:
        sp.texel_m = args.texel_m

    subjects = pick_subjects(chain.world, args.cities, args.towns)
    subjects += pick_crossing(chain.world, chain, args.crossing, subjects, sp)
    results = []
    for subj in subjects:
        r = run_one(chain, subj, sp)
        results.append((subj, r))
        print_result(r)

    worst = max((row["max"] for _, r in results for row in r["report"].values()), default=0.0)
    crossing = sum(len(r["keys"]) > 1 for _, r in results)
    print(f"\n{len(results)} frames · {crossing} cross a local map's edge · worst residual {worst:.2e}")
    print("time: " + ", ".join(f"{k} {v:.1f} s" for k, v in chain.seconds.items())
          + f", settlement {sum(r['seconds'] for _, r in results):.1f} s")

    if args.order_check:
        print("\norder check (A9): neighbours first, on a fresh world")
        fresh = make_chain(args.seed)
        for subj, r in results:
            if len(r["keys"]) < 2:
                continue
            for key in r["keys"][1:]:
                fresh.local(*key)
            fresh.local(*r["keys"][0])
            moved_km = float(np.linalg.norm(current_position(fresh.world, subj) - r["position"])) * 6371.0
            print(f"  {r['name']}: moved {moved_km * 1000:.0f} m")


if __name__ == "__main__":
    main()
