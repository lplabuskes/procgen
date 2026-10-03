"""Visual check on a local map — coarse vs. refined, features, villages, conservation.

A local map is one regional hex refined the way a region refines the globe
(``docs/local.md``): the region's fields upsampled onto the local hex textures,
detail added above the region's Nyquist, and every regional texel's average
forced back to its exact value; then the region's rivers and roads re-routed
over it and villages sited on it.  This renders one, side by side:

  • left — the region's fields upsampled only (no added relief, the water
    masks copied from the regional texel), so the detail is visible by contrast;
  • right — the real local map, with its rivers, roads, towns, cities and
    villages drawn over it (villages coloured by kind, as in the viewer).

and prints the local statistics, the **conservation report** against the region
(per regional texel; must stay at float32 noise), the shoreline movement, and
the **village budget report** (rural + villages per global texel ∩ map; must be
exact) with the village counts, sizes and spacing.

Usage:
    python3 scripts/debug_local.py                     # seed 4, the densest land's hex
    python3 scripts/debug_local.py 31 --region 20 --hex 103
    python3 scripts/debug_local.py --layer "Rural Density"
    python3 scripts/debug_local.py --relief 4.0        # override the local relief factor
    python3 scripts/debug_local.py --out /tmp/local.png
"""
from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from scipy.spatial import KDTree  # noqa: E402

from core.display import TextureRecipe  # noqa: E402
from localmap import (  # noqa: E402
    LocalMap, LocalParams, build_local_fields, build_local_map, local_conservation_report,
    village_budget_report,
)
from localmap.geometry import PARENT_CENTER, PARENT_RADIUS  # noqa: E402
from localmap.villages import VILLAGE_KIND_NAMES  # noqa: E402
from realmgen.towns.builder import rural_density_recipe  # noqa: E402
from region.geometry import RegionGeometry, ecef_to_cell  # noqa: E402
from region.grid import RegionHexGrid  # noqa: E402
from region.rivers import build_region_rivers, river_length_km  # noqa: E402
from region.streams import build_region_streams  # noqa: E402
from region.roads import build_region_roads  # noqa: E402
from region.upsample import EARTH_RADIUS_KM, RegionFields, RegionUpsampleParams, build_region_fields  # noqa: E402
from debug_region import LAYERS, draw_region  # noqa: E402  (shared with the region check)
from _common import get_realm_world  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
# Village dot colours, as the viewer draws them (viz/mapview/overlays.py).
VILLAGE_COLORS = ("#d6be78", "#6eaaeb", "#5acdbe", "#c8966e")


def local_layers(world) -> dict[str, TextureRecipe]:  # type: ignore[no-untyped-def]
    """debug_region's layers plus Rural Density, scaled like the world map's."""
    rho = np.asarray(world.mesh.face_textures["rural_density"])[..., 0]
    water = np.asarray(world.mesh.face_textures["ocean_mask"])[..., 0] > 0.5
    d_max = float(np.percentile(rho[~water], 99)) if (~water).any() else 1.0
    return {**LAYERS, "Rural Density": rural_density_recipe(d_max)}


def hex_of_face(world, face: int, n_cols: int) -> tuple[int, int]:  # type: ignore[no-untyped-def]
    """``(region_id, hex_index)`` of the regional hex a global face's centroid lies in."""
    c = np.asarray(world.mesh.face_centroids[face], dtype=np.float64)
    c = c / np.linalg.norm(c)
    region_id = int(ecef_to_cell(c[None], world.antimeridian)[0])
    geometry = RegionGeometry(region_id, world.antimeridian)
    grid = RegionHexGrid(geometry, n_cols)
    s, t, _ = geometry.ecef_to_local(c[None])
    return region_id, int(np.argmin((grid.centers[:, 0] - s[0]) ** 2 + (grid.centers[:, 1] - t[0]) ** 2))


def land_face_density(world) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[no-untyped-def]
    """``(faces, mean rural density)`` of the faces with any rural people."""
    rho = np.asarray(world.mesh.face_textures["rural_density"])[..., 0].mean(axis=(1, 2))
    faces = np.flatnonzero(rho > 0)
    return faces, rho[faces]


def build_local(
    world, region_id: int, hex_index: int, params: Optional[LocalParams] = None,  # type: ignore[no-untyped-def]
    region_params: Optional[RegionUpsampleParams] = None,
) -> tuple[RegionFields, LocalMap]:
    """The region (fields, rivers, roads) and one hex's local map, as the viewer builds them."""
    region = build_region_fields(world, region_id, region_params)
    rivers = build_region_rivers(region, world)
    roads = build_region_roads(region, world, rivers)
    streams = build_region_streams(region, world, rivers, (params or LocalParams()).streams)
    local = build_local_map(world, region, hex_index, params, region_rivers=rivers, region_roads=roads,
                            region_streams=streams)
    return region, local


def mask_outside_parent(ax) -> None:  # type: ignore[no-untyped-def]
    """Paint over the local square outside the parent hexagon, and outline it."""
    hexagon = [
        (PARENT_CENTER + PARENT_RADIUS * math.cos(math.radians(60 * i)),
         PARENT_CENTER + PARENT_RADIUS * math.sin(math.radians(60 * i)))
        for i in range(6)
    ]
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    verts = square + [square[0]] + hexagon[::-1] + [hexagon[-1]]
    codes = ([MplPath.MOVETO] + [MplPath.LINETO] * 3 + [MplPath.CLOSEPOLY]
             + [MplPath.MOVETO] + [MplPath.LINETO] * 5 + [MplPath.CLOSEPOLY])
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="white", edgecolor="none", zorder=5))
    ax.add_patch(PathPatch(MplPath(hexagon + [hexagon[0]], closed=True), fill=False,
                           edgecolor=(0.2, 0.25, 0.3), lw=0.8, zorder=6))


def draw_features(ax, local: LocalMap, world) -> None:  # type: ignore[no-untyped-def]
    """Rivers, roads, towns, cities and villages in the local ``(s, t)`` frame."""
    geo = local.fields.geometry

    def st(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s, t, _ = geo.ecef_to_local(np.asarray(points, dtype=np.float64).reshape(-1, 3))
        return s, t

    channels = [*local.rivers.paths, *local.rivers.streams]
    top = max((float(np.max(f)) for _, f, _ in channels), default=1.0)
    for positions, flows, in_water in channels:
        s, t = st(positions)
        for k in np.flatnonzero(~np.asarray(in_water, dtype=bool)):
            ax.plot(s[k:k + 2], t[k:k + 2], color="#4a9be8",
                    lw=0.4 + 2.2 * math.sqrt(float(flows[k]) / top), zorder=3, solid_capstyle="round")
    for path in local.roads.polylines:
        s, t = st(path)
        ax.plot(s, t, color="#d9a520", lw=1.3, zorder=3)
    v = local.villages
    if len(v):
        s, t = st(v.positions)
        size = 2.0 + 6.0 * (v.population - v.population.min()) / max(float(np.ptp(v.population)), 1.0)
        ax.scatter(s, t, s=size, c=[VILLAGE_COLORS[k] for k in v.kind], edgecolors="#302a22",
                   linewidths=0.25, zorder=4)
    if len(local.town_positions):
        s, t = st(local.town_positions)
        ax.scatter(s, t, s=22, c="white", edgecolors="black", linewidths=0.6, zorder=4)
    if len(local.city_positions):
        s, t = st(local.city_positions)
        ax.scatter(s, t, s=70, c="#5b9be0", edgecolors="black", linewidths=1.0, zorder=4)


def report_shoreline(local: RegionFields, region: RegionFields) -> None:
    """How far the local water masks moved off the region's."""
    assert local.parent_anchors is not None
    anchors = local.parent_anchors.reshape(-1)
    inside = local.domain.reshape(-1)
    for key in ("ocean_mask", "lake_mask"):
        if key not in local.textures or key not in region.textures:
            continue
        refined = local.textures[key].reshape(-1) > 0.5
        coarse = region.textures[key].reshape(-1)[anchors] > 0.5
        print(f"  {key}: {refined[inside].mean():>6.2%} of the map (region {coarse[inside].mean():.2%}) · "
              f"flooded {(refined & ~coarse)[inside].mean():.2%}, dried {(~refined & coarse)[inside].mean():.2%}")


def report_villages(local: LocalMap) -> None:
    v = local.villages
    budget = village_budget_report(local.fields, v)
    area = float(local.fields.domain.sum()) * local.fields.texel_area_km2
    print(f"  villages: {len(v)} ({len(v) / area * 100:.1f} per 100 km²) holding "
          f"{budget['in_villages']:,.0f} of {budget['people']:,.0f} rural people "
          f"({budget['village_share']:.1%}; rural density {budget['people'] / area:.1f}/km²)")
    if len(v):
        print(f"    size: mean {v.population.mean():.0f}, median {np.median(v.population):.0f}, "
              f"range {v.population.min():.0f}–{v.population.max():.0f}")
    if len(v) > 1:
        nn = KDTree(v.positions).query(v.positions, k=2)[0][:, 1] * EARTH_RADIUS_KM
        print(f"    nearest neighbour: median {np.median(nn):.1f} km (p10 {np.percentile(nn, 10):.1f}, "
              f"p90 {np.percentile(nn, 90):.1f})")
    kinds = np.bincount(v.kind, minlength=len(VILLAGE_KIND_NAMES))
    print("    kinds: " + ", ".join(f"{n} {k}" for n, k in zip(VILLAGE_KIND_NAMES, kinds)))
    empty = v.budget_placed == 0
    print(f"    budgets: {int(budget['budgets'])} (global texel ∩ map), {int(empty.sum())} with no village "
          f"holding {v.budget_people[empty].sum() / max(budget['people'], 1e-9):.1%} of the people · "
          f"max residual {budget['max_relative_residual']:.1e} (relative)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seed", nargs="?", type=int, default=4)
    parser.add_argument("--region", type=int, default=None, help="region id (default: the densest land's)")
    parser.add_argument("--hex", type=int, default=None, help="regional hex index (default: the densest land's)")
    parser.add_argument("--layer", default="Biome", help="a debug_region layer or 'Rural Density'")
    parser.add_argument("--relief", type=float, default=None, help="override LocalUpsampleParams.relief_factor")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    world = get_realm_world(args.seed, args.rebuild_cache)
    layers = local_layers(world)
    if args.layer not in layers:
        raise SystemExit(f"unknown layer {args.layer!r}; one of: {', '.join(layers)}")

    region_params = RegionUpsampleParams()
    if args.region is None or args.hex is None:
        faces, rho = land_face_density(world)
        region_id, hex_index = hex_of_face(world, int(faces[np.argmax(rho)]), int(region_params.n_cols))
        region_id = args.region if args.region is not None else region_id
        hex_index = args.hex if args.hex is not None else hex_index
    else:
        region_id, hex_index = args.region, args.hex

    params = LocalParams()
    if args.relief is not None:
        params.upsample.relief_factor = args.relief
    region, local = build_local(world, region_id, hex_index, params, region_params)

    # The honest "no refinement" baseline, as in debug_region.
    coarse_params = dataclasses.replace(
        params.upsample, relief_factor=0.0, orographic_gain=0.0, refine_water_masks=False,
    )
    coarse = build_local_fields(world, region, hex_index, coarse_params)
    fields = local.fields
    display = dataclasses.replace(
        fields, textures={**fields.textures, "rural_density": local.villages.rural_density},
    )

    delta = (fields.textures["elevation"] - coarse.textures["elevation"])[..., 0]
    land = fields.domain & (fields.textures["ocean_mask"][..., 0] <= 0.5)
    print(f"seed {args.seed} · region {region_id} · hex {hex_index} · layer {args.layer}")
    print(f"  {fields.n_hexes} hexes × {fields.tex_size}² texels (~{fields.texel_area_km2:.3f} km²/texel, "
          f"{fields.detail_octaves} detail octaves)")
    print(f"  added relief: land std {delta[land].std() * 1000:.0f} m, max {np.abs(delta[land]).max() * 1000:.0f} m"
          if land.any() else "  no land")
    report_shoreline(fields, region)
    print("  conservation residual per regional texel (downsample must recover the region):")
    for name, stats in local_conservation_report(fields, region).items():
        print(f"    {name:<14} max {stats['max']:.3e}  mean {stats['mean']:.3e}  over {stats['n_anchors']} anchors")
    stream_km = sum(river_length_km(p) for p in local.rivers.streams)
    print(f"  rivers {len(local.rivers.paths)} ({len(local.rivers.confluences)} confluences, "
          f"{len(local.rivers.mouths)} mouths) · streams {len(local.rivers.streams)} ({stream_km:.0f} km)"
          f" · roads {len(local.roads.polylines)} · "
          f"towns {len(local.town_index)} · cities {len(local.city_index)}")
    report_villages(local)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8))
    recipe = layers[args.layer]
    draw_region(axes[0], coarse, args.layer, "region fields, upsampled only", recipe)
    draw_region(axes[1], display, args.layer, f"+ detail ({fields.detail_octaves} octaves), features, villages",
                recipe)
    draw_features(axes[1], local, world)
    for ax in axes:
        mask_outside_parent(ax)
    fig.suptitle(
        f"Local map · seed {args.seed} · region {region_id} hex {hex_index} · {args.layer} · "
        f"{len(local.villages)} villages",
        fontsize=11,
    )
    fig.tight_layout()
    slug = args.layer.lower().replace(" ", "_")
    out = args.out or RESULTS_DIR / f"local_{args.seed}_{region_id}_{hex_index}_{slug}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
