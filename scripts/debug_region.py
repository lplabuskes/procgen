"""Visual check on regional upsampling — coarse vs. refined, plus conservation.

Renders one region side by side: left with ``relief_factor = 0`` (upsampled
only), right with the real params, each hexagon drawn as ``RegionView`` paints it.

Prints the **conservation report**, the critical invariant: averaging each global
texel's regional samples must give back the global value (``docs/region_fields.md``).

Usage:
    python3 scripts/debug_region.py                       # seed 4, auto-pick the most land-rich region
    python3 scripts/debug_region.py 31 --region 20        # explicit region id
    python3 scripts/debug_region.py --layer Elevation     # any name in LAYERS below
    python3 scripts/debug_region.py --tex-size 96 --n-cols 20
    python3 scripts/debug_region.py --relief 4.0          # override the detail relief factor
    python3 scripts/debug_region.py --out /tmp/region.png
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from core.display import SchemeSpec, TextureRecipe  # noqa: E402
from region.display import build_layer_texture  # noqa: E402
from region.geometry import N_REGIONS, ecef_to_cell  # noqa: E402
from region.upsample import (  # noqa: E402
    RegionFields, RegionUpsampleParams, build_region_fields, conservation_report,
)
from worldgen.climate.display import (  # noqa: E402
    ANNUAL_TEMPERATURE_RECIPE, BIOME_RECIPE, PRECIPITATION_RECIPE,
    SUMMER_TEMPERATURE_RECIPE, WINTER_TEMPERATURE_RECIPE,
)
from worldgen.danger.display import DANGER_RECIPE  # noqa: E402
from worldgen.geology.display import ELEVATION_RECIPE  # noqa: E402
from worldgen.resources.display import (  # noqa: E402
    DOMINANT_CROP_RECIPE, FERTILITY_RECIPE, FOREST_DENSITY_RECIPE,
)
from _common import get_physical_world  # noqa: E402

# The world-map layers a region refines, by scheme name (the stages' own recipes).
LAYERS: dict[str, TextureRecipe] = {
    "Elevation": ELEVATION_RECIPE,
    "Biome": BIOME_RECIPE,
    "Annual Temperature": ANNUAL_TEMPERATURE_RECIPE,
    "Summer Temperature": SUMMER_TEMPERATURE_RECIPE,
    "Winter Temperature": WINTER_TEMPERATURE_RECIPE,
    "Annual Precipitation": PRECIPITATION_RECIPE,
    "Fertility": FERTILITY_RECIPE,
    "Forest Density": FOREST_DENSITY_RECIPE,
    "Dominant Crop": DOMINANT_CROP_RECIPE,
    "Danger": DANGER_RECIPE,
}

RESULTS_DIR = Path(__file__).parent / "results"


def pick_land_region(world) -> int:  # type: ignore[no-untyped-def]
    """Region holding the most land faces — the one worth looking at."""
    cells = ecef_to_cell(world.mesh.face_centroids, world.antimeridian)
    is_ocean = np.array(
        [bool(m.get("is_ocean", True)) for m in world.mesh.face_metadata], dtype=bool
    )
    land_counts = np.bincount(cells[~is_ocean], minlength=N_REGIONS)
    return int(np.argmax(land_counts))


def draw_region(  # type: ignore[no-untyped-def]
    ax, fields: RegionFields, layer: str, title: str, recipe: Optional[TextureRecipe] = None,
) -> None:
    """Blit every hexagon's texture, clipped to the hexagon, in local (s, t) space.

    ``recipe`` overrides the ``LAYERS`` entry for ``layer`` (a layer not listed there)."""
    spec = SchemeSpec(lambda _meta: (0.0, 0.0, 0.0, 1.0), recipe=recipe or LAYERS[layer])
    rgba = np.clip(build_layer_texture(fields, spec), 0.0, 1.0)
    r = fields.grid.radius
    corners = np.array([
        [math.cos(math.radians(60 * i)), math.sin(math.radians(60 * i))] for i in range(6)
    ]) * r
    for i, (s, t) in enumerate(fields.grid.centers):
        patch = Polygon(corners + np.array([s, t]), closed=True, transform=ax.transData)
        image = ax.imshow(
            rgba[i], extent=(s - r, s + r, t - r, t + r), origin="upper",
            interpolation="nearest", zorder=1,
        )
        image.set_clip_path(patch)
        ax.add_patch(Polygon(corners + np.array([s, t]), closed=True, fill=False,
                             edgecolor=(0, 0, 0, 0.18), lw=0.4, zorder=2))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def report_shoreline(fields: RegionFields, world) -> None:  # type: ignore[no-untyped-def]
    """How far the refined water masks moved off the global ones.

    A few percent is the expected sub-texel wiggle; more means the refined relief
    is fighting the global masks.
    """
    anchors = fields.anchors.reshape(-1)
    for key in ("ocean_mask", "lake_mask"):
        if key not in fields.textures or key not in world.mesh.face_textures:
            continue
        refined = fields.textures[key].reshape(-1) > 0.5
        coarse = np.asarray(world.mesh.face_textures[key]).reshape(-1)[anchors] > 0.5
        inside = fields.in_hex.reshape(-1)
        gained = (refined & ~coarse)[inside].mean()
        lost = (~refined & coarse)[inside].mean()
        print(f"  {key}: {refined[inside].mean():>6.2%} of the region "
              f"(global {coarse[inside].mean():.2%}) · "
              f"flooded {gained:.2%}, dried {lost:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seed", nargs="?", type=int, default=4)
    parser.add_argument("--region", type=int, default=None,
                        help=f"region id in [0, {N_REGIONS}); default: most land-rich")
    parser.add_argument("--layer", default="Biome", choices=list(LAYERS))
    parser.add_argument("--tex-size", type=int, default=None)
    parser.add_argument("--n-cols", type=int, default=None)
    parser.add_argument("--relief", type=float, default=None,
                        help="override RegionUpsampleParams.relief_factor")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    world = get_physical_world(args.seed, args.rebuild_cache)
    region_id = args.region if args.region is not None else pick_land_region(world)

    params = RegionUpsampleParams()
    if args.tex_size is not None:
        params.tex_size = args.tex_size
    if args.n_cols is not None:
        params.n_cols = args.n_cols
    if args.relief is not None:
        params.relief_factor = args.relief

    # "No refinement" baseline: no added relief, water masks copied from the anchor texel.
    coarse_params = RegionUpsampleParams(**vars(params))
    coarse_params.relief_factor = 0.0
    coarse_params.orographic_gain = 0.0
    coarse_params.refine_water_masks = False

    refined = build_region_fields(world, region_id, params)
    coarse = build_region_fields(world, region_id, coarse_params)
    missing = [k for k in LAYERS[args.layer].sources if k not in refined.textures]
    if missing:
        raise SystemExit(f"{args.layer}: the cached world has no {', '.join(missing)} — try --rebuild-cache")

    delta = (refined.textures["elevation"] - coarse.textures["elevation"])[..., 0]
    land = refined.textures.get("ocean_mask")
    land_mask = land[..., 0] <= 0 if land is not None else np.ones_like(delta, dtype=bool)

    print(f"seed {args.seed} · region {region_id} · layer {args.layer}")
    print(f"  {refined.n_hexes} hexes × {refined.tex_size}² texels "
          f"(~{refined.texel_area_km2:.1f} km²/texel, {refined.detail_octaves} detail octaves)")
    print(f"  added relief: std {delta.std() * 1000:.0f} m "
          f"(land {delta[land_mask].std() * 1000:.0f} m), max {np.abs(delta).max() * 1000:.0f} m")
    report_shoreline(refined, world)
    print("  conservation residual per global texel (downsample must recover the globe):")
    for name, stats in conservation_report(refined, world).items():
        print(f"    {name:<14} max {stats['max']:.3e}  mean {stats['mean']:.3e}  "
              f"over {stats['n_anchors']} anchors")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8))
    draw_region(axes[0], coarse, args.layer, "global field, upsampled only")
    draw_region(axes[1], refined, args.layer,
                f"+ slope-proportional detail ({refined.detail_octaves} octaves)")
    fig.suptitle(
        f"Region {region_id} · seed {args.seed} · {args.layer} · "
        f"{refined.n_hexes}×{refined.tex_size}² texels (~{refined.texel_area_km2:.0f} km² each)",
        fontsize=11,
    )
    fig.tight_layout()

    out = args.out or RESULTS_DIR / f"region_{args.seed}_{region_id}_{args.layer.lower()}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
