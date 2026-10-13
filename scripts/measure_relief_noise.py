"""How noisy is the region tier's relief, and how much of that is conservation's doing?

Conservation forces each global texel's regional texels to average back to the
globe.  Whatever of the region's detail a global texel can see (its per-texel
mean) is taken out again, which pushes what is left toward texel scale.  The
globe's relief band (``docs/plans/global_relief_band.md``) puts that part on the
globe instead, so the correction should shrink.

Per seed, for the region with the most land, the roughest (mean land ruggedness)
and the smoothest, this refines the region's terrain (``refine_terrain``: relief,
conservation, shore reshaping, water masks; no rivers or carving) and reports
over texels well inland by the globe (elevation > 50 m):

  • detail rms — refined minus the bilinear parent (net of the band), before conservation;
  • removed — share of the detail's variance in its per-global-texel means (what
    conservation must take out);
  • correction rms and its correlation with the detail;
  • lag-1 — neighbour-texel correlation, before → after conservation;
  • texel-scale share — variance left after a 3×3 box mean, before → after;
  • band rms — the globe's relief band under the region (0 without one);

and over the dry texels: highest, lowest, share below sea level (with the
globe's dry texels for reference).

Usage:
    python3 scripts/measure_relief_noise.py                 # seeds 4, 31, 35
    python3 scripts/measure_relief_noise.py 1 2 --rebuild-cache

Results appended to ``scripts/results/relief_noise_results.json``.
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from region.geometry import N_REGIONS, RegionGeometry, ecef_to_cell
from region.grid import RegionHexGrid
from region.source import GlobalSource
from region.upsample import _SEED_SALT_REGION, RegionUpsampleParams, refine_terrain
from worldgen.geology.channels import IS_OCEAN, RUGGEDNESS_FACE
from _common import add_seed_args, get_physical_world, load_history, resolve_seeds, save_history

DEFAULT_SEEDS = [4, 31, 35]
RESULTS_PATH = Path(__file__).parent / "results" / "relief_noise_results.json"
INLAND_KM = 0.05


def pick_regions(world) -> dict[str, int]:  # type: ignore[no-untyped-def]
    """The regions with the most land, the roughest and the smoothest land (≥ 20 land faces)."""
    cells = ecef_to_cell(world.mesh.face_centroids, world.antimeridian)
    land = ~np.asarray(IS_OCEAN.read(world.mesh), dtype=bool)
    rug = np.asarray(RUGGEDNESS_FACE.read(world.mesh), dtype=np.float64)
    counts = np.bincount(cells[land], minlength=N_REGIONS)
    mean = np.bincount(cells[land], weights=rug[land], minlength=N_REGIONS) / np.maximum(counts, 1)
    ok = counts >= 20
    return {
        "most_land": int(np.argmax(counts)),
        "roughest": int(np.argmax(np.where(ok, mean, -1.0))),
        "smoothest": int(np.argmin(np.where(ok, mean, 9.0))),
    }


def lag1(field: np.ndarray, mask: np.ndarray) -> float:
    """Mean neighbour correlation along both texture axes, over pairs inside ``mask``."""
    out = []
    for ax in (1, 2):
        n = field.shape[ax]
        a, b = np.take(field, range(n - 1), axis=ax), np.take(field, range(1, n), axis=ax)
        m = np.take(mask, range(n - 1), axis=ax) & np.take(mask, range(1, n), axis=ax)
        out.append(np.corrcoef(a[m], b[m])[0, 1])
    return float(np.mean(out))


def texel_scale_share(field: np.ndarray, mask: np.ndarray) -> float:
    """Share of ``field``'s variance (over ``mask``) left after removing a 3×3 box mean."""
    w = uniform_filter(mask.astype(np.float64), size=3, axes=(1, 2))
    smooth = uniform_filter(np.where(mask, field, 0.0), size=3, axes=(1, 2)) / np.maximum(w, 1e-9)
    return float((field - smooth)[mask].var() / max(field[mask].var(), 1e-18))


def measure_region(world, region_id: int, params: RegionUpsampleParams) -> dict:  # type: ignore[no-untyped-def]
    source = GlobalSource(world)
    tex = int(params.tex_size)
    grid = RegionHexGrid(RegionGeometry(region_id, world.antimeridian), int(params.n_cols))
    positions = grid.texture_positions(tex)
    in_hex = grid.texture_in_hexagon(tex)
    terrain = refine_terrain(
        source, positions, in_hex, grid.texel_spacing_rad(tex), params,
        int(world.seed or 0) ^ _SEED_SALT_REGION, neighbours=grid.texel_neighbours(tex),
    )
    shape = in_hex.shape
    loc = terrain.located
    coarse = terrain.elev_coarse[:, 0].reshape(shape)
    raw = terrain.raw[:, 0].reshape(shape)
    final = terrain.elevation[:, 0].reshape(shape)
    target = source.anchor_values("elevation", loc)[:, 0].reshape(shape)
    band = (source.coarse("relief_band", loc)[:, 0].reshape(shape) if source.has("relief_band")
            else np.zeros(shape))
    rug = source.coarse("ruggedness", loc)[:, 0].reshape(shape) if source.has("ruggedness") else np.zeros(shape)
    ocean = terrain.textures["ocean_mask"][..., 0] > 0
    lake = terrain.textures["lake_mask"][..., 0] > 0 if "lake_mask" in terrain.textures else np.zeros(shape, bool)

    inland = in_hex & (target > INLAND_KM)
    detail, after, corr = raw - coarse, final - coarse, final - raw
    anchors = loc.anchors.reshape(shape)[inland]
    d = detail[inland]
    counts = np.bincount(anchors)
    well = counts[anchors] >= 4
    amean = (np.bincount(anchors, weights=d) / np.maximum(counts, 1))[anchors]
    dry = in_hex & ~ocean & ~lake
    return {
        "region": region_id,
        "rho": float(rug[inland].mean()),
        "detail_rms_m": float(d.std() * 1000),
        "removed_share": float(amean[well].var() / max(d[well].var(), 1e-18)),
        "correction_rms_m": float(corr[inland].std() * 1000),
        "correction_vs_detail": float(np.corrcoef(corr[inland], d)[0, 1]),
        "lag1_before": lag1(detail, inland), "lag1_after": lag1(after, inland),
        "texel_share_before": texel_scale_share(detail, inland),
        "texel_share_after": texel_scale_share(after, inland),
        "band_rms_m": float(band[inland].std() * 1000),
        "dry_max_m": float(final[dry].max() * 1000), "dry_min_m": float(final[dry].min() * 1000),
        "dry_below_sea": float((final[dry] < 0).mean()),
    }


def globe_dry(world) -> dict:  # type: ignore[no-untyped-def]
    tex = world.mesh.face_textures
    elev = np.asarray(tex["elevation"])[..., 0]
    dry = ~(np.asarray(tex["ocean_mask"])[..., 0] > 0)
    if "lake_mask" in tex:
        dry &= ~(np.asarray(tex["lake_mask"])[..., 0] > 0)
    return {"dry_max_m": float(elev[dry].max() * 1000), "dry_min_m": float(elev[dry].min() * 1000),
            "dry_below_sea": float((elev[dry] < 0).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_seed_args(parser)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    params = RegionUpsampleParams()
    rows = []
    for seed in resolve_seeds(args, DEFAULT_SEEDS):
        world = get_physical_world(seed, args.rebuild_cache)
        g = globe_dry(world)
        print(f"seed {seed}  globe dry: max {g['dry_max_m']:.0f} m  min {g['dry_min_m']:.0f} m  "
              f"below sea {g['dry_below_sea'] * 100:.2f}%")
        row = {"seed": seed, "globe": g, "regions": {}}
        for name, rid in pick_regions(world).items():
            r = measure_region(world, rid, params)
            row["regions"][name] = r
            print(f"  {name:<9} r{rid:<3} ρ {r['rho']:.2f} | detail {r['detail_rms_m']:4.0f} m, removed "
                  f"{r['removed_share']:.2f}, correction {r['correction_rms_m']:4.0f} m "
                  f"(r {r['correction_vs_detail']:+.2f}), band {r['band_rms_m']:4.0f} m | lag1 "
                  f"{r['lag1_before']:.2f}→{r['lag1_after']:.2f}, texel-scale {r['texel_share_before']:.2f}→"
                  f"{r['texel_share_after']:.2f} | dry {r['dry_min_m']:.0f}…{r['dry_max_m']:.0f} m, "
                  f"below sea {r['dry_below_sea'] * 100:.2f}%")
        rows.append(row)
    history = load_history(RESULTS_PATH)
    history.append({"timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "seeds": rows})
    save_history(history, RESULTS_PATH)
    print(f"appended to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
