"""Measure the globe's Wetland biome: how much land, where, fed by what, in what sizes.

Re-scores the cached physical world (``saved_worlds/``) with the current
``WetlandParams``, so tuning needs no rebuild (``--rebuild-cache`` after other
stage changes).  Reports, per seed: Wetland share of land, share per climate biome,
the rain-fed share (the score's rain term alone clears the threshold), and the
largest connected wetland complexes.

Usage:
    python3 scripts/measure_wetlands.py              # vetted seeds 4, 31, 35
    python3 scripts/measure_wetlands.py 1 2 3        # specific seeds
    python3 scripts/measure_wetlands.py --random 5   # 5 random seeds

Earth reference (wetlands, inland + coastal):
    total ~12 M km², ~6–9% of land (finer than a ~14 km texel; whole-texel target ~4–6%)
    West Siberian Lowland peat ~1 M km² · Hudson Bay Lowlands ~0.3 M km²
    Pantanal ~0.15–0.2 M km² · Sudd ~0.03–0.13 M km² (seasonal) · Everglades ~0.01 M km²
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from _common import add_seed_args, avg, get_physical_world, load_history, resolve_seeds, save_history
from worldgen.climate.classification import BIOME_NAMES, WETLAND_BIOME
from worldgen.geology.relief_band import trend_elevation
from worldgen.hydrology.rivers import inter_face_adjacency, texel_neighbor_csr
from worldgen.wetland.builder import WetlandParams, _PetParams, wetland_score

RESULTS_DIR = Path(__file__).parent / "results"
JSON_PATH = RESULTS_DIR / "wetland_results.json"
DEFAULT_SEEDS = [4, 31, 35]


def measure(seed: int, rebuild: bool, params: WetlandParams) -> dict:
    world = get_physical_world(seed, rebuild)
    mesh, tex = world.mesh, world.mesh.face_textures
    n, h, w = tex["elevation"].shape[:3]
    ocean, lake = tex["ocean_mask"][..., 0] > 0.5, tex["lake_mask"][..., 0] > 0.5
    surface = np.where(lake, np.asarray(tex["filled_height"], dtype=np.float64)[..., 0], 0.0)
    streams = world.get_feature("streams")
    terms = wetland_score(
        mesh, params, _PetParams(),
        elevation=np.asarray(tex["elevation"], dtype=np.float64)[..., 0], water=ocean | lake, surface_km=surface,
        temperature=tex["temperature"], precipitation=tex["precipitation"],
        channel_nodes=streams.nodes[streams.channel] if streams is not None else None,
        channel_flow=streams.flow[streams.channel] if streams is not None else None,
        slope_elevation=trend_elevation(mesh)[..., 0],
    )
    inside = mesh.face_texture_in_polygon(h, w)
    land = ~(ocean | lake) & inside
    frozen = np.asarray(tex["temperature"], dtype=np.float64).max(axis=-1) < 0.0
    wet = (terms["score"] >= params.threshold) & land & ~frozen
    texel_km2 = float(np.mean(mesh.face_areas)) * 6371.0 ** 2 / float(np.mean(inside.sum(axis=(1, 2))))

    # The climate biome under the wetland: the texture's, wetland stamped or not (wetland-stage
    # worlds keep the face's climate biome, so read the face biome where the texel is stamped).
    biome = np.asarray(tex["biome"])[..., 0].copy()
    face_biome = np.array([m.get("biome", 3) for m in mesh.face_metadata])
    stamped = biome == WETLAND_BIOME
    biome[stamped] = np.broadcast_to(face_biome[:, None, None], biome.shape)[stamped]
    by_biome = {}
    for b, name in enumerate(BIOME_NAMES):
        sel = land & (biome == b)
        if sel.sum() >= 500:
            by_biome[name] = float(wet[sel].mean())

    indptr, flat = texel_neighbor_csr(n, h, inter_face_adjacency(mesh, h))
    ids = np.flatnonzero(wet.reshape(-1))
    slot = np.full(n * h * w, -1)
    slot[ids] = np.arange(len(ids))
    src = np.repeat(np.arange(n * h * w), np.diff(indptr))
    keep = (slot[src] >= 0) & (slot[flat] >= 0)
    graph = csr_matrix((np.ones(int(keep.sum())), (slot[src[keep]], slot[flat[keep]])), shape=(len(ids),) * 2)
    _, label = connected_components(graph, directed=False)
    sizes = np.sort(np.bincount(label))[::-1] * texel_km2 if len(ids) else np.zeros(0)

    rain_fed = (terms["flat"] * terms["rain"] >= params.threshold) & wet
    return {
        "seed": seed,
        "wetland_land_frac": float(wet.sum() / max(land.sum(), 1)),
        "wetland_km2": float(wet.sum() * texel_km2),
        "rain_fed_frac": float(rain_fed.sum() / max(wet.sum(), 1)),
        "largest_km2": [float(s) for s in sizes[:5]],
        "complexes_over_10k_km2": int((sizes >= 1e4).sum()),
        "by_biome": by_biome,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_seed_args(parser)
    parser.add_argument("--rebuild-cache", action="store_true", help="rebuild the cached physical worlds")
    args = parser.parse_args()
    params = WetlandParams()
    rows = []
    for seed in resolve_seeds(args, DEFAULT_SEEDS):
        r = measure(seed, args.rebuild_cache, params)
        rows.append(r)
        big = ", ".join(f"{s / 1e3:.0f}k" for s in r["largest_km2"])
        print(f"seed {seed}: wetland {r['wetland_land_frac']:.1%} of land ({r['wetland_km2'] / 1e6:.1f} M km²), "
              f"rain-fed {r['rain_fed_frac']:.0%}, {r['complexes_over_10k_km2']} complexes ≥10k km², "
              f"largest {big} km²")
        print("   by climate biome: " + ", ".join(f"{k} {v:.0%}" for k, v in r["by_biome"].items()))
    print(f"mean: wetland {avg(rows, 'wetland_land_frac'):.1%} of land (Earth ~6–9% finer-grained; "
          f"whole-texel target ~4–6%), rain-fed {avg(rows, 'rain_fed_frac'):.0%}")

    history = load_history(JSON_PATH)
    history.append({"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                    "params": vars(params), "results": rows})
    save_history(history, JSON_PATH)


if __name__ == "__main__":
    main()
