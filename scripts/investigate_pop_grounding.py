"""One-off investigation: ground the abstract population model in real people.

The carrying-capacity core is:

    cap[k] = Σ_i shares[k,i] · fertility[i] · face_area[i]      (face_area in steradians)

To turn this into people we need (a) physical face areas in km² and (b) a density
anchor (people per km² at fertility = 1). This script reports the physical scale of
the mesh, the land/fertility distribution, and the realm carrying-capacity + final
population distributions, for the river-vetted seeds 4/31/35.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.pipeline import BuildContext
from core.world import World
from realmgen.builder import RealmParams, RealmPipeline
from realmgen.population.calibration import (
    compute_carrying_capacity,
    compute_hinterland_shares,
)
from worldgen import GlobeParams, GlobePipeline, load_globe, save_globe
from worldgen.cost.channels import IS_WATER
from worldgen.resources.channels import FERTILITY_MEAN

EARTH_RADIUS_KM = 6371.0
EARTH_R2 = EARTH_RADIUS_KM ** 2  # km² per steradian
CACHE_DIR = Path(__file__).parent.parent / "saved_worlds"
SEEDS = [4, 31, 35]


def get_world(seed: int) -> World:
    path = CACHE_DIR / f"world_{seed}.npz"
    if not path.exists():
        params = GlobeParams()
        params.seed = seed
        gp = GlobePipeline(params)
        gp.build(BuildContext(world=World(params.mesh, seed=params.seed)))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        save_globe(gp, path)
    _, world = load_globe(path)
    return world


def main() -> None:
    print(f"\nMesh: icosphere_depth={GlobeParams().mesh.icosphere_depth}")
    for seed in SEEDS:
        print(f"\n{'='*70}\nSEED {seed}\n{'='*70}")
        world = get_world(seed)
        ctx = BuildContext(world=world, seed=seed)
        mesh = ctx.mesh
        n = len(mesh.faces)

        face_areas_sr = np.asarray(mesh.face_areas, dtype=np.float64)
        face_areas_km2 = face_areas_sr * EARTH_R2
        is_water = IS_WATER.read(mesh)
        land = ~is_water
        fert = FERTILITY_MEAN.read(mesh).clip(0.0, 1.0).astype(np.float64)

        total_area = face_areas_km2.sum()
        land_area = face_areas_km2[land].sum()
        # fertility on land faces only (fert includes aquatic on coastal water)
        land_fert = fert[land]
        # "effective fertile area" = Σ fertility·area over land (km²-equivalent at fert=1)
        fertile_area_land = (fert[land] * face_areas_km2[land]).sum()

        print(f"  faces={n}  mean face area={face_areas_km2.mean():,.0f} km²")
        print(f"  total surface={total_area/1e6:,.1f} M km²  (Earth ~510 M)")
        print(f"  land faces={int(land.sum())}  land area={land_area/1e6:,.1f} M km² "
              f"({100*land_area/total_area:.1f}% land)")
        print(f"  land fertility: mean={land_fert.mean():.3f} median={np.median(land_fert):.3f} "
              f"p90={np.percentile(land_fert, 90):.3f} max={land_fert.max():.3f}")
        print(f"  Σ fertility·area over land = {fertile_area_land/1e6:,.2f} M km²-equiv "
              f"(={100*fertile_area_land/land_area:.0f}% of land area)")

        # Fertility→density contrast: for people = D_max·fert^γ·area, what D_max hits
        # a 400M medieval world, and what peak/median densities result?
        print("  fertility-exponent sweep (target world = 400M people):")
        a_land = face_areas_km2[land]
        f_land = fert[land]
        for g in (1.0, 1.5, 2.0, 3.0):
            eff = (f_land ** g * a_land).sum()      # km²-equiv
            d_max = 400e6 / eff                     # people/km² at fert=1
            d_top = d_max * 0.9 ** g                # density of a near-max (0.9) face
            d_med = d_max * 0.3 ** g                # density of a median (~0.3) face
            print(f"    γ={g:<3}: D_max={d_max:6.1f}  → fert0.9={d_top:6.1f}/km²  "
                  f"fert0.3={d_med:5.2f}/km²")

        # Run realm pipeline
        realm = RealmPipeline(RealmParams())
        realm.build(ctx)
        ps = realm.population_stage
        settlements = ps.settlements or []
        pops = np.array([s.population for s in settlements], dtype=np.float64)

        # Recompute carrying capacity explicitly (population stage doesn't store it).
        # Rebuild hinterland shares with the final settlements to get raw cap units.
        from worldgen.cost.channels import COST_SURFACES
        from worldgen.cost.surfaces import build_land_edge_costs
        from realmgen.placement.seeder import build_cost_graph
        pp = RealmParams().population
        cs = COST_SURFACES.read(world)
        hinterland_land_cost = np.where(is_water, pp.hinterland_water_cost, cs.land_base_cost)
        hl_costs = build_land_edge_costs(
            mesh, hinterland_land_cost, np.ones(n, dtype=bool),
            cs.river_flow_dir, cs.river_flow_mag, cs.river_navigability,
        )
        hl_graph = build_cost_graph(mesh.face_adjacency, hl_costs)
        shares, _ = compute_hinterland_shares(
            n, settlements, hl_graph,
            max_distance=pp.hinterland_max_distance,
            influence_scale=pp.hinterland_influence_scale,
            reach_scale=pp.hinterland_reach_scale,
        )
        cap = compute_carrying_capacity(shares, fert)  # raw units (steradian)

        print(f"\n  settlements={len(settlements)}")
        print(f"  carrying_cap (raw, steradian units): "
              f"sum={cap.sum():.4f} mean={cap.mean():.5f} max={cap.max():.5f}")
        print(f"  Σ cap in km²-equiv (×R²) = {cap.sum()*EARTH_R2/1e6:.2f} M km²-equiv "
              f"({100*cap.sum()*EARTH_R2/fertile_area_land:.0f}% of land fertile-area claimed)")
        print(f"  population (with trade bonus): "
              f"sum={pops.sum():.4f} max={pops.max():.5f} "
              f"trade_inflation={pops.sum()/cap.sum():.2f}×")

        # What D_max [people/km² at fert=1] reproduces a given total population?
        # total_people = D_max · R² · Σ cap   (carrying-only, before trade bonus)
        cap_km2 = cap.sum() * EARTH_R2  # km²-equiv claimed
        for target_total, label in [(70e6, "medieval Europe ~70M"),
                                    (400e6, "world c.1300 ~400M")]:
            d_max = target_total / cap_km2
            print(f"    → to hit {label}: D_max={d_max:.1f} people/km² at fert=1")


if __name__ == "__main__":
    main()
