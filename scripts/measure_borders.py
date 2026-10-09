"""Measure how physical polity borders are — exclaves, river and ridge adherence.

Caches the physical world per seed in ``saved_worlds/`` and re-runs only the
realm stage.  Reports, per seed:

- **Exclaves**: land-connected pieces of a polity's territory that don't hold its
  seat, at the sovereign-domain and the direct-governor level, split into
  *across-lake* (the piece and the polity's seat piece touch the same lake),
  *overseas* (on another landmass) and *land-cut* (same landmass, cut off by a
  neighbour).
- **River share**: share of sovereign border-line length within ``--river-km`` of
  a river with flow ≥ ``--river-flow`` (the drawn lines, ``polity_borders`` depth 0).
- **Ridge share**: the same for the ridge lines (``realmgen.refine.ridges``, the
  ``"ridges"`` feature) with relief ≥ ``--ridge-relief`` km.

Usage:
    python3 scripts/measure_borders.py                 # seeds 4, 31, 35
    python3 scripts/measure_borders.py 1 2 3 --detail
    python3 scripts/measure_borders.py --rebuild-cache
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree

from core.pipeline import BuildContext
from realmgen.builder import RealmPipeline
from realmgen.channels import POLITY_ID, SOVEREIGN_ID
from realmgen.refine.ridges import ridge_lines
from worldgen.cost.channels import IS_WATER
from worldgen.cost.surfaces import EARTH_RADIUS_KM
from worldgen.hydrology.channels import LAKE_MASK
from _common import (add_seed_args, add_set_arg, avg, get_physical_world, load_history, realm_params,
                     resolve_seeds, save_history)

DEFAULT_SEEDS = [4, 31, 35]
RESULTS = Path(__file__).parent / "results" / "border_results.json"


def _components(adj, mask: np.ndarray) -> np.ndarray:
    """Connected-component label per face over ``adj`` restricted to ``mask`` (-1 outside)."""
    n = len(adj)
    rows, cols = [], []
    for i, nb in enumerate(adj):
        if mask[i]:
            for j in nb:
                if mask[j]:
                    rows.append(i)
                    cols.append(j)
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    _, lab = connected_components(g, directed=False)
    return np.where(mask, lab, -1)


def exclaves(adj, owner: np.ndarray, seats: dict, land: np.ndarray, landmass: np.ndarray,
             lake_id: np.ndarray) -> dict:
    """Counts/faces of seatless land pieces of each owner, by kind."""
    out = {"n": 0, "faces": 0, "lake": 0, "overseas": 0, "landcut": 0, "sizes": []}
    for pid, seat in seats.items():
        mask = (owner == pid) & land
        if not mask.any():
            continue
        comp = _components(adj, mask)
        seat_comp = comp[seat] if mask[seat] else -2
        seat_lakes = _touching_lakes(adj, comp == seat_comp, lake_id)
        for c in np.unique(comp[comp >= 0]):
            if c == seat_comp:
                continue
            piece = comp == c
            size = int(piece.sum())
            out["n"] += 1
            out["faces"] += size
            out["sizes"].append(size)
            if _touching_lakes(adj, piece, lake_id) & seat_lakes:
                out["lake"] += 1
            elif landmass[np.flatnonzero(piece)[0]] != landmass[seat]:
                out["overseas"] += 1
            else:
                out["landcut"] += 1
    return out


def _touching_lakes(adj, piece: np.ndarray, lake_id: np.ndarray) -> set:
    found = set()
    for f in np.flatnonzero(piece):
        for j in adj[f]:
            if lake_id[j] >= 0:
                found.add(int(lake_id[j]))
    return found


def line_share(borders: dict, lines, *, on_km: float, min_strength: float,
               near_km: float = 60.0) -> tuple:
    """``(on, on | near)``: sovereign border-line length share within ``on_km`` of a line
    segment (``(positions, strength, ...)``: river flow, ridge relief) of strength ≥
    ``min_strength``, overall and among the length within ``near_km`` of one."""
    segs = np.asarray(borders.get(0, np.zeros((0, 6))), dtype=np.float64)
    if segs.size == 0:
        return float("nan"), float("nan")
    pts = []
    step = 2.0 / EARTH_RADIUS_KM   # densify lines to ~2 km
    for line in lines or []:
        pos, strength = np.asarray(line[0], dtype=np.float64), np.asarray(line[1], dtype=np.float64)
        for k in range(min(len(strength), len(pos) - 1)):
            if strength[k] < min_strength:
                continue
            a, b = pos[k], pos[k + 1]
            m = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
            t = np.linspace(0.0, 1.0, m + 1)[:, None]
            pts.append(a + t * (b - a))
    if not pts:
        return 0.0, float("nan")
    tree = KDTree(np.concatenate(pts))
    a, b = segs[:, :3], segs[:, 3:]
    length = np.linalg.norm(b - a, axis=1)
    d, _ = tree.query(0.5 * (a + b), k=1)
    on = d * EARTH_RADIUS_KM <= on_km
    near = d * EARTH_RADIUS_KM <= near_km
    return (float(length[on].sum() / max(length.sum(), 1e-12)),
            float(length[on].sum() / max(length[near].sum(), 1e-12)))


def measure_seed(seed: int, rebuild: bool, detail: bool, river_km: float, river_flow: float,
                 ridge_relief: float, overrides: list) -> dict:
    world = get_physical_world(seed, rebuild)
    realm = RealmPipeline(realm_params(overrides))
    realm.build(BuildContext(world=world, seed=seed))
    mesh = world.mesh
    adj = mesh.face_adjacency
    pols = world.get_feature("polities") or []
    pof = np.asarray(POLITY_ID.read(mesh), dtype=np.int64)
    sof = np.asarray(SOVEREIGN_ID.read(mesh), dtype=np.int64)
    water = np.asarray(IS_WATER.read(mesh), dtype=bool)
    land = ~water
    lake_face = water & (LAKE_MASK.read(mesh)[..., 0].mean(axis=(1, 2)) > 0.5)
    lake_id = _components(adj, lake_face)
    landmass = _components(adj, land)

    sov_seats = {p.polity_id: p.capital_face for p in pols if p.liege is None}
    all_seats = {p.polity_id: p.capital_face for p in pols}
    ex_sov = exclaves(adj, sof, sov_seats, land, landmass, lake_id)
    ex_dir = exclaves(adj, pof, all_seats, land, landmass, lake_id)
    borders = world.get_feature("polity_borders") or {}
    rs, rs_near = line_share(borders, world.get_feature("rivers"), on_km=river_km, min_strength=river_flow)
    ridges = world.get_feature("ridges") or ridge_lines(mesh, world.get_feature("streams"),
                                                        world.get_feature("river_texels"))
    rg, rg_near = line_share(borders, ridges, on_km=river_km, min_strength=ridge_relief)

    print(
        f"seed {seed:>4}: {len(sov_seats):>2} sov / {len(pols):>3} pol | "
        f"sov exclaves {ex_sov['n']:>3} ({ex_sov['faces']:>3} faces; lake {ex_sov['lake']}, "
        f"overseas {ex_sov['overseas']}, land-cut {ex_sov['landcut']}) | "
        f"direct exclaves {ex_dir['n']:>3} ({ex_dir['faces']:>3} faces; lake {ex_dir['lake']}, "
        f"overseas {ex_dir['overseas']}, land-cut {ex_dir['landcut']}) | "
        f"river share {rs * 100:4.1f}% (of near {rs_near * 100:4.1f}%) | "
        f"ridge share {rg * 100:4.1f}% (of near {rg_near * 100:4.1f}%)"
    )
    if detail:
        print(f"    sovereign exclave sizes: {sorted(ex_sov['sizes'], reverse=True)}")
        print(f"    direct exclave sizes:    {sorted(ex_dir['sizes'], reverse=True)}")
    return {
        "seed": seed, "sovereigns": len(sov_seats), "polities": len(pols),
        "sov_exclaves": ex_sov["n"], "sov_exclave_faces": ex_sov["faces"],
        "sov_lake": ex_sov["lake"], "sov_overseas": ex_sov["overseas"], "sov_landcut": ex_sov["landcut"],
        "dir_exclaves": ex_dir["n"], "dir_exclave_faces": ex_dir["faces"],
        "dir_lake": ex_dir["lake"], "dir_overseas": ex_dir["overseas"], "dir_landcut": ex_dir["landcut"],
        "river_share": rs, "river_share_near": rs_near, "ridge_share": rg, "ridge_share_near": rg_near,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_seed_args(parser)
    parser.add_argument("--detail", action="store_true", help="Print exclave sizes")
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild cached physical worlds")
    parser.add_argument("--river-km", type=float, default=8.0, help="Border-to-river distance counted as on it")
    parser.add_argument("--river-flow", type=float, default=2000.0, help="Min river flow that counts")
    parser.add_argument("--ridge-relief", type=float, default=0.5, help="Min ridge relief (km) that counts")
    parser.add_argument("--label", default="", help="Tag stored with the results")
    add_set_arg(parser)
    args = parser.parse_args()
    rows = [measure_seed(s, args.rebuild_cache, args.detail, args.river_km, args.river_flow, args.ridge_relief,
                         args.set)
            for s in resolve_seeds(args, DEFAULT_SEEDS)]
    print(
        f"mean: sov exclaves {avg(rows, 'sov_exclaves'):.1f} | direct exclaves {avg(rows, 'dir_exclaves'):.1f} "
        f"| river share {avg(rows, 'river_share') * 100:.1f}% (of near {avg(rows, 'river_share_near') * 100:.1f}%) "
        f"| ridge share {avg(rows, 'ridge_share') * 100:.1f}% (of near {avg(rows, 'ridge_share_near') * 100:.1f}%)"
    )
    history = load_history(RESULTS)
    history.append({"date": datetime.datetime.now().isoformat(timespec="seconds"),
                    "label": args.label, "overrides": args.set, "rows": rows})
    save_history(history, RESULTS)


if __name__ == "__main__":
    main()
