"""Pack/unpack helpers for realm-stage serialization.

The generic persistence framework (``core/persistence.py``) already round-trips
per-face metadata (suitability, population, polity_id, …) and textures
(``polity_fill``). What it can't reach are the realm's *feature* objects — the
``Settlement`` / ``Polity`` lists and the route / border geometry. These helpers
flatten those into npz-friendly arrays and rebuild them on load, mirroring the
flat-plus-offsets idiom the hydrology stage uses for rivers.

Each helper namespaces its keys (``settle_*`` / ``pol_*`` / ``route_*`` /
``border_*``) so the per-stage dicts merge without collision.
"""
from __future__ import annotations

from typing import Dict, List, Mapping

import numpy as np

from realmgen.placement.types import Settlement
from realmgen.polity.types import Polity


# ── Ragged int-list helper ───────────────────────────────────────────────────


def _pack_ragged(name: str, lists: List[List[int]]) -> Dict[str, np.ndarray]:
    flat: List[int] = []
    offsets: List[int] = [0]
    for lst in lists:
        flat.extend(int(x) for x in lst)
        offsets.append(offsets[-1] + len(lst))
    return {
        f"{name}_flat": np.array(flat, dtype=np.int64),
        f"{name}_offsets": np.array(offsets, dtype=np.int64),
    }


def _unpack_ragged(name: str, arrays: Mapping[str, np.ndarray]) -> List[List[int]]:
    flat = arrays[f"{name}_flat"]
    offs = arrays[f"{name}_offsets"]
    return [flat[offs[i]:offs[i + 1]].tolist() for i in range(len(offs) - 1)]


# ── Settlements ───────────────────────────────────────────────────────────────


def pack_settlements(settlements: List[Settlement]) -> Dict[str, np.ndarray]:
    return {
        "settle_face_id": np.array([s.face_id for s in settlements], dtype=np.int32),
        "settle_position": np.array(
            [s.position for s in settlements], dtype=np.float32
        ).reshape(-1, 3),
        "settle_tier": np.array([s.tier for s in settlements], dtype=np.int32),
        "settle_score": np.array([s.score for s in settlements], dtype=np.float64),
        "settle_population": np.array([s.population for s in settlements], dtype=np.float64),
        "settle_defensibility": np.array([s.defensibility for s in settlements], dtype=np.float64),
        "settle_has_walls": np.array([s.has_walls for s in settlements], dtype=np.bool_),
        "settle_wall_strength": np.array([s.wall_strength for s in settlements], dtype=np.float64),
        "settle_site_kind": np.array([s.site_kind for s in settlements], dtype=np.int8),
        "settle_site_position": np.array(
            [s.site_position for s in settlements], dtype=np.float32
        ).reshape(-1, 3),
        "settle_region_position": np.array(
            [s.region_position for s in settlements], dtype=np.float32
        ).reshape(-1, 3),
        "settle_refined": np.array([s.refined for s in settlements], dtype=np.bool_),
        "settle_local_refined": np.array([s.local_refined for s in settlements], dtype=np.bool_),
    }


def unpack_settlements(arrays: Mapping[str, np.ndarray]) -> List[Settlement]:
    fid = arrays["settle_face_id"]
    pos = arrays["settle_position"]
    tier = arrays["settle_tier"]
    score = arrays["settle_score"]
    pop = arrays["settle_population"]
    defens = arrays["settle_defensibility"]
    walls = arrays["settle_has_walls"]
    wall_strength = arrays["settle_wall_strength"]
    return [
        Settlement(
            face_id=int(fid[i]),
            position=np.asarray(pos[i], dtype=np.float64),
            tier=int(tier[i]),
            score=float(score[i]),
            population=float(pop[i]),
            defensibility=float(defens[i]),
            has_walls=bool(walls[i]),
            wall_strength=float(wall_strength[i]),
            site_kind=int(arrays["settle_site_kind"][i]),
            site_position=np.asarray(arrays["settle_site_position"][i], dtype=np.float64),
            region_position=np.asarray(arrays["settle_region_position"][i], dtype=np.float64),
            refined=bool(arrays["settle_refined"][i]),
            local_refined=bool(arrays["settle_local_refined"][i]),
        )
        for i in range(len(fid))
    ]


# ── Polities ──────────────────────────────────────────────────────────────────


def pack_polities(polities: List[Polity]) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {
        "pol_id": np.array([p.polity_id for p in polities], dtype=np.int32),
        "pol_capital_face": np.array([p.capital_face for p in polities], dtype=np.int32),
        "pol_capital_pos": np.array(
            [p.capital_position for p in polities], dtype=np.float32
        ).reshape(-1, 3),
        "pol_cohesion": np.array([p.cohesion for p in polities], dtype=np.float64),
        "pol_infrastructure": np.array([p.infrastructure for p in polities], dtype=np.float64),
        "pol_legitimacy": np.array([p.legitimacy for p in polities], dtype=np.float64),
        "pol_maritime_reach": np.array([p.maritime_reach for p in polities], dtype=np.float64),
        "pol_autonomy": np.array([p.autonomy for p in polities], dtype=np.float64),
        "pol_power": np.array([p.power for p in polities], dtype=np.float64),
        "pol_domain_power": np.array([p.domain_power for p in polities], dtype=np.float64),
        "pol_liege": np.array(
            [-1 if p.liege is None else p.liege for p in polities], dtype=np.int32
        ),
        "pol_depth": np.array([p.depth for p in polities], dtype=np.int32),
        "pol_n_faces": np.array([p.n_faces for p in polities], dtype=np.int32),
    }
    arrays.update(_pack_ragged("pol_territory", [p.territory_faces for p in polities]))
    arrays.update(_pack_ragged("pol_members", [p.member_settlement_faces for p in polities]))
    arrays.update(_pack_ragged("pol_vassals", [p.vassals for p in polities]))
    return arrays


def unpack_polities(arrays: Mapping[str, np.ndarray]) -> List[Polity]:
    pid = arrays["pol_id"]
    cap_face = arrays["pol_capital_face"]
    cap_pos = arrays["pol_capital_pos"]
    cohesion = arrays["pol_cohesion"]
    infra = arrays["pol_infrastructure"]
    legitimacy = arrays["pol_legitimacy"]
    maritime = arrays["pol_maritime_reach"]
    autonomy = arrays["pol_autonomy"]
    power = arrays["pol_power"]
    domain_power = arrays["pol_domain_power"]
    liege = arrays["pol_liege"]
    depth = arrays["pol_depth"]
    n_faces = arrays["pol_n_faces"]
    territory = _unpack_ragged("pol_territory", arrays)
    members = _unpack_ragged("pol_members", arrays)
    vassals = _unpack_ragged("pol_vassals", arrays)
    return [
        Polity(
            polity_id=int(pid[i]),
            capital_face=int(cap_face[i]),
            capital_position=np.asarray(cap_pos[i], dtype=np.float64),
            cohesion=float(cohesion[i]),
            infrastructure=float(infra[i]),
            legitimacy=float(legitimacy[i]),
            maritime_reach=float(maritime[i]),
            liege=None if int(liege[i]) < 0 else int(liege[i]),
            autonomy=float(autonomy[i]),
            depth=int(depth[i]),
            power=float(power[i]),
            domain_power=float(domain_power[i]),
            n_faces=int(n_faces[i]),
            territory_faces=list(territory[i]),
            member_settlement_faces=list(members[i]),
            vassals=list(vassals[i]),
        )
        for i in range(len(pid))
    ]


# ── Trade routes ──────────────────────────────────────────────────────────────


def pack_routes(routes: list) -> Dict[str, np.ndarray]:
    return {
        "route_endpoints": np.array(
            [[*r[0], *r[1]] for r in routes], dtype=np.float64
        ).reshape(-1, 6),
        "route_flow": np.array([r[2] for r in routes], dtype=np.float64),
    }


def unpack_routes(arrays: Mapping[str, np.ndarray]) -> list:
    eps = arrays["route_endpoints"]
    flow = arrays["route_flow"]
    return [
        (np.asarray(eps[i, :3]), np.asarray(eps[i, 3:]), float(flow[i]))
        for i in range(len(flow))
    ]


# ── Polity borders (depth → (M, 6) segments) ──────────────────────────────────


def pack_borders(borders_by_depth: Mapping[int, np.ndarray]) -> Dict[str, np.ndarray]:
    segs: List[np.ndarray] = []
    depths: List[np.ndarray] = []
    for depth, arr in borders_by_depth.items():
        a = np.asarray(arr, dtype=np.float64).reshape(-1, 6)
        if a.shape[0] == 0:
            continue
        segs.append(a)
        depths.append(np.full(a.shape[0], int(depth), dtype=np.int32))
    if not segs:
        return {}
    return {
        "border_segs": np.concatenate(segs, axis=0),
        "border_depth": np.concatenate(depths),
    }


def unpack_borders(arrays: Mapping[str, np.ndarray]) -> Dict[int, np.ndarray]:
    segs = arrays["border_segs"]
    depth = arrays["border_depth"]
    return {int(d): segs[depth == d] for d in np.unique(depth)}
