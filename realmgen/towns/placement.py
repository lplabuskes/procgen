"""Pure functions for the town stage: rural density, town sizes, town sites.

Everything works per global face and conserves that face's people; unsited
town people go back to the countryside.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from scipy.spatial import KDTree

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is an optional accelerator
    def njit(func=None, **_):  # type: ignore[misc]
        return func if func is not None else (lambda f: f)

from realmgen.refine.sites import SiteField
from realmgen.towns.types import TOWN_MARKET, TOWN_MINING, TOWN_PORT, TOWN_RIVER, TOWN_ROUTE, Towns

if TYPE_CHECKING:
    from realmgen.towns.builder import TownParams

# Town exclusion radii grow with size as ``(pop / reference) ** SPACING_EXPONENT``:
# a 7k town claims ~1.6× the room of a 1k one, a 500k city ~2.9× a 7k one.
SPACING_EXPONENT = 0.25
# Density floor (relative to the face mean): mining camps and waypoints don't need farmland.
_DENSITY_FLOOR = 0.05
# RNG stream tag for a face's mining-town site noise, seeded (seed, face, tag).
_MINING_STREAM = 5


def texel_owners(centroids: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """``(N, H, W)`` nearest-face-centroid owner of every texel in ``positions``."""
    shape = positions.shape[:-1]
    _, owner = KDTree(np.asarray(centroids, dtype=np.float64)).query(
        np.asarray(positions, dtype=np.float64).reshape(-1, 3), workers=-1,
    )
    return np.asarray(owner, dtype=np.int32).reshape(shape)


def shore_texels(land: np.ndarray) -> np.ndarray:
    """``(N, H, W)`` land texels with water among their 8 neighbours in the same face texture."""
    water = np.pad(~land, ((0, 0), (1, 1), (1, 1)), mode="edge")
    _, h, w = land.shape
    near = np.zeros(land.shape, dtype=bool)
    for di in range(3):
        for dj in range(3):
            near |= water[:, di:di + h, dj:dj + w]
    return land & near


def compute_rural_density(
    rural: np.ndarray,
    face_areas_km2: np.ndarray,
    fertility: Optional[np.ndarray],
    land: np.ndarray,
    owner: np.ndarray,
    fertility_exponent: float,
    fishing: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-texel rural density and its fishing part (people/km²), ``(N, H, W)`` each.

    Farmers (``rural − fishing``) follow ``fertility ** γ`` over land; fishers
    live on the shore. Each is scaled per face so the face's owned texels (equal
    area each) hold exactly its people; over-hanging texels take their owner's
    scale, so the field agrees across seams. All-barren faces spread farmers
    evenly; shoreless faces put fishers with the farmers.
    """
    n = rural.shape[0]
    rural = np.asarray(rural, dtype=np.float64)
    fish = np.zeros(n) if fishing is None else np.clip(np.asarray(fishing, dtype=np.float64), 0.0, rural)
    fert = np.zeros(land.shape) if fertility is None else np.clip(fertility, 0.0, 1.0)
    owned = owner == np.arange(n)[:, None, None]
    n_owned = owned.sum(axis=(1, 2))
    texel_area = np.asarray(face_areas_km2, dtype=np.float64) / np.maximum(n_owned, 1)

    def mass(raw: np.ndarray) -> np.ndarray:
        return (raw * owned).sum(axis=(1, 2)) * texel_area

    def spread(people: np.ndarray, raw: np.ndarray) -> np.ndarray:
        m = mass(raw)
        return raw * np.where(m > 0, people / np.maximum(m, 1e-300), 0.0)[owner]

    farm_raw = np.where(land, fert ** fertility_exponent, 0.0)
    barren = (mass(farm_raw) <= 0) & (rural > 0)
    if barren.any():
        farm_raw = np.where(barren[owner] & land, 1.0, farm_raw)
    shore_raw = shore_texels(land).astype(np.float64)
    no_shore = (mass(shore_raw) <= 0) & (fish > 0)
    if no_shore.any():
        shore_raw = np.where(no_shore[owner], farm_raw, shore_raw)
    fishing_density = spread(fish, shore_raw)
    return spread(rural - fish, farm_raw) + fishing_density, fishing_density


def compute_mining_density(
    points: np.ndarray,
    faces: np.ndarray,
    workers: np.ndarray,
    positions: np.ndarray,
    face_areas_km2: np.ndarray,
    land: np.ndarray,
    owner: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Miner density ``(N, H, W)`` people/km² and unsited miners per face ``(N,)``.

    Each deposit's ``workers`` go on the owned land texel nearest it, at the same
    per-texel area as :func:`compute_rural_density`. Faces with no owned land
    return their miners as unsited.
    """
    n = land.shape[0]
    density = np.zeros(land.shape, dtype=np.float64)
    unsited = np.zeros(n, dtype=np.float64)
    owned = owner == np.arange(n)[:, None, None]
    texel_area = np.asarray(face_areas_km2, dtype=np.float64) / np.maximum(owned.sum(axis=(1, 2)), 1)
    for p, f, m in zip(np.asarray(points, dtype=np.float64), np.asarray(faces), np.asarray(workers)):
        if m <= 0:
            continue
        f = int(f)
        ok = owned[f] & land[f]
        if not ok.any():
            unsited[f] += m
            continue
        idx = np.flatnonzero(ok)
        best = idx[int(np.argmax(positions[f].reshape(-1, 3)[idx] @ p))]
        density[f].reshape(-1)[best] += m / texel_area[f]
    return density, unsited


def draw_town_sizes(
    budget: float,
    rng: np.random.Generator,
    *,
    town_min: float,
    town_max: float,
    alpha: float,
    cap: float,
) -> np.ndarray:
    """Town populations spending ``budget`` people, largest first.

    Truncated-Pareto (``pdf ∝ x^-(α+1)`` on ``[town_min, town_max]``); the
    overspending draw is cut to what's left, and a remainder below ``town_min``
    becomes a min-size town with probability ``remainder / town_min`` (unbiased).
    The total never exceeds ``cap``.
    """
    sizes: list[float] = []
    remaining = float(budget)
    tail = 1.0 - (town_min / town_max) ** alpha
    while remaining >= town_min:
        x = town_min / (1.0 - rng.random() * tail) ** (1.0 / alpha)
        x = min(x, remaining)
        sizes.append(x)
        remaining -= x
    if remaining > 0 and rng.random() < remaining / town_min and sum(sizes) + town_min <= cap:
        sizes.append(town_min)
    return np.sort(np.asarray(sizes, dtype=np.float64))[::-1]


def _jitter(pos: np.ndarray, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Displace unit vectors uniformly within a tangent disc of ``radius``."""
    r = rng.normal(size=pos.shape)
    r -= (r * pos).sum(axis=1, keepdims=True) * pos
    r /= np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-12)
    out = pos + r * (radius * np.sqrt(rng.random(pos.shape[0])))[:, None]
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def site_log_weight_and_kind(
    rho: np.ndarray, coast: np.ndarray, confluence: np.ndarray, river: np.ndarray,
    stream: np.ndarray, route: np.ndarray, mine: np.ndarray, params: "TownParams",
) -> tuple[np.ndarray, np.ndarray]:
    """Log site weight and town kind of one face's candidate sites.

    ``rho`` is rural density (normalised here by the face mean). Water flags
    don't stack: pass ``river`` without confluences and ``stream`` without either.
    """
    mean = float(rho.mean()) if rho.size else 0.0
    rho_n = rho / mean if mean > 0 else np.ones_like(rho)
    weight = (
        (rho_n + _DENSITY_FLOOR) * (
            1.0 + params.port_bonus * coast + params.confluence_bonus * confluence
            + params.river_bonus * river + params.stream_bonus * stream + params.route_bonus * route
        )
        + params.mining_bonus * mine
    )
    kind = np.select(
        [coast, mine, confluence | river | stream, route],
        [TOWN_PORT, TOWN_MINING, TOWN_RIVER, TOWN_ROUTE],
        default=TOWN_MARKET,
    ).astype(np.int8)
    return np.log(weight), kind


class FaceCandidates:
    """One face's town sites: positions, log site weight, and kind."""

    def __init__(self, positions: np.ndarray, log_weight: np.ndarray, kind: np.ndarray) -> None:
        self.positions = positions
        self.log_weight = log_weight
        self.kind = kind


def _build_candidates(
    faces: Sequence[int], sites: SiteField, density: np.ndarray, params: "TownParams",
    mineral_tree: Optional[KDTree], route_tree: Optional[KDTree],
    mining_radius: float, route_radius: float, seed: int,
) -> tuple[dict[int, FaceCandidates], dict[int, np.random.Generator]]:
    """Candidates per face, with each face's site RNG (reused for its Gumbel noise)."""
    sites.prefetch(faces)
    rngs: dict[int, np.random.Generator] = {}
    per_face = []
    for f in faces:
        fs = sites.face(f)
        idx = fs.sites()
        rng = np.random.default_rng([int(seed), int(f)])
        rngs[f] = rng
        pos = fs.positions[idx]
        # Coast texels aren't jittered: they could step into the sea.
        coast = fs.coast[idx]
        jittered = np.where(coast[:, None], pos, _jitter(pos, params.site_jitter * fs.spacing, rng))
        per_face.append((f, fs, idx, pos, jittered))

    all_pos = np.concatenate([p for *_, p, _j in per_face]) if per_face else np.zeros((0, 3))
    all_jit = np.concatenate([j for *_, j in per_face]) if per_face else np.zeros((0, 3))
    n_all = all_pos.shape[0]
    route = (route_tree.query(all_pos, workers=-1)[0] < route_radius
             if route_tree is not None and n_all else np.zeros(n_all, dtype=bool))
    mine = (mineral_tree.query(all_pos, workers=-1)[0] < mining_radius
            if mineral_tree is not None and n_all else np.zeros(n_all, dtype=bool))
    owner = sites.owner(all_jit) if n_all else np.zeros(0, dtype=np.int64)

    out: dict[int, FaceCandidates] = {}
    start = 0
    for f, fs, idx, pos, jittered in per_face:
        sl = slice(start, start + idx.size)
        start += idx.size
        # A jitter must not carry a site out of its face.
        positions = np.where((owner[sl] == f)[:, None] | ~fs.owned[idx][:, None], jittered, pos)

        conf = fs.confluence[idx]
        river = fs.river[idx] & ~conf
        log_weight, kind = site_log_weight_and_kind(
            density[f].reshape(-1)[idx], fs.coast[idx], conf, river, fs.stream[idx] & ~conf & ~river,
            route[sl], mine[sl], params,
        )
        out[f] = FaceCandidates(positions, log_weight, kind)
    return out, rngs


def _mining_candidates(
    cand: FaceCandidates, points: np.ndarray, workers: np.ndarray, mining_radius: float,
) -> FaceCandidates:
    """Mining-town sites: within ``2 × mining_radius`` of a worked deposit,
    weighted ``Σ workers · exp(−(d / radius)²)``."""
    d2 = ((cand.positions[:, None, :] - points[None, :, :]) ** 2).sum(axis=2) / mining_radius ** 2
    w = (workers[None, :] * np.exp(-d2)).sum(axis=1)
    reach = (d2 < 4.0).any(axis=1) & (w > 0)
    log_weight = np.where(reach, np.log(np.maximum(w, 1e-300)), -np.inf)
    return FaceCandidates(cand.positions, log_weight, np.full(cand.positions.shape[0], TOWN_MINING, dtype=np.int8))


def place_towns(
    requests: Sequence[tuple[int, np.ndarray]],
    *,
    sites: SiteField,
    density: np.ndarray,
    adjacency: Sequence[Sequence[int]],
    city_positions: np.ndarray,
    city_faces: np.ndarray,
    city_radii: np.ndarray,
    hinterland: np.ndarray,
    params: "TownParams",
    town_spacing: float,
    mining_radius: float,
    route_radius: float,
    mineral_points: Optional[np.ndarray],
    route_points: Optional[np.ndarray],
    seed: int,
    mining_requests: Sequence[tuple[int, np.ndarray]] = (),
    mine_points: Optional[np.ndarray] = None,
    mine_faces: Optional[np.ndarray] = None,
    mine_workers: Optional[np.ndarray] = None,
) -> tuple[Towns, np.ndarray, np.ndarray]:
    """Site every requested town; returns ``(towns, unplaced_people_per_face,
    is_mining_request)`` — the last flags the towns sited for ``mining_requests``.

    ``requests``/``mining_requests`` are ``(face, sizes)``; mining towns are
    sited among sites near the face's worked deposits. Radii are unit-sphere
    chords: ``town_spacing`` for a ``town_min`` town, ``city_radii`` per city.
    See :func:`site_largest_first`.
    """
    mineral_tree = KDTree(mineral_points) if mineral_points is not None and len(mineral_points) else None
    route_tree = KDTree(route_points) if route_points is not None and len(route_points) else None
    faces = sorted({int(f) for f, _ in requests} | {int(f) for f, _ in mining_requests})
    face_cands, face_rngs = _build_candidates(
        faces, sites, density, params, mineral_tree, route_tree, mining_radius, route_radius, seed,
    )
    all_requests = [(int(f), sz) for f, sz in requests]
    candidates = [face_cands[f] for f, _ in all_requests]
    rngs = [face_rngs[f] for f, _ in all_requests]
    if mining_requests:
        assert mine_points is not None and mine_faces is not None and mine_workers is not None
        m_faces = np.asarray(mine_faces)
        m_workers = np.asarray(mine_workers, dtype=np.float64)
        for f, sizes in mining_requests:
            on = (m_faces == int(f)) & (m_workers > 0)
            all_requests.append((int(f), sizes))
            candidates.append(_mining_candidates(
                face_cands[int(f)], np.asarray(mine_points, dtype=np.float64)[on], m_workers[on], mining_radius))
            rngs.append(np.random.default_rng([int(seed), int(f), _MINING_STREAM]))
    placed, unplaced = site_largest_first(
        all_requests, candidates, rngs,
        adjacency=adjacency,
        disc_positions=city_positions, disc_faces=city_faces, disc_radii=city_radii,
        town_spacing=town_spacing, town_min=float(params.town_min),
    )
    is_mining = np.asarray([r >= len(requests) for r, *_ in placed], dtype=bool)
    return _towns_from([(f, size, pos, kind) for _, f, size, pos, kind in placed], hinterland), unplaced, is_mining


def site_largest_first(
    requests: Sequence[tuple[int, np.ndarray]],
    candidates: Sequence[FaceCandidates],
    rngs: Sequence[np.random.Generator],
    *,
    adjacency: Sequence[Sequence[int]],
    disc_positions: np.ndarray,
    disc_faces: np.ndarray,
    disc_radii: np.ndarray,
    town_spacing: float,
    town_min: float,
) -> tuple[list[tuple[int, int, float, np.ndarray, int]], np.ndarray]:
    """Site ``requests`` (``(face, sizes)``) on their ``candidates`` (one per
    request, all on that request's face), globally largest first; returns
    ``([(request, face, size, position, kind)], unplaced_per_face)``.

    Each town takes the Gumbel-max of its log site weights over sites clear of
    every disc placed so far (fixed ``disc_*`` cities, then earlier towns). A town
    claims ``town_spacing · (size / town_min)^0.25``; a pair must be the larger
    radius apart. Only the face and its neighbours are checked, which is exact
    while every radius is shorter than a face's width.
    """
    n = len(adjacency)
    order = sorted(
        ((float(size), int(f), r, j) for r, (f, sizes) in enumerate(requests) for j, size in enumerate(sizes)),
        key=lambda t: (-t[0], t[1], t[2], t[3]),
    )

    # Flat candidate arrays, ``cand_off[r]:cand_off[r + 1]`` being request r's,
    # and each request's Gumbel noise (one row per town) flat from ``gum_off[r]``.
    n_cand = np.array([candidates[r].positions.shape[0] for r in range(len(requests))], dtype=np.int64)
    cand_off = np.zeros(len(requests) + 1, dtype=np.int64)
    np.cumsum(n_cand, out=cand_off[1:])
    cand_pos = (np.concatenate([np.asarray(c.positions, dtype=np.float64).reshape(-1, 3) for c in candidates])
                if len(candidates) else np.zeros((0, 3)))
    cand_logw = (np.concatenate([np.asarray(c.log_weight, dtype=np.float64).reshape(-1) for c in candidates])
                 if len(candidates) else np.zeros(0))
    gumbel = [
        rngs[r].gumbel(size=(len(sizes), candidates[r].positions.shape[0])).reshape(-1)
        for r, (_, sizes) in enumerate(requests)
    ]
    gum_off = np.zeros(len(requests) + 1, dtype=np.int64)
    np.cumsum([g.size for g in gumbel], out=gum_off[1:])
    gum = np.concatenate(gumbel) if gumbel else np.zeros(0)

    # Per face: room for its exclusion discs (fixed ones — cities — then towns),
    # ``disc_off[f]:disc_off[f] + n_disc[f]`` live.
    n_towns = np.zeros(n, dtype=np.int64)
    for f, sizes in requests:
        n_towns[int(f)] += len(sizes)
    disc_faces = np.asarray(disc_faces, dtype=np.int64)
    n_fixed = np.bincount(disc_faces, minlength=n) if disc_faces.size else np.zeros(n, dtype=np.int64)
    disc_off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(n_towns + n_fixed, out=disc_off[1:])
    by_face = np.argsort(disc_faces, kind="stable")
    rank = np.arange(by_face.size) - np.searchsorted(disc_faces[by_face], disc_faces[by_face])
    disc_pos = np.zeros((int(disc_off[-1]), 3))
    disc_rad = np.zeros(int(disc_off[-1]))
    slots = disc_off[disc_faces[by_face]] + rank
    disc_pos[slots] = np.asarray(disc_positions, dtype=np.float64).reshape(-1, 3)[by_face]
    disc_rad[slots] = np.asarray(disc_radii, dtype=np.float64).reshape(-1)[by_face]
    n_disc = n_fixed.copy()

    adj_ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum([len(a) for a in adjacency], out=adj_ptr[1:])
    adj_idx = np.fromiter((g for a in adjacency for g in a), dtype=np.int64, count=int(adj_ptr[-1]))

    town = np.array(order, dtype=np.float64).reshape(-1, 4)
    chosen, unplaced = _site_towns(
        town[:, 0].copy(), town[:, 1].astype(np.int64), town[:, 2].astype(np.int64), town[:, 3].astype(np.int64),
        cand_off, cand_pos, cand_logw, gum_off, gum, disc_off, disc_pos, disc_rad, n_disc,
        adj_ptr, adj_idx, float(town_spacing), float(town_min), float(SPACING_EXPONENT), n,
    )
    placed: list[tuple[int, int, float, np.ndarray, int]] = []
    for t, c in enumerate(chosen.tolist()):
        if c < 0:
            continue
        size, f, r, _j = order[t]
        k = c - int(cand_off[r])
        placed.append((r, f, size, candidates[r].positions[k], int(candidates[r].kind[k])))
    return placed, unplaced


@njit(cache=True)
def _site_towns(size, face, req, row, cand_off, cand_pos, cand_logw, gum_off, gum,
                disc_off, disc_pos, disc_rad, n_disc, adj_ptr, adj_idx,
                town_spacing, town_min, exponent, n_faces):  # pragma: no cover - compiled
    """The compiled :func:`site_largest_first` loop.

    Returns ``(chosen candidate per town or −1, unplaced people per face)``.
    """
    n_t = size.shape[0]
    chosen = np.full(n_t, -1, dtype=np.int64)
    unplaced = np.zeros(n_faces)
    for t in range(n_t):
        f, r = face[t], req[t]
        c0, c1 = cand_off[r], cand_off[r + 1]
        radius = town_spacing * (size[t] / town_min) ** exponent
        g0 = gum_off[r] + row[t] * (c1 - c0)
        best = -1
        best_score = -np.inf
        for c in range(c0, c1):
            score = cand_logw[c] + gum[g0 + c - c0]
            if not score > best_score:          # argmax keeps the first of equals
                continue
            ok = True
            for a in range(adj_ptr[f] - 1, adj_ptr[f + 1]):
                g = f if a < adj_ptr[f] else adj_idx[a]
                for q in range(disc_off[g], disc_off[g] + n_disc[g]):
                    rr = max(disc_rad[q], radius)
                    # Unit vectors: |a − b| ≥ r  ⇔  a·b ≤ 1 − r²/2.
                    dot = (cand_pos[c, 0] * disc_pos[q, 0] + cand_pos[c, 1] * disc_pos[q, 1]
                           + cand_pos[c, 2] * disc_pos[q, 2])
                    if dot > 1.0 - 0.5 * rr * rr:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                best = c
                best_score = score
        if best < 0:
            unplaced[f] += size[t]
            continue
        chosen[t] = best
        q = disc_off[f] + n_disc[f]
        disc_pos[q, 0] = cand_pos[best, 0]
        disc_pos[q, 1] = cand_pos[best, 1]
        disc_pos[q, 2] = cand_pos[best, 2]
        disc_rad[q] = radius
        n_disc[f] += 1
    return chosen, unplaced


def _towns_from(placed: list[tuple[int, float, np.ndarray, int]], hinterland: np.ndarray) -> Towns:
    if not placed:
        return Towns.empty()
    faces = np.asarray([f for f, *_ in placed], dtype=np.int32)
    return Towns(
        positions=np.stack([p for _, _, p, _ in placed]),
        population=np.asarray([size for _, size, _, _ in placed], dtype=np.float64),
        face_id=faces,
        parent=np.asarray(hinterland, dtype=np.int32)[faces],
        kind=np.asarray([k for *_, k in placed], dtype=np.int8),
    )
