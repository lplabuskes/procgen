"""Pure functions for the town stage: rural density, town sizes, town sites.

Towns are carved out of the rural population, so every function here works per
global face and conserves that face's people: rural density spreads the face's
rural population over its texels, town sizes spend a share of it, and whatever
a town could not be sited for goes back to the countryside.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from scipy.spatial import KDTree

from realmgen.refine.sites import SiteField
from realmgen.towns.types import TOWN_MARKET, TOWN_MINING, TOWN_PORT, TOWN_RIVER, TOWN_ROUTE, Towns

if TYPE_CHECKING:
    from realmgen.towns.builder import TownParams

# Town exclusion radii grow with size as ``(pop / reference) ** SPACING_EXPONENT``:
# a 7k town claims ~1.6× the room of a 1k one, a 500k city ~2.9× a 7k one.
SPACING_EXPONENT = 0.25
# Density floor (relative to the face mean) so barren-but-land sites keep a
# nonzero chance: mining camps and waypoints do not need farmland.
_DENSITY_FLOOR = 0.05


def texel_owners(centroids: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """``(N, H, W)`` nearest-face-centroid owner of every texel in ``positions``."""
    shape = positions.shape[:-1]
    _, owner = KDTree(np.asarray(centroids, dtype=np.float64)).query(
        np.asarray(positions, dtype=np.float64).reshape(-1, 3), workers=-1,
    )
    return np.asarray(owner, dtype=np.int32).reshape(shape)


def compute_rural_density(
    rural: np.ndarray,
    face_areas_km2: np.ndarray,
    fertility: Optional[np.ndarray],
    land: np.ndarray,
    owner: np.ndarray,
    fertility_exponent: float,
) -> np.ndarray:
    """Per-texel rural density (people/km²), ``(N, H, W)``.

    Density follows the rural model's own food base, ``fertility ** γ``, over
    land texels, scaled per face so the texels a face *owns* hold exactly its
    rural population (each owned texel standing for an equal share of the face's
    area). An over-hanging texel takes its owner face's scale, so the field is a
    function of position and agrees across face seams. A face whose owned land
    is all barren spreads its people evenly over that land instead.
    """
    n = rural.shape[0]
    fert = np.zeros(land.shape) if fertility is None else np.clip(fertility, 0.0, 1.0)
    raw = np.where(land, fert ** fertility_exponent, 0.0)
    owned = owner == np.arange(n)[:, None, None]
    n_owned = owned.sum(axis=(1, 2))
    texel_area = np.asarray(face_areas_km2, dtype=np.float64) / np.maximum(n_owned, 1)

    mass = (raw * owned).sum(axis=(1, 2)) * texel_area
    barren = (mass <= 0) & (rural > 0)
    if barren.any():
        raw = np.where(barren[owner] & land, 1.0, raw)
        mass = (raw * owned).sum(axis=(1, 2)) * texel_area
    scale = np.where(mass > 0, rural / np.maximum(mass, 1e-300), 0.0)
    return raw * scale[owner]


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

    Sizes are truncated-Pareto (``pdf ∝ x^-(α+1)`` on ``[town_min, town_max]``;
    α = 1 is rank-size). The draw that would overspend is cut to what is left,
    and a remainder below ``town_min`` becomes one more minimum-size town with
    probability ``remainder / town_min`` — unbiased, so a thinly peopled face
    still gets a town now and then. The total never exceeds ``cap``.
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


class _FaceCandidates:
    """One face's town sites: jittered positions, log site weight, and kind."""

    def __init__(self, positions: np.ndarray, log_weight: np.ndarray, kind: np.ndarray) -> None:
        self.positions = positions
        self.log_weight = log_weight
        self.kind = kind


def _build_candidates(
    faces: Sequence[int], sites: SiteField, density: np.ndarray, params: "TownParams",
    mineral_tree: Optional[KDTree], route_tree: Optional[KDTree],
    mining_radius: float, route_radius: float, seed: int,
) -> tuple[dict[int, _FaceCandidates], dict[int, np.random.Generator]]:
    """Candidates for every face in ``faces``, with each face's site RNG (the
    same stream then draws that face's Gumbel noise)."""
    sites.prefetch(faces)
    rngs: dict[int, np.random.Generator] = {}
    per_face = []
    for f in faces:
        fs = sites.face(f)
        idx = fs.sites()
        rng = np.random.default_rng([int(seed), int(f)])
        rngs[f] = rng
        pos = fs.positions[idx]
        # Coast texels keep their centre: a jitter could step off into the sea.
        coast = fs.coast[idx]
        jittered = np.where(coast[:, None], pos, _jitter(pos, params.site_jitter * fs.spacing, rng))
        per_face.append((f, fs, idx, pos, jittered))

    # One batched query per tree over every face's sites.
    all_pos = np.concatenate([p for *_, p, _j in per_face]) if per_face else np.zeros((0, 3))
    all_jit = np.concatenate([j for *_, j in per_face]) if per_face else np.zeros((0, 3))
    n_all = all_pos.shape[0]
    route = (route_tree.query(all_pos, workers=-1)[0] < route_radius
             if route_tree is not None and n_all else np.zeros(n_all, dtype=bool))
    mine = (mineral_tree.query(all_pos, workers=-1)[0] < mining_radius
            if mineral_tree is not None and n_all else np.zeros(n_all, dtype=bool))
    owner = sites.owner(all_jit) if n_all else np.zeros(0, dtype=np.int64)

    out: dict[int, _FaceCandidates] = {}
    start = 0
    for f, fs, idx, pos, jittered in per_face:
        sl = slice(start, start + idx.size)
        start += idx.size
        # A jitter must not carry a site out of its face.
        positions = np.where((owner[sl] == f)[:, None] | ~fs.owned[idx][:, None], jittered, pos)

        rho = density[f].reshape(-1)[idx]
        mean = float(rho.mean()) if rho.size else 0.0
        rho_n = rho / mean if mean > 0 else np.ones_like(rho)
        coast = fs.coast[idx]
        conf = fs.confluence[idx]
        river = fs.river[idx] & ~conf
        weight = (
            (rho_n + _DENSITY_FLOOR) * (
                1.0 + params.port_bonus * coast + params.confluence_bonus * conf
                + params.river_bonus * river + params.route_bonus * route[sl]
            )
            + params.mining_bonus * mine[sl]
        )
        # What the site offered, most distinctive first.
        kind = np.select(
            [coast, mine[sl], conf | river, route[sl]],
            [TOWN_PORT, TOWN_MINING, TOWN_RIVER, TOWN_ROUTE],
            default=TOWN_MARKET,
        ).astype(np.int8)
        out[f] = _FaceCandidates(positions, np.log(weight), kind)
    return out, rngs


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
) -> tuple[Towns, np.ndarray]:
    """Site every requested town; returns ``(towns, unplaced_people_per_face)``.

    ``requests`` is ``(face, sizes)`` per face. Towns are sited globally largest
    first, so the biggest towns take the best sites. Each picks among its face's
    owned land texels by weighted random choice (Gumbel-max over the log site
    weight — denser farmland, harbours, rivers, trade routes and mineral
    deposits all raise it), skipping any site within the exclusion radius of a
    town or city already there. Radii are chord distances on the unit sphere:
    ``town_spacing`` for a ``town_min`` town, ``city_radii`` per city, and a pair
    must be the larger of their two radii apart.

    Exclusion is only checked against the face and its neighbours, which is exact
    while every radius is shorter than a face's width.
    """
    n = len(adjacency)
    town_min = float(params.town_min)
    order = sorted(
        ((float(size), int(f), j) for f, sizes in requests for j, size in enumerate(sizes)),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    mineral_tree = KDTree(mineral_points) if mineral_points is not None and len(mineral_points) else None
    route_tree = KDTree(route_points) if route_points is not None and len(route_points) else None

    # Per face: exclusion discs already placed (cities, then towns), in
    # preallocated buffers — ``disc_pos[f][:n_disc[f]]`` are live.
    n_towns = np.zeros(n, dtype=np.int64)
    for f, sizes in requests:
        n_towns[int(f)] += len(sizes)
    city_faces = np.asarray(city_faces, dtype=np.int64)
    n_cities = np.bincount(city_faces, minlength=n) if city_faces.size else np.zeros(n, dtype=np.int64)
    disc_pos = [np.empty((int(c), 3)) for c in n_towns + n_cities]
    disc_rad = [np.empty(int(c)) for c in n_towns + n_cities]
    n_disc = np.zeros(n, dtype=np.int64)

    def _add_disc(f: int, p: np.ndarray, r: float) -> None:
        k = n_disc[f]
        disc_pos[f][k] = p
        disc_rad[f][k] = r
        n_disc[f] = k + 1

    for p, f, r in zip(city_positions, city_faces, city_radii):
        _add_disc(int(f), np.asarray(p, dtype=np.float64), float(r))

    candidates, rngs = _build_candidates(
        [int(f) for f, _ in requests], sites, density, params,
        mineral_tree, route_tree, mining_radius, route_radius, seed,
    )
    gumbel = {
        int(f): rngs[int(f)].gumbel(size=(len(sizes), candidates[int(f)].positions.shape[0]))
        for f, sizes in requests
    }

    out_pos: list[np.ndarray] = []
    out_pop: list[float] = []
    out_face: list[int] = []
    out_kind: list[int] = []
    unplaced = np.zeros(n, dtype=np.float64)

    for size, f, j in order:
        cand = candidates[f]
        if cand.positions.shape[0] == 0:
            unplaced[f] += size
            continue

        radius = town_spacing * (size / town_min) ** SPACING_EXPONENT
        score = cand.log_weight + gumbel[f][j]
        near = [g for g in (f, *adjacency[f]) if n_disc[g]]
        if near:
            near_pos = np.concatenate([disc_pos[g][:n_disc[g]] for g in near])
            near_rad = np.maximum(np.concatenate([disc_rad[g][:n_disc[g]] for g in near]), radius)
            # Unit vectors: |a − b| ≥ r  ⇔  a·b ≤ 1 − r²/2.
            ok = (cand.positions @ near_pos.T <= 1.0 - 0.5 * near_rad ** 2).all(axis=1)
            score = np.where(ok, score, -np.inf)
        best = int(np.argmax(score))
        if not np.isfinite(score[best]):
            unplaced[f] += size
            continue

        p = cand.positions[best]
        _add_disc(f, p, radius)
        out_pos.append(p)
        out_pop.append(size)
        out_face.append(f)
        out_kind.append(int(cand.kind[best]))

    if not out_pop:
        return Towns.empty(), unplaced
    faces = np.asarray(out_face, dtype=np.int32)
    towns = Towns(
        positions=np.stack(out_pos),
        population=np.asarray(out_pop, dtype=np.float64),
        face_id=faces,
        parent=np.asarray(hinterland, dtype=np.int32)[faces],
        kind=np.asarray(out_kind, dtype=np.int8),
    )
    return towns, unplaced
