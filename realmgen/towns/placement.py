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


def shore_texels(land: np.ndarray) -> np.ndarray:
    """``(N, H, W)`` land texels with water among their 8 neighbours in the same
    face texture (which over-hangs its face, so a polygon-edge texel sees the
    real neighbouring water)."""
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

    The farmers — ``rural − fishing`` — follow the rural model's own food base,
    ``fertility ** γ``, over land texels; the fishers (``fishing``, per face)
    live on the shore (:func:`shore_texels`).  Each is scaled per face so the
    texels a face *owns* hold exactly its people (each owned texel standing for
    an equal share of the face's area). An over-hanging texel takes its owner
    face's scale, so the field is a function of position and agrees across face
    seams. A face whose owned land is all barren spreads its farmers evenly over
    that land instead, and one with no owned shore puts its fishers with them.
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


def site_log_weight_and_kind(
    rho: np.ndarray, coast: np.ndarray, confluence: np.ndarray, river: np.ndarray,
    stream: np.ndarray, route: np.ndarray, mine: np.ndarray, params: "TownParams",
) -> tuple[np.ndarray, np.ndarray]:
    """Log site weight and town kind of one face's candidate sites.

    ``rho`` is rural density, normalised here by the face mean; the flags are
    what each site offers. The water ones do not stack: pass ``river`` without
    the confluences and ``stream`` without either. A stream site makes a river
    town, like a river site.
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
    # What the site offered, most distinctive first.
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
    mineral_tree = KDTree(mineral_points) if mineral_points is not None and len(mineral_points) else None
    route_tree = KDTree(route_points) if route_points is not None and len(route_points) else None
    candidates, rngs = _build_candidates(
        [int(f) for f, _ in requests], sites, density, params,
        mineral_tree, route_tree, mining_radius, route_radius, seed,
    )
    placed, unplaced = site_largest_first(
        requests, candidates, rngs,
        adjacency=adjacency,
        disc_positions=city_positions, disc_faces=city_faces, disc_radii=city_radii,
        town_spacing=town_spacing, town_min=float(params.town_min),
    )
    return _towns_from(placed, hinterland), unplaced


def site_largest_first(
    requests: Sequence[tuple[int, np.ndarray]],
    candidates: dict[int, FaceCandidates],
    rngs: dict[int, np.random.Generator],
    *,
    adjacency: Sequence[Sequence[int]],
    disc_positions: np.ndarray,
    disc_faces: np.ndarray,
    disc_radii: np.ndarray,
    town_spacing: float,
    town_min: float,
) -> tuple[list[tuple[int, float, np.ndarray, int]], np.ndarray]:
    """Site ``requests`` (``(face, sizes)``) on their faces' ``candidates``, globally
    largest first; returns ``([(face, size, position, kind)], unplaced_per_face)``.

    Each town takes the Gumbel-max of its face's log site weights (noise drawn
    from ``rngs[face]``) over the sites clear of every exclusion disc already
    placed — the fixed ``disc_*`` ones (cities) and earlier towns. A town of
    ``size`` claims ``town_spacing · (size / town_min)^0.25``; a pair must be the
    larger of their two radii apart. Only the face and its neighbours are
    checked, which is exact while every radius is shorter than a face's width.
    """
    n = len(adjacency)
    order = sorted(
        ((float(size), int(f), j) for f, sizes in requests for j, size in enumerate(sizes)),
        key=lambda t: (-t[0], t[1], t[2]),
    )

    # Per face: exclusion discs already placed (fixed ones, then towns), in
    # preallocated buffers — ``disc_pos[f][:n_disc[f]]`` are live.
    n_towns = np.zeros(n, dtype=np.int64)
    for f, sizes in requests:
        n_towns[int(f)] += len(sizes)
    disc_faces = np.asarray(disc_faces, dtype=np.int64)
    n_fixed = np.bincount(disc_faces, minlength=n) if disc_faces.size else np.zeros(n, dtype=np.int64)
    disc_pos = [np.empty((int(c), 3)) for c in n_towns + n_fixed]
    disc_rad = [np.empty(int(c)) for c in n_towns + n_fixed]
    n_disc = np.zeros(n, dtype=np.int64)

    def _add_disc(f: int, p: np.ndarray, r: float) -> None:
        k = n_disc[f]
        disc_pos[f][k] = p
        disc_rad[f][k] = r
        n_disc[f] = k + 1

    for p, f, r in zip(disc_positions, disc_faces, disc_radii):
        _add_disc(int(f), np.asarray(p, dtype=np.float64), float(r))

    gumbel = {
        int(f): rngs[int(f)].gumbel(size=(len(sizes), candidates[int(f)].positions.shape[0]))
        for f, sizes in requests
    }

    placed: list[tuple[int, float, np.ndarray, int]] = []
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
        placed.append((f, size, p, int(cand.kind[best])))
    return placed, unplaced


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
