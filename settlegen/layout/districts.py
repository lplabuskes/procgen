"""Districts per block: market, quay, elite, craft quarters by industry, graveyards, temples, squares, houses;
and the everyday trades spread through the houses."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

from settlegen.layout.params import SettlementLayoutParams
from settlegen.layout.types import Block
from settlegen.profile.builder import SettlementProfile

if TYPE_CHECKING:
    from settlegen.site.builder import SettlementSite


# Craft quarters: industry → (district, noxious, by the water).
CRAFTS: dict[str, tuple[str, bool, bool]] = {
    "tanning": ("tanners", True, True),
    "dyeing": ("dyers", True, True),
    "smithing": ("smiths", True, False),
    "butchery": ("butchers", True, True),
    "brewing": ("brewers", False, True),
    "milling": ("millers", False, True),
    "sugar": ("sugar mills", False, True),
    "weaving": ("weavers", False, False),
    "timber": ("carpenters", False, False),
    "shipping": ("shipwrights", False, True),
    "salt works": ("salters", False, True),
    "fishing": ("fishers", False, True),
    "carving": ("artisans", False, False),
    "pearling": ("artisans", False, True),
}


# Districts with no lots ("temple" blocks join the sacred precinct in the layout).
OPEN_DISTRICTS = ("graveyard", "square", "temple")


# A block under this (m²) is too small for a temple precinct.
_TEMPLE_MIN_M2 = 600.0


# Districts whose lots run back to fill a block (merchants' and the elite's long plots).
DEEP_LOT_DISTRICTS = ("quay", "market", "elite")


# Districts whose deep blocks are opened up by closes and alleys.
LANE_DISTRICTS = ("residential", "suburb", *sorted({v[0] for v in CRAFTS.values()}), *DEEP_LOT_DISTRICTS)


DISTRICTS = ("market", "quay", "elite", "residential", "suburb", "gardens", "graveyard", "square", "temple",
             *sorted({v[0] for v in CRAFTS.values()}))


def _nearest_cluster(score: np.ndarray, centroids: np.ndarray, free: np.ndarray, n: int) -> np.ndarray:
    """Up to ``n`` free blocks: the best-scoring one and those nearest it."""
    idx = np.flatnonzero(free & np.isfinite(score))
    if idx.size == 0 or n <= 0:
        return np.zeros(0, dtype=np.int64)
    seed = idx[int(np.argmax(score[idx]))]
    d = np.linalg.norm(centroids[idx] - centroids[seed], axis=1)
    return idx[np.argsort(d, kind="stable")[:n]]


def assign_districts(
    site: "SettlementSite", profile: SettlementProfile, blocks: list[Polygon], in_core: np.ndarray,
    anchors: dict[str, np.ndarray], plaza: MultiPolygon, precinct: MultiPolygon, rng: np.random.Generator,
    params: SettlementLayoutParams, walled: bool = True,
) -> list[str]:
    """Each block's district.  See the module docstring: a town keeps to a market,
    a quay, its two leading trades, an elite quarter only by a castle and its
    churchyard; suburbs lie only outside walls."""
    n = len(blocks)
    district = np.array(["" for _ in range(n)], dtype=object)
    if n == 0:
        return []
    geoms = np.asarray(blocks, dtype=object)
    centroids = np.asarray([(p.x, p.y) for p in shapely.centroid(geoms)])
    centre = np.asarray(anchors["market"] if "market" in anchors else centroids.mean(axis=0), dtype=np.float64)
    radius = max(math.sqrt(max(site.footprint.area, 1.0) / math.pi), 1.0)
    wind = np.asarray(profile.wind_xy, dtype=np.float64)
    speed = float(np.linalg.norm(wind))
    downwind = (centroids - centre) @ (wind / speed) / radius if speed > 0.5 else np.zeros(n)
    water = site.water.bodies
    d_water = shapely.distance(geoms, water) if not water.is_empty else np.full(n, np.inf)
    downstream = np.zeros(n)
    rivers = [ln for ln in site.water.lines if len(ln.xy) >= 2 and not ln.minor]
    if rivers:
        main = max(rivers, key=lambda ln: float(np.mean(ln.width_m)))
        line = LineString(main.xy)
        s = shapely.line_locate_point(line, shapely.points(centroids))
        downstream = (s - line.project(Point(float(centre[0]), float(centre[1])))) / radius
        downstream = np.where(shapely.distance(line, shapely.points(centroids)) < radius, downstream, 0.0)
    sparse = site.sparse
    in_sparse = shapely.contains_xy(sparse, centroids[:, 0], centroids[:, 1]) if not sparse.is_empty \
        else np.zeros(n, dtype=bool)
    free = np.ones(n, dtype=bool)
    village = profile.tier == 5
    town = profile.tier == 4

    def take(idx: np.ndarray, name: str) -> None:
        idx = np.asarray(idx, dtype=np.int64)
        idx = idx[free[idx]]
        district[idx] = name
        free[idx] = False

    # Market: round the plaza.
    if not plaza.is_empty:
        d = shapely.distance(geoms, plaza)
        near = np.flatnonzero(d <= 1.5 * float(params.street_m) + 1.0)
        take(near[np.argsort(d[near], kind="stable")][:8 if town else 12], "market")
    # Quay: along the water by the waterfront.
    if "waterfront" in anchors and not village:
        reach = float(np.clip(200.0 * math.sqrt(profile.population / 5000.0), 150.0, 1200.0))
        near = np.linalg.norm(centroids - anchors["waterfront"], axis=1) <= reach
        take(np.flatnonzero(near & (d_water <= float(params.quay_m))), "quay")
    # Elite: core, near the citadel (else the precinct, the market), upwind.
    if not village and (not town or "citadel" in anchors):
        focus = anchors.get("citadel", anchors.get("sacred", centre))
        score = -np.linalg.norm(centroids - focus, axis=1) / radius - 0.5 * downwind
        score = np.where(in_core & free, score, -np.inf)
        k = int(round(float(params.elite_share) * int(in_core.sum())))
        take(_nearest_cluster(score, centroids, free & in_core, min(k, 4) if town else k), "elite")
    # Craft quarters.
    crafts = [(i.name, i.weight) for i in profile.industries if i.name in CRAFTS]
    if town:
        crafts = [c for c in sorted(crafts, key=lambda c: -c[1]) if c[1] >= 0.4][:2]
    elif not village:
        crafts.append(("butchery", 0.5))
    for name, weight in crafts:
        quarter, noxious, wet = CRAFTS[name]
        if quarter == "shipwrights" and "waterfront" not in anchors:
            continue
        k = int(round(weight * float(params.craft_share) * n))
        if k == 0 and weight >= 0.3:
            k = 1
        score = rng.uniform(0.0, 0.1, n) - np.linalg.norm(centroids - centre, axis=1) / (3.0 * radius)
        if noxious:
            score += downwind + 0.5 * downstream + 0.5 * ~in_core
        if wet:
            score += np.where(d_water <= float(params.quay_m), 1.0, -1.0)
        take(_nearest_cluster(score, centroids, free, k), quarter)
    # Graveyards: the churchyard by the precinct, the rest outside on the edge.
    n_graves = 1 if town or village else max(1, int(round(profile.population / float(params.graveyard_people))))
    if not precinct.is_empty:
        d = np.where(free, shapely.distance(geoms, precinct), np.inf)
        if np.isfinite(d).any():
            take(np.array([int(np.argmin(d))]), "graveyard")
            n_graves -= 1
    if n_graves > 0:
        edge = np.linalg.norm(centroids - centre, axis=1) / radius
        areas = shapely.area(geoms)
        cand = np.flatnonzero(free & ~in_core & (edge >= np.quantile(edge, 0.6)) & (areas <= np.median(areas)))
        if cand.size:
            take(rng.choice(cand, size=min(n_graves, cand.size), replace=False), "graveyard")
    # Temple precincts besides the main one: blocks of a middling size off the sparse ground,
    # each as far from the other sacred grounds as can be, the dense middle by preference.
    areas = shapely.area(geoms)
    n_temples = 0 if village else \
        min(int(math.sqrt(profile.population / float(params.temple_people))), int(params.temple_max))
    cand = np.flatnonzero(free & ~in_sparse & (areas >= np.quantile(areas, 0.2)) & (areas <= np.quantile(areas, 0.7))
                          & (areas >= _TEMPLE_MIN_M2))
    if n_temples and cand.size:
        held = [np.asarray(g.centroid.coords[0]) for g in shapely.get_parts(precinct)] or [centre]
        edge = np.linalg.norm(centroids[cand] - centre, axis=1) / radius
        weight = rng.uniform(0.7, 1.0, cand.size) * (1.0 - 0.5 * np.clip(edge, 0.0, 1.0))
        for _ in range(min(n_temples, cand.size)):
            d = np.min(np.linalg.norm(centroids[cand][:, None] - np.asarray(held)[None], axis=2), axis=1)
            score = np.where(free[cand], d * weight, -np.inf)
            k = int(np.argmax(score))
            if not np.isfinite(score[k]):
                break
            take(cand[k:k + 1], "temple")
            held.append(centroids[cand[k]])
    # Small squares in the core.
    n_squares = 0 if town or village else int(round(profile.population / float(params.square_people)))
    cand = np.flatnonzero(free & in_core & (areas <= np.quantile(areas, 0.3)))
    if n_squares and cand.size:
        take(rng.choice(cand, size=min(n_squares, cand.size), replace=False), "square")
    rest = np.flatnonzero(free)
    outside = ~in_core[rest] & walled
    district[rest] = np.where(in_sparse[rest], "gardens", np.where(outside, "suburb", "residential"))
    return [str(d) for d in district]


# Districts whose lots the everyday trades spread through.
TRADE_DISTRICTS = ("residential", "suburb", "market", "quay")


def disperse_trades(
    blocks: Sequence[Block], counts: dict[str, int], fronts: BaseGeometry, rng: np.random.Generator,
    max_share: float,
) -> int:
    """Tag lots of the houses (:data:`TRADE_DISTRICTS`) with ``counts`` of each trade, at most
    ``max_share`` of them in all: each trade spread out over the place (no two of it nearer
    than half the spacing its count would have evenly), preferring lots on ``fronts`` (the
    arterials and the market).  Sets the blocks' ``uses``; the number tagged."""
    refs = [(bi, li) for bi, b in enumerate(blocks) if b.district in TRADE_DISTRICTS for li in range(len(b.lots))]
    total = sum(counts.values())
    if not refs or total == 0:
        return 0
    lots = np.asarray([blocks[bi].lots[li] for bi, li in refs], dtype=object)
    xy = shapely.get_coordinates(shapely.centroid(lots))
    on_front = np.zeros(len(refs), dtype=bool)
    if not fronts.is_empty:
        # The fronts' outlines as single segments: an index on them finds the lots by them fast.
        rings = shapely.get_parts(shapely.get_rings(shapely.get_parts(fronts)))
        segs = [np.asarray(shapely.linestrings(np.stack([c[:-1], c[1:]], axis=1)), dtype=object)
                for c in (shapely.get_coordinates(r) for r in rings) if len(c) >= 2]
        tree = shapely.STRtree(np.concatenate(segs))
        on_front[tree.query(lots, predicate="dwithin", distance=1.0)[0]] = True
    ground = float(shapely.area(lots).sum())
    scale = min(1.0, max_share * len(refs) / total)
    uses = np.full(len(refs), "", dtype=object)
    for name in sorted(counts):
        k = int(round(counts[name] * scale))
        if k == 0:
            continue
        r = 0.5 * math.sqrt(ground / k)
        cells = np.floor(xy / r).astype(np.int64).tolist()
        pts = xy.tolist()
        grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
        got = 0
        score = on_front + rng.uniform(0.0, 1.0, len(refs))
        for i in np.argsort(-score, kind="stable").tolist():
            if uses[i]:
                continue
            (cx, cy), (x, y) = cells[i], pts[i]
            if any(math.hypot(px - x, py - y) < r
                   for dx in (-1, 0, 1) for dy in (-1, 0, 1) for px, py in grid.get((cx + dx, cy + dy), ())):
                continue
            uses[i] = name
            grid.setdefault((cx, cy), []).append((x, y))
            got += 1
            if got == k:
                break
    for (bi, li), use in zip(refs, uses):
        if use:
            b = blocks[bi]
            if b.uses is None:
                b.uses = np.full(len(b.lots), "", dtype=object)
            b.uses[li] = use
    return int(np.count_nonzero(uses))
