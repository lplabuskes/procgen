"""Districts per block: market, quay, elite, craft quarters by industry, graveyards, squares, houses."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from settlegen.layout.params import SettlementLayoutParams
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


# Districts with no lots.
OPEN_DISTRICTS = ("graveyard", "square")


# Districts whose deep blocks are opened up by closes and alleys.
LANE_DISTRICTS = ("residential", "suburb", *sorted({v[0] for v in CRAFTS.values()}))


DISTRICTS = ("market", "quay", "elite", "residential", "suburb", "gardens", "graveyard", "square",
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
    # Small squares in the core.
    n_squares = 0 if town or village else int(round(profile.population / float(params.square_people)))
    areas = shapely.area(geoms)
    cand = np.flatnonzero(free & in_core & (areas <= np.quantile(areas, 0.3)))
    if n_squares and cand.size:
        take(rng.choice(cand, size=min(n_squares, cand.size), replace=False), "square")
    rest = np.flatnonzero(free)
    outside = ~in_core[rest] & walled
    district[rest] = np.where(in_sparse[rest], "gardens", np.where(outside, "suburb", "residential"))
    return [str(d) for d in district]
