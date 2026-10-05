"""Texel-level settlement siting.

Moves each settlement from its face centroid to the best owned land texel of its
face, scored as a weighted sum of harbour (ocean, else lake), confluence or
river (additive with coast, so a river mouth beats a bare shore), 3×3
fertility, a mild pull to the face centre (keeps cities off seams), and a small
per-``(seed, face)`` jitter (avoids a visible lattice at region scale).

Cities are sited largest first; sites within ``min_spacing`` of an already-sited
city are rejected (if all are, the farthest is taken). ``Settlement.site_kind``
records what the site offered so the region/local tiers can re-site without
losing a harbour or river.
"""
from __future__ import annotations

import numpy as np

from realmgen.placement.types import (
    SITE_KIND_CONFLUENCE, SITE_KIND_LAKE_COAST, SITE_KIND_OCEAN_COAST, SITE_KIND_RIVER,
)
from realmgen.refine.sites import SiteField

SITE_OCEAN = 3.0
SITE_LAKE = 2.0
SITE_CONFLUENCE = 2.0
SITE_RIVER = 1.5
SITE_FERTILITY = 1.0
SITE_CENTRE = 0.5
SITE_JITTER = 0.3


def position_key(pos: np.ndarray) -> tuple[float, float, float]:
    """Hashable key for a unit-sphere position, stable for centroid-derived points."""
    p = np.asarray(pos, dtype=np.float64).ravel()
    return (round(float(p[0]), 9), round(float(p[1]), 9), round(float(p[2]), 9))


def site_kind(sites, i: int) -> int:
    """The ``SITE_KIND_*`` flags of texel ``i`` of one face's sites."""
    return (
        SITE_KIND_OCEAN_COAST * bool(sites.coast_ocean[i]) | SITE_KIND_LAKE_COAST * bool(sites.coast_lake[i])
        | SITE_KIND_RIVER * bool(sites.river[i]) | SITE_KIND_CONFLUENCE * bool(sites.confluence[i])
    )


def city_site_scores(sites, idx: np.ndarray, centroid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Weighted site score of the texels ``idx`` of one face (higher is better)."""
    ocean = sites.coast_ocean[idx]
    lake = sites.coast_lake[idx] & ~ocean
    conf = sites.confluence[idx]
    river = sites.river[idx] & ~conf
    d_centre = np.linalg.norm(sites.positions[idx] - centroid[None, :], axis=1)
    radius = float(d_centre.max()) if d_centre.size else 0.0
    centre = d_centre / radius if radius > 0 else np.zeros_like(d_centre)
    return (
        SITE_OCEAN * ocean + SITE_LAKE * lake
        + SITE_CONFLUENCE * conf + SITE_RIVER * river
        + SITE_FERTILITY * sites.fertility[idx]
        - SITE_CENTRE * centre
        + SITE_JITTER * rng.random(idx.size)
    )


def refine_settlement_positions(
    settlements,
    *,
    sites: SiteField,
    min_spacing: float = 0.0,
    seed: int = 0,
) -> dict[tuple[float, float, float], np.ndarray]:
    """Site each settlement on its best in-face texel; mutate ``position`` in place.

    ``min_spacing`` is the minimum chord distance between two cities. Returns
    ``{old_position_key: new_position}`` for settlements that moved.
    """
    moved: dict[tuple[float, float, float], np.ndarray] = {}
    if not settlements:
        return moved

    sites.prefetch(s.face_id for s in settlements if 0 <= int(s.face_id) < sites.n)
    order = sorted(range(len(settlements)), key=lambda k: (-float(settlements[k].population), k))
    placed: list[np.ndarray] = []

    for k in order:
        s = settlements[k]
        f = int(s.face_id)
        if f < 0 or f >= sites.n:
            continue
        face = sites.face(f)
        idx = face.sites()
        if idx.size == 0:
            continue  # all-water texel grid; leave at centroid

        rng = np.random.default_rng([int(seed), f])
        score = city_site_scores(face, idx, sites.centroids[f], rng)
        if placed and min_spacing > 0:
            others = np.stack(placed)
            gap = np.linalg.norm(
                face.positions[idx][:, None, :] - others[None, :, :], axis=2,
            ).min(axis=1)
            ok = gap >= min_spacing
            # Every site is crowded: take the least-crowded one instead.
            score = np.where(ok, score, -np.inf) if ok.any() else gap
        best = idx[int(np.argmax(score))]
        new_pos = face.positions[best].copy()
        placed.append(new_pos)

        if not np.array_equal(new_pos, np.asarray(s.position, dtype=np.float64)):
            moved[position_key(s.position)] = new_pos
        s.set_site(new_pos, site_kind(face, int(best)))

    return moved


def remap_routes(routes, moved: dict) -> list:
    """Copy of trade routes with endpoints moved to refined settlement positions."""
    if not routes:
        return routes
    out = []
    for start_pos, end_pos, flow in routes:
        new_start = moved.get(position_key(start_pos), start_pos)
        new_end = moved.get(position_key(end_pos), end_pos)
        out.append((np.asarray(new_start), np.asarray(new_end), flow))
    return out
