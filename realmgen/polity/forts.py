"""Defensive-only sites — border forts, strait forts and frontier outposts.

Placed by :class:`~realmgen.polity.builder.PolityStage` on its first-pass
sovereign partition, then fed back: a fort makes its crossing dearer for
authority projection (:func:`fortify_land_costs`, :func:`fortify_landings`) and
the forest is grown again, so fortified frontiers tend to hold.

- **Border fort**: on the hot side of a busy, permeable, evenly matched land
  crossing between sovereigns (the :func:`~realmgen.polity.tension.border_crossings`
  score, × terrain ruggedness).
- **Strait fort**: on a shore of a sea passage — ocean whose removal splits the
  nearby sea in two (:func:`find_passages`) — weighted by the lane's trade, its
  narrowness and whether its shores belong to different sovereigns.
- **Frontier outpost**: on a settled face beside land too dangerous to settle.

Each holds a small garrison carved out of its face's farmers (``fort_population``).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from core.params import param_field as _f, paramclass

FORT_BORDER, FORT_STRAIT, FORT_FRONTIER = range(3)
FORT_KIND_NAMES = ("Border fort", "Strait fort", "Frontier outpost")
FORT_KIND_COLORS: list[tuple[float, float, float, float]] = [
    (0.78, 0.22, 0.20, 1.0),  # brick red  — border
    (0.25, 0.40, 0.75, 1.0),  # navy       — strait
    (0.55, 0.62, 0.25, 1.0),  # olive      — frontier
]


@paramclass
class FortParams:
    """Defensive sites and how hard they make their crossings (consumed by PolityStage)."""
    enabled: bool = _f(True, label="Place forts (and re-grow polities around them)", min=0, max=1)
    border_min_score: float = _f(0.12, label="Border fort: min crossing score", min=0.0, max=1.0)
    rugged_weight: float = _f(0.5, label="Border fort: ruggedness bonus", min=0.0, max=3.0)
    strait_min_score: float = _f(0.2, label="Strait fort: min score", min=0.0, max=1.0)
    # A passage of cut radius r is about 2r + 1 faces (~220 km each) across.
    strait_max_radius: int = _f(2, label="Strait fort: max passage cut radius (faces)", min=0, max=3)
    strait_uncontested_weight: float = _f(
        0.5, label="Strait fort: weight when one sovereign holds every shore", min=0.0, max=1.0)
    frontier_min_score: float = _f(0.2, label="Frontier outpost: min score", min=0.0, max=1.0)
    spacing_hops: int = _f(2, label="Min face hops between forts", min=1, max=6)
    resistance: float = _f(4.0, label="Fortified crossing extra cost", min=0.0, max=50.0)
    border_garrison: float = _f(300.0, label="Border fort garrison (people)", min=0.0, max=5_000.0, step=10.0)
    strait_garrison: float = _f(250.0, label="Strait fort garrison (people)", min=0.0, max=5_000.0, step=10.0)
    frontier_garrison: float = _f(120.0, label="Frontier outpost garrison (people)", min=0.0, max=5_000.0, step=10.0)


@dataclass
class Forts:
    """Every fort on the globe, as parallel arrays (``F`` forts).

    ``facing`` is the face it watches: across the border, across the strait, or
    the dangerous neighbour.
    """

    positions: np.ndarray   # (F, 3) unit-sphere
    face_id: np.ndarray     # (F,) owning global face
    facing: np.ndarray      # (F,) watched face
    kind: np.ndarray        # (F,) FORT_* code
    score: np.ndarray       # (F,) placement score
    garrison: np.ndarray    # (F,) people

    def __len__(self) -> int:
        return int(self.face_id.shape[0])

    @classmethod
    def empty(cls) -> "Forts":
        return cls(
            positions=np.zeros((0, 3), dtype=np.float64),
            face_id=np.zeros(0, dtype=np.int32),
            facing=np.zeros(0, dtype=np.int32),
            kind=np.zeros(0, dtype=np.int8),
            score=np.zeros(0, dtype=np.float64),
            garrison=np.zeros(0, dtype=np.float64),
        )

    def pack(self) -> Dict[str, np.ndarray]:
        return {
            "fort_position": self.positions.astype(np.float32),
            "fort_face_id": self.face_id.astype(np.int32),
            "fort_facing": self.facing.astype(np.int32),
            "fort_kind": self.kind.astype(np.int8),
            "fort_score": self.score.astype(np.float64),
            "fort_garrison": self.garrison.astype(np.float64),
        }

    @classmethod
    def unpack(cls, arrays: Mapping[str, np.ndarray]) -> "Forts":
        return cls(
            positions=np.asarray(arrays["fort_position"], dtype=np.float64).reshape(-1, 3),
            face_id=np.asarray(arrays["fort_face_id"], dtype=np.int32),
            facing=np.asarray(arrays["fort_facing"], dtype=np.int32),
            kind=np.asarray(arrays["fort_kind"], dtype=np.int8),
            score=np.asarray(arrays["fort_score"], dtype=np.float64),
            garrison=np.asarray(arrays["fort_garrison"], dtype=np.float64),
        )


@dataclass
class FortSite:
    """A chosen fort before siting: ``face`` watches ``facing``; ``via`` is the
    face its position leans toward (the first water face for a strait fort)."""
    face: int
    facing: int
    via: int
    kind: int
    score: float


# ── Candidates ────────────────────────────────────────────────────────────────


def border_candidates(
    src: np.ndarray, dst: np.ndarray, crossing: np.ndarray,
    rugged_norm: np.ndarray, *, rugged_weight: float, min_score: float,
) -> List[FortSite]:
    """Border crossings (from :func:`border_crossings`) on the defending face ``src``,
    scored ``crossing × (1 + rugged_weight · rugged[src]) / (1 + rugged_weight)``."""
    score = crossing * (1.0 + rugged_weight * rugged_norm[src]) / (1.0 + rugged_weight)
    keep = np.flatnonzero(score >= min_score)
    return [FortSite(int(src[k]), int(dst[k]), int(dst[k]), FORT_BORDER, float(score[k])) for k in keep]


# How far (sea hops) a passage test looks, and how much open sea each side must hold.
_PASSAGE_REACH = 8
_PASSAGE_MIN_OPEN = 3


def find_passages(
    face_adjacency: Sequence[Sequence[int]], sea: np.ndarray, *, max_radius: int,
    reach: int = _PASSAGE_REACH, min_open: int = _PASSAGE_MIN_OPEN,
) -> np.ndarray:
    """``(N,)`` int: per coastal sea face, the smallest cut radius ``r ≤ max_radius``
    at which it is a passage, else ``-1``.

    A face is a passage at ``r`` when removing the sea within ``r`` hops of it splits
    the rest of its ``reach``-hop sea neighbourhood into ≥ 2 pieces each holding ≥
    ``min_open`` open-sea faces (sea all round). A sea that wraps round an island, a
    bay with no far side, or a one-face-wide arm of the sea into the land isn't one.
    """
    sea = np.asarray(sea, dtype=bool)
    open_sea = np.array([sea[f] and all(sea[g] for g in adj) for f, adj in enumerate(face_adjacency)])
    out = np.full(len(face_adjacency), -1, dtype=np.int64)
    for w in np.flatnonzero(sea & ~open_sea).tolist():   # the narrows are coastal
        dist = {w: 0}
        q = deque([w])
        while q:
            f = q.popleft()
            if dist[f] == reach:
                continue
            for g in face_adjacency[f]:
                if sea[g] and g not in dist:
                    dist[g] = dist[f] + 1
                    q.append(g)
        for r in range(int(max_radius) + 1):
            rest = {f for f, d in dist.items() if d > r}
            seen: set = set()
            sides = 0
            for start in rest:
                if start in seen:
                    continue
                n_open, q = 0, deque([start])
                seen.add(start)
                while q:
                    f = q.popleft()
                    n_open += int(open_sea[f])
                    for g in face_adjacency[f]:
                        if g in rest and g not in seen:
                            seen.add(g)
                            q.append(g)
                sides += n_open >= min_open
            if sides >= 2:
                out[w] = r
                break
    return out


def strait_candidates(
    face_adjacency: Sequence[Sequence[int]],
    sovereign_of_face: np.ndarray,
    is_water: np.ndarray,
    passage: np.ndarray,
    norm_flow: np.ndarray,
    *, flow_ref: float, artery_weight: float, uncontested_weight: float, min_score: float,
) -> List[FortSite]:
    """Claimed shores at the key face of each strait (a connected run of ``passage``
    faces): its narrowest, then busiest, face with a claimed shore — the narrows a
    fort commands, not the whole channel.

    ``score = artery(peak flow in the strait) × narrowness × contest``: narrowness
    ``1 / (1 + r/2)`` at the key face's cut radius ``r``; contest 1 when the strait's
    shores hold ≥ 2 sovereigns, else ``uncontested_weight``. A fort watches the
    nearest key-face shore of another sovereign (else the key face).
    """
    sov = np.asarray(sovereign_of_face)
    water = np.asarray(is_water, dtype=bool)
    flow = np.asarray(norm_flow, dtype=np.float64)
    passage = np.asarray(passage)
    fref = max(float(flow_ref), 1e-9)
    aw = float(np.clip(artery_weight, 0.0, 1.0))

    def claimed(f: int) -> List[int]:
        return [int(g) for g in face_adjacency[f] if not water[g] and int(sov[g]) >= 0]

    out: List[FortSite] = []
    seen: set = set()
    for start in np.flatnonzero(passage >= 0).tolist():
        if start in seen:
            continue
        strait, q = [], deque([start])
        seen.add(start)
        while q:
            f = q.popleft()
            strait.append(f)
            for g in face_adjacency[f]:
                if passage[g] >= 0 and g not in seen:
                    seen.add(g)
                    q.append(g)
        held = [f for f in strait if claimed(f)]
        if not held:
            continue
        holders = {int(sov[g]) for f in held for g in claimed(f)}
        key = min(held, key=lambda f: (int(passage[f]), -float(flow[f]), f))
        shores = claimed(key)
        narrow = 1.0 / (1.0 + 0.5 * float(passage[key]))
        artery = (1.0 - aw) + aw * float(np.tanh(max(flow[f] for f in strait) / fref))
        contest = 1.0 if len(holders) >= 2 else float(uncontested_weight)
        score = artery * narrow * contest
        if score < min_score:
            continue
        for g in sorted(shores):
            others = [h for h in shores if int(sov[h]) != int(sov[g])]
            facing = min(others, key=lambda h: (_hops_apart(face_adjacency, g, h), h)) if others else key
            out.append(FortSite(g, facing, key, FORT_STRAIT, score))
    return out


def _hops_apart(face_adjacency: Sequence[Sequence[int]], a: int, b: int, limit: int = 12) -> int:
    """Face hops from ``a`` to ``b`` (``limit + 1`` if farther)."""
    dist = {a: 0}
    q = deque([a])
    while q:
        f = q.popleft()
        if f == b:
            return dist[f]
        if dist[f] == limit:
            continue
        for g in face_adjacency[f]:
            if g not in dist:
                dist[g] = dist[f] + 1
                q.append(g)
    return limit + 1


def frontier_candidates(
    face_adjacency: Sequence[Sequence[int]],
    sovereign_of_face: np.ndarray,
    is_water: np.ndarray,
    danger: np.ndarray,
    rural: np.ndarray,
    *, danger_threshold: float, min_score: float,
) -> List[FortSite]:
    """Settled faces beside land at or above ``danger_threshold`` (unsettleable).

    ``score = peak neighbouring danger × sqrt(rural / p90 rural)`` (clipped to 1):
    outposts guard people, not empty marches.
    """
    sov = np.asarray(sovereign_of_face)
    water = np.asarray(is_water, dtype=bool)
    d = np.asarray(danger, dtype=np.float64)
    r = np.asarray(rural, dtype=np.float64)
    land_r = r[~water & (r > 0)]
    r_ref = float(np.percentile(land_r, 90.0)) if land_r.size else 1.0
    people = np.sqrt(np.clip(r / max(r_ref, 1e-9), 0.0, 1.0))
    out: List[FortSite] = []
    for i, adj in enumerate(face_adjacency):
        if water[i] or int(sov[i]) < 0 or d[i] >= danger_threshold:
            continue
        wild = [int(j) for j in adj if not water[j] and d[j] >= danger_threshold]
        if not wild:
            continue
        j = max(wild, key=lambda f: (d[f], -f))
        score = float(d[j]) * float(people[i])
        if score >= min_score:
            out.append(FortSite(i, j, j, FORT_FRONTIER, score))
    return out


def select_sites(
    candidates: Sequence[FortSite],
    face_adjacency: Sequence[Sequence[int]],
    *, spacing_hops: int, excluded: Optional[set] = None,
    blocked: Optional[np.ndarray] = None,
) -> List[FortSite]:
    """Greedy best-first, no two forts within ``spacing_hops - 1`` hops of each other.

    ``blocked`` (``(N,)`` bool, updated in place) carries the exclusion zones of
    forts already chosen; ``excluded`` faces (cities) never take one.
    """
    n = len(face_adjacency)
    blocked = np.zeros(n, dtype=bool) if blocked is None else blocked
    excluded = excluded or set()
    order = sorted(candidates, key=lambda c: (-c.score, c.face, c.facing))
    chosen: List[FortSite] = []
    for c in order:
        if blocked[c.face] or c.face in excluded:
            continue
        chosen.append(c)
        ring = {c.face}
        blocked[c.face] = True
        for _ in range(int(spacing_hops) - 1):
            ring = {int(j) for f in ring for j in face_adjacency[f]} - ring
            blocked[list(ring)] = True
    return chosen


# ── Siting ────────────────────────────────────────────────────────────────────

# How far from its face centroid toward ``via`` a fort's target point lies.
_LEAN = {FORT_BORDER: 0.35, FORT_STRAIT: 0.5, FORT_FRONTIER: 0.35}
# Hill search radius about the target, × the face's mean centroid-to-neighbour distance.
_HILL_RADIUS = 0.3


def site_forts(
    sites: Sequence[FortSite],
    centroids: np.ndarray,
    texel_positions: np.ndarray,
    land: np.ndarray,
    owned: np.ndarray,
    elevation: Optional[np.ndarray],
) -> np.ndarray:
    """``(F, 3)`` positions on owned land texels.

    The texel arrays are per site: ``texel_positions (F, H, W, 3)``, ``land``,
    ``owned`` (texel's nearest centroid is the site's face) and ``elevation``
    ``(F, H, W)``. Each fort leans from its centroid toward ``via``; a strait fort
    takes the nearest land texel to that point (the shore), others the highest
    candidate within a hill radius of it (else the nearest). No land: the point.
    """
    c = np.asarray(centroids, dtype=np.float64)
    out = np.zeros((len(sites), 3), dtype=np.float64)
    for k, s in enumerate(sites):
        ci, cv = c[s.face], c[s.via]
        target = ci + _LEAN[s.kind] * (cv - ci)
        target /= np.linalg.norm(target)
        cand = land[k] & owned[k]
        if not cand.any():
            out[k] = target
            continue
        pts = texel_positions[k][cand]
        dist = np.linalg.norm(pts - target, axis=1)
        pick = int(np.argmin(dist))
        if s.kind != FORT_STRAIT and elevation is not None:
            radius = _HILL_RADIUS * float(np.linalg.norm(cv - ci))
            near = dist <= max(radius, float(dist[pick]))
            pick = int(np.argmax(np.where(near, elevation[k][cand], -np.inf)))
        p = pts[pick]
        out[k] = p / np.linalg.norm(p)
    return out


# ── Garrisons and resistance ──────────────────────────────────────────────────


def garrisons(
    sites: Sequence[FortSite], farmers: np.ndarray, base: Mapping[int, float],
) -> np.ndarray:
    """``base[kind] × (0.5 + score / kind's top score)``, never more than the face's
    farmers left after earlier forts on it."""
    left = np.maximum(np.asarray(farmers, dtype=np.float64).copy(), 0.0)
    top: Dict[int, float] = {}
    for s in sites:
        top[s.kind] = max(top.get(s.kind, 0.0), s.score)
    out = np.zeros(len(sites), dtype=np.float64)
    for k, s in enumerate(sites):
        want = float(base.get(s.kind, 0.0)) * (0.5 + s.score / max(top[s.kind], 1e-12))
        out[k] = min(want, float(left[s.face]))
        left[s.face] -= out[k]
    return out


def fortify_land_costs(
    face_adjacency: Sequence[Sequence[int]],
    land_edge_costs: Sequence[np.ndarray],
    sovereign_of_face: np.ndarray,
    fort_faces: Sequence[int],
    resistance: float,
) -> List[np.ndarray]:
    """Copy of ``land_edge_costs`` with ``resistance`` added, both ways, to every edge
    from a fort face into another sovereign's (or unclaimed) land."""
    out = [np.array(c, dtype=np.float64, copy=True) for c in land_edge_costs]
    sov = np.asarray(sovereign_of_face)
    for f in fort_faces:
        f = int(f)
        for idx, g in enumerate(face_adjacency[f]):
            g = int(g)
            if int(sov[g]) == int(sov[f]):
                continue
            out[f][idx] += resistance
            back = list(face_adjacency[g]).index(f)
            out[g][back] += resistance
    return out


def fortify_landings(graph: csr_matrix, fort_faces: Sequence[int], n: int, resistance: float) -> csr_matrix:
    """Hybrid ``(2N, 2N)`` graph with ``resistance`` added to every land↔water
    transition at the fort faces (land nodes ``< n``, sea nodes ``≥ n``)."""
    if len(fort_faces) == 0 or resistance <= 0.0:
        return graph
    coo = graph.tocoo(copy=True)
    forts = np.zeros(2 * n, dtype=bool)
    forts[np.asarray(fort_faces, dtype=np.int64)] = True
    hit = (forts[coo.row] & (coo.col >= n)) | (forts[coo.col] & (coo.row >= n))
    coo.data[hit] += resistance
    return csr_matrix((coo.data, (coo.row, coo.col)), shape=(2 * n, 2 * n))
