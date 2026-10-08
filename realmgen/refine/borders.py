"""Continuous polity border polylines, contoured on a seamless global lattice.

Contours the same position-only ``fill.PolityLabelField`` the fill texture
samples, so the lines bound the fill's regions. Face textures can't be contoured
directly (they over-hang their hexagons and their grids don't line up across
seams), so the field is sampled on a seamless fine icosphere lattice at about
texel spacing:

1. A lattice edge with differing non-water end labels is a crossing at the
   shallowest depth where their ancestors differ (depth 0 includes polity vs
   unclaimed). Land against water is coast, not border.
2. Each crossing is bisected against the label field onto the true boundary.
3. Marching triangles: two crossings link to each other; one (coast) or three
   (junction) link to the triangle centre. No ambiguous cases, so pieces chain
   into continuous lines without welding.
4. Polylines are Chaikin-smoothed with junctions held fixed.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Callable, List, Optional

import numpy as np

from realmgen.refine.fill import WATER, PolityLabelField

# Angle subtended by an icosahedron edge; each subdivision halves it.
_ICOSA_EDGE_RAD = 1.1071487177940904


def _chain_edges(edges, stops=frozenset()) -> List[list]:
    """Chain undirected ``(a, b)`` edges into vertex polylines.

    Walks stop at vertices of degree != 2 and in ``stops``; a closed loop comes
    back as ``[v, ..., v]`` and junctions appear in every incident chain.
    """
    adj: dict = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    visited: set = set()

    def ekey(a, b):
        return (a, b) if a < b else (b, a)

    def is_stop(v) -> bool:
        return len(adj[v]) != 2 or v in stops

    def walk(start, nxt) -> list:
        chain = [start]
        prev, cur = start, nxt
        while True:
            chain.append(cur)
            visited.add(ekey(prev, cur))
            if is_stop(cur):
                break
            nbrs = [w for w in adj[cur] if ekey(cur, w) not in visited]
            if not nbrs:
                break
            prev, cur = cur, nbrs[0]
        return chain

    chains: List[list] = []
    for v in [v for v in adj if is_stop(v)]:
        for w in adj[v]:
            if ekey(v, w) not in visited:
                chains.append(walk(v, w))
    # Remaining edges are pure cycles.
    for a, b in edges:
        if ekey(a, b) not in visited:
            chains.append(walk(a, b))
    return chains


def _chaikin(points: List[np.ndarray], iterations: int, closed: bool) -> List[np.ndarray]:
    """Chaikin corner-cutting; open chains keep their endpoints fixed so junctions stay joined."""
    pts = points
    for _ in range(max(iterations, 0)):
        if len(pts) < 3:
            break
        new: List[np.ndarray] = []
        if closed:
            n = len(pts)
            for i in range(n):
                p, q = pts[i], pts[(i + 1) % n]
                new.append(0.75 * p + 0.25 * q)
                new.append(0.25 * p + 0.75 * q)
        else:
            new.append(pts[0])
            for i in range(len(pts) - 1):
                p, q = pts[i], pts[i + 1]
                new.append(0.75 * p + 0.25 * q)
                new.append(0.25 * p + 0.75 * q)
            new.append(pts[-1])
        pts = new
    return pts


# ── ancestor lookup ───────────────────────────────────────────────────────────


def _ancestor_lut(polities: List) -> tuple[np.ndarray, int]:
    """``(lut, max_depth)``; ``lut[d, polity_id]`` = depth-d ancestor (clamped to the leaf), -1 if unused."""
    pol_by_id = {p.polity_id: p for p in polities}
    max_depth = max(p.depth for p in polities)
    max_pid = max(p.polity_id for p in polities)
    lut = np.full((max_depth + 1, max_pid + 1), -1, dtype=np.int64)
    for p in polities:
        chain: list[int] = []
        cur = p
        while cur is not None:
            chain.append(cur.polity_id)
            cur = pol_by_id.get(cur.liege) if cur.liege is not None else None
        chain.reverse()
        for d in range(max_depth + 1):
            lut[d, p.polity_id] = chain[min(d, len(chain) - 1)]
    return lut, max_depth


def _edge_border_depth(la: np.ndarray, lb: np.ndarray,
                       lut: np.ndarray) -> np.ndarray:
    """Per-edge shallowest border depth (-1 = not a border) for label arrays.

    Coastlines are never borders; polity-vs-unclaimed land is depth 0.
    """
    shape = la.shape
    fa = la.ravel()
    fb = lb.ravel()
    out = np.full(fa.shape, -1, dtype=np.int64)

    nonwater = (fa != WATER) & (fb != WATER)
    diff = (fa != fb) & nonwater
    out[diff & ((fa < 0) | (fb < 0))] = 0
    both = diff & (fa >= 0) & (fb >= 0)
    if np.any(both):
        a = fa[both]
        b = fb[both]
        anc_a = lut[:, a]                 # (D, k)
        anc_b = lut[:, b]
        differ = anc_a != anc_b           # (D, k)
        any_d = differ.any(axis=0)
        first_d = np.argmax(differ, axis=0)
        out[both] = np.where(any_d, first_d, -1)
    return out.reshape(shape)


# ── lattice ───────────────────────────────────────────────────────────────────


def _unique_edges(tris: np.ndarray, n_verts: int) -> tuple[np.ndarray, np.ndarray]:
    """``(edges (E, 2), tri_edges (F, 3))`` for a triangle array.

    ``tri_edges[f, k]`` indexes the edge from corner ``k`` to corner ``(k + 1) % 3``.
    """
    a = tris
    b = np.roll(tris, -1, axis=1)
    lo = np.minimum(a, b).astype(np.int64)
    hi = np.maximum(a, b).astype(np.int64)
    keys = (lo * n_verts + hi).ravel()
    uniq, inverse = np.unique(keys, return_inverse=True)
    edges = np.stack([uniq // n_verts, uniq % n_verts], axis=1)
    return edges, inverse.reshape(tris.shape)


@lru_cache(maxsize=2)
def _icosphere_lattice(depth: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(verts, tris, edges, tri_edges)`` of an icosphere subdivided ``depth`` times.

    Vectorised because ``Mesh.icosphere`` is far too slow at ~10⁶ vertices.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    tris = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    for _ in range(depth):
        n = len(verts)
        edges, te = _unique_edges(tris, n)
        mids = verts[edges[:, 0]] + verts[edges[:, 1]]
        mids /= np.linalg.norm(mids, axis=1, keepdims=True)
        m = te + n                     # midpoint vertex of edge (k, k+1)
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        mab, mbc, mca = m[:, 0], m[:, 1], m[:, 2]
        tris = np.concatenate([
            np.stack([a, mab, mca], axis=1),
            np.stack([mab, b, mbc], axis=1),
            np.stack([mca, mbc, c], axis=1),
            np.stack([mab, mbc, mca], axis=1),
        ])
        verts = np.vstack([verts, mids])
    edges, te = _unique_edges(tris, len(verts))
    return verts, tris, edges, te


def lattice_depth_for(n_faces: int, tex_size: int) -> int:
    """Icosphere depth whose edge length is no longer than a face texel."""
    face_width = np.sqrt(4.0 * np.pi / max(n_faces, 1))
    spacing = face_width / max(tex_size - 1, 1)
    return max(0, int(np.ceil(np.log2(_ICOSA_EDGE_RAD / spacing))))


# ── tracing ───────────────────────────────────────────────────────────────────


def _bisect_crossings(verts: np.ndarray, edges: np.ndarray, la: np.ndarray,
                      lb: np.ndarray, label_fn: Optional[Callable], steps: int) -> np.ndarray:
    """Point on each edge where the label turns from ``la`` into ``lb``.

    Bisects against ``label_fn`` (``(P, 3) → (P,)``); an edge whose midpoint hits a
    third label keeps its bracket. Without ``label_fn`` the crossing is the midpoint.
    """
    p0 = verts[edges[:, 0]]
    p1 = verts[edges[:, 1]]
    lo = np.zeros(len(edges))
    hi = np.ones(len(edges))
    if label_fn is not None:
        active = np.ones(len(edges), dtype=bool)
        for _ in range(steps):
            idx = np.flatnonzero(active)
            if idx.size == 0:
                break
            t = 0.5 * (lo[idx] + hi[idx])
            pts = (1.0 - t)[:, None] * p0[idx] + t[:, None] * p1[idx]
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            lab = np.asarray(label_fn(pts))
            is_a = lab == la[idx]
            is_b = lab == lb[idx]
            lo[idx[is_a]] = t[is_a]
            hi[idx[is_b]] = t[is_b]
            active[idx[~(is_a | is_b)]] = False
    t = 0.5 * (lo + hi)
    pts = (1.0 - t)[:, None] * p0 + t[:, None] * p1
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _lattice_links(verts: np.ndarray, tris: np.ndarray, edges: np.ndarray, tri_edges: np.ndarray,
                   labels: np.ndarray, lut: np.ndarray, label_fn: Optional[Callable], bisect_steps: int):
    """Marching-triangle links: ``(points, ids (K, 2), depth (K,), junctions, pair (P, 2))`` or None.

    ``pair`` holds a crossing point's two labels (``-2`` for triangle-centre hubs).
    """
    labels = np.asarray(labels, dtype=np.int64)
    la = labels[edges[:, 0]]
    lb = labels[edges[:, 1]]
    ebd = _edge_border_depth(la, lb, lut)                    # (E,) -1 = no crossing
    cross = np.flatnonzero(ebd >= 0)
    if cross.size == 0:
        return None
    cross_pts = _bisect_crossings(verts, edges[cross], la[cross], lb[cross],
                                  label_fn, bisect_steps)
    point_id = np.full(len(edges), -1, dtype=np.int64)
    point_id[cross] = np.arange(cross.size)

    is_cross = ebd >= 0
    touched = np.flatnonzero(is_cross[tri_edges].any(axis=1))
    te = tri_edges[touched]                                  # (T, 3)
    bd = ebd[te]                                             # (T, 3)
    n_cross = (bd >= 0).sum(axis=1)

    # Two crossings: same border, hence same depth.
    two = n_cross == 2
    te2, bd2 = te[two], bd[two]
    col = np.argsort(bd2 < 0, axis=1, kind="stable")[:, :2]  # the two crossing columns
    rows = np.arange(len(te2))[:, None]
    ids2 = point_id[te2[rows, col]]                          # (T2, 2)
    depth2 = bd2[rows[:, 0], col[:, 0]]

    # One crossing (coast) or three (junction): link each to the triangle centre.
    hub = ~two
    hub_tris = touched[hub]
    centres = verts[tris[hub_tris]].sum(axis=1)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    hub_id = cross.size + np.arange(hub_tris.size)
    te_h, bd_h = te[hub], bd[hub]
    r, c = np.nonzero(bd_h >= 0)
    ids_h = np.stack([point_id[te_h[r, c]], hub_id[r]], axis=1)
    depth_h = bd_h[r, c]
    junctions = frozenset(hub_id[n_cross[hub] == 3].tolist())

    points = np.vstack([cross_pts, centres])
    pair = np.full((len(points), 2), -2, dtype=np.int64)
    pair[:cross.size, 0] = la[cross]
    pair[:cross.size, 1] = lb[cross]
    return (points, np.concatenate([ids2, ids_h]), np.concatenate([depth2, depth_h]), junctions, pair)


def trace_lattice_borders(verts: np.ndarray, tris: np.ndarray, edges: np.ndarray,
                          tri_edges: np.ndarray, labels: np.ndarray, polities: List,
                          *, iterations: int = 2, label_fn: Optional[Callable] = None,
                          bisect_steps: int = 5) -> dict:
    """Return ``{depth: (M, 6)}`` smoothed border segments contoured on a triangle lattice.

    ``labels`` per vertex are polity id, ``UNCLAIMED`` or ``WATER``;
    ``edges``/``tri_edges`` come from ``_unique_edges(tris)``. ``label_fn`` is the
    land label field the labels came from, used to bisect crossings.
    """
    if not polities:
        return {}
    lut, max_depth = _ancestor_lut(polities)
    links = _lattice_links(verts, tris, edges, tri_edges, labels, lut, label_fn, bisect_steps)
    if links is None:
        return {}
    points, all_ids, all_depth, junctions, _ = links

    out: dict[int, np.ndarray] = {}
    for depth in range(max_depth + 1):
        sel = all_ids[all_depth == depth]
        if len(sel) == 0:
            continue
        segs = _smooth_chains(sel, points, junctions, iterations)
        if len(segs):
            out[depth] = segs
    return out


def trace_lattice_chains(verts: np.ndarray, tris: np.ndarray, edges: np.ndarray,
                         tri_edges: np.ndarray, labels: np.ndarray, polities: List,
                         *, label_fn: Optional[Callable] = None, bisect_steps: int = 5) -> list:
    """Unsmoothed border chains ``[(points (K, 3), (a, b), depth)]`` between junctions / coasts.

    ``(a, b)`` is the chain's commonest label pair on its crossings (``a < b``).
    """
    if not polities:
        return []
    lut, max_depth = _ancestor_lut(polities)
    links = _lattice_links(verts, tris, edges, tri_edges, labels, lut, label_fn, bisect_steps)
    if links is None:
        return []
    points, all_ids, all_depth, junctions, pair = links
    out: list = []
    for depth in range(max_depth + 1):
        sel = all_ids[all_depth == depth]
        if len(sel) == 0:
            continue
        edges_d = {(int(a), int(b)) if a < b else (int(b), int(a)) for a, b in sel}
        for chain in _chain_edges(edges_d, stops=junctions):
            if len(chain) < 3:
                continue
            ids = np.asarray(chain, dtype=np.int64)
            pr = pair[ids]
            pr = pr[pr[:, 0] != -2]
            if len(pr) == 0:
                continue
            pr = np.sort(pr, axis=1)
            uniq, count = np.unique(pr, axis=0, return_counts=True)
            a, b = uniq[np.argmax(count)]
            out.append((points[ids], (int(a), int(b)), depth))
    return out


def _lattice_labels(mesh, field: PolityLabelField, ocean_tex: np.ndarray, lake_tex: Optional[np.ndarray],
                    lattice_depth: Optional[int], lattice_offset: int, plain: bool = False):
    """``(verts, tris, edges, tri_edges, labels)`` of the global lattice, water from the masks."""
    from region.sampling import GlobalSampler

    if lattice_depth is None:
        lattice_depth = max(0, lattice_depth_for(len(mesh.faces), int(ocean_tex.shape[2])) + lattice_offset)
    verts, tris, edges, tri_edges = _icosphere_lattice(int(lattice_depth))

    sampler = GlobalSampler(mesh)
    faces, u, v = sampler.locate(verts)
    labels = (field.plain_land_labels if plain else field.land_labels)(verts, owner=faces)
    water = sampler.sample(ocean_tex, faces, u, v)[:, 0] > 0.5
    if lake_tex is not None:
        water |= sampler.sample(lake_tex, faces, u, v)[:, 0] > 0.5
    labels[water] = WATER
    return verts, tris, edges, tri_edges, labels


def _smooth_chains(ids: np.ndarray, points: np.ndarray, junctions: frozenset,
                   iterations: int) -> np.ndarray:
    """Chain ``(K, 2)`` point-id links into polylines, Chaikin-smooth, emit ``(M, 6)``."""
    edges = {(int(a), int(b)) if a < b else (int(b), int(a)) for a, b in ids}
    segs: list = []
    for chain in _chain_edges(edges, stops=junctions):
        closed = len(chain) > 3 and chain[0] == chain[-1] and chain[0] not in junctions
        keys = chain[:-1] if closed else chain
        if len(keys) < 2:
            continue
        pts = _chaikin([points[k] for k in keys], iterations, closed)
        arr = np.asarray(pts)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)
        nxt = np.roll(arr, -1, axis=0) if closed else arr[1:]
        cur = arr if closed else arr[:-1]
        segs.append(np.concatenate([cur, nxt], axis=1))
    if not segs:
        return np.empty((0, 6), dtype=np.float32)
    return np.concatenate(segs).astype(np.float32)


def smooth_polity_borders(mesh, polities: List, field: PolityLabelField,
                          ocean_tex: np.ndarray, lake_tex: Optional[np.ndarray],
                          *, iterations: int = 2, lattice_depth: Optional[int] = None,
                          lattice_offset: int = 0, bisect_steps: int = 5) -> dict:
    """Return ``{depth: (M, 6)}`` continuous, Chaikin-smoothed polity borders.

    Lattice is about texel-fine, shifted by ``lattice_offset`` levels
    (``lattice_depth`` overrides). Water from ``ocean_tex``/``lake_tex`` so
    borders end at the texel coastline.
    """
    if not polities:
        return {}
    verts, tris, edges, tri_edges, labels = _lattice_labels(mesh, field, ocean_tex, lake_tex,
                                                            lattice_depth, lattice_offset)
    return trace_lattice_borders(verts, tris, edges, tri_edges, labels, polities,
                                 iterations=iterations, label_fn=field.land_labels,
                                 bisect_steps=bisect_steps)


def plain_border_chains(mesh, polities: List, field: PolityLabelField,
                        ocean_tex: np.ndarray, lake_tex: Optional[np.ndarray],
                        *, lattice_depth: Optional[int] = None, lattice_offset: int = 0,
                        bisect_steps: int = 5) -> list:
    """Border chains of the field before any river following (``trace_lattice_chains``)."""
    if not polities:
        return []
    verts, tris, edges, tri_edges, labels = _lattice_labels(mesh, field, ocean_tex, lake_tex,
                                                            lattice_depth, lattice_offset, plain=True)
    return trace_lattice_chains(verts, tris, edges, tri_edges, labels, polities,
                                label_fn=field.plain_land_labels, bisect_steps=bisect_steps)
