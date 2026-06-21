"""Continuous border polylines traced directly on the texel polity classification.

The polity *fill* (``fill.py``) classifies every texel to a polity via an
authority-weighted Voronoi.  The borders must be the edges *of that texel
classification* — otherwise the lines drift off the fill they are meant to bound,
and stop short of the texel coastline.  This module traces them from the same
``(N, H, W)`` label grid:

1. For each hierarchy depth (depth 0 = sovereign borders, including polity vs.
   unclaimed *land*; depth d > 0 = internal subtree borders), an edge between two
   adjacent texels is a border at the **shallowest** depth where their depth-d
   ancestors differ.  Coastlines (land vs. water) are *not* borders.
2. Within each face, the crossings around a 2×2 texel cell are joined
   (marching-squares style): two crossings → a straight link between their edge
   midpoints; otherwise each crossing links to the cell centre (handles junctions
   and coastal terminations).  Crossing midpoints on a shared mesh edge are the
   **same 3D point in both faces** (``face_texture_positions`` makes shared-edge
   texels coincident), so the per-face pieces chain into lines that are continuous
   across face seams.
3. The chained polylines are Chaikin-smoothed to round off the texel staircase.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List

import numpy as np

from realmgen.refine.fill import WATER
from realmgen.refine.settlements import position_key


def _chain_edges(edges: set) -> List[list]:
    """Chain undirected ``(a, b)`` edges into vertex-key polylines.

    Walks stop at junction/endpoint vertices (degree != 2), so a closed loop
    comes back as ``[v, ..., v]`` and shared junctions appear in every incident
    chain — keeping the pieces joined.
    """
    adj: dict = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    visited: set = set()

    def ekey(a, b):
        return (a, b) if a < b else (b, a)

    def walk(start, nxt) -> list:
        chain = [start]
        prev, cur = start, nxt
        while True:
            chain.append(cur)
            visited.add(ekey(prev, cur))
            if len(adj[cur]) != 2:
                break
            nbrs = [w for w in adj[cur] if ekey(cur, w) not in visited]
            if not nbrs:
                break
            prev, cur = cur, nbrs[0]
        return chain

    chains: List[list] = []
    # Open chains: anchor at junctions / endpoints first.
    for v in [v for v in adj if len(adj[v]) != 2]:
        for w in adj[v]:
            if ekey(v, w) not in visited:
                chains.append(walk(v, w))
    # Pure cycles with no junction: start anywhere on each remaining edge.
    for a, b in edges:
        if ekey(a, b) not in visited:
            chains.append(walk(a, b))
    return chains


def _chaikin(points: List[np.ndarray], iterations: int, closed: bool) -> List[np.ndarray]:
    """Chaikin corner-cutting.  Open chains keep their endpoints fixed (so chains
    sharing a junction vertex stay joined); closed loops cut every corner."""
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
    """``lut[d, polity_id]`` = depth-d ancestor id (clamped to the leaf).

    Rows cover ``0..max_depth``; columns cover ``0..max_polity_id``.  Unused
    polity-id columns stay -1.
    """
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

    Coastlines (either side ``WATER``) are never borders.  Polity-vs-unclaimed
    land is a depth-0 border; polity-vs-polity is the shallowest depth where
    their ancestor chains diverge.
    """
    shape = la.shape
    fa = la.ravel()
    fb = lb.ravel()
    out = np.full(fa.shape, -1, dtype=np.int64)

    nonwater = (fa != WATER) & (fb != WATER)
    diff = (fa != fb) & nonwater
    # polity vs unclaimed land → depth 0
    out[diff & ((fa < 0) | (fb < 0))] = 0
    # polity vs polity → shallowest diverging depth
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


# ── tracing ───────────────────────────────────────────────────────────────────


def trace_label_borders(positions: np.ndarray, labels: np.ndarray,
                        polities: List, *, iterations: int = 2) -> dict:
    """Return ``{depth: (M, 6)}`` smoothed border segments from a texel label grid.

    ``positions`` is ``(N, H, W, 3)`` unit-sphere texel positions; ``labels`` is
    the matching ``(N, H, W)`` per-texel polity classification.
    """
    if not polities:
        return {}
    labels = np.asarray(labels)
    P = np.asarray(positions, dtype=np.float64)
    lut, max_depth = _ancestor_lut(polities)

    # Per-edge border depth.  H-edges link (i, j)|(i, j+1); V-edges (i, j)|(i+1, j).
    hbd = _edge_border_depth(labels[:, :, :-1], labels[:, :, 1:], lut)   # (N,H,W-1)
    vbd = _edge_border_depth(labels[:, :-1, :], labels[:, 1:, :], lut)   # (N,H-1,W)

    # Edge midpoints and cell centres (shared-edge midpoints coincide across faces).
    hmid = 0.5 * (P[:, :, :-1, :] + P[:, :, 1:, :])                      # (N,H,W-1,3)
    vmid = 0.5 * (P[:, :-1, :, :] + P[:, 1:, :, :])                      # (N,H-1,W,3)
    ctr = 0.25 * (P[:, :-1, :-1, :] + P[:, :-1, 1:, :]
                  + P[:, 1:, :-1, :] + P[:, 1:, 1:, :])                  # (N,H-1,W-1,3)

    out: dict[int, np.ndarray] = {}
    for depth in range(max_depth + 1):
        segs = _trace_depth(depth, hbd, vbd, hmid, vmid, ctr, iterations)
        if segs:
            out[depth] = np.asarray(segs, dtype=np.float32)
    return out


def _trace_depth(depth: int, hbd, vbd, hmid, vmid, ctr, iterations: int) -> list:
    # Per-cell crossing flags for the four cell edges (cells span i,j in
    # 0..H-2, 0..W-2).
    top = hbd[:, :-1, :] == depth                       # (N,H-1,W-1)
    bot = hbd[:, 1:, :] == depth
    left = vbd[:, :, :-1] == depth
    right = vbd[:, :, 1:] == depth
    cnt = top.astype(np.int8) + bot + left + right
    cells = np.argwhere(cnt > 0)
    if cells.size == 0:
        return []

    m_top = hmid[:, :-1, :, :]
    m_bot = hmid[:, 1:, :, :]
    m_left = vmid[:, :, :-1, :]
    m_right = vmid[:, :, 1:, :]

    edges: set = set()
    pt_of: dict = {}

    def add(p, q):
        kp, kq = position_key(p), position_key(q)
        if kp == kq:
            return
        pt_of[kp] = p
        pt_of[kq] = q
        edges.add((kp, kq) if kp < kq else (kq, kp))

    for f, i, j in cells:
        mids = []
        if top[f, i, j]:
            mids.append(m_top[f, i, j])
        if bot[f, i, j]:
            mids.append(m_bot[f, i, j])
        if left[f, i, j]:
            mids.append(m_left[f, i, j])
        if right[f, i, j]:
            mids.append(m_right[f, i, j])
        if len(mids) == 2:
            add(mids[0], mids[1])
        else:
            c = ctr[f, i, j]
            for m in mids:
                add(m, c)

    segs: list = []
    for chain in _chain_edges(edges):
        closed = len(chain) > 3 and chain[0] == chain[-1]
        keys = chain[:-1] if closed else chain
        if len(keys) < 2:
            continue
        pts = [pt_of[k] for k in keys]
        sm = _chaikin(pts, iterations, closed)
        sm = [p / np.linalg.norm(p) for p in sm]
        n = len(sm)
        rng = range(n) if closed else range(n - 1)
        for i in rng:
            segs.append(np.concatenate([sm[i], sm[(i + 1) % n]]))
    return segs


def smooth_polity_borders(mesh, polities: List, labels: np.ndarray,
                          *, iterations: int = 2) -> dict:
    """Return ``{depth: (M, 6)}`` continuous, Chaikin-smoothed border segments
    traced on the texel ``labels`` grid (see :func:`trace_label_borders`)."""
    if not polities:
        return {}
    labels = np.asarray(labels)
    _, H, W = labels.shape
    positions = np.asarray(mesh.face_texture_positions(H, W), dtype=np.float64)
    return trace_label_borders(positions, labels, polities, iterations=iterations)
