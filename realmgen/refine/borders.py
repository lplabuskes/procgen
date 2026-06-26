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
2. Each face's 16×16 texture is a *square* that over-hangs its hexagon.  Tracing
   the whole square would draw every shared-edge border twice — once in each
   face's over-hang (a doubled line).  We instead **de-double by Voronoi
   ownership** (``_texel_owner``): each texel belongs to its nearest face centroid,
   and every texel-edge is drawn from only the lower-id of the two face-cells its
   endpoints fall in.  Unlike clipping the over-hang away, this keeps the over-hang
   labels, so a border that runs *along* a seam (which exists only in the
   over-hang) is preserved and drawn exactly once instead of vanishing.
3. Within each face, the crossings around a 2×2 texel cell are joined
   (marching-squares style): two crossings → a straight link between their edge
   midpoints; otherwise each crossing links to the cell centre (handles junctions
   and coastal terminations).  Where a border crosses a seam the two faces' pieces
   meet ~1 texel apart and are joined into continuous lines by welding their
   terminations (``_weld_seam_endpoints``) across mesh-adjacent faces.
4. The chained polylines are Chaikin-smoothed to round off the texel staircase.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List

import numpy as np
from scipy.spatial import KDTree

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


def _texel_owner(P: np.ndarray, centroids: np.ndarray, adj: list,
                 in_poly: np.ndarray | None = None) -> np.ndarray:
    """Per-texel owning face id, used to draw each border edge from only one face.

    Each face's square texture over-hangs its hexagon into neighbouring cells, so
    the *same* boundary gets drawn in several faces' over-hangs (the doubled line).
    We de-duplicate by ownership: a texel inside its own face's rendered polygon
    (``in_poly``) is owned by that face — definitively, matching the exact render
    clip; an over-hang texel (outside the polygon) is owned by the mesh neighbour it
    spilled into, found by nearest centroid. Using the polygon (not a bare
    centroid-Voronoi) is what avoids the thin sliver between the centroid bisector
    and the real hexagon seam where both faces would otherwise draw a short doubled
    stretch (a ``lens`` artifact). The neighbour candidates are ordered by ascending
    face id so the ``argmax`` tie-break resolves to the lowest id, keeping the
    per-edge owner test symmetric across the two faces.
    """
    N, H, W, _ = P.shape
    C = np.asarray(centroids, dtype=np.float64)
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    owner = np.empty((N, H, W), dtype=np.int64)
    flat = P.reshape(N, H * W, 3)
    inside = in_poly.reshape(N, H * W) if in_poly is not None else None
    for f in range(N):
        if inside is None:
            # No polygon mask: nearest centroid among own face and neighbours
            # (a bare centroid-Voronoi — leaves a thin seam sliver, hence lenses).
            cand = np.array(sorted({f, *adj[f]}), dtype=np.int64)
            owner[f] = cand[np.argmax(flat[f] @ C[cand].T, axis=1)].reshape(H, W)
            continue
        own = np.full(H * W, f, dtype=np.int64)
        out = ~inside[f]
        if out.any():
            # Over-hang texels: nearest neighbour centroid (own face excluded — the
            # texel is outside its own polygon, so it belongs to a neighbour).
            nbrs = np.array(sorted(adj[f]), dtype=np.int64)
            own[out] = nbrs[np.argmax(flat[f][out] @ C[nbrs].T, axis=1)]
        owner[f] = own.reshape(H, W)
    return owner


def trace_label_borders(positions: np.ndarray, labels: np.ndarray,
                        polities: List, *, iterations: int = 2,
                        weld_factor: float = 2.5,
                        adj: list | None = None,
                        centroids: np.ndarray | None = None,
                        in_poly: np.ndarray | None = None) -> dict:
    """Return ``{depth: (M, 6)}`` smoothed border segments from a texel label grid.

    ``positions`` is ``(N, H, W, 3)`` unit-sphere texel positions; ``labels`` is
    the matching ``(N, H, W)`` per-texel polity classification.  ``weld_factor``
    (× texel spacing) is the tolerance for joining cross-face dangling endpoints at
    shared mesh edges (``_weld_seam_endpoints``); ``0`` disables welding.  ``adj``
    is the per-face neighbour list (``mesh.face_adjacency``) used both to restrict
    welds to genuinely adjacent faces and to compute texel ownership.

    Each face's 16×16 texture is a *square* that over-hangs its hexagon, so without
    de-duplication a border running along a shared edge gets drawn once in **each**
    face's over-hang — a doubled line.  We instead keep the over-hang labels (so
    seam-running borders are never deleted) and draw each texel-edge from only
    **one** face: the lower-id of the two face-cells its endpoints fall in
    (``_texel_owner``).  ``in_poly`` (``mesh.face_texture_in_polygon``) makes that
    ownership match the exact render clip — a texel inside its own hexagon is owned
    by that face, the over-hang by the neighbour — which avoids the thin sliver
    between the centroid bisector and the real seam where both faces would otherwise
    draw a short doubled stretch; ``centroids`` resolve the over-hang.  The remaining
    ~1-texel seam crossings are joined by the weld.

    A residual artifact survives: where the per-face fill Voronoi disagrees by a
    texel across a seam, a border is drawn as two near-parallel copies (a thin
    ``lens``, ≤ ~2 texels). These are left in place — they read as a faint doubling
    on the line, far less disruptive than the visible gaps that collapsing them used
    to punch into the through-border. The real cure is cross-seam-consistent fill
    labels in ``fill.py``.
    """
    if not polities:
        return {}
    labels = np.asarray(labels)
    P = np.asarray(positions, dtype=np.float64)
    lut, max_depth = _ancestor_lut(polities)

    # Texel spacing → seam-weld tolerance (a crossing pair across a seam lands
    # within ~1–2 texels; see _weld_seam_endpoints).
    spacing = float(np.median(np.linalg.norm(P[:, :, 1:, :] - P[:, :, :-1, :], axis=-1)))
    weld_tol = weld_factor * spacing if weld_factor > 0.0 else 0.0

    # Per-edge border depth.  H-edges link (i, j)|(i, j+1); V-edges (i, j)|(i+1, j).
    hbd = _edge_border_depth(labels[:, :, :-1], labels[:, :, 1:], lut)   # (N,H,W-1)
    vbd = _edge_border_depth(labels[:, :-1, :], labels[:, 1:, :], lut)   # (N,H-1,W)

    # De-double by ownership: draw each edge only from the lower-id of the two
    # face-cells its two texels fall in.  Interior edges (both texels own-face) are
    # always drawn; over-hang copies of a neighbour's interior border are dropped;
    # a seam-running border is kept by exactly one of the two faces.
    frontier = None
    if centroids is not None and adj is not None:
        owner = _texel_owner(P, centroids, adj, in_poly)
        fid = np.arange(labels.shape[0], dtype=np.int64)[:, None, None]
        h_owns = np.minimum(owner[:, :, :-1], owner[:, :, 1:]) == fid     # (N,H,W-1)
        v_owns = np.minimum(owner[:, :-1, :], owner[:, 1:, :]) == fid     # (N,H-1,W)
        hbd = np.where(h_owns, hbd, -1)
        vbd = np.where(v_owns, vbd, -1)
        # A 2×2 cell is on the ownership frontier (a seam) if any of its texels is
        # owned by another face — that is where a border crossing the seam stops on
        # this face and must weld to its neighbour-face counterpart.
        notmine = owner != fid
        frontier = (notmine[:, :-1, :-1] | notmine[:, :-1, 1:]
                    | notmine[:, 1:, :-1] | notmine[:, 1:, 1:])           # (N,H-1,W-1)

    # Edge midpoints and cell centres (shared-edge midpoints coincide across faces).
    hmid = 0.5 * (P[:, :, :-1, :] + P[:, :, 1:, :])                      # (N,H,W-1,3)
    vmid = 0.5 * (P[:, :-1, :, :] + P[:, 1:, :, :])                      # (N,H-1,W,3)
    ctr = 0.25 * (P[:, :-1, :-1, :] + P[:, :-1, 1:, :]
                  + P[:, 1:, :-1, :] + P[:, 1:, 1:, :])                  # (N,H-1,W-1,3)

    # Depth-d ancestor of every texel label (negative water/unclaimed pass
    # through), so the saddle resolver can read the four corner regions of a cell.
    claimed = labels >= 0

    out: dict[int, np.ndarray] = {}
    for depth in range(max_depth + 1):
        anc_d = labels.copy()
        anc_d[claimed] = lut[depth, labels[claimed]]
        segs = _trace_depth(depth, hbd, vbd, hmid, vmid, ctr, anc_d, iterations,
                            weld_tol=weld_tol, adj=adj, frontier=frontier)
        if segs:
            out[depth] = np.asarray(segs, dtype=np.float32)
    return out


def _weld_seam_endpoints(edges: set, pt_of: dict, key_face: dict,
                         tol: float, adj: list | None = None) -> tuple[set, dict]:
    """Weld dangling border endpoints onto the neighbour face's seam border.

    With the trace de-doubled by ownership, a border crossing a shared edge stops
    when ownership flips at the seam, leaving a ~1-texel gap to where the neighbour
    face draws the same border (``face_texture_positions`` does not place the two
    faces' edge texels exactly coincident — the pairing is many-to-many and up to ~2
    texels apart). The ownership tie-break lets the lower-id face carry a
    seam-crossing border straight *through* the seam, so the neighbour's matching
    point is usually **mid-chain**, not another loose end — a T-join the old
    endpoint-to-endpoint weld could not make.

    So every degree-1 endpoint that lies on an ownership *frontier* cell is snapped
    onto the nearest frontier border *vertex* of a **mesh-adjacent** face (any
    degree) within ``tol``, merging the dangle there. Two mutually-near loose ends
    (a genuine 3-face / pentagon-corner junction) collapse together; a dangle facing
    a continuous neighbour chain joins it as a T. The adjacency guard keeps the
    generous tolerance from fusing unrelated borders.
    """
    deg: dict = defaultdict(int)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1

    fkeys = [k for k in key_face if k in pt_of]        # all frontier vertices
    dangles = [k for k in fkeys if deg.get(k, 0) == 1]
    if not dangles or len(fkeys) < 2:
        return edges, pt_of

    fpts = np.array([pt_of[k] for k in fkeys])
    tree = KDTree(fpts)
    parent = {k: k for k in fkeys}
    rank = {k: deg.get(k, 0) for k in fkeys}           # union toward higher degree

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] > rank[rb]:
            ra, rb = rb, ra            # rb is the higher-degree (anchor) root
        parent[ra] = rb

    def faces_adjacent(fa, fb) -> bool:
        return adj is None or (fa is not None and fb is not None and fb in adj[fa])

    welded = False
    for d in dangles:
        fd = key_face.get(d)
        idxs = tree.query_ball_point(pt_of[d], tol)
        # nearest eligible frontier vertex on a different, mesh-adjacent face
        best, best_d2 = None, np.inf
        pd = pt_of[d]
        for ix in idxs:
            t = fkeys[ix]
            if t == d or find(t) == find(d):
                continue
            ft = key_face.get(t)
            if ft == fd or not faces_adjacent(fd, ft):
                continue
            d2 = float(np.sum((pd - fpts[ix]) ** 2))
            if d2 < best_d2:
                best, best_d2 = t, d2
        if best is not None:
            union(d, best)
            welded = True
    if not welded:
        return edges, pt_of

    clusters: dict = defaultdict(list)
    for k in fkeys:
        clusters[find(k)].append(k)
    remap: dict = {}
    for root, members in clusters.items():
        if len(members) < 2:
            continue
        anchors = [m for m in members if deg.get(m, 0) >= 2]
        if anchors:
            # Keep a continuous chain in place; snap the loose ends onto it.
            ck = max(anchors, key=lambda m: deg[m])
        else:
            mean = np.mean([pt_of[m] for m in members], axis=0)
            ck = position_key(mean)
            pt_of[ck] = mean
        for m in members:
            remap[m] = ck

    new_edges: set = set()
    for a, b in edges:
        a, b = remap.get(a, a), remap.get(b, b)
        if a != b:
            new_edges.add((a, b) if a < b else (b, a))
    return new_edges, pt_of


def _trace_depth(depth: int, hbd, vbd, hmid, vmid, ctr, anc, iterations: int,
                 weld_tol: float = 0.0, adj: list | None = None,
                 frontier: np.ndarray | None = None) -> list:
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
    key_face: dict = {}           # frontier crossing key -> emitting face
    _mark_f = -1                  # face to tag added endpoints with (-1 = none)

    def add(p, q):
        kp, kq = position_key(p), position_key(q)
        if kp == kq:
            return
        pt_of[kp] = p
        pt_of[kq] = q
        edges.add((kp, kq) if kp < kq else (kq, kp))
        if _mark_f >= 0:
            # On the clip frontier: any of these endpoints may be the hexagon-edge
            # termination that must weld to the neighbour face's counterpart.
            key_face[kp] = _mark_f
            key_face[kq] = _mark_f

    for f, i, j in cells:
        t, b, lft, rgt = top[f, i, j], bot[f, i, j], left[f, i, j], right[f, i, j]
        _mark_f = f if (frontier is not None and frontier[f, i, j]) else -1
        if int(t) + int(b) + int(lft) + int(rgt) == 4:
            # 2×2 saddle.  A two-region checkerboard (TL==BR, TR==BL) routed as
            # four spokes to the centre draws a self-crossing X (a visible knot).
            # Resolve it into two non-crossing links that keep the equal diagonal
            # connected.  Genuine 3-/4-region junctions keep the centre hub.
            tl, tr = anc[f, i, j], anc[f, i, j + 1]
            bl, br = anc[f, i + 1, j], anc[f, i + 1, j + 1]
            if tl == br and tr == bl and tl != tr:
                # keep TL/BR connected: cut the TR and BL corners
                add(m_top[f, i, j], m_right[f, i, j])
                add(m_bot[f, i, j], m_left[f, i, j])
                continue
        mids = []
        if t:
            mids.append(m_top[f, i, j])
        if b:
            mids.append(m_bot[f, i, j])
        if lft:
            mids.append(m_left[f, i, j])
        if rgt:
            mids.append(m_right[f, i, j])
        if len(mids) == 2:
            add(mids[0], mids[1])
        else:
            c = ctr[f, i, j]
            for m in mids:
                add(m, c)

    if weld_tol > 0.0:
        edges, pt_of = _weld_seam_endpoints(edges, pt_of, key_face, weld_tol, adj)

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
    adj = getattr(mesh, "face_adjacency", None)
    centroids = getattr(mesh, "face_centroids", None)
    in_poly = (mesh.face_texture_in_polygon(H, W)
               if hasattr(mesh, "face_texture_in_polygon") else None)
    return trace_label_borders(positions, labels, polities, iterations=iterations,
                               adj=adj, centroids=centroids, in_poly=in_poly)
