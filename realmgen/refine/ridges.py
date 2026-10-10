"""Mountain ridges as polylines: the drainage divides between stream basins.

Each texel belongs to the basin of the first stream-network channel its downhill
path reaches (``DRAINAGE_LANDING``), named by that channel's main stem
(``stream_stems``: the world rivers, then the streams below them), or to the
coastal basin where it reaches the sea or an endorheic sink first. The basin
boundaries are contoured on the global lattice the polity borders use (marching
triangles, ``borders._lattice_links``); chains between junctions are joined through
each junction by pairing its two highest-relief arms, so a crest carries on past a
spur.

A ridge point's **relief** is how far the divide stands above the valleys either
side: ``min(e_a, e_b) − max(floor_a, floor_b)`` over its crossing's two texels, a
floor being the height of the texel's landing channel (sea level for the coastal
basin). Averaged along the line, it plays a river's flow in border following
(``river_follow``): a low pass between two basins barely holds a border.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Optional

import numpy as np

from realmgen.refine.borders import _chain_edges, _icosphere_lattice, _lattice_links, lattice_depth_for
from realmgen.refine.fill import WATER
from worldgen.geology.channels import ELEVATION, OCEAN_MASK
from worldgen.hydrology.channels import DRAINAGE_LANDING, FILLED_HEIGHT, LAKE_MASK
from worldgen.hydrology.rivers import stream_stems
from worldgen.hydrology.types import StreamNetwork, laplacian_smooth_sphere

# The basin of texels draining to the sea / an endorheic sink before any channel.
COASTAL = -1
# Laplacian passes over a ridge line (as a river's): the lattice contour steps texel by texel.
_SMOOTH_ITERS = 8
# Points in the running mean of relief along a line.
_RELIEF_WINDOW = 5
# Points from a junction whose relief scores that arm when pairing a junction's arms.
_ARM_POINTS = 6
# Crossings below this share of ``min_relief_km`` are dropped before chaining: most of
# the divides are low (median ~0.05 km), and the line work scales with what's kept.
_PRUNE_SHARE = 0.5


def basin_texels(mesh, streams: StreamNetwork, river_texels: Optional[list],
                 stem_ids: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """``(basin, floor)`` per flat texel: the landing channel's stem id (``stream_stems``;
    ``stem_ids`` its ids, already computed), ``COASTAL`` without one; and its height
    (km), 0 for the coastal basin."""
    landing = np.rint(np.asarray(DRAINAGE_LANDING.read(mesh), dtype=np.float64).reshape(-1)).astype(np.int64)
    ids = stream_stems(streams, river_texels)[1] if stem_ids is None else stem_ids
    basin = np.full(landing.size, COASTAL, dtype=np.int64)
    if len(streams):
        order = np.argsort(streams.nodes, kind="stable")
        sorted_nodes = streams.nodes[order]
        at = np.clip(np.searchsorted(sorted_nodes, landing), 0, len(sorted_nodes) - 1)
        found = (landing >= 0) & (sorted_nodes[at] == landing)
        basin[found] = ids[order[at[found]]]
    basin[basin < 0] = COASTAL
    height = FILLED_HEIGHT if FILLED_HEIGHT.present(mesh) else ELEVATION
    filled = np.asarray(height.read(mesh), dtype=np.float64).reshape(-1)
    floor = np.where(basin >= 0, filled[np.maximum(landing, 0)], 0.0)
    return basin, floor


def _running_mean(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) < 2 or window <= 1:
        return x.copy()
    k = min(int(window), len(x))
    pad = np.concatenate([np.full(k // 2, x[0]), x, np.full(k - 1 - k // 2, x[-1])])
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def _fill_nan(x: np.ndarray) -> np.ndarray:
    """Hub points (no crossing of their own) take the mean of their line neighbours."""
    ok = ~np.isnan(x)
    if ok.all() or not ok.any():
        return np.nan_to_num(x)
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def _join_at_junctions(chains: List[list], junctions: frozenset, relief: np.ndarray) -> List[list]:
    """Chains of point ids joined through each junction's two highest-relief arms."""
    arms: dict = defaultdict(list)     # junction point → [(score, chain, at_end)]
    for c, ch in enumerate(chains):
        if ch[0] == ch[-1] and ch[0] not in junctions:
            continue                   # a closed loop
        for at_end, j in ((False, ch[0]), (True, ch[-1])):
            if j in junctions:
                near = ch[::-1][1:_ARM_POINTS + 1] if at_end else ch[1:_ARM_POINTS + 1]
                r = relief[np.asarray(near, dtype=np.int64)]
                arms[j].append((float(np.nanmean(r)) if np.any(~np.isnan(r)) else 0.0, c, at_end))
    link: dict = {}                    # (chain, at_end) → (chain, at_end)
    for j, a in arms.items():
        if len(a) < 2:
            continue
        a.sort(key=lambda t: -t[0])
        (_, c1, e1), (_, c2, e2) = a[0], a[1]
        if c1 == c2:
            continue                   # a loop's two ends: leave it open at the junction
        link[(c1, e1)] = (c2, e2)
        link[(c2, e2)] = (c1, e1)

    seen = np.zeros(len(chains), dtype=bool)
    out: List[list] = []

    def walk(c: int, enter_end: bool) -> list:
        """From chain ``c`` entered at its ``enter_end`` (True: its last point) onwards."""
        line: list = []
        while c is not None and not seen[c]:
            seen[c] = True
            ch = chains[c][::-1] if enter_end else chains[c]
            line.extend(ch if not line else ch[1:])
            nxt = link.get((c, not enter_end))
            if nxt is None:
                break
            c, enter_end = nxt
        return line

    # Lines from their free ends, then what's left (cycles through junctions, loops).
    for c in range(len(chains)):
        for start_end in (False, True):
            if not seen[c] and (c, start_end) not in link:
                out.append(walk(c, start_end))
    for c in range(len(chains)):
        if not seen[c]:
            out.append(walk(c, False))
    return out


def ridge_lines(mesh, streams: Optional[StreamNetwork], river_texels: Optional[list], *,
                lattice_depth: Optional[int] = None, lattice_offset: int = 0,
                min_relief_km: float = 0.0, located: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
                stem_ids: Optional[np.ndarray] = None) -> list:
    """``[(positions (K, 3) float32, relief_km (K − 1,) float32)]``: the drainage divides.

    ``located``: ``GlobalSampler.locate`` of the lattice's vertices, and ``stem_ids``
    ``stream_stems``' ids, when already computed.

    Crossings below ``_PRUNE_SHARE × min_relief_km`` are dropped before chaining (the
    lines break there) and lines whose relief never reaches ``min_relief_km`` are left
    out. Empty without the stream network or the drainage landing texture.
    """
    if streams is None or not DRAINAGE_LANDING.present(mesh) or not OCEAN_MASK.present(mesh):
        return []
    from region.sampling import GlobalSampler

    ocean_tex = OCEAN_MASK.read(mesh)
    h, w = int(ocean_tex.shape[1]), int(ocean_tex.shape[2])
    if lattice_depth is None:
        lattice_depth = max(0, lattice_depth_for(len(mesh.faces), w) + lattice_offset)
    verts, tris, edges, tri_edges = _icosphere_lattice(int(lattice_depth))

    sampler = GlobalSampler(mesh)
    texel = sampler.anchors(*(located or sampler.locate(verts)), h, w)
    basin, floor = basin_texels(mesh, streams, river_texels, stem_ids)
    elev = np.asarray(ELEVATION.read(mesh), dtype=np.float64).reshape(-1)
    water = np.asarray(ocean_tex).reshape(-1) > 0.5
    if LAKE_MASK.present(mesh):
        water |= np.asarray(LAKE_MASK.read(mesh)).reshape(-1) > 0.5
    labels = basin[texel]
    labels[water[texel]] = WATER

    lut = np.arange(max(int(labels.max()), 0) + 1, dtype=np.int64)[None, :]
    links = _lattice_links(verts, tris, edges, tri_edges, labels, lut, None, 0)
    if links is None:
        return []
    points, ids, _, junctions, _ = links

    # Relief at each crossing (the same edges, in the same order, as _lattice_links').
    la, lb = labels[edges[:, 0]], labels[edges[:, 1]]
    cross = np.flatnonzero((la != lb) & (la != WATER) & (lb != WATER))
    ta, tb = texel[edges[cross, 0]], texel[edges[cross, 1]]
    relief = np.full(len(points), np.nan)
    relief[:cross.size] = np.maximum(np.minimum(elev[ta], elev[tb]) - np.maximum(floor[ta], floor[tb]), 0.0)

    low = relief < _PRUNE_SHARE * float(min_relief_km)          # NaN (hubs) is never low
    ids = ids[~(low[ids[:, 0]] | low[ids[:, 1]])]
    pairs = {(int(a), int(b)) if a < b else (int(b), int(a)) for a, b in ids}
    out: list = []
    for line in _join_at_junctions(_chain_edges(pairs, stops=junctions), junctions, relief):
        if len(line) < 3:
            continue
        k = np.asarray(line, dtype=np.int64)
        r = _running_mean(_fill_nan(relief[k]), _RELIEF_WINDOW)
        seg = 0.5 * (r[:-1] + r[1:])
        if seg.max(initial=0.0) < min_relief_km:
            continue
        pos = np.stack(laplacian_smooth_sphere(list(points[k]), iterations=_SMOOTH_ITERS, alpha=0.5))
        out.append((pos.astype(np.float32), seg.astype(np.float32)))
    return out
