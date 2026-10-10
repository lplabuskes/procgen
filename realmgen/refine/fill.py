"""Polity classification of the land surface — the shared source of truth.

The label is a function of position on the sphere only (``PolityLabelField``),
never of which face's texture holds the point, so over-hanging textures agree
across seams and ``borders.py`` contours the same partition.

A point is labelled among its owner face (nearest centroid) and that face's
claimed-land neighbours by an authority-weighted power diagram over domain-warped
positions, so stronger polities bulge and boundaries wiggle off the bisectors.

Big rivers: runs of border beside one are moved onto it (``RiverSwaps``,
:mod:`realmgen.refine.river_follow`), applied after the plain labels.

Labels: polity id (``>= 0``), ``UNCLAIMED`` (-1), ``WATER`` (-2, from the texel
masks). Land texels of a face-level water face (orphan land) go to its
claimed-land neighbours.
"""
from __future__ import annotations

import copy
from functools import lru_cache
from typing import List, Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree

from core.display import POSITIONS_KEY, RIVER_SEGMENTS_KEY, TextureRecipe
from realmgen.refine.river_follow import RiverSwaps
from realmgen.polity.display import make_polity_tree_color
from realmgen.polity.types import UNCLAIMED_COLOR
from utils.simplex import fbm_sum, make_perm
from worldgen.geology.display import OCEAN_COLOR

WATER = -2
UNCLAIMED = -1
# Nearest other-polity faces voting in ``labels_without``.
_WITHOUT_K = 12
# Specks: points within this many lattice spacings of a speck vertex are checked, and
# the same-label vertices up to ``_SPECK_HOPS`` edges from a speck kept to decide them
# (enough to hold every vertex nearer such a point than the speck is).
_SPECK_REACH = 2.0
_SPECK_HOPS = 5
# Points per KD-tree query below which one thread beats the pool's start-up.
_THREADED_MIN_POINTS = 2048


def _workers(n: int) -> int:
    return -1 if n >= _THREADED_MIN_POINTS else 1


@lru_cache(maxsize=None)
def _warp_perm(seed: int) -> np.ndarray:
    perm = make_perm(seed)
    perm.flags.writeable = False
    return perm


def domain_warp(
    positions: np.ndarray,
    *,
    amplitude: float,
    base_frequency: float,
    octaves: int,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Displace 3-D ``positions`` by a coherent fBm-simplex vector field.

    A pure function of position, so it never introduces seams.
    """
    if amplitude <= 0.0 or octaves <= 0:
        return np.asarray(positions, dtype=np.float64)
    flat = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    freqs, amps, total = [], [], 0.0
    f, a = float(base_frequency), 1.0
    for _ in range(int(octaves)):
        freqs.append(f)
        amps.append(a)
        total += a
        a *= persistence
        f *= lacunarity
    freqs_arr, amps_arr = np.asarray(freqs), np.asarray(amps)
    total = total or 1.0
    disp = np.empty_like(flat)
    for c in range(3):
        perm = _warp_perm(seed * 3 + c + 1)
        disp[:, c] = fbm_sum(flat, perm, freqs_arr, amps_arr) / total
    warped = flat + amplitude * disp
    return warped.reshape(np.asarray(positions).shape)


def _polity_color_map(polities: List) -> dict:
    """``polity_id -> rgb`` using the exact "Polities" scheme colours."""
    pol_by_id = {p.polity_id: p for p in polities}

    def root_id(p):
        cur = p
        while cur.liege is not None and cur.liege in pol_by_id:
            cur = pol_by_id[cur.liege]
        return cur.polity_id

    color_fn = make_polity_tree_color(polities)
    out: dict[int, np.ndarray] = {}
    for p in polities:
        meta = {"sovereign_id": root_id(p), "polity_id": p.polity_id, "is_water": False}
        out[p.polity_id] = np.asarray(color_fn(meta)[:3], dtype=np.float32)
    return out


class PolityLabelField:
    """The polity label as a function of position on the sphere.

    ``land_labels`` ignores water; callers overlay their own water masks.
    Unclaimed land faces, and water faces with no claimed-land neighbour, label
    their whole cell ``UNCLAIMED``.
    """

    # Points labelled per call; bounds the (P, K, 3) Voronoi work array.
    _CHUNK = 200_000

    def __init__(
        self,
        mesh,
        polity_of_face: np.ndarray,
        is_water: np.ndarray,
        authority: np.ndarray,
        *,
        control_floor: float = 1e-3,
        warp_amplitude: float = 0.0,
        warp_frequency: float = 40.0,
        warp_octaves: int = 3,
        warp_seed: int = 0,
    ) -> None:
        centroids = np.asarray(mesh.face_centroids, dtype=np.float64)
        self.centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        self._tree = KDTree(self.centroids)
        self.pof = np.asarray(polity_of_face, dtype=np.int64)
        water = np.asarray(is_water, dtype=bool)
        self._water = water
        claimed_land = (~water) & (self.pof >= 0)
        self._claimed_land = claimed_land
        # Per excluded polity: the other claimed-land faces and their tree (``labels_without``).
        self._without: dict = {}
        control = np.maximum(np.asarray(authority, dtype=np.float64), control_floor)
        # Power diagram: a point goes to the candidate with the least |x − c|² − power.
        # Every two-polity boundary is a straight line (before the warp); ``power`` =
        # ½·L²·ln(control), L the mean neighbour spacing, shifts it toward the weaker
        # face as |x − c| / control did near the bisector. (That multiplicative form
        # curved the boundaries into arcs round the weaker centroids, which met in long
        # thin cusps at three-polity corners.)
        adj0 = mesh.face_adjacency
        src = np.repeat(np.arange(len(adj0)), [len(a) for a in adj0])
        dst = np.fromiter((j for a in adj0 for j in a), dtype=np.int64, count=len(src))
        spacing2 = float(np.mean(np.sum((self.centroids[src] - self.centroids[dst]) ** 2, axis=1))) if len(src) else 0.0
        self._power = 0.5 * spacing2 * np.log(control)
        self._warp_amplitude = float(warp_amplitude)
        self._warp_frequency = float(warp_frequency)
        self._warp_octaves = int(warp_octaves)
        self._warp_seed = int(warp_seed)

        # Candidates per owner: itself + neighbours, claimed land only (an
        # unclaimed-land owner has none).
        adj = mesh.face_adjacency
        n = len(self.pof)
        k = 1 + max((len(a) for a in adj), default=0)
        cand = np.full((n, k), -1, dtype=np.int64)
        for f in range(n):
            cand[f, 0] = f
            cand[f, 1:1 + len(adj[f])] = adj[f]
        valid = (cand >= 0) & claimed_land[np.maximum(cand, 0)]
        valid[(~water) & (self.pof < 0)] = False
        self._cand = np.where(valid, cand, -1)

        # Owners whose candidates all share one polity need no vote.
        cand_pid = np.where(valid, self.pof[np.maximum(cand, 0)], -1)
        first = np.where(valid.any(axis=1), cand_pid.max(axis=1), -1)
        uniform = np.all(~valid | (cand_pid == first[:, None]), axis=1)
        self._fixed = np.where(valid.any(axis=1), first, UNCLAIMED)   # label when uniform
        self._vote = ~uniform

        # Borders moved onto rivers run by run (``river_follow``); set by ``with_swaps``.
        self.swaps: Optional[RiverSwaps] = None
        # Tiny enclaves handed to their surroundings; set by ``with_specks``.
        self.specks: Optional[Specks] = None

    def with_rivers(self, segments: Optional[np.ndarray]) -> "PolityLabelField":
        """This field with its swaps moved onto packed ``(M, 8)`` river ``segments`` (``pack_river_segments``).

        The region / local tiers pass the rivers they draw (re-routed off the world
        centreline), so borders sit on those.
        """
        if self.swaps is None:
            return self
        other = copy.copy(self)
        other.swaps = self.swaps.with_rivers(segments)
        return other

    def with_swaps(self, swaps: Optional[RiverSwaps]) -> "PolityLabelField":
        """This field with ``swaps`` (borders moved onto rivers) applied after the plain labels."""
        other = copy.copy(self)
        other.swaps = swaps if swaps is not None and len(swaps) else None
        return other

    def with_specks(self, specks: Optional["Specks"]) -> "PolityLabelField":
        """This field with ``specks`` (``find_specks``) relabelled after the swaps."""
        other = copy.copy(self)
        other.specks = specks if specks is not None and len(specks) else None
        return other

    def plain_land_labels(self, points: np.ndarray, owner: Optional[np.ndarray] = None) -> np.ndarray:
        """``(P,)`` land labels before any river handling: the warped power diagram."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        own = self.owner(pts) if owner is None else np.asarray(owner, dtype=np.int64)
        return self._plain_labels(pts, own)

    def labels_without(self, points: np.ndarray, polity: int) -> np.ndarray:
        """``(P,)`` the polity each point would go to were ``polity`` not there.

        The warped power diagram over the nearest claimed-land faces of other
        polities; a river swap that moves a junction gives ``polity``'s land across the
        river to these.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if polity not in self._without:
            faces = np.flatnonzero(self._claimed_land & (self.pof != polity))
            self._without[polity] = (faces, KDTree(self.centroids[faces]) if faces.size else None)
        faces, tree = self._without[polity]
        if tree is None or len(pts) == 0:
            return np.full(len(pts), UNCLAIMED, dtype=np.int64)
        k = min(_WITHOUT_K, faces.size)
        warped = self._warp(pts)
        # Candidates nearest the warped point (the one scored), so neighbouring points
        # vote among the same faces.
        _, nn = tree.query(warped, k=k, workers=_workers(len(pts)))
        c = faces[np.asarray(nn).reshape(len(pts), k)]
        score = ((warped[:, None, :] - self.centroids[c]) ** 2).sum(axis=-1) - self._power[c]
        return self.pof[c[np.arange(len(pts)), np.argmin(score, axis=1)]]

    def is_water_at(self, points: np.ndarray) -> np.ndarray:
        """``(P,)`` whether each point's owner face is water (face level)."""
        return self._water[self.owner(np.asarray(points, dtype=np.float64).reshape(-1, 3))]

    def owner(self, points: np.ndarray) -> np.ndarray:
        """Nearest-centroid face of each ``(P, 3)`` point."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        _, faces = self._tree.query(pts, k=1, workers=_workers(len(pts)))
        return np.asarray(faces, dtype=np.int64)

    def land_labels(self, points: np.ndarray, owner: Optional[np.ndarray] = None,
                    plain: Optional[np.ndarray] = None) -> np.ndarray:
        """``(P,)`` land label of each ``(P, 3)`` unit-sphere point (water ignored).

        ``plain``: the points' ``plain_land_labels``, already computed.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        own = self.owner(pts) if owner is None else np.asarray(owner, dtype=np.int64)
        out = self._plain_labels(pts, own) if plain is None else np.array(plain, dtype=np.int64)
        if self.swaps is not None:
            out = self.swaps.apply(pts, own, out)
        if self.specks is not None:
            out = self.specks.apply(pts, out)
        return out

    def _plain_labels(self, pts: np.ndarray, own: np.ndarray) -> np.ndarray:
        """Labels before the river swaps: the warped power diagram over the 1-ring."""
        out = self._fixed[own].copy()
        idx = np.flatnonzero(self._vote[own])
        for s in range(0, idx.size, self._CHUNK):
            sel = idx[s:s + self._CHUNK]
            cand = self._cand[own[sel]]                                       # (p, k)
            ok = cand >= 0
            c = np.maximum(cand, 0)
            d2 = ((self._warp(pts[sel])[:, None, :] - self.centroids[c]) ** 2).sum(axis=-1)
            score = np.where(ok, d2 - self._power[c], np.inf)
            winner = c[np.arange(sel.size), np.argmin(score, axis=1)]
            out[sel] = self.pof[winner]
        return out

    def _warp(self, pts: np.ndarray) -> np.ndarray:
        return domain_warp(pts, amplitude=self._warp_amplitude, base_frequency=self._warp_frequency,
                           octaves=self._warp_octaves, seed=self._warp_seed)

    def texel_labels(self, positions: np.ndarray, ocean_tex: np.ndarray,
                     lake_tex: Optional[np.ndarray]) -> np.ndarray:
        """``(N, H, W)`` int32 labels for face-texture ``positions``, water from the masks."""
        pos = np.asarray(positions, dtype=np.float64)
        label = self.land_labels(pos.reshape(-1, 3)).reshape(pos.shape[:-1])
        label[ocean_tex[..., 0] > 0.5] = WATER
        if lake_tex is not None:
            label[lake_tex[..., 0] > 0.5] = WATER
        return label.astype(np.int32)


class Specks:
    """Tiny land enclaves found on a lattice, handed to the polity round them.

    A point carrying a speck's label takes the replacement when, of the lattice
    vertices with that label near the speck (``_SPECK_HOPS`` edges out), the nearest
    is the speck's: the field's enclave runs past its vertices' own cells, and a
    nearer same-label vertex outside the speck is land that stays. Every tier
    relabels the same places and nothing else.
    """

    def __init__(self, verts: np.ndarray, is_speck: np.ndarray, label: np.ndarray, replace: np.ndarray,
                 spacing: float) -> None:
        self.verts = np.asarray(verts, dtype=np.float64)
        self.is_speck = np.asarray(is_speck, dtype=bool)
        self.label = np.asarray(label, dtype=np.int64)
        self.replace = np.asarray(replace, dtype=np.int64)
        self.spacing = float(spacing)
        sv = np.flatnonzero(self.is_speck)
        self._speck_tree = KDTree(self.verts[sv]) if sv.size else None
        # Per speck label: the kept vertices with that label and their tree.
        self._by_label: dict = {}
        for lab in np.unique(self.label[sv]):
            idx = np.flatnonzero(self.label == lab)
            self._by_label[int(lab)] = (idx, KDTree(self.verts[idx]))

    def __len__(self) -> int:
        return int(self.is_speck.sum())

    def apply(self, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if self._speck_tree is None or len(points) == 0:
            return labels
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        d, _ = self._speck_tree.query(pts, k=1, distance_upper_bound=_SPECK_REACH * self.spacing,
                                      workers=_workers(len(pts)))
        cand = np.flatnonzero(np.isfinite(d))
        out = labels.copy()
        for lab in np.unique(labels[cand]):
            entry = self._by_label.get(int(lab))
            if entry is None:
                continue
            idx, tree = entry
            sel = cand[labels[cand] == lab]
            _, j = tree.query(pts[sel], k=1, workers=_workers(len(sel)))
            v = idx[np.asarray(j, dtype=np.int64)]
            hit = self.is_speck[v]
            out[sel[hit]] = self.replace[v[hit]]
        return out


def find_specks(verts: np.ndarray, edges: np.ndarray, labels: np.ndarray, max_verts: int,
                keep: Optional[np.ndarray] = None) -> Optional[Specks]:
    """Land pieces of ``≤ max_verts`` lattice vertices of one label (over ``edges``), each
    handed to its commonest land neighbour label; None when there are none.

    Pieces without a land neighbour (islands) and pieces within a lattice spacing of
    a ``keep`` point (``(S, 3)``, settlements) stay.
    """
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    land = labels >= 0
    a, b = edges[:, 0], edges[:, 1]
    same = land[a] & (labels[a] == labels[b])
    g = coo_matrix((np.ones(int(same.sum())), (a[same], b[same])), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    size = np.bincount(comp[land], minlength=comp.max() + 1)
    small = np.zeros(comp.max() + 1, dtype=bool)
    small[np.flatnonzero((size > 0) & (size <= max_verts))] = True
    in_speck = land & small[comp]
    spacing = float(np.median(np.linalg.norm(verts[a[:1000]] - verts[b[:1000]], axis=1)))
    if keep is not None and len(keep) and in_speck.any():
        sv = np.flatnonzero(in_speck)
        d, j = KDTree(verts[sv]).query(np.asarray(keep, dtype=np.float64).reshape(-1, 3),
                                       distance_upper_bound=spacing)
        d, j = np.atleast_1d(d), np.atleast_1d(j)
        small[comp[sv[j[np.isfinite(d)]]]] = False
        in_speck = land & small[comp]
    if not in_speck.any():
        return None
    # Neighbour labels across each speck's rim: the commonest land one replaces it.
    cross = land[a] & land[b] & (labels[a] != labels[b])
    rim_a = cross & in_speck[a]
    rim_b = cross & in_speck[b]
    sp = np.concatenate([comp[a[rim_a]], comp[b[rim_b]]])
    nb = np.concatenate([labels[b[rim_a]], labels[a[rim_b]]])
    replace = np.full(comp.max() + 1, -1, dtype=np.int64)
    if sp.size:
        key = np.unique(np.stack([sp, nb], axis=1), axis=0, return_counts=True)
        pairs, counts = key
        order = np.lexsort((-counts, pairs[:, 0]))
        pairs = pairs[order]
        first = np.ones(len(pairs), dtype=bool)
        first[1:] = pairs[1:, 0] != pairs[:-1, 0]
        replace[pairs[first, 0]] = pairs[first, 1]
    in_speck &= replace[comp] >= 0
    if not in_speck.any():
        return None
    # The specks' vertices and those a few edges out, so a point's nearest same-label vertex decides.
    near = in_speck.copy()
    for _ in range(_SPECK_HOPS):
        touch = near[a] | near[b]
        near[a[touch]] = True
        near[b[touch]] = True
    idx = np.flatnonzero(near)
    return Specks(verts[idx], in_speck[idx], labels[idx], replace[comp[idx]], spacing)


def compute_texel_labels(
    mesh,
    polity_of_face: np.ndarray,
    is_water: np.ndarray,
    authority: np.ndarray,
    ocean_tex: np.ndarray,
    lake_tex: Optional[np.ndarray],
    *,
    control_floor: float = 1e-3,
    warp_amplitude: float = 0.0,
    warp_frequency: float = 40.0,
    warp_octaves: int = 3,
    warp_seed: int = 0,
) -> np.ndarray:
    """Return the ``(N, H, W)`` int32 per-texel label grid (see ``PolityLabelField``)."""
    field = PolityLabelField(
        mesh, polity_of_face, is_water, authority, control_floor=control_floor,
        warp_amplitude=warp_amplitude, warp_frequency=warp_frequency,
        warp_octaves=warp_octaves, warp_seed=warp_seed,
    )
    H, W = int(ocean_tex.shape[1]), int(ocean_tex.shape[2])
    return field.texel_labels(mesh.face_texture_positions(H, W), ocean_tex, lake_tex)


def build_polity_fill_texture(
    mesh,
    polities: List,
    polity_of_face: np.ndarray,
    is_water: np.ndarray,
    authority: np.ndarray,
    ocean_tex: np.ndarray,
    lake_tex: Optional[np.ndarray],
    *,
    control_floor: float = 1e-3,
    warp_amplitude: float = 0.0,
    warp_frequency: float = 40.0,
    warp_octaves: int = 3,
    warp_seed: int = 0,
    labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return an ``(N, H, W, 4)`` float16 RGBA polity-fill texture (``labels`` reused if given)."""
    if labels is None:
        labels = compute_texel_labels(
            mesh, polity_of_face, is_water, authority, ocean_tex, lake_tex,
            control_floor=control_floor, warp_amplitude=warp_amplitude,
            warp_frequency=warp_frequency, warp_octaves=warp_octaves,
            warp_seed=warp_seed,
        )
    return colorize_labels(labels, polities, ocean_tex, lake_tex).astype(np.float16)


def colorize_labels(labels: np.ndarray, polities: List, ocean_tex: np.ndarray,
                    lake_tex: Optional[np.ndarray]) -> np.ndarray:
    """``(..., 4)`` float32 RGBA for a label grid, in the "Polities" scheme colours."""
    labels = np.asarray(labels)
    pid_rgb = _polity_color_map(polities)
    unclaimed = np.asarray(UNCLAIMED_COLOR[:3], dtype=np.float32)
    ocean = np.asarray(OCEAN_COLOR[:3], dtype=np.float32)

    max_pid = max((p.polity_id for p in polities), default=-1)
    color_lut = np.tile(unclaimed, (max_pid + 2, 1))  # row 0..max_pid; row -1 spare
    for pid, rgb in pid_rgb.items():
        if 0 <= pid <= max_pid:
            color_lut[pid] = rgb

    rgba = np.empty((*labels.shape, 4), dtype=np.float32)
    rgba[..., 3] = 1.0
    flat = labels.reshape(-1)
    rgb_flat = np.tile(unclaimed, (flat.size, 1))
    claimed = flat >= 0
    rgb_flat[claimed] = color_lut[flat[claimed]]
    rgba[..., :3] = rgb_flat.reshape(*labels.shape, 3)

    # Water last, overriding any polity colour.
    rgba[ocean_tex[..., 0] > 0.5, :3] = ocean
    if lake_tex is not None:
        from worldgen.hydrology.builder import LAKE_COLOR
        rgba[lake_tex[..., 0] > 0.5, :3] = np.asarray(LAKE_COLOR[:3], dtype=np.float32)
    return rgba


def polity_fill_recipe(field: PolityLabelField, polities: List) -> TextureRecipe:
    """The "Polities" fill as a recipe over texel positions and water masks.

    Lets a region view colour its own finer texels (and refined shoreline)
    instead of magnifying the global texture, borders following the rivers it draws
    when it passes them under ``RIVER_SEGMENTS_KEY``.
    """
    def fn(textures) -> np.ndarray:
        rivers = textures.get(RIVER_SEGMENTS_KEY)
        f = field if rivers is None else field.with_rivers(rivers)
        labels = f.texel_labels(textures[POSITIONS_KEY], textures["ocean_mask"], textures.get("lake_mask"))
        return colorize_labels(labels, polities, textures["ocean_mask"], textures.get("lake_mask"))

    return TextureRecipe(fn, sources=(POSITIONS_KEY, "ocean_mask"))
