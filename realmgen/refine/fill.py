"""Polity classification of the land surface — the shared source of truth.

The label is a function of position on the sphere only (``PolityLabelField``),
never of which face's texture holds the point, so over-hanging textures agree
across seams and ``borders.py`` contours the same partition.

A point is labelled among its owner face (nearest centroid) and that face's
claimed-land neighbours by an authority-weighted Voronoi over domain-warped
positions, so stronger polities bulge and boundaries wiggle off the bisectors.

Big rivers are barriers (:mod:`realmgen.refine.barriers`): near one, a point
takes the label from a little way out on its own bank, so a border within reach
lies on the river.

Labels: polity id (``>= 0``), ``UNCLAIMED`` (-1), ``WATER`` (-2, from the texel
masks). Land texels of a face-level water face (orphan land) go to its
claimed-land neighbours.
"""
from __future__ import annotations

import copy
from typing import List, Optional

import numpy as np
from scipy.spatial import KDTree

from core.display import POSITIONS_KEY, RIVER_SEGMENTS_KEY, TextureRecipe
from realmgen.refine.barriers import RiverBarriers, build_river_barriers, pack_river_segments
from realmgen.refine.river_follow import RiverSwaps
from realmgen.polity.display import make_polity_tree_color
from realmgen.polity.types import UNCLAIMED_COLOR
from utils.simplex import fbm_sum, make_perm
from worldgen.geology.display import OCEAN_COLOR

WATER = -2
UNCLAIMED = -1


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
        perm = make_perm(seed * 3 + c + 1)
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
        rivers=None,
        river_reach_km: float = 0.0,
        river_min_flow: float = 1000.0,
        river_full_flow: float = 8000.0,
        river_snap_fraction: float = 0.6,
        river_cross_weight: float = 3.0,
    ) -> None:
        centroids = np.asarray(mesh.face_centroids, dtype=np.float64)
        self.centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        self._tree = KDTree(self.centroids)
        self.pof = np.asarray(polity_of_face, dtype=np.int64)
        water = np.asarray(is_water, dtype=bool)
        self._water = water
        claimed_land = (~water) & (self.pof >= 0)
        control = np.maximum(np.asarray(authority, dtype=np.float64), control_floor)
        self._inv_w2 = 1.0 / control ** 2
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

        # Near a river the vote widens to the 2-ring (a candidate across the river is
        # penalised, so the winner may be further off; every nearby cell must see it).
        ring2 = [sorted({f, *adj[f], *(g for h in adj[f] for g in adj[h])}) for f in range(n)]
        k2 = max((len(r) for r in ring2), default=1)
        cand2 = np.full((n, k2), -1, dtype=np.int64)
        for f, r in enumerate(ring2):
            cand2[f, :len(r)] = r
        valid2 = (cand2 >= 0) & claimed_land[np.maximum(cand2, 0)]
        valid2[(~water) & (self.pof < 0)] = False
        self._cand2 = np.where(valid2, cand2, -1)
        pid2 = np.where(valid2, self.pof[np.maximum(cand2, 0)], -1)
        first2 = pid2.max(axis=1)
        self._vote2 = valid2.any(axis=1) & ~np.all(~valid2 | (pid2 == first2[:, None]), axis=1)
        c2 = self.centroids[np.maximum(cand2, 0)]
        self._ring2_radius = np.where(cand2 >= 0, np.linalg.norm(c2 - self.centroids[:, None, :], axis=-1),
                                      0.0).max(axis=1)
        self._cross_weight = float(river_cross_weight)

        self._river_params = dict(reach_km=float(river_reach_km), min_flow=float(river_min_flow),
                                  full_flow=float(river_full_flow), snap_fraction=float(river_snap_fraction))
        self._set_barriers(build_river_barriers(pack_river_segments(rivers), **self._river_params))
        # Borders moved onto rivers run by run (``river_follow``); set by ``with_swaps``.
        self.swaps: Optional[RiverSwaps] = None

    def _set_barriers(self, barriers: Optional[RiverBarriers]) -> None:
        self.barriers = barriers
        # Per owner face, the segments its 2-ring chords can cross (vote-2 owners only).
        self._river_segs: list = [None] * len(self.pof)
        if barriers is not None and self._cross_weight > 0.0:
            faces = np.flatnonzero(self._vote2)
            near = barriers.segments_near(self.centroids[faces], self._ring2_radius[faces])
            for f, segs in zip(faces.tolist(), near):
                self._river_segs[f] = segs

    def with_rivers(self, segments: Optional[np.ndarray]) -> "PolityLabelField":
        """This field with packed ``(M, 7)`` river ``segments`` (``pack_river_segments``) as its barriers.

        The region / local tiers pass the rivers they draw (re-routed off the world
        centreline), so borders sit on those.
        """
        other = copy.copy(self)
        other._set_barriers(build_river_barriers(segments, **self._river_params))
        if self.swaps is not None:
            other.swaps = self.swaps.with_rivers(segments)
        return other

    def with_swaps(self, swaps: Optional[RiverSwaps]) -> "PolityLabelField":
        """This field with ``swaps`` (borders moved onto rivers) applied after the plain labels."""
        other = copy.copy(self)
        other.swaps = swaps if swaps is not None and len(swaps) else None
        return other

    def plain_land_labels(self, points: np.ndarray, owner: Optional[np.ndarray] = None) -> np.ndarray:
        """``(P,)`` land labels before any river handling: the warped, weighted Voronoi."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        own = self.owner(pts) if owner is None else np.asarray(owner, dtype=np.int64)
        return self._plain_labels(pts, own)

    def is_water_at(self, points: np.ndarray) -> np.ndarray:
        """``(P,)`` whether each point's owner face is water (face level)."""
        return self._water[self.owner(np.asarray(points, dtype=np.float64).reshape(-1, 3))]

    def owner(self, points: np.ndarray) -> np.ndarray:
        """Nearest-centroid face of each ``(P, 3)`` point."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        _, faces = self._tree.query(pts, k=1, workers=-1)
        return np.asarray(faces, dtype=np.int64)

    def land_labels(self, points: np.ndarray, owner: Optional[np.ndarray] = None) -> np.ndarray:
        """``(P,)`` land label of each ``(P, 3)`` unit-sphere point (water ignored)."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        own = self.owner(pts) if owner is None else np.asarray(owner, dtype=np.int64)
        out = self._plain_labels(pts, own)
        if self.barriers is not None:
            idx, target = self.barriers.snap_targets(pts)
            if idx.size:
                out[idx] = self._plain_labels(target, self.owner(target))
        if self.swaps is not None:
            out = self.swaps.apply(pts, own, out)
        return out

    def _plain_labels(self, pts: np.ndarray, own: np.ndarray) -> np.ndarray:
        """Labels before the river snap: the warped, weighted Voronoi, rivers cutting cells.

        Owners with a river in their 2-ring vote over the 2-ring with crossings
        penalised (``_river_vote``); the rest over the 1-ring.
        """
        river = np.zeros(own.size, dtype=bool)
        if self.barriers is not None and self._cross_weight > 0.0:
            river = np.array([s is not None for s in self._river_segs], dtype=bool)[own]
        out = self._fixed[own].copy()
        idx = np.flatnonzero(self._vote[own] & ~river)
        for s in range(0, idx.size, self._CHUNK):
            sel = idx[s:s + self._CHUNK]
            cand = self._cand[own[sel]]                                       # (p, k)
            ok = cand >= 0
            c = np.maximum(cand, 0)
            d2 = ((self._warp(pts[sel])[:, None, :] - self.centroids[c]) ** 2).sum(axis=-1)
            score = np.where(ok, d2 * self._inv_w2[c], np.inf)
            winner = c[np.arange(sel.size), np.argmin(score, axis=1)]
            out[sel] = self.pof[winner]
        rows = np.flatnonzero(river)
        if rows.size:
            out[rows] = self._river_vote(pts[rows], self._warp(pts[rows]), own[rows])
        return out

    def _warp(self, pts: np.ndarray) -> np.ndarray:
        return domain_warp(pts, amplitude=self._warp_amplitude, base_frequency=self._warp_frequency,
                           octaves=self._warp_octaves, seed=self._warp_seed)

    def _river_vote(self, pts: np.ndarray, warped: np.ndarray, own: np.ndarray) -> np.ndarray:
        """2-ring vote with ``exp(cross_weight · strength)`` per river a candidate's chord crosses."""
        barriers = self.barriers
        assert barriers is not None
        out = np.empty(own.size, dtype=np.int64)
        order = np.argsort(own, kind="stable")
        bounds = np.flatnonzero(np.diff(own[order])) + 1
        for grp in np.split(order, bounds):     # one owner: shared candidates and segments
            f = int(own[grp[0]])
            cand = self._cand2[f]
            cand = cand[cand >= 0]
            d2 = ((warped[grp][:, None, :] - self.centroids[cand]) ** 2).sum(axis=-1)
            score = d2 * self._inv_w2[cand]
            segs = self._river_segs[f]
            first = np.argmin(score, axis=1)
            winner = cand[first]
            # Crossings only raise a score: a nearest candidate on the point's own bank
            # wins outright; the rest walk candidates best-first, stopping once no
            # unpenalised score can beat the best penalised one.
            crossed = np.flatnonzero(barriers.crossings(segs, pts[grp], self.centroids[winner]) > 0.0)
            if crossed.size:
                sub = score[crossed]
                rank = np.argsort(sub, axis=1)
                rows = np.arange(crossed.size)
                best = np.full(crossed.size, np.inf)
                win = winner[crossed]
                for r in range(cand.size):
                    k = rank[:, r]
                    base = sub[rows, k]
                    live = np.flatnonzero(base < best)
                    if live.size == 0:
                        break
                    cross = barriers.crossings(segs, pts[grp[crossed[live]]], self.centroids[cand[k[live]]])
                    pen = base[live] * np.exp(self._cross_weight * cross)
                    better = pen < best[live]
                    best[live[better]] = pen[better]
                    win[live[better]] = cand[k[live[better]]]
                winner[crossed] = win
            out[grp] = self.pof[winner]
        return out

    def texel_labels(self, positions: np.ndarray, ocean_tex: np.ndarray,
                     lake_tex: Optional[np.ndarray]) -> np.ndarray:
        """``(N, H, W)`` int32 labels for face-texture ``positions``, water from the masks."""
        pos = np.asarray(positions, dtype=np.float64)
        label = self.land_labels(pos.reshape(-1, 3)).reshape(pos.shape[:-1])
        label[ocean_tex[..., 0] > 0.5] = WATER
        if lake_tex is not None:
            label[lake_tex[..., 0] > 0.5] = WATER
        return label.astype(np.int32)


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
    instead of magnifying the global texture, snapping to the rivers it draws
    when it passes them under ``RIVER_SEGMENTS_KEY``.
    """
    def fn(textures) -> np.ndarray:
        rivers = textures.get(RIVER_SEGMENTS_KEY)
        f = field if rivers is None else field.with_rivers(rivers)
        labels = f.texel_labels(textures[POSITIONS_KEY], textures["ocean_mask"], textures.get("lake_mask"))
        return colorize_labels(labels, polities, textures["ocean_mask"], textures.get("lake_mask"))

    return TextureRecipe(fn, sources=(POSITIONS_KEY, "ocean_mask"))
