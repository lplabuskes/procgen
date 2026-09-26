"""Polity classification of the land surface — the shared source of truth.

The polity a point belongs to is a **function of its position on the sphere**
(``PolityLabelField``), never of which face's texture grid happens to hold it.
That is what keeps everything consistent across the mesh seams: two faces'
over-hanging square textures that sample the same spot of ground always agree,
and the border tracer (``borders.py``) contours the very same field on its own
seamless lattice, so the lines and the fill describe one partition.

A point is labelled by its **owner** face (nearest centroid) and that face's
neighbours: claimed land is split among its own polity and its claimed-land
neighbours by an authority-weighted (multiplicatively-weighted) Voronoi, so the
stronger polity bulges into the weaker one. The positions fed to that Voronoi
are domain-warped so the boundary wiggles organically instead of following the
straight bisectors between face centroids.

Labels: a polity id (``>= 0``), ``UNCLAIMED`` (-1) for claimed-by-nobody land, or
``WATER`` (-2) for ocean/lake. Water comes from the per-texel masks, so a
face-level *water* face (more than half its texels water) can still hold genuine
land texels — the *orphan land* — which go to its claimed-land neighbours.

``build_polity_fill_texture`` colourises the texel labels into the ``(N, H, W, 4)``
RGBA "Polities" fill texture.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.spatial import KDTree

from core.display import POSITIONS_KEY, TextureRecipe
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

    Used to perturb the texel positions fed to the polity Voronoi so the
    classification boundary wiggles organically instead of hugging the straight
    perpendicular bisectors between face centroids (the polygonal staircase).

    The displacement is a pure function of position, so coincident points get the
    *same* warp — the warp never introduces seams of its own.
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

    ``land_labels(points)`` ignores water (it answers "whose land would this be");
    callers overlay the water masks themselves, at whatever resolution they hold
    them. Unclaimed land faces, and water faces with no claimed-land neighbour,
    label their whole cell ``UNCLAIMED``.
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
        claimed_land = (~water) & (self.pof >= 0)
        control = np.maximum(np.asarray(authority, dtype=np.float64), control_floor)
        self._inv_w2 = 1.0 / control ** 2
        self._warp_amplitude = float(warp_amplitude)
        self._warp_frequency = float(warp_frequency)
        self._warp_octaves = int(warp_octaves)
        self._warp_seed = int(warp_seed)

        # Voronoi candidates per owner face: itself plus its neighbours, kept only
        # where they are claimed land (so a water owner votes among its claimed
        # neighbours — the orphan land — and an unclaimed-land owner has none).
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

    def owner(self, points: np.ndarray) -> np.ndarray:
        """Nearest-centroid face of each ``(P, 3)`` point."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        _, faces = self._tree.query(pts, k=1)
        return np.asarray(faces, dtype=np.int64)

    def land_labels(self, points: np.ndarray, owner: Optional[np.ndarray] = None) -> np.ndarray:
        """``(P,)`` land label of each ``(P, 3)`` unit-sphere point (water ignored)."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        own = self.owner(pts) if owner is None else np.asarray(owner, dtype=np.int64)
        out = self._fixed[own].copy()
        idx = np.flatnonzero(self._vote[own])
        for s in range(0, idx.size, self._CHUNK):
            sel = idx[s:s + self._CHUNK]
            warped = domain_warp(pts[sel], amplitude=self._warp_amplitude,
                                 base_frequency=self._warp_frequency,
                                 octaves=self._warp_octaves, seed=self._warp_seed)  # (p, 3)
            cand = self._cand[own[sel]]                                       # (p, k)
            ok = cand >= 0
            c = np.maximum(cand, 0)
            d2 = ((warped[:, None, :] - self.centroids[c]) ** 2).sum(axis=-1)
            score = np.where(ok, d2 * self._inv_w2[c], np.inf)
            winner = c[np.arange(sel.size), np.argmin(score, axis=1)]
            out[sel] = self.pof[winner]
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
    """Return the ``(N, H, W)`` int32 per-texel label grid (see ``PolityLabelField``).

    Values: ``>= 0`` polity id, ``UNCLAIMED`` (-1) for unclaimed land, ``WATER``
    (-2) for ocean/lake texels.
    """
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
    """Return an ``(N, H, W, 4)`` float16 RGBA polity-fill texture.

    Colourises the per-texel labels (computed here, or reused via ``labels``).
    """
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
    """``(..., 4)`` float32 RGBA for a label grid of any leading shape.

    Polity ids take the exact "Polities" scheme colours, unclaimed land is grey,
    and ocean / lake texels (from the masks) their own blues.
    """
    labels = np.asarray(labels)
    pid_rgb = _polity_color_map(polities)
    unclaimed = np.asarray(UNCLAIMED_COLOR[:3], dtype=np.float32)
    ocean = np.asarray(OCEAN_COLOR[:3], dtype=np.float32)

    # Colour lookup by polity id (unclaimed grey for gaps / negative labels).
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

    # Texel-accurate coastlines from the ocean / lake masks (runs last so it
    # overrides any polity colour painted onto a water texel).
    rgba[ocean_tex[..., 0] > 0.5, :3] = ocean
    if lake_tex is not None:
        from worldgen.hydrology.builder import LAKE_COLOR
        rgba[lake_tex[..., 0] > 0.5, :3] = np.asarray(LAKE_COLOR[:3], dtype=np.float32)
    return rgba


def polity_fill_recipe(field: PolityLabelField, polities: List) -> TextureRecipe:
    """The "Polities" fill as a recipe over texel positions and water masks.

    The label is a function of position, so any grid of texels can be coloured
    directly — a region view evaluates it at its own, finer texels (with its
    refined shoreline) instead of magnifying the global fill texture.
    """
    def fn(textures) -> np.ndarray:
        labels = field.texel_labels(textures[POSITIONS_KEY], textures["ocean_mask"],
                                    textures.get("lake_mask"))
        return colorize_labels(labels, polities, textures["ocean_mask"], textures.get("lake_mask"))

    return TextureRecipe(fn, sources=(POSITIONS_KEY, "ocean_mask"))
