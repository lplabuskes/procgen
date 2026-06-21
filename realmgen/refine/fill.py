"""Texel-resolution polity classification — the shared source of truth.

``compute_texel_labels`` assigns **every texel** a label: a polity id (``>= 0``),
``UNCLAIMED`` (-1) for claimed-by-nobody land, or ``WATER`` (-2) for ocean/lake.
Land texels are classified to the nearest claimed polity by an authority-weighted
(multiplicatively-weighted) Voronoi over the surrounding claimed-land face
centroids, so the polity colour covers the land all the way to the texel coastline
and the stronger polity bulges into the weaker one at texel resolution.

Crucially this also covers the *orphan land* problem: a face counts as water at
face level once **more than half** its texels are water (``is_ocean_per_face``),
so a coastal "water" face can still hold a large minority of genuine land texels.
Those texels are reassigned to whichever claimed-land neighbour wins the weighted
Voronoi, instead of being painted ocean.

Labels along shared mesh edges are reconciled across faces (via
``mesh.boundary_pixel_pairs``) so coincident edge texels agree — this is what lets
the border tracer (``borders.py``) draw continuous lines across face seams.

``build_polity_fill_texture`` colourises the labels into the ``(N, H, W, 4)`` RGBA
"Polities" fill texture; ``borders.py`` traces the texel region boundaries into
smooth polylines.  Both consume the *same* labels, so the lines fall exactly on
the fill's region edges.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

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

    The displacement is a pure function of position, so coincident texels on a
    shared mesh edge get the *same* warp — the warp never introduces seams of its
    own; cross-face label disagreements are still reconciled downstream.
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
    reconcile_edges: bool = True,
) -> np.ndarray:
    """Return an ``(N, H, W)`` int32 per-texel label grid.

    Values: ``>= 0`` polity id, ``UNCLAIMED`` (-1) for unclaimed land, ``WATER``
    (-2) for ocean/lake texels.

    When ``warp_amplitude > 0`` the texel positions fed to the polity Voronoi are
    domain-warped (see :func:`domain_warp`) so the classification boundary erodes
    the polygonal face-bisector staircase into an organic wiggle. The warp affects
    only the *classification decision*; rendered border geometry still uses the
    true texel positions, so lines stay on the warped fill's region edges.
    """
    n = len(mesh.faces)
    H, W = int(ocean_tex.shape[1]), int(ocean_tex.shape[2])
    positions = np.asarray(mesh.face_texture_positions(H, W), dtype=np.float64)
    warped = domain_warp(
        positions, amplitude=warp_amplitude, base_frequency=warp_frequency,
        octaves=warp_octaves, seed=warp_seed,
    ) if warp_amplitude > 0.0 else positions
    centroids = np.asarray(mesh.face_centroids, dtype=np.float64)
    control = np.maximum(np.asarray(authority, dtype=np.float64), control_floor)
    pof = np.asarray(polity_of_face, dtype=np.int64)
    is_water = np.asarray(is_water, dtype=bool)

    def claimed_land(f: int) -> bool:
        return (not is_water[f]) and int(pof[f]) >= 0

    label = np.full((n, H, W), UNCLAIMED, dtype=np.int64)
    for f in range(n):
        if claimed_land(f):
            label[f, :, :] = int(pof[f])

    # Per-texel reclassification.  Two cases need it:
    #   • claimed-land *border* faces — texels split between this polity and a
    #     differing claimed neighbour (the bulge);
    #   • face-level *water* faces with a claimed-land neighbour — their genuine
    #     land texels (minority < 50%) belong to that neighbour, not the ocean.
    for f in range(n):
        claimed_nb = [nb for nb in mesh.face_adjacency[f] if claimed_land(nb)]
        if claimed_land(f):
            cand = [f] + claimed_nb
            if all(int(pof[c]) == int(pof[f]) for c in cand):
                continue  # interior — uniform base label is correct
        elif is_water[f] and claimed_nb:
            cand = claimed_nb  # orphan land → nearest claimed polity
        else:
            continue  # unclaimed land or open-ocean face — keep base label
        cand_arr = np.asarray(cand, dtype=np.int64)
        C = centroids[cand_arr]                                   # (k, 3)
        wsq = control[cand_arr] ** 2                              # (k,)
        P = warped[f].reshape(-1, 3)                              # (HW, 3)
        d2 = ((P[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)  # (HW, k)
        winners = cand_arr[np.argmin(d2 / wsq[None, :], axis=1)]  # (HW,) face ids
        label[f] = pof[winners].reshape(H, W)

    # Water texels (per-texel ocean/lake masks) override any land label.
    label[ocean_tex[..., 0] > 0.5] = WATER
    if lake_tex is not None:
        label[lake_tex[..., 0] > 0.5] = WATER

    # Reconcile coincident texels along shared mesh edges so the border tracer
    # produces continuous lines across face seams.  Only land-vs-land
    # disagreements are touched (water status is continuous across faces).
    if reconcile_edges and H == W and hasattr(mesh, "boundary_pixel_pairs"):
        fa, fb, ra, ca, rb, cb = mesh.boundary_pixel_pairs(H)
        if len(fa):
            la = label[fa, ra, ca]
            lb = label[fb, rb, cb]
            fix = (la != lb) & (la != WATER) & (lb != WATER)
            idx = np.where(fix)[0]
            label[fb[idx], rb[idx], cb[idx]] = la[idx]

    return label.astype(np.int32)


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
    n = len(mesh.faces)
    H, W = int(ocean_tex.shape[1]), int(ocean_tex.shape[2])
    if labels is None:
        labels = compute_texel_labels(
            mesh, polity_of_face, is_water, authority, ocean_tex, lake_tex,
            control_floor=control_floor, warp_amplitude=warp_amplitude,
            warp_frequency=warp_frequency, warp_octaves=warp_octaves,
            warp_seed=warp_seed,
        )
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

    rgba = np.empty((n, H, W, 4), dtype=np.float32)
    rgba[..., 3] = 1.0
    flat = labels.reshape(-1)
    rgb_flat = np.tile(unclaimed, (flat.size, 1))
    claimed = flat >= 0
    rgb_flat[claimed] = color_lut[flat[claimed]]
    rgba[..., :3] = rgb_flat.reshape(n, H, W, 3)

    # Texel-accurate coastlines from the ocean / lake masks (runs last so it
    # overrides any polity colour painted onto a water texel).
    rgba[ocean_tex[..., 0] > 0.5, :3] = ocean
    if lake_tex is not None:
        from worldgen.hydrology.builder import LAKE_COLOR
        rgba[lake_tex[..., 0] > 0.5, :3] = np.asarray(LAKE_COLOR[:3], dtype=np.float32)

    return rgba.astype(np.float16)
