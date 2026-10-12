"""Terrain ruggedness — how rough the ground is below the globe's resolution.

An index in [0, 1] per face (0 floodplain-smooth, 1 alpine), rasterised seam-free
like elevation.  The elevation stage writes the tectonic **seed** here; later
stages refine it in place (``docs/plans/ruggedness.md``).  The globe's own relief
doesn't read it: it steers the detail the region, local and settlement tiers add.

The seed sums, per face:

- ``base`` — the master knob (an additive shift of the index, i.e. a uniform
  scale of sub-grid relief, which maps from the index logarithmically);
- **orogenic** — convergent boundaries at their convergence rate (collisions in
  full, arcs and trenches at ``arc_factor``), e-folding over ``orogenic_decay_hops``;
- **rift** — divergent boundaries on continental plates, likewise;
- **volcanic** — volcanic activity;
- **uplift** — height above sea level (``elevation / uplift_ref_km``, clipped to
  [0, 1]): high ground is cut by its rivers, so plateaus far from any boundary
  aren't smooth;
- **interior** — subtracted with distance from any plate boundary (shields, cratons);
- **noise** — low-frequency simplex, so the field isn't a function of tectonics alone;

then diffused over the face graph and clipped.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from core.params import param_field as _f, paramclass
from utils.simplex import fbm_sum, make_perm
from worldgen.resources.terrain import face_hops_from_seeds

if TYPE_CHECKING:
    from core.mesh import PolygonMesh

_MAX_HOPS = 12


@paramclass
class RuggednessParams:
    # Ruggedness before any cause; shifts the whole field (relief scales as a power of it).
    base: float = _f(0.3, label="Ruggedness", min=-0.5, max=1.0, step=0.05)
    # Convergent boundaries: weight at full convergence rate; arcs/trenches vs collisions; e-folding (~220 km faces).
    orogenic_weight: float = _f(0.5, label="Orogenic weight", min=0.0, max=1.5, step=0.05)
    arc_factor: float = _f(0.7, label="Arc / trench share of orogenic", min=0.0, max=1.0, step=0.05)
    orogenic_decay_hops: float = _f(2.0, label="Orogenic decay (faces)", min=0.1, max=10.0, step=0.1)
    # Divergent boundaries on continental plates (rift valleys and their shoulders).
    rift_weight: float = _f(0.25, label="Rift weight", min=0.0, max=1.0, step=0.05)
    rift_decay_hops: float = _f(1.5, label="Rift decay (faces)", min=0.1, max=10.0, step=0.1)
    volcanic_weight: float = _f(0.3, label="Volcanic weight", min=0.0, max=1.0, step=0.05)
    # Height above sea level (km) at which the uplift term is full.
    uplift_weight: float = _f(0.3, label="Uplift weight", min=0.0, max=1.0, step=0.05)
    uplift_ref_km: float = _f(2.0, label="Uplift full at (km)", min=0.1, max=8.0, step=0.1)
    # Subtracted as 1 − exp(−hops / decay) from the nearest plate boundary of any kind.
    interior_weight: float = _f(0.2, label="Plate-interior smoothing", min=0.0, max=1.0, step=0.05)
    interior_decay_hops: float = _f(3.0, label="Plate-interior decay (faces)", min=0.1, max=12.0, step=0.1)
    # Low-frequency variety: noise in ~[-1, 1] times this weight; frequency 5 → features ~1,000–2,000 km.
    noise_weight: float = _f(0.3, label="Noise weight", min=0.0, max=1.0, step=0.05)
    noise_frequency: float = _f(5.0, label="Noise frequency", min=0.5, max=30.0, step=0.5)
    # Neighbour-mean diffusion over the face graph before clipping and rasterising.
    smoothing_passes: int = _f(3, label="Smoothing passes", min=0, max=20)
    smoothing_alpha: float = _f(0.5, label="Smoothing strength", min=0.0, max=1.0, step=0.05)


def decayed_max(
    adjacency: list, strength: np.ndarray, decay_hops: float, max_hops: int = _MAX_HOPS
) -> np.ndarray:
    """``max over sources s of strength[s] · exp(-hops(s, ·) / decay_hops)``, by max-plus relaxation."""
    factor = float(np.exp(-1.0 / max(decay_hops, 1e-6)))
    value = np.asarray(strength, dtype=np.float64).copy()
    for _ in range(max_hops):
        nxt = value.copy()
        for i, nbrs in enumerate(adjacency):
            if nbrs:
                nxt[i] = max(nxt[i], factor * float(value[nbrs].max()))
        if np.array_equal(nxt, value):
            break
        value = nxt
    return value


def diffuse(values: np.ndarray, adjacency: list, passes: int, alpha: float) -> np.ndarray:
    """``passes`` rounds of blending each face ``alpha`` of the way to its neighbours' mean."""
    out = np.asarray(values, dtype=np.float64).copy()
    for _ in range(max(int(passes), 0)):
        nxt = out.copy()
        for i, nbrs in enumerate(adjacency):
            if nbrs:
                nxt[i] += alpha * (out[nbrs].mean() - out[i])
        out = nxt
    return out


def boundary_strengths(
    plate_types: Sequence[str], boundary_data: Sequence[dict], rate_norm: float, arc_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``(orogenic, rift)`` per-face source strengths in [0, 1] from the boundary classification.

    Orogenic: convergent faces at ``|normal| / rate_norm``, × ``arc_factor`` unless a
    continental face meets another continental plate (a collision).  Rift: divergent
    faces on continental plates, at the same rate.
    """
    n = len(boundary_data)
    orogenic = np.zeros(n, dtype=np.float64)
    rift = np.zeros(n, dtype=np.float64)
    norm = max(float(rate_norm), 1e-9)
    for i, bd in enumerate(boundary_data):
        kind = bd["boundary_type"]
        rate = min(abs(float(bd["normal_avg"])) / norm, 1.0)
        if kind == "convergent":
            collision = plate_types[i] == "continental" and "continental" in bd["neighbor_plate_types"]
            orogenic[i] = rate * (1.0 if collision else float(arc_factor))
        elif kind == "divergent" and plate_types[i] == "continental":
            rift[i] = rate
    return orogenic, rift


def ruggedness_seed(
    mesh: "PolygonMesh",
    params: RuggednessParams,
    seed: int,
    *,
    plate_types: Sequence[str],
    boundary_data: Sequence[dict],
    rate_norm: float,
    volcanism: Optional[np.ndarray] = None,
    elevation: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per-face seed ruggedness in [0, 1] and each term's (undiffused) contribution.

    ``elevation``: per-face km (the tectonic value, before texture noise); missing → no uplift term.
    """
    p = params
    adjacency = mesh.face_adjacency
    n = len(mesh.faces)
    pos = np.asarray(mesh.face_centroids, dtype=np.float64)
    pos = pos / np.maximum(np.linalg.norm(pos, axis=1, keepdims=True), 1e-12)
    f0 = float(p.noise_frequency)
    noise = fbm_sum(pos, make_perm(seed), np.array([f0, 2.0 * f0]), np.array([1.0, 0.5])) / 1.5

    orogenic, rift = boundary_strengths(plate_types, boundary_data, rate_norm, float(p.arc_factor))
    terms: dict[str, np.ndarray] = {
        "noise": float(p.noise_weight) * noise,
        "orogenic": float(p.orogenic_weight) * decayed_max(adjacency, orogenic, float(p.orogenic_decay_hops)),
        "rift": float(p.rift_weight) * decayed_max(adjacency, rift, float(p.rift_decay_hops)),
    }
    if volcanism is not None:
        terms["volcanic"] = float(p.volcanic_weight) * np.clip(np.asarray(volcanism, dtype=np.float64), 0.0, 1.0)
    if elevation is not None:
        terms["uplift"] = float(p.uplift_weight) * np.clip(
            np.asarray(elevation, dtype=np.float64) / max(float(p.uplift_ref_km), 1e-9), 0.0, 1.0)
    boundary = np.array([bd["boundary_type"] != "interior" for bd in boundary_data], dtype=bool)
    if boundary.any():
        hops = face_hops_from_seeds(mesh, boundary, max_hops=_MAX_HOPS).astype(np.float64)
        terms["interior"] = -float(p.interior_weight) * (
            1.0 - np.exp(-hops / max(float(p.interior_decay_hops), 1e-6)))

    raw = np.full(n, float(p.base)) + sum(terms.values())
    out = diffuse(raw, adjacency, p.smoothing_passes, float(p.smoothing_alpha))
    return np.clip(out, 0.0, 1.0), terms


def ruggedness_texture(mesh: "PolygonMesh", face_values: np.ndarray, tex_size: int) -> np.ndarray:
    """``(N, H, W, 1)`` float32: ``face_values`` rasterised as a smooth field of position, seams averaged."""
    from worldgen.geology.elevation import rasterize_face_values, smooth_texture_seams
    tex = rasterize_face_values(mesh, np.asarray(face_values, dtype=np.float64), tex_size)
    smooth_texture_seams(mesh, tex, n_passes=1)
    return np.clip(tex, 0.0, 1.0).astype(np.float32)
