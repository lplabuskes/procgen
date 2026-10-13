"""Terrain character — jagged vs rolling, beside ruggedness's how-much.

An index in [0, 1] per face (0 rolling, 1 jagged), rasterised seam-free like
ruggedness.  Ruggedness sets how much sub-grid relief there is; this sets its
shape: the relief spectrum's Hurst exponent (``region/upsample.py::hurst_exponent``).
Jagged ground keeps its relief at fine scales (sharp crests, broken slopes);
rolling ground puts it in the long wavelengths (soil-mantled, rounded hills).
0.5 is the neutral character the spectrum was calibrated on.

The elevation stage writes the tectonic **seed** here; the ruggedness stage
refines it with climate (``worldgen/ruggedness/builder.py``).  The seed sums,
per face:

- ``base`` — the neutral character;
- **young** — convergent boundaries at their convergence rate (collisions in full,
  arcs and trenches at ``arc_factor``), e-folding over ``young_decay_hops``:
  active orogens are young, steep and sharp;
- **old** — subtracted with distance from any plate boundary: the ranges and
  shields of plate interiors have been worn down and rounded;

then diffused over the face graph and clipped.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from core.params import param_field as _f, paramclass
from worldgen.geology.ruggedness import _MAX_HOPS, boundary_strengths, decayed_max, diffuse
from worldgen.resources.terrain import face_hops_from_seeds

if TYPE_CHECKING:
    from core.mesh import PolygonMesh

# Neutral character: the spectrum's own ``hurst`` (and the default where a world has no field).
NEUTRAL_JAGGEDNESS = 0.5


@paramclass
class JaggednessParams:
    # Character before any cause (0 rolling, 1 jagged; 0.5 is the spectrum's own Hurst exponent).  Above
    # neutral because most land is plate interior: seeds 4/31 land median ~0.5 after the climate terms.
    base: float = _f(0.62, label="Jaggedness", min=0.0, max=1.0, step=0.05)
    # Convergent boundaries: weight at full convergence rate; arcs/trenches vs collisions; e-folding (~220 km faces).
    young_weight: float = _f(0.4, label="Young orogen weight", min=0.0, max=1.0, step=0.05)
    arc_factor: float = _f(0.7, label="Arc / trench share of young", min=0.0, max=1.0, step=0.05)
    young_decay_hops: float = _f(2.0, label="Young orogen decay (faces)", min=0.1, max=10.0, step=0.1)
    # Subtracted as 1 − exp(−hops / decay) from the nearest plate boundary of any kind.
    old_weight: float = _f(0.35, label="Plate-interior rounding", min=0.0, max=1.0, step=0.05)
    old_decay_hops: float = _f(3.0, label="Plate-interior decay (faces)", min=0.1, max=12.0, step=0.1)
    # Neighbour-mean diffusion over the face graph before clipping and rasterising.
    smoothing_passes: int = _f(2, label="Smoothing passes", min=0, max=20)
    smoothing_alpha: float = _f(0.5, label="Smoothing strength", min=0.0, max=1.0, step=0.05)


def jaggedness_seed(
    mesh: "PolygonMesh",
    params: JaggednessParams,
    *,
    plate_types: Sequence[str],
    boundary_data: Sequence[dict],
    rate_norm: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per-face seed jaggedness in [0, 1] and each term's (undiffused) contribution."""
    p = params
    adjacency = mesh.face_adjacency
    n = len(mesh.faces)
    orogenic, _ = boundary_strengths(plate_types, boundary_data, rate_norm, float(p.arc_factor))
    terms: dict[str, np.ndarray] = {
        "young": float(p.young_weight) * decayed_max(adjacency, orogenic, float(p.young_decay_hops)),
    }
    boundary = np.array([bd["boundary_type"] != "interior" for bd in boundary_data], dtype=bool)
    if boundary.any():
        hops = face_hops_from_seeds(mesh, boundary, max_hops=_MAX_HOPS).astype(np.float64)
        terms["old"] = -float(p.old_weight) * (1.0 - np.exp(-hops / max(float(p.old_decay_hops), 1e-6)))

    raw = np.full(n, float(p.base)) + sum(terms.values())
    out = diffuse(raw, adjacency, p.smoothing_passes, float(p.smoothing_alpha))
    return np.clip(out, 0.0, 1.0), terms
