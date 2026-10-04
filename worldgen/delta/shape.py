"""Delta shape — whether the river or the waves win at its mouth.

A delta's plan is set by the balance between the sediment its river delivers
and the waves that rework it (Galloway's triangle; tides are not modelled):

- **River-dominated → bird's-foot.**  In calm, sheltered water the river's
  channels build long levee-fringed fingers straight out to sea, with open bays
  between them (Mississippi).
- **Balanced → arcuate.**  A rounded fan (Nile, Niger).
- **Wave-dominated → cuspate.**  Waves sweep the sediment along shore into
  beach ridges, so the delta barely protrudes: a blunt cusp at the mouth with
  straight flanks running along the coast (São Francisco, Tiber, Ebro).

**Wave power** at a mouth comes from the onshore wind over the open sea in
front of it: for bearings within 90° of the fan's axis, a ray is marched over
ocean faces until it meets land (the *fetch*, capped at ``fetch_cap_km``), and
each bearing contributes ``cos φ · (onshore wind³ + swell³) · fetch / cap``
(deep-water wave power grows as the wind speed cubed).
``swell_speed`` stands in for the swell distant storms send in, whatever the
local wind.  **River power** is ``(flow / 1000) ** river_exponent``.  The
**wave dominance** is ``W / (W + river_weight · R)``, with ``W`` the wave power
over ``(7 m/s)³``: 0 is all river, 1 all wave.

The flood's priority is a blend of three shape metrics, weighted by where the
dominance falls between ``birdfoot_below`` and ``cuspate_above`` (with a soft
``shape_blend`` either side):

- *arcuate* — distance from the apex, raised off the axis (the original fan);
- *bird's-foot* — seaward of the old mouth, distance along the nearest of a few
  *finger* rays fanned out from the mouth, with leaving the finger's narrow
  band or running past its length dear (``finger_cost``);
- *cuspate* — seaward of the old mouth, distance forward plus distance along
  the coast × ``cot(cusp_angle / 2)``, so the front is two straight beaches
  meeting at a blunt point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.spatial import KDTree

if TYPE_CHECKING:
    from worldgen.delta.builder import DeltaParams

EARTH_RADIUS_KM = 6371.0
# Wave power is counted in units of this wind speed cubed (the trades).
_REFERENCE_WIND = 7.0
# Bearings sampled across the seaward half-plane.
_N_BEARINGS = 13
# Fetch rays step this many face spacings, and start this far out (the mouth's
# own face is usually land-majority).
_FETCH_STEP_FACES = 0.5
_FETCH_START_FACES = 0.75

BIRDFOOT, ARCUATE, CUSPATE = "birdfoot", "arcuate", "cuspate"
SHAPE_NAMES = {BIRDFOOT: "bird's-foot", ARCUATE: "arcuate", CUSPATE: "cuspate"}


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-15)


def _tangent(v: np.ndarray, at: np.ndarray) -> np.ndarray:
    return _unit(v - at * float(at @ v))


def _smoothstep(lo: float, hi: float, x: float) -> float:
    t = float(np.clip((x - lo) / (hi - lo), 0.0, 1.0)) if hi != lo else float(x >= hi)
    return t * t * (3.0 - 2.0 * t)


class WaveField:
    """Face-level open sea and winds, for measuring wave power at a coast."""

    def __init__(self, centroids: np.ndarray, sea: np.ndarray, winds: Optional[np.ndarray],
                 face_spacing_rad: float) -> None:
        self.centroids = np.asarray(centroids, dtype=np.float64)
        self.sea = np.asarray(sea, dtype=bool)
        # (N, K, 3) per-season winds, or none (then only swell).
        self.winds = None if winds is None else np.asarray(winds, dtype=np.float64)
        self.tree = KDTree(self.centroids)
        self.spacing = float(face_spacing_rad)

    def power(self, mouth: np.ndarray, axis: np.ndarray, params: "DeltaParams") -> float:
        """Wave power at ``mouth`` from the sea in front of it, in ``(7 m/s)³``."""
        mouth = _unit(np.asarray(mouth, dtype=np.float64))
        a = _tangent(np.asarray(axis, dtype=np.float64), mouth)
        b = np.cross(mouth, a)
        cap = float(params.fetch_cap_km) / EARTH_RADIUS_KM
        step = _FETCH_STEP_FACES * self.spacing
        arcs = np.arange(_FETCH_START_FACES * self.spacing, cap + 0.5 * step, step)
        swell3 = float(params.swell_speed) ** 3
        total, weights = 0.0, 0.0
        for phi in np.linspace(-0.5 * np.pi, 0.5 * np.pi, _N_BEARINGS):
            d = np.cos(phi) * a + np.sin(phi) * b
            w = float(np.cos(phi))
            weights += w
            if w <= 1e-9:
                continue
            ray = np.cos(arcs)[:, None] * mouth[None, :] + np.sin(arcs)[:, None] * d[None, :]
            _, faces = self.tree.query(ray)
            faces = np.asarray(faces, dtype=np.int64)
            land = np.flatnonzero(~self.sea[faces])
            n_sea = int(land[0]) if land.size else len(faces)
            if n_sea == 0:
                continue
            fetch = min(arcs[n_sea - 1] + 0.5 * step, cap) / cap
            onshore3 = 0.0
            if self.winds is not None:
                over = self.winds[faces[:n_sea]]                         # (S, K, 3)
                # The direction the waves travel at each sample: back toward the mouth.
                back = -_ray_tangents(ray[:n_sea], mouth)                 # (S, 3)
                along = np.einsum("skc,sc->sk", over, back)
                onshore3 = float(np.mean(np.maximum(along, 0.0) ** 3))
            total += w * (onshore3 + swell3) * fetch
        return total / max(weights, 1e-12) / _REFERENCE_WIND ** 3


def _ray_tangents(points: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Unit tangents at ``points`` of the great circles from ``origin``, pointing away from it."""
    t = points * (points @ origin)[:, None] - origin[None, :]
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-15)


def wave_dominance(wave_power: float, flow: float, params: "DeltaParams") -> float:
    """0 (river-dominated) … 1 (wave-dominated)."""
    river = float(params.river_weight) * (max(flow, 0.0) / 1000.0) ** float(params.river_exponent)
    return float(wave_power / max(wave_power + river, 1e-12))


@dataclass
class DeltaShape:
    """How one delta's footprint is shaped."""

    wave_dominance: float = 0.5
    birdfoot: float = 0.0          # blend weights; arcuate = 1 − birdfoot − cuspate
    cuspate: float = 0.0
    head: Optional[np.ndarray] = None    # (3,) the old mouth, where fingers and cusp start
    # (F, 3) finger directions, tangent at ``head``, and their lengths (km).
    finger_dirs: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    finger_km: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def arcuate(self) -> float:
        return max(1.0 - self.birdfoot - self.cuspate, 0.0)

    @property
    def name(self) -> str:
        return max(((self.birdfoot, BIRDFOOT), (self.arcuate, ARCUATE), (self.cuspate, CUSPATE)))[1]


def finger_count(flow: float, params: "DeltaParams") -> int:
    k = float(params.finger_base) + float(params.finger_per_decade) * np.log10(max(flow, 1.0) / 1000.0)
    return int(np.clip(round(k), 2, int(params.finger_max)))


def make_shape(dominance: float, head: np.ndarray, axis: np.ndarray, flow: float, size_km: float,
               params: "DeltaParams", rng: np.random.Generator) -> DeltaShape:
    """Blend weights for ``dominance``, and the fingers a bird's-foot would grow."""
    h = 0.5 * float(params.shape_blend)
    bird = 1.0 - _smoothstep(float(params.birdfoot_below) - h, float(params.birdfoot_below) + h, dominance)
    cusp = _smoothstep(float(params.cuspate_above) - h, float(params.cuspate_above) + h, dominance)
    at = _unit(np.asarray(head, dtype=np.float64))
    shape = DeltaShape(wave_dominance=float(dominance), birdfoot=bird, cuspate=cusp, head=at)
    if bird <= 0.0:
        return shape
    a = _tangent(np.asarray(axis, dtype=np.float64), at)
    b = np.cross(at, a)
    n = finger_count(flow, params)
    spread = np.deg2rad(float(params.finger_spread_deg))
    gap = 2.0 * spread / max(n - 1, 1)
    phis = np.linspace(-spread, spread, n) + rng.uniform(-0.2, 0.2, n) * gap
    # The finger nearest the axis is the longest; the rest fall short of it by up to ``finger_jitter``.
    lengths = float(params.finger_length) * size_km * (1.0 - float(params.finger_jitter) * rng.uniform(0, 1, n))
    lengths[int(np.argmin(np.abs(phis)))] = float(params.finger_length) * size_km
    shape.finger_dirs = np.cos(phis)[:, None] * a[None, :] + np.sin(phis)[:, None] * b[None, :]
    shape.finger_km = lengths
    return shape


def shape_metric(
    pos: np.ndarray, apex: np.ndarray, axis: np.ndarray, arcuate: np.ndarray,
    shape: DeltaShape, spacing_km: float, params: "DeltaParams",
) -> np.ndarray:
    """The flood's distance metric (km) at ``pos``, blending the three shapes.

    ``arcuate`` is the arcuate metric already computed at ``pos``.
    """
    if shape.head is None or (shape.birdfoot <= 0.0 and shape.cuspate <= 0.0):
        return arcuate
    head = shape.head
    to_head = float(np.linalg.norm(head - apex)) * EARTH_RADIUS_KM
    a = _tangent(axis, head)
    rel = (pos - head[None, :]) * EARTH_RADIUS_KM
    forward = rel @ a
    seaward = forward > 0.0
    out = shape.arcuate * arcuate
    if shape.cuspate > 0.0:
        lateral = np.linalg.norm(rel - forward[:, None] * a[None, :], axis=1)
        k = 1.0 / np.tan(0.5 * np.deg2rad(float(params.cusp_angle_deg)))
        cusp = np.where(seaward, to_head + forward + k * lateral, arcuate)
        out = out + shape.cuspate * cusp
    if shape.birdfoot > 0.0:
        half = float(params.finger_half_width) * spacing_km
        cost = float(params.finger_cost)
        best = np.full(len(pos), np.inf)
        for d, length in zip(shape.finger_dirs, shape.finger_km):
            along = rel @ d
            perp = np.linalg.norm(rel - along[:, None] * d[None, :], axis=1)
            r = to_head + along + cost * (np.maximum(perp - half, 0.0) + np.maximum(along - length, 0.0))
            best = np.minimum(best, np.where(along >= 0.0, r, np.inf))
        bird = np.where(seaward & np.isfinite(best), best, arcuate)
        out = out + shape.birdfoot * bird
    return out


def finger_tips(pos: np.ndarray, taken: np.ndarray, shape: DeltaShape, spacing_km: float,
                params: "DeltaParams") -> np.ndarray:
    """``(F, 3)`` the farthest plain texel along each finger that was actually built."""
    if shape.head is None or shape.birdfoot < 0.5 or not taken.any():
        return np.zeros((0, 3))
    rel = (pos[taken] - shape.head[None, :]) * EARTH_RADIUS_KM
    reach = (float(params.finger_half_width) + 1.0) * spacing_km
    tips = []
    for d in shape.finger_dirs:
        along = rel @ d
        perp = np.linalg.norm(rel - along[:, None] * d[None, :], axis=1)
        on = (perp <= reach) & (along >= 1.5 * spacing_km)
        if on.any():
            tips.append(pos[taken][np.flatnonzero(on)[int(np.argmax(along[on]))]])
    return np.asarray(tips, dtype=np.float64).reshape(-1, 3)
