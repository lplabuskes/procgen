"""Annually-averaged surface ocean currents and their climate feedback.

Intentionally simple (Climate-Cookbook level, not a GCM):

- The surface current at an ocean face is `ekman_coupling` times the
  annual-mean wind, rotated 45° via Ekman drift (right in the northern
  hemisphere, left in the southern).
- The current vector is projected onto outgoing edges of the face graph
  and restricted to ocean neighbours; land blocks flow and the remaining
  ocean neighbours absorb the redistributed weight. Faces whose current
  points entirely into land become natural pile-up sinks.
- The local annual-mean insolation is used as the water's "thermal
  payload" and advected through the resulting sub-stochastic Markov
  field with a per-iteration mixing factor `advection_alpha`.
- The departure of advected vs local insolation, scaled to °C, is the
  per-ocean-face temperature anomaly. A coast-tracking Dijkstra carries
  that anomaly inland with exponential decay over `inland_decay_km`.
- The precipitation delta is proportional to the temperature anomaly
  (warm current → wet, cold current → dry). It is passed back as a
  multiplicative `(1 + delta)` correction.

No seasonal breakdown — the model produces a single per-face (N,)
anomaly that the climate builder broadcasts across seasons via the
`coastal_current_temp_delta` and `current_wet_delta` hooks already
exposed by `surface_temperatures` and `surface_precipitation`.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from worldgen.climate.winds import _tangent_frames
from worldgen.mesh.mymesh import PolygonMesh


EARTH_RADIUS_KM = 6371.0


@dataclass
class CurrentsResult:
    currents: np.ndarray              # (N, 3) m/s tangent vectors (0 on land)
    temperature_delta: np.ndarray     # (N,) °C
    precipitation_delta: np.ndarray   # (N,) multiplicative wet delta
    advected_insolation: np.ndarray   # (N,) W/m² post-advection (for inspection)


def _ekman_currents(face_centroids: np.ndarray,
                    annual_winds: np.ndarray,
                    is_ocean: np.ndarray,
                    ekman_coupling: float) -> np.ndarray:
    """Annual-mean wind → annual-mean surface current via 45° Ekman deflection."""
    east, north = _tangent_frames(face_centroids)
    we = (annual_winds * east).sum(axis=1)
    wn = (annual_winds * north).sum(axis=1)

    # "Right of wind" (NH) is clockwise in the (east, north) plane viewed from
    # outside the sphere with north up — i.e. a negative-angle rotation.
    hemi_sign = np.where(face_centroids[:, 2] >= 0.0, 1.0, -1.0)
    angle = -0.25 * np.pi * hemi_sign
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    ce = ekman_coupling * (cos_a * we - sin_a * wn)
    cn = ekman_coupling * (sin_a * we + cos_a * wn)

    ce = np.where(is_ocean, ce, 0.0)
    cn = np.where(is_ocean, cn, 0.0)
    return ce[:, None] * east + cn[:, None] * north


def _build_advection_edges(mesh: PolygonMesh,
                           currents: np.ndarray,
                           is_ocean: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sparse flow edges (src, dst, weight) on the ocean-only face graph.

    Each edge weight is `max(0, current[src] · direction_to_dst)` — the
    component of the source's current vector that actually points toward
    the destination. Normalisation happens at the destination side inside
    `_advect_field` so each face takes a weighted *average* of its inflow
    rather than a sum (which would blow up at convergence sinks).
    """
    centroids = mesh.face_centroids.astype(np.float64)
    src_list: list[int] = []
    dst_list: list[int] = []
    w_list: list[float] = []
    for f in range(len(mesh.faces)):
        if not is_ocean[f]:
            continue
        cvec = currents[f]
        if float(np.dot(cvec, cvec)) < 1e-20:
            continue
        n_hat = centroids[f]
        for nb in mesh.face_adjacency[f]:
            if not is_ocean[nb]:
                continue
            d = centroids[nb] - n_hat
            d -= d.dot(n_hat) * n_hat            # project onto tangent plane at f
            dn = float(np.linalg.norm(d))
            if dn < 1e-12:
                continue
            w = float(cvec.dot(d) / dn)
            if w > 0.0:
                src_list.append(f)
                dst_list.append(nb)
                w_list.append(w)
    if not src_list:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64))
    return (np.asarray(src_list, dtype=np.int64),
            np.asarray(dst_list, dtype=np.int64),
            np.asarray(w_list, dtype=np.float64))


def _advect_field(local: np.ndarray,
                  is_ocean: np.ndarray,
                  src: np.ndarray,
                  dst: np.ndarray,
                  weights: np.ndarray,
                  iterations: int,
                  alpha: float) -> np.ndarray:
    """Damped Jacobi iteration with per-destination inflow averaging.

    advected[f] = (1-α)·local[f] + α·(Σ_in w·advected[upstream] / Σ_in w)

    Per-destination normalisation keeps the operator row-stochastic over
    faces with any inflow (zero rows elsewhere), so the steady state is
    bounded between min and max of `local`. The (1-α) anchor also keeps
    the system contracting in the presence of cycles. Faces with no
    inflow (sources / blocked-by-land currents) are pinned to their own
    local value — they emit but do not absorb.
    """
    n = len(local)
    advected = local.copy().astype(np.float64)
    base = (1.0 - alpha) * local

    inflow_total = np.zeros(n, dtype=np.float64)
    if src.size:
        np.add.at(inflow_total, dst, weights)
    has_inflow = inflow_total > 0.0
    inv_inflow = np.zeros(n, dtype=np.float64)
    inv_inflow[has_inflow] = 1.0 / inflow_total[has_inflow]

    for _ in range(iterations):
        contribution = np.zeros(n, dtype=np.float64)
        if src.size:
            np.add.at(contribution, dst, weights * advected[src])
        avg_inflow = contribution * inv_inflow
        new = np.where(has_inflow, base + alpha * avg_inflow, local)
        new[~is_ocean] = local[~is_ocean]
        advected = new
    return advected


def _propagate_inland(ocean_delta: np.ndarray,
                      mesh: PolygonMesh,
                      is_ocean: np.ndarray,
                      decay_km: float) -> tuple[np.ndarray, np.ndarray]:
    """Dijkstra from every ocean face, tracking the source ocean for each land face.

    Returns (full_delta, distance_km). For ocean faces, full_delta equals
    `ocean_delta`. For land faces it is `ocean_delta[parent] · exp(-d/decay)`.
    Unreachable land (none in practice) ends up at zero.
    """
    n = len(mesh.faces)
    centroids = mesh.face_centroids.astype(np.float64)
    dist = np.full(n, np.inf, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int64)
    heap: list[tuple[float, int]] = []
    for i in np.flatnonzero(is_ocean):
        dist[i] = 0.0
        parent[i] = int(i)
        heapq.heappush(heap, (0.0, int(i)))
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v in mesh.face_adjacency[u]:
            cos_ang = float(np.clip(centroids[u] @ centroids[v], -1.0, 1.0))
            edge = EARTH_RADIUS_KM * np.arccos(cos_ang)
            nd = d + edge
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = parent[u]
                heapq.heappush(heap, (nd, int(v)))
    dist[~np.isfinite(dist)] = 0.0

    delta_full = np.zeros(n, dtype=np.float64)
    has_parent = parent >= 0
    decay = max(float(decay_km), 1e-6)
    delta_full[has_parent] = (
        ocean_delta[parent[has_parent]] * np.exp(-dist[has_parent] / decay)
    )
    return delta_full, dist


def compute_currents(mesh: PolygonMesh,
                     face_elev_km: np.ndarray,
                     face_centroids: np.ndarray,
                     winds_per_season: np.ndarray,
                     face_insolation: np.ndarray,
                     *,
                     sea_level: float = 0.0,
                     ekman_coupling: float = 0.03,
                     advection_alpha: float = 0.7,
                     advection_iterations: int = 200,
                     temperature_scale: float = 0.04,
                     precipitation_scale: float = 0.02,
                     inland_decay_km: float = 600.0) -> CurrentsResult:
    """Run the annual-mean ocean current model and return per-face deltas.

    Args:
        mesh: dual `PolygonMesh` (after elevation has been built).
        face_elev_km: (N,) per-face elevation in km.
        face_centroids: (N, 3) unit-sphere face positions.
        winds_per_season: (N, K, 3) seasonal wind vectors from `zonal_winds`.
        face_insolation: (N,) annual-mean insolation in W/m² (mean of the
            seasonal `seasonal_insolation` output along the season axis).
        sea_level: elevation threshold defining the ocean mask.
        ekman_coupling: surface-current speed as a fraction of wind speed.
        advection_alpha: per-iteration mixing fraction in the Jacobi advection.
        advection_iterations: number of Jacobi sweeps; ~200 is plenty for
            current depths/icosphere sizes used in this project.
        temperature_scale: °C per (W/m² of advected-vs-local insolation gap).
        precipitation_scale: multiplicative wet-delta per °C of anomaly.
        inland_decay_km: e-folding distance for inland propagation of the
            ocean anomaly.
    """
    n = len(mesh.faces)
    is_ocean = face_elev_km <= sea_level

    annual_winds = winds_per_season.astype(np.float64).mean(axis=1)        # (N, 3)
    currents = _ekman_currents(face_centroids.astype(np.float64),
                               annual_winds,
                               is_ocean,
                               ekman_coupling)                              # (N, 3)

    src, dst, w = _build_advection_edges(mesh, currents, is_ocean)
    advected = _advect_field(face_insolation.astype(np.float64),
                             is_ocean,
                             src, dst, w,
                             advection_iterations,
                             advection_alpha)

    ocean_dQ = np.zeros(n, dtype=np.float64)
    ocean_dQ[is_ocean] = advected[is_ocean] - face_insolation[is_ocean]
    ocean_dT = temperature_scale * ocean_dQ

    temp_delta, _ = _propagate_inland(ocean_dT, mesh, is_ocean, inland_decay_km)
    precip_delta = precipitation_scale * temp_delta

    return CurrentsResult(
        currents=currents.astype(np.float32),
        temperature_delta=temp_delta.astype(np.float32),
        precipitation_delta=precip_delta.astype(np.float32),
        advected_insolation=advected.astype(np.float32),
    )
