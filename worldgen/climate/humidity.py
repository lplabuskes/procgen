"""Atmospheric moisture transport: SST-driven evaporation advected along seasonal winds.

Continental drying, ITCZ convergence and rain shadows emerge from transport:

1. ``evaporation_source`` — ocean faces emit moisture proportional to SST.
2. ``advect_moisture``    — Jacobi transport along wind; a background decay
                            is the precipitation sink.
3. ``moisture_to_precipitation`` — q_air × orographic factor × scale → mm/season.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is an optional accelerator
    def njit(func=None, **_):  # type: ignore[misc]
        return func if func is not None else (lambda f: f)


def _adjacency_csr(face_adjacency: list) -> tuple[np.ndarray, np.ndarray]:
    """``(indptr, indices)`` int64 CSR of an adjacency list, neighbour order kept."""
    counts = np.fromiter((len(a) for a in face_adjacency), dtype=np.int64, count=len(face_adjacency))
    indptr = np.zeros(len(face_adjacency) + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    indices = np.fromiter((j for a in face_adjacency for j in a), dtype=np.int64, count=int(indptr[-1]))
    return indptr, indices


def evaporation_source(
    face_temps_c: np.ndarray,
    face_elev_km: np.ndarray,
    evaporation_scale: float = 1.0,
    sea_level: float = 0.0,
    is_ocean: np.ndarray | None = None,
) -> np.ndarray:
    """Moisture emission rate for ocean faces; zero on land.

    Temperature factor is linear over [−25 °C, 27 °C].

    Args:
        face_temps_c:     (N, K) seasonal face temperatures in °C.
        face_elev_km:     (N,) mean face elevation in km.
        evaporation_scale: Global wetness multiplier (1.0 = Earth-like default).
        is_ocean:         Optional (N,) bool; replaces the elevation-based ocean test.

    Returns:
        (N, K) float32 dimensionless moisture availability in [0, evaporation_scale].
    """
    if is_ocean is None:
        is_ocean = face_elev_km <= sea_level                      # (N,)
    t_factor = np.clip((face_temps_c + 25.0) / 52.0, 0.0, 1.0)  # (N, K)
    source = (t_factor * evaporation_scale).astype(np.float32)
    source[~is_ocean] = 0.0
    return source


def _build_wind_edges(
    face_centroids: np.ndarray,
    face_adjacency: list,
    winds_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outgoing advection edges for one season.

    Edge i → j exists when wind at i has a positive component along i→j;
    weights are that flux, normalised to sum to 1 per source face.

    Returns:
        (src, dst, weight) int32 / int32 / float32 arrays.
    """
    indptr, indices = _adjacency_csr(face_adjacency)
    src, dst, wt, n_out = _wind_edges_kernel(
        np.asarray(face_centroids, dtype=np.float64), indptr, indices,
        np.asarray(winds_k, dtype=np.float64),
    )
    return src[:n_out], dst[:n_out], wt[:n_out]


@njit(cache=True)
def _wind_edges_kernel(centroids, indptr, indices, winds):  # pragma: no cover - compiled
    m = indices.shape[0]
    src = np.empty(m, dtype=np.int32)
    dst = np.empty(m, dtype=np.int32)
    wt = np.empty(m, dtype=np.float32)
    flux = np.empty(m, dtype=np.float64)
    n_out = 0
    for i in range(indptr.shape[0] - 1):
        total = 0.0
        first = n_out
        for e in range(indptr[i], indptr[i + 1]):
            j = indices[e]
            f = (winds[i, 0] * (centroids[j, 0] - centroids[i, 0])
                 + winds[i, 1] * (centroids[j, 1] - centroids[i, 1])
                 + winds[i, 2] * (centroids[j, 2] - centroids[i, 2]))
            if f > 0.0:
                dst[n_out] = j
                flux[n_out] = f
                total += f
                n_out += 1
        if total > 0.0:
            inv = 1.0 / total
            for e in range(first, n_out):
                src[e] = i
                wt[e] = np.float32(flux[e] * inv)
        else:
            n_out = first
    return src, dst, wt, n_out


def advect_moisture(
    evaporation: np.ndarray,
    face_centroids: np.ndarray,
    face_adjacency: list,
    winds: np.ndarray,
    n_iter: int = 200,
    decay_rate: float = 0.025,
    _transport_step: float = 0.5,
) -> np.ndarray:
    """Advect atmospheric moisture to near-steady-state along seasonal winds.

    Each iteration: every face sends ``transport_step × q`` downwind, loses
    ``decay_rate × q`` to precipitation, and ocean faces are topped back up to
    their evaporation source.

    Args:
        evaporation:      (N, K) source strengths from ``evaporation_source``.
        face_centroids:   (N, 3) unit-sphere centroids.
        face_adjacency:   adjacency list from ``PolygonMesh``.
        winds:            (N, K, 3) seasonal wind vectors in m/s.
        n_iter:           Jacobi iterations (200 ≈ steady state at ~10 k faces).
        decay_rate:       Fraction removed per step; sets inland drying rate.
        _transport_step:  Fraction moved per step; 0.5 for stability (CFL < 1), not a knob.

    Returns:
        (N, K) float32 equilibrium specific humidity (dimensionless).
    """
    N, K = evaporation.shape
    is_ocean = evaporation.sum(axis=1) > 0   # (N,) bool — has a source term
    q_air = np.zeros((N, K), dtype=np.float32)

    for k in range(K):
        src, dst, w = _build_wind_edges(face_centroids, face_adjacency, winds[:, k, :])
        evap_k = evaporation[:, k]            # (N,)
        q_k = evap_k.copy()

        q_air[:, k] = _advect_season(
            q_k, evap_k, src, dst, w, is_ocean, int(n_iter),
            q_k.dtype.type(_transport_step), q_k.dtype.type(1.0 - decay_rate),
        )

    return q_air


@njit(cache=True)
def _advect_season(q, evap, src, dst, w, is_ocean, n_iter, step, keep):  # pragma: no cover - compiled
    """``n_iter`` Jacobi transport steps of one season (see :func:`advect_moisture`).

    Runs in the field's own precision, adding in edge order.
    """
    n = q.shape[0]
    for _ in range(n_iter):
        q_sent = q * step
        q_new = q - q_sent
        for e in range(src.shape[0]):
            q_new[dst[e]] += q_sent[src[e]] * w[e]
        q_new *= keep
        for i in range(n):
            if is_ocean[i] and evap[i] > q_new[i]:
                q_new[i] = evap[i]
        q = q_new
    return q


def smooth_humidity(
    q_air: np.ndarray,
    face_adjacency: list,
    n_passes: int = 10,
    alpha: float = 0.4,
) -> np.ndarray:
    """Laplacian graph smoothing of the humidity field.

    Diffuses ITCZ stripes and softens face-edge discontinuities in the biome map.

    Args:
        q_air:          (N, K) equilibrium humidity from advect_moisture.
        alpha:          neighbour-mean blend weight (0 = no change, 1 = full replace).

    Returns:
        (N, K) float32 smoothed humidity.
    """
    q = q_air.astype(np.float32)
    indptr, indices = _adjacency_csr(face_adjacency)
    for _ in range(n_passes):
        q = _smooth_pass(q, indptr, indices, np.float32(alpha))
    return q


@njit(cache=True)
def _smooth_pass(q, indptr, indices, alpha):  # pragma: no cover - compiled
    """One pass of :func:`smooth_humidity`, in float32 like the numpy original.

    The neighbour mean sums in float32 in adjacency order and divides in float64
    (numpy's ``mean`` divides by an integer count); the blend is float32.
    """
    out = q.copy()
    for i in range(q.shape[0]):
        s0, s1 = indptr[i], indptr[i + 1]
        if s1 == s0:
            continue
        for k in range(q.shape[1]):
            acc = q[indices[s0], k]
            for e in range(s0 + 1, s1):
                acc = np.float32(acc + q[indices[e], k])
            mean = np.float32(np.float64(acc) / np.float64(s1 - s0))
            out[i, k] = np.float32(out[i, k] + alpha * np.float32(mean - q[i, k]))
    return out


def per_pixel_humidity(
    q_air: np.ndarray,
    mesh,
    tex_size: int,
) -> np.ndarray:
    """Rasterize (N, K) face humidity to (N, H, W, K) float32 via shared-vertex interpolation."""
    from worldgen.geology.elevation import rasterize_face_values, smooth_texture_seams

    h_tex = rasterize_face_values(mesh, q_air.astype(np.float64), tex_size)  # (N, H, W, K)
    smooth_texture_seams(mesh, h_tex, n_passes=1)
    return h_tex


def moisture_to_precipitation(
    q_air: np.ndarray,
    winds: np.ndarray,
    elevation_gradient: np.ndarray,
    orographic_k: float,
    precipitation_scale: float,
) -> np.ndarray:
    """Convert equilibrium humidity to face-level precipitation (mm/season).

    Orographic enhancement ``exp(k × wind·∇h)`` on windward slopes.

    Args:
        q_air:               (N, K) from ``advect_moisture``.
        winds:               (N, K, 3) m/s seasonal wind vectors.
        elevation_gradient:  (N, 3) face elevation gradient (km / km).
        orographic_k:        Orographic enhancement coefficient.
        precipitation_scale: Annual precipitation at full moisture (mm / yr).

    Returns:
        (N, K) float32 mm / season.
    """
    uplift = np.einsum("nki,ni->nk", winds, elevation_gradient)     # (N, K)
    oro = np.exp(orographic_k * np.clip(uplift, 0.0, None))         # (N, K)
    mm_per_season = precipitation_scale / 4.0
    return (q_air * oro * mm_per_season).astype(np.float32)
