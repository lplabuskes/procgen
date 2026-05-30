"""Atmospheric moisture transport and humidity model.

Replaces the parameterised zonal precipitation baselines and continentality
decay with an SST-driven evaporation source advected along seasonal winds.
Continental drying, ITCZ convergence, and rain shadows all emerge from the
transport physics rather than per-parameter tuning.

Pipeline
--------
1. ``evaporation_source`` — ocean faces emit moisture proportional to SST.
2. ``advect_moisture``    — conservative Jacobi transport along seasonal wind
                            vectors; a background decay rate acts as the
                            precipitation sink that dries continental interiors.
3. ``moisture_to_precipitation`` — q_air × orographic factor × scale → mm/season.
"""
from __future__ import annotations

import numpy as np


def evaporation_source(
    face_temps_c: np.ndarray,
    face_elev_km: np.ndarray,
    evaporation_scale: float = 1.0,
    sea_level: float = 0.0,
    is_ocean: np.ndarray | None = None,
) -> np.ndarray:
    """Moisture emission rate for ocean faces; zero on land.

    The temperature factor is linear in [−25 °C, 27 °C], giving stronger
    evaporation from warm tropical oceans and negligible evaporation from cold
    polar seas.  Land faces are zeroed regardless of temperature.

    Args:
        face_temps_c:     (N, K) seasonal face temperatures in °C.
        face_elev_km:     (N,) mean face elevation in km.
        evaporation_scale: Global wetness multiplier (1.0 = Earth-like default).
        is_ocean:         Optional (N,) bool override. When provided, replaces
                          the elevation-based ocean test.

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

    An edge (i → j) exists when the wind vector at face i has a positive
    component in the direction i→j.  Outgoing weights are normalised to sum
    to 1 per source face so that ``transport_step × q`` is distributed among
    all downwind neighbours in proportion to their directional flux.

    Returns:
        (src, dst, weight) int32 / int32 / float32 arrays.
    """
    N = len(face_adjacency)
    src_list: list[int] = []
    dst_list: list[int] = []
    wt_list: list[float] = []

    for i in range(N):
        c_i = face_centroids[i]
        w_i = winds_k[i]
        local: list[tuple[int, float]] = []
        total_out = 0.0
        for j in face_adjacency[i]:
            flux = float(np.dot(w_i, face_centroids[j] - c_i))
            if flux > 0.0:
                local.append((j, flux))
                total_out += flux
        if total_out > 0.0:
            inv = 1.0 / total_out
            for j, flux in local:
                src_list.append(i)
                dst_list.append(j)
                wt_list.append(flux * inv)

    return (
        np.array(src_list, dtype=np.int32),
        np.array(dst_list, dtype=np.int32),
        np.array(wt_list, dtype=np.float32),
    )


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

    Each iteration:
      1. Each face sends ``transport_step × q`` to its downwind neighbours
         (distributed by normalised flux weight) and loses that amount locally.
      2. Background precipitation removes ``decay_rate × q`` from every face.
      3. Ocean faces are clamped back to their evaporation source, replenishing
         moisture lost to advection and decay (continuous evaporation).

    Continentality drying emerges because inland faces receive decreasing
    amounts each hop; convergent zones (ITCZ) accumulate moisture from multiple
    source directions; divergent zones (horse latitudes, anticyclones) dry out
    because outflow exceeds inflow.

    Args:
        evaporation:      (N, K) source strengths from ``evaporation_source``.
        face_centroids:   (N, 3) unit-sphere centroids.
        face_adjacency:   adjacency list from ``PolygonMesh``.
        winds:            (N, K, 3) seasonal wind vectors in m/s.
        n_iter:           Jacobi iterations; 200 reaches steady state on a
                          depth-4 icosphere dual (~10 k faces).
        decay_rate:       Background precipitation fraction removed per step;
                          controls how quickly moisture declines inland.
        _transport_step:  Fraction of each face's moisture moved per step.
                          Kept at 0.5 for stability (CFL < 1); not a tuning knob.

    Returns:
        (N, K) float32 equilibrium specific humidity (dimensionless).
    """
    N, K = evaporation.shape
    is_ocean = evaporation.sum(axis=1) > 0   # (N,) bool — has a source term
    q_air = np.zeros((N, K), dtype=np.float32)

    for k in range(K):
        src, dst, w = _build_wind_edges(face_centroids, face_adjacency, winds[:, k, :])
        evap_k = evaporation[:, k]            # (N,)
        q_k = evap_k.copy()                   # initialise at source values

        for _ in range(n_iter):
            # Conservative outflow: each face loses transport_step × q total.
            q_sent = q_k * _transport_step    # (N,) amount leaving each face
            q_new = q_k - q_sent              # deduct before adding inflow

            # Distribute outflow to downwind neighbours.
            if len(src):
                np.add.at(q_new, dst, q_sent[src] * w)

            # Background precipitation sink.
            q_new *= 1.0 - decay_rate

            # Continuous evaporation: replenish ocean faces to their source level.
            np.maximum(q_new, evap_k, where=is_ocean, out=q_new)

            q_k = q_new

        q_air[:, k] = q_k

    return q_air


def smooth_humidity(
    q_air: np.ndarray,
    face_adjacency: list,
    n_passes: int = 10,
    alpha: float = 0.4,
) -> np.ndarray:
    """Laplacian graph smoothing of the humidity field.

    Blends each face toward the mean of its immediate neighbours, diffusing
    ITCZ stripes and softening face-edge discontinuities in the biome map.

    Args:
        q_air:          (N, K) equilibrium humidity from advect_moisture.
        face_adjacency: adjacency list from PolygonMesh.
        n_passes:       smoothing iterations.
        alpha:          neighbour-mean blend weight (0 = no change, 1 = full replace).

    Returns:
        (N, K) float32 smoothed humidity.
    """
    q = q_air.astype(np.float32)
    for _ in range(n_passes):
        q_next = q.copy()
        for i, neighbors in enumerate(face_adjacency):
            if neighbors:
                q_next[i] += alpha * (q[list(neighbors)].mean(axis=0) - q[i])
        q = q_next
    return q


def per_pixel_humidity(
    q_air: np.ndarray,
    mesh,
    tex_size: int,
) -> np.ndarray:
    """Rasterize face-level humidity to per-pixel texture via shared-vertex barycentric interpolation.

    Face humidity values are averaged onto shared vertices so adjacent faces
    interpolate from the same boundary values, eliminating hard seams.

    Args:
        q_air:    (N, K) smoothed humidity from smooth_humidity.
        mesh:     PolygonMesh.
        tex_size: Output texture resolution (H = W = tex_size).

    Returns:
        (N, H, W, K) float32.
    """
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

    Applies an orographic enhancement factor identical to the old model
    (``exp(k × wind·∇h)``) so windward slopes receive extra precipitation.
    Divides the annual-scale constant by 4 for per-season output to stay
    consistent with the rest of the climate pipeline.

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
