"""Annually-averaged surface ocean currents and their climate feedback.

Barotropic Stommel streamfunction model (with β-plane western intensification):

- A scalar streamfunction ψ is defined per ocean face, with ψ=0 on the
  ocean–land boundary so each basin is naturally closed.
- The streamfunction satisfies L·ψ − β·∂ψ/∂e = −curl(τ), where τ is the
  quadratic-drag wind stress (C_D·|u|·u), L is the Voronoi-weighted graph
  Laplacian (FVM approximation of −∇²), and β = 2Ω·cos(φ)/R is the
  planetary vorticity gradient. The non-symmetric system is solved by GMRES.
- Velocity is recovered from ψ via u = k̂ × ∇ψ on each ocean face, giving
  a divergence-free flow with western-boundary intensification.
- The advection step (insolation → SST anomaly) and the inland-decay
  step are unchanged: they only consume the velocity vector.

No seasonal breakdown — the model produces a single per-face (N,)
anomaly that the climate builder broadcasts across seasons via the
`coastal_current_temp_delta` and `current_wet_delta` hooks already
exposed by `surface_temperatures` and `surface_precipitation`.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence, TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

from worldgen.climate._gradient import _tangent_frames, _tangent_gradient
from core.mesh import PolygonMesh

if TYPE_CHECKING:
    from worldgen.climate.builder import CurrentsParams


EARTH_RADIUS_KM = 6371.0
EARTH_ROTATION_RAD_S = 7.292e-5


def _adjacency_pairs(mesh) -> tuple[np.ndarray, np.ndarray]:
    """``(src, dst)`` int64 of every directed face adjacency, in adjacency order."""
    counts = np.fromiter((len(a) for a in mesh.face_adjacency), dtype=np.int64, count=len(mesh.faces))
    src = np.repeat(np.arange(len(mesh.faces), dtype=np.int64), counts)
    dst = np.fromiter((j for a in mesh.face_adjacency for j in a), dtype=np.int64, count=int(counts.sum()))
    return src, dst


def _voronoi_edges(mesh, is_ocean: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(src, dst, w)`` for each directed ocean→any edge: w = edge_len / centroid_dist.

    In adjacency order, ocean faces only. A neighbour sharing no polygon edge
    (degenerate topology) is skipped.
    """
    n = len(mesh.faces)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    centroids = np.asarray(mesh.face_centroids, dtype=np.float64)

    # Every polygon edge, keyed by its sorted vertex pair; a key held by two
    # faces is the edge they share.
    sizes = np.fromiter((len(f) for f in mesh.faces), dtype=np.int64, count=n)
    fv = np.fromiter((v for f in mesh.faces for v in f), dtype=np.int64, count=int(sizes.sum()))
    face_of = np.repeat(np.arange(n, dtype=np.int64), sizes)
    starts = np.repeat(np.cumsum(sizes) - sizes, sizes)
    nxt = fv[starts + (np.arange(fv.size) - starts + 1) % np.repeat(sizes, sizes)]
    va, vb = np.minimum(fv, nxt), np.maximum(fv, nxt)
    order = np.argsort(va * len(verts) + vb, kind="stable")
    key = (va * len(verts) + vb)[order]
    same = np.flatnonzero(key[1:] == key[:-1])
    e0, e1 = order[same], order[same + 1]
    pair_key = np.concatenate([face_of[e0] * n + face_of[e1], face_of[e1] * n + face_of[e0]])
    pair_edge = np.concatenate([e0, e0])
    by_key = np.argsort(pair_key, kind="stable")
    pair_key, pair_edge = pair_key[by_key], pair_edge[by_key]

    src, dst = _adjacency_pairs(mesh)
    ocean = is_ocean[src]
    src, dst = src[ocean], dst[ocean]
    q = src * n + dst
    loc = np.minimum(np.searchsorted(pair_key, q), max(pair_key.size - 1, 0))
    found = pair_key[loc] == q if pair_key.size else np.zeros(q.size, dtype=bool)
    src, dst, e = src[found], dst[found], pair_edge[loc[found]]

    edge_len = EARTH_RADIUS_KM * np.arccos(np.clip(np.sum(verts[va[e]] * verts[vb[e]], axis=1), -1.0, 1.0))
    d_km = EARTH_RADIUS_KM * np.arccos(np.clip(np.sum(centroids[src] * centroids[dst], axis=1), -1.0, 1.0))
    return src, dst, edge_len / np.maximum(d_km, 1e-6)


@dataclass
class CurrentsResult:
    currents: np.ndarray              # (N, 3) m/s tangent vectors (0 on land)
    streamfunction: np.ndarray        # (N,) ψ in raw solver units (0 on land)
    temperature_delta: np.ndarray     # (N,) °C
    precipitation_delta: np.ndarray   # (N,) multiplicative wet delta
    advected_insolation: np.ndarray   # (N,) W/m² post-advection (for inspection)


def _wind_stress_curl(face_centroids: np.ndarray,
                      face_adjacency: Sequence[Sequence[int]],
                      annual_winds: np.ndarray) -> np.ndarray:
    """Curl of the quadratic-drag wind stress, per face, in (m²/s²)/km."""
    east, north = _tangent_frames(face_centroids)
    C_D = 1.3e-3
    we = np.sum(annual_winds * east, axis=1)
    wn = np.sum(annual_winds * north, axis=1)
    speed = np.sqrt(we ** 2 + wn ** 2)
    tau_e = C_D * speed * we
    tau_n = C_D * speed * wn
    tau_en = np.stack([tau_e, tau_n], axis=1)                           # (N, 2)
    grad = _tangent_gradient(face_centroids, face_adjacency, tau_en)    # (N, 2, 2)
    # grad[:, :, 0] = ∇τ_e = [∂τ_e/∂e, ∂τ_e/∂n]
    # grad[:, :, 1] = ∇τ_n = [∂τ_n/∂e, ∂τ_n/∂n]
    return grad[:, 0, 1] - grad[:, 1, 0]


def _solve_streamfunction(mesh: PolygonMesh,
                          face_centroids: np.ndarray,
                          is_ocean: np.ndarray,
                          rhs: np.ndarray,
                          *,
                          beta_scale: float = 1.0,
                          tol: float = 1e-6,
                          max_iter: int = 200) -> np.ndarray:
    """Sparse solve of (L - β·D_east)·ψ = rhs on the ocean face graph, ψ=0 on land.

    L is the Voronoi-weighted graph Laplacian (w = edge_len / centroid_dist).
    The β·D_east term (Stommel planetary vorticity gradient) breaks symmetry,
    producing western-boundary intensification. Land rows are identity.
    """
    n = len(mesh.faces)
    centroids = face_centroids.astype(np.float64)

    # Voronoi-weighted Laplacian
    src, dst, w = _voronoi_edges(mesh, is_ocean)
    if not src.size:
        return np.zeros(n, dtype=np.float64)

    # Laplacian: diagonal − off-diagonal ocean→ocean entries; identity for land.
    diag_vals = np.bincount(src, weights=w, minlength=n)
    diag_vals[~is_ocean] = 1.0

    ocean_dst_mask = is_ocean[dst]
    src_o = src[ocean_dst_mask]
    dst_o = dst[ocean_dst_mask]
    w_o = w[ocean_dst_mask]

    idx_n = np.arange(n, dtype=np.int64)
    L = sp.csr_matrix(
        (np.concatenate([diag_vals, -w_o]),
         (np.concatenate([idx_n, src_o]), np.concatenate([idx_n, dst_o]))),
        shape=(n, n),
    )

    # β-plane east-derivative: finite difference to nearest-east neighbour
    lat = np.arcsin(np.clip(centroids[:, 2], -1.0, 1.0))
    beta_arr = beta_scale * 2.0 * EARTH_ROTATION_RAD_S * np.cos(lat) / EARTH_RADIUS_KM

    east_axes, _ = _tangent_frames(centroids)
    a_src, a_dst = _adjacency_pairs(mesh)
    proj = np.sum((centroids[a_dst] - centroids[a_src]) * east_axes[a_src], axis=1) * EARTH_RADIUS_KM
    # Each face's most-eastward neighbour (the first, on a tie).
    order = np.lexsort((np.arange(a_src.size), -proj, a_src))
    first = order[np.flatnonzero(np.r_[True, a_src[order][1:] != a_src[order][:-1]])]
    east_dst = np.full(n, -1, dtype=np.int64)
    east_proj = np.ones(n, dtype=np.float64)
    east_dst[a_src[first]] = a_dst[first]
    east_proj[a_src[first]] = proj[first]

    has_east = (east_dst >= 0) & is_ocean
    he_idx = np.where(has_east)[0].astype(np.int64)
    inv_ep = 1.0 / east_proj[he_idx]
    D_east = sp.csr_matrix(
        (np.concatenate([-inv_ep, inv_ep]),
         (np.concatenate([he_idx, he_idx]),
          np.concatenate([he_idx, east_dst[he_idx]]))),
        shape=(n, n),
    )

    # Stommel operator: A = L - diag(β) · D_east
    A = L - sp.diags(beta_arr) @ D_east

    # Scale rhs by EARTH_RADIUS_KM so that ψ has the same magnitude as the
    # old 1/d Laplacian (which had units 1/km, making L*ψ match curl(τ) in
    # (m²/s²)/km; here L is dimensionless so we restore the km factor explicitly).
    b = np.zeros(n, dtype=np.float64)
    b[is_ocean] = rhs[is_ocean] * EARTH_RADIUS_KM

    # Direct solve; every ocean basin borders land, which pins ψ. Only a world
    # with no land leaves A singular — fall back to GMRES's best effort there.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", spla.MatrixRankWarning)
        psi = np.asarray(spla.spsolve(A.tocsc(), b), dtype=np.float64)
    if not np.all(np.isfinite(psi)):
        psi, _ = spla.gmres(A, b, rtol=tol, restart=min(max_iter, n), maxiter=max_iter)
    psi[~is_ocean] = 0.0
    return psi


def _streamfunction_currents(face_centroids: np.ndarray,
                             face_adjacency: Sequence[Sequence[int]],
                             psi: np.ndarray,
                             is_ocean: np.ndarray) -> np.ndarray:
    """Recover u = k̂ × ∇ψ in each face's tangent plane."""
    east, north = _tangent_frames(face_centroids)
    grad = _tangent_gradient(face_centroids, face_adjacency, psi)       # (N, 2)
    u_east = -grad[:, 1]
    u_north = grad[:, 0]
    u_3d = u_east[:, None] * east + u_north[:, None] * north
    u_3d[~is_ocean] = 0.0
    return u_3d


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

    if src.size == 0:
        return advected

    inflow_total = np.zeros(n, dtype=np.float64)
    np.add.at(inflow_total, dst, weights)
    has_inflow = inflow_total > 0.0
    inv_inflow = np.zeros(n, dtype=np.float64)
    inv_inflow[has_inflow] = 1.0 / inflow_total[has_inflow]

    # Normalise weights by destination inflow total and build a sparse
    # row-stochastic matrix W so each iteration is a single sparse matvec.
    norm_w = weights * inv_inflow[dst]
    W = sp.csr_matrix((norm_w, (dst, src)), shape=(n, n))

    for _ in range(iterations):
        avg_inflow = W @ advected
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

    # Build face-distance graph + virtual multi-source node at index n (weight 0
    # to every ocean face) for single-call multi-source Dijkstra.
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for u in range(n):
        for v in mesh.face_adjacency[u]:
            cos_ang = float(np.clip(centroids[u] @ centroids[v], -1.0, 1.0))
            rows.append(u)
            cols.append(int(v))
            data.append(EARTH_RADIUS_KM * np.arccos(cos_ang))
    ocean_idx = np.flatnonzero(is_ocean)
    virt = n
    for i in ocean_idx:
        rows.append(virt)
        cols.append(int(i))
        data.append(0.0)

    graph = sp.csr_matrix((data, (rows, cols)), shape=(n + 1, n + 1))
    dist_full, pred_full = sp_dijkstra(
        graph, directed=True, indices=virt, return_predecessors=True
    )
    dist = dist_full[:n]
    pred = pred_full[:n]

    # Walk the predecessor tree in distance order to assign each face the
    # nearest ocean face as its parent (ocean faces are their own parents).
    ocean_parent = np.full(n, -1, dtype=np.int64)
    for f in np.argsort(dist):
        if is_ocean[f]:
            ocean_parent[f] = f
        else:
            p = int(pred[f])
            if 0 <= p < n:
                ocean_parent[f] = ocean_parent[p]

    dist[~np.isfinite(dist)] = 0.0
    decay = max(float(decay_km), 1e-6)
    delta_full = np.zeros(n, dtype=np.float64)
    has_parent = ocean_parent >= 0
    delta_full[has_parent] = (
        ocean_delta[ocean_parent[has_parent]] * np.exp(-dist[has_parent] / decay)
    )
    return delta_full, dist


def compute_currents(mesh: PolygonMesh,
                     face_elev_km: np.ndarray,
                     face_centroids: np.ndarray,
                     winds_per_season: np.ndarray,
                     face_insolation: np.ndarray,
                     *,
                     params: "CurrentsParams",
                     sea_level: float = 0.0,
                     is_ocean: np.ndarray | None = None) -> CurrentsResult:
    """Run the annual-mean ocean current model and return per-face deltas.

    Args:
        mesh: dual `PolygonMesh` (after elevation has been built).
        face_elev_km: (N,) per-face elevation in km.
        face_centroids: (N, 3) unit-sphere face positions.
        winds_per_season: (N, K, 3) seasonal wind vectors from `zonal_winds`.
        face_insolation: (N,) annual-mean insolation in W/m² (mean of the
            seasonal `seasonal_insolation` output along the season axis).
        params: CurrentsParams — β scale, advection mixing/iterations, the
            temperature/precipitation feedback scales, and inland decay.
        sea_level: elevation threshold defining the ocean mask.
        is_ocean: optional precomputed ocean mask; derived from sea_level if None.
    """
    p = params
    n = len(mesh.faces)
    if is_ocean is None:
        is_ocean = face_elev_km <= sea_level

    annual_winds = winds_per_season.astype(np.float64).mean(axis=1)         # (N, 3)
    centroids = face_centroids.astype(np.float64)

    # Stommel balance: L·ψ − β·∂ψ/∂e = −curl(τ), where L is the Voronoi
    # Laplacian (≈ −∇²) and the β term drives western-boundary intensification.
    curl_tau = _wind_stress_curl(centroids, mesh.face_adjacency, annual_winds)
    psi = _solve_streamfunction(mesh, centroids, is_ocean, -curl_tau,
                                beta_scale=p.beta_scale)
    currents = _streamfunction_currents(centroids, mesh.face_adjacency, psi, is_ocean)

    src, dst, w = _build_advection_edges(mesh, currents, is_ocean)
    advected = _advect_field(face_insolation.astype(np.float64),
                             is_ocean,
                             src, dst, w,
                             p.advection_iterations,
                             p.advection_alpha)

    ocean_dQ = np.zeros(n, dtype=np.float64)
    ocean_dQ[is_ocean] = advected[is_ocean] - face_insolation[is_ocean]
    ocean_dT = p.temperature_scale * ocean_dQ

    temp_delta, _ = _propagate_inland(ocean_dT, mesh, is_ocean, p.inland_decay_km)
    precip_delta = p.precipitation_scale * temp_delta

    return CurrentsResult(
        currents=currents.astype(np.float32),
        streamfunction=psi.astype(np.float32),
        temperature_delta=temp_delta.astype(np.float32),
        precipitation_delta=precip_delta.astype(np.float32),
        advected_insolation=advected.astype(np.float32),
    )
