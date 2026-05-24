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

import heapq
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from worldgen.climate.winds import _tangent_frames
from worldgen.mesh.mymesh import PolygonMesh


EARTH_RADIUS_KM = 6371.0
EARTH_ROTATION_RAD_S = 7.292e-5


def _build_voronoi_weights(mesh, is_ocean: np.ndarray) -> dict:
    """For each directed ocean→any edge, compute w = edge_len / centroid_dist.

    Returns dict (f, nb) -> float for ocean faces only. Edges with degenerate
    topology (shared vertex count != 2) are skipped.
    """
    verts = np.array(mesh.vertices, dtype=np.float64)
    face_vert_sets = [set(f) for f in mesh.faces]
    centroids = mesh.face_centroids.astype(np.float64)
    weights: dict = {}
    for f in range(len(mesh.faces)):
        if not is_ocean[f]:
            continue
        fverts = face_vert_sets[f]
        for nb in mesh.face_adjacency[f]:
            shared = list(fverts & face_vert_sets[nb])
            if len(shared) != 2:
                continue
            edge_len = EARTH_RADIUS_KM * np.arccos(
                float(np.clip(verts[shared[0]] @ verts[shared[1]], -1.0, 1.0))
            )
            d_km = EARTH_RADIUS_KM * np.arccos(
                float(np.clip(centroids[f] @ centroids[nb], -1.0, 1.0))
            )
            weights[(f, nb)] = edge_len / max(d_km, 1e-6)
    return weights


def _gmres(matvec, b: np.ndarray, tol: float = 1e-6, max_iter: int = 200) -> np.ndarray:
    """GMRES (no restart) for Ax = b starting from x = 0."""
    n = len(b)
    b_norm = float(np.linalg.norm(b))
    if b_norm < 1e-30:
        return np.zeros(n, dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)
    r = b.copy()
    r_norm = float(np.linalg.norm(r))
    if r_norm / b_norm < tol:
        return x

    Q = np.zeros((n, max_iter + 1), dtype=np.float64)
    H = np.zeros((max_iter + 1, max_iter), dtype=np.float64)
    Q[:, 0] = r / r_norm
    cs = np.zeros(max_iter, dtype=np.float64)
    sn = np.zeros(max_iter, dtype=np.float64)
    g = np.zeros(max_iter + 1, dtype=np.float64)
    g[0] = r_norm
    m = 0
    for j in range(max_iter):
        w = matvec(Q[:, j])
        for i in range(j + 1):
            H[i, j] = float(Q[:, i] @ w)
            w -= H[i, j] * Q[:, i]
        H[j + 1, j] = float(np.linalg.norm(w))
        if H[j + 1, j] > 1e-12:
            Q[:, j + 1] = w / H[j + 1, j]
        for i in range(j):
            h_ij = H[i, j]
            H[i, j] = cs[i] * h_ij + sn[i] * H[i + 1, j]
            H[i + 1, j] = -sn[i] * h_ij + cs[i] * H[i + 1, j]
        denom = float(np.sqrt(H[j, j] ** 2 + H[j + 1, j] ** 2))
        if denom > 1e-12:
            cs[j] = H[j, j] / denom
            sn[j] = H[j + 1, j] / denom
        else:
            cs[j], sn[j] = 1.0, 0.0
        H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
        H[j + 1, j] = 0.0
        g[j + 1] = -sn[j] * g[j]
        g[j] = cs[j] * g[j]
        m = j + 1
        if abs(g[j + 1]) / b_norm < tol:
            break
    y = np.linalg.solve(H[:m, :m], g[:m])
    return x + Q[:, :m] @ y


@dataclass
class CurrentsResult:
    currents: np.ndarray              # (N, 3) m/s tangent vectors (0 on land)
    streamfunction: np.ndarray        # (N,) ψ in raw solver units (0 on land)
    temperature_delta: np.ndarray     # (N,) °C
    precipitation_delta: np.ndarray   # (N,) multiplicative wet delta
    advected_insolation: np.ndarray   # (N,) W/m² post-advection (for inspection)


def _tangent_gradient(face_centroids: np.ndarray,
                      face_adjacency: Sequence[Sequence[int]],
                      face_field: np.ndarray,
                      earth_radius_km: float = EARTH_RADIUS_KM) -> np.ndarray:
    """Per-face least-squares gradient of a scalar (or vector) face field.

    Args:
        face_centroids: (N, 3) unit-sphere face positions.
        face_adjacency: per-face neighbour index lists.
        face_field: (N,) for a scalar, or (N, K) for K independent scalar
            fields (e.g. the east/north components of a vector field).

    Returns:
        (N, 2) or (N, 2, K) gradient in the per-face (east, north) basis,
        in units of (field-unit) / km. Faces with degenerate neighbourhoods
        get the minimum-norm pinv solution (typically zero).
    """
    n = face_centroids.shape[0]
    east, north = _tangent_frames(face_centroids)

    src_list: list[int] = []
    dst_list: list[int] = []
    for i, nbrs in enumerate(face_adjacency):
        for j in nbrs:
            src_list.append(i)
            dst_list.append(int(j))
    if not src_list:
        out_shape = (n, 2) if face_field.ndim == 1 else (n, 2, face_field.shape[1])
        return np.zeros(out_shape, dtype=np.float64)
    src = np.asarray(src_list, dtype=np.int64)
    dst = np.asarray(dst_list, dtype=np.int64)

    rel = face_centroids[dst].astype(np.float64) - face_centroids[src].astype(np.float64)
    u_e = np.sum(rel * east[src], axis=1) * earth_radius_km
    v_e = np.sum(rel * north[src], axis=1) * earth_radius_km
    r = np.stack([u_e, v_e], axis=1)                                    # (E, 2)

    M = np.zeros((n, 2, 2), dtype=np.float64)
    np.add.at(M, src, r[:, :, None] * r[:, None, :])
    M_inv = np.linalg.pinv(M)

    if face_field.ndim == 1:
        df = face_field[dst] - face_field[src]                          # (E,)
        b = np.zeros((n, 2), dtype=np.float64)
        np.add.at(b, src, r * df[:, None])
        return (M_inv @ b[..., None])[..., 0]
    df = face_field[dst] - face_field[src]                              # (E, K)
    b = np.zeros((n, 2, face_field.shape[1]), dtype=np.float64)
    np.add.at(b, src, r[:, :, None] * df[:, None, :])
    return M_inv @ b


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
    """GMRES solve of (L - β·D_east)·ψ = rhs on the ocean face graph, ψ=0 on land.

    L is the Voronoi-weighted graph Laplacian (w = edge_len / centroid_dist).
    The β·D_east term (Stommel planetary vorticity gradient) breaks symmetry,
    producing western-boundary intensification. Land rows are identity.
    """
    n = len(mesh.faces)
    centroids = face_centroids.astype(np.float64)

    # Voronoi-weighted Laplacian
    voronoi_weights = _build_voronoi_weights(mesh, is_ocean)
    src_list: list[int] = []
    dst_list: list[int] = []
    w_list: list[float] = []
    for f in range(n):
        if not is_ocean[f]:
            continue
        for nb in mesh.face_adjacency[f]:
            wt = voronoi_weights.get((f, nb))
            if wt is None:
                continue
            src_list.append(f)
            dst_list.append(int(nb))
            w_list.append(wt)
    if not src_list:
        return np.zeros(n, dtype=np.float64)
    src = np.asarray(src_list, dtype=np.int64)
    dst = np.asarray(dst_list, dtype=np.int64)
    w = np.asarray(w_list, dtype=np.float64)

    diag = np.zeros(n, dtype=np.float64)
    np.add.at(diag, src, w)
    diag[~is_ocean] = 1.0

    ocean_dst_mask = is_ocean[dst]
    src_o = src[ocean_dst_mask]
    dst_o = dst[ocean_dst_mask]
    w_o = w[ocean_dst_mask]

    def L_matvec(psi: np.ndarray) -> np.ndarray:
        out = diag * psi
        if src_o.size:
            np.subtract.at(out, src_o, w_o * psi[dst_o])
        return out

    # β-plane east-derivative: finite difference to nearest-east neighbour
    lat = np.arcsin(np.clip(centroids[:, 2], -1.0, 1.0))
    beta_arr = beta_scale * 2.0 * EARTH_ROTATION_RAD_S * np.cos(lat) / EARTH_RADIUS_KM

    east_axes, _ = _tangent_frames(centroids)
    east_proj = np.zeros(n, dtype=np.float64)
    east_dst = np.full(n, -1, dtype=np.int64)
    for f in range(n):
        if not is_ocean[f]:
            continue
        best_proj = -np.inf
        for nb in mesh.face_adjacency[f]:
            proj = float((centroids[nb] - centroids[f]) @ east_axes[f]) * EARTH_RADIUS_KM
            if proj > best_proj:
                best_proj = proj
                east_dst[f] = nb
        east_proj[f] = best_proj if east_dst[f] >= 0 else 1.0

    has_east = (east_dst >= 0) & is_ocean

    def d_east_matvec(psi: np.ndarray) -> np.ndarray:
        out = np.zeros(n, dtype=np.float64)
        out[has_east] = (psi[east_dst[has_east]] - psi[has_east]) / east_proj[has_east]
        return out

    def matvec_stommel(psi: np.ndarray) -> np.ndarray:
        return L_matvec(psi) - beta_arr * d_east_matvec(psi)

    # Scale rhs by EARTH_RADIUS_KM so that ψ has the same magnitude as the
    # old 1/d Laplacian (which had units 1/km, making L*ψ match curl(τ) in
    # (m²/s²)/km; here L is dimensionless so we restore the km factor explicitly).
    b = np.zeros(n, dtype=np.float64)
    b[is_ocean] = rhs[is_ocean] * EARTH_RADIUS_KM

    psi = _gmres(matvec_stommel, b, tol=tol, max_iter=max_iter)
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
                     beta_scale: float = 1.0,
                     advection_alpha: float = 1.0,
                     advection_iterations: int = 400,
                     temperature_scale: float = 0.06,
                     precipitation_scale: float = 0.03,
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
        beta_scale: scale factor on the planetary vorticity gradient β. Set
            to 0 for symmetric gyres, ~1 for Stommel western intensification.
        advection_alpha: per-iteration mixing fraction in the Jacobi advection.
        advection_iterations: number of Jacobi sweeps; ~400 is plenty for
            current depths/icosphere sizes used in this project.
        temperature_scale: °C per (W/m² of advected-vs-local insolation gap).
        precipitation_scale: multiplicative wet-delta per °C of anomaly.
        inland_decay_km: e-folding distance for inland propagation of the
            ocean anomaly.
    """
    n = len(mesh.faces)
    is_ocean = face_elev_km <= sea_level

    annual_winds = winds_per_season.astype(np.float64).mean(axis=1)         # (N, 3)
    centroids = face_centroids.astype(np.float64)

    # Stommel balance: L·ψ − β·∂ψ/∂e = −curl(τ), where L is the Voronoi
    # Laplacian (≈ −∇²) and the β term drives western-boundary intensification.
    curl_tau = _wind_stress_curl(centroids, mesh.face_adjacency, annual_winds)
    psi = _solve_streamfunction(mesh, centroids, is_ocean, -curl_tau,
                                beta_scale=beta_scale)
    currents = _streamfunction_currents(centroids, mesh.face_adjacency, psi, is_ocean)

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
        streamfunction=psi.astype(np.float32),
        temperature_delta=temp_delta.astype(np.float32),
        precipitation_delta=precip_delta.astype(np.float32),
        advected_insolation=advected.astype(np.float32),
    )
