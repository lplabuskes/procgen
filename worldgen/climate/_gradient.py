"""Shared tangent-frame and least-squares gradient utilities for climate modules."""
from __future__ import annotations

from typing import Sequence

import numpy as np

EARTH_RADIUS_KM = 6371.0


def _tangent_frames(face_centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-face east/north unit vectors. At the poles falls back to a fixed frame."""
    z_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    east = np.cross(z_hat, face_centroids)                 # (N, 3)
    east_norm = np.linalg.norm(east, axis=1, keepdims=True)
    polar = east_norm[:, 0] < 1e-6
    fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    east = np.where(polar[:, None], fallback, east / np.maximum(east_norm, 1e-12))
    north = np.cross(face_centroids, east)                 # (N, 3)
    north /= np.maximum(np.linalg.norm(north, axis=1, keepdims=True), 1e-12)
    return east, north


def _tangent_gradient(face_centroids: np.ndarray,
                      face_adjacency: Sequence[Sequence[int]],
                      face_field: np.ndarray,
                      earth_radius_km: float = EARTH_RADIUS_KM) -> np.ndarray:
    """Per-face least-squares gradient; returns (N, 2) or (N, 2, K) in (field-unit)/km."""
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
