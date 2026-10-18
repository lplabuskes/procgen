"""Filters over stacks of square textures: the last two axes are a texture's rows and columns."""
from __future__ import annotations

import numpy as np


def texture_slope(elevation: np.ndarray, texel_km: float) -> np.ndarray:
    """``(M, T, T)`` gradient magnitude (km/km) of an ``(M, T, T)`` elevation, per texture."""
    dz_r, dz_c = np.gradient(np.asarray(elevation, dtype=np.float64), axis=(1, 2))
    return np.hypot(dz_r, dz_c) / max(texel_km, 1e-9)


def neighbour_any(mask: np.ndarray) -> np.ndarray:
    """True where any 4-neighbour (no wrap) is True, over the last two axes of ``(..., H, W)``."""
    out = np.zeros_like(mask)
    out[..., 1:, :] |= mask[..., :-1, :]
    out[..., :-1, :] |= mask[..., 1:, :]
    out[..., :, 1:] |= mask[..., :, :-1]
    out[..., :, :-1] |= mask[..., :, 1:]
    return out


def neighbour_min4(values: np.ndarray) -> np.ndarray:
    """Per-texel minimum over the 4 neighbours of an ``(M, T, T)`` array (edge replicates)."""
    padded = np.pad(values, ((0, 0), (1, 1), (1, 1)), mode="edge")
    return np.minimum(
        np.minimum(padded[:, :-2, 1:-1], padded[:, 2:, 1:-1]),
        np.minimum(padded[:, 1:-1, :-2], padded[:, 1:-1, 2:]),
    )


def neighbour_min8(values: np.ndarray) -> np.ndarray:
    """Per-node minimum over the 8 neighbours of an ``(n, n)`` array (edges: +inf)."""
    padded = np.pad(values, 1, mode="constant", constant_values=np.inf)
    n = values.shape[0]
    out = np.full_like(values, np.inf)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                out = np.minimum(out, padded[1 + dr:1 + dr + n, 1 + dc:1 + dc + n])
    return out


def box_mean(values: np.ndarray, r: int) -> np.ndarray:
    """``(2r+1)``² mean over each texture of an ``(M, T, T)`` array, edges replicated."""
    out = np.asarray(values, dtype=np.float64)
    for axis in (1, 2):
        pad = [(0, 0)] * 3
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis)
        n = out.shape[axis]
        hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
        lo = np.take(c, np.arange(n), axis=axis)
        out = (hi - lo) / (2 * r + 1)
    return out


def box_filter(texture: np.ndarray, size: int) -> np.ndarray:
    """``size``×``size`` mean over each texture of an ``(M, T, T, C)`` array.

    Edges replicate; only the invisible over-hang corners see that.
    """
    if size <= 1:
        return texture
    r = size // 2
    padded = np.pad(np.asarray(texture, dtype=np.float64), ((0, 0), (r, r), (r, r), (0, 0)), mode="edge")
    _, h, w, _ = texture.shape
    cols = np.zeros((padded.shape[0], h + 2 * r, w, padded.shape[3]), dtype=np.float64)
    for j in range(size):
        cols += padded[:, :, j:j + w, :]
    total = np.zeros(texture.shape, dtype=np.float64)
    for i in range(size):
        total += cols[:, i:i + h, :, :]
    return (total / float(size * size)).astype(texture.dtype)


def weighted_box_mean3(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted 3×3 box mean over the last two axes of ``(..., H, W)`` (edges use what exists)."""
    num = np.zeros_like(values)
    den = np.zeros_like(values)
    vw = values * weights
    h, w = values.shape[-2:]
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            dst_i = slice(max(0, -di), h - max(0, di))
            dst_j = slice(max(0, -dj), w - max(0, dj))
            src_i = slice(max(0, di), h - max(0, -di))
            src_j = slice(max(0, dj), w - max(0, -dj))
            num[..., dst_i, dst_j] += vw[..., src_i, src_j]
            den[..., dst_i, dst_j] += weights[..., src_i, src_j]
    return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
