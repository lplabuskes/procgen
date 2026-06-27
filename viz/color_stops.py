"""Utility functions for stop-based color interpolation."""
from __future__ import annotations

import numpy as np


def interp_stops_scalar(value: float, stops: list) -> tuple:
    """Linear interpolation through a list of (anchor, RGBA) stops for a scalar."""
    if value <= stops[0][0]:
        return stops[0][1]
    if value >= stops[-1][0]:
        return stops[-1][1]
    for i in range(len(stops) - 1):
        e0, c0 = stops[i]
        e1, c1 = stops[i + 1]
        if e0 == e1:
            continue
        if e0 <= value <= e1:
            t = (value - e0) / (e1 - e0)
            return tuple(c0[j] + t * (c1[j] - c0[j]) for j in range(4))
    return stops[-1][1]


def interp_stops(values: np.ndarray, stops: list) -> np.ndarray:
    """Linear interpolation through a list of (anchor, RGBA) stops for an array.

    Returns (*values.shape, 4) float32.
    """
    shape = values.shape
    result = np.empty((*shape, 4), dtype=np.float32)
    result[...] = stops[-1][1]
    for i in range(len(stops) - 1):
        a0, c0 = stops[i]
        a1, c1 = stops[i + 1]
        mask = (values >= a0) & (values < a1)
        t = np.where(mask, (values - a0) / max(a1 - a0, 1e-9), 0.0)
        for ch in range(4):
            result[..., ch] = np.where(mask, c0[ch] + t * (c1[ch] - c0[ch]), result[..., ch])
    result[values <= stops[0][0]] = np.array(stops[0][1], dtype=np.float32)
    result[values >= stops[-1][0]] = np.array(stops[-1][1], dtype=np.float32)
    return result


def gradient_scheme(meta_key: str, stops: list, default: float = 0.0):
    """Return a scheme function that reads meta[meta_key] as float and interpolates through stops."""
    def fn(meta: dict) -> tuple:
        v = float(meta.get(meta_key, default))
        return interp_stops_scalar(v, stops)
    return fn


def categorical_scheme(meta_key: str, palette: np.ndarray, n: int, default: int = 0):
    """Return a scheme function that reads meta[meta_key] as int and looks up in palette."""
    def fn(meta: dict) -> tuple:
        idx = max(0, min(int(meta.get(meta_key, default)), n - 1))
        return tuple(palette[idx].tolist())
    return fn


def gradient_texture(channel_tex: np.ndarray, stops: list) -> np.ndarray:
    """(N, H, W, 1) single-channel → (N, H, W, 4) RGBA via stops interpolation."""
    return interp_stops(channel_tex[..., 0], stops)


def categorical_texture(idx_tex: np.ndarray, palette: np.ndarray, n: int) -> np.ndarray:
    """(N, H, W, 1) int index texture → (N, H, W, 4) RGBA via palette lookup."""
    idx = np.clip(idx_tex[..., 0].astype(np.int32), 0, n - 1)
    return palette[idx].copy()


def palette_legend_entries(names: list[str], palette: np.ndarray) -> list[tuple[str, tuple]]:
    """(name, rgba) pairs for a categorical palette in index order."""
    return [(name, tuple(palette[i].tolist())) for i, name in enumerate(names)]
