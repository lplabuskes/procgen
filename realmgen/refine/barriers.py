"""River segments in packed form, and the flow ramp that scales a river's pull.

``pack_river_segments`` turns river polylines (world ``"rivers"``, region / local
paths) into ``(M, 7|8)`` segment rows: the form the region / local tiers pass
under ``RIVER_SEGMENTS_KEY`` and that :mod:`realmgen.refine.river_follow` and
:mod:`realmgen.polity.barriers` read.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def pack_river_segments(rivers: Optional[Iterable], *, min_flow: float = 0.0,
                        river_ids: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    """``(M, 7)`` ``[a, b, flow]`` rows of the segments of ``rivers`` above ``min_flow``.

    ``rivers`` is the ``(positions, flows, in_lake)`` polyline form (world
    ``"rivers"``, region / local paths); in-lake segments are kept, so a border
    following a river follows it through a lake (only land is labelled). With
    ``river_ids`` (per path, per vertex: ``RegionRivers.river_ids()``) a column 7
    holds each segment's world river index (-1 none).
    """
    a, b, flow, rid = _river_segments(rivers, min_flow=min_flow, river_ids=river_ids)
    cols = [a, b, flow[:, None]] + ([rid[:, None].astype(np.float64)] if river_ids is not None else [])
    return np.concatenate(cols, axis=1)


def _river_segments(rivers: Optional[Iterable], *, min_flow: float,
                    river_ids: Optional[Sequence[np.ndarray]] = None,
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a_parts, b_parts, f_parts, r_parts = [], [], [], []
    for k, path in enumerate(rivers or []):
        pos = np.asarray(path[0], dtype=np.float64)
        flow = np.asarray(path[1], dtype=np.float64)
        m = min(flow.size, pos.shape[0] - 1)
        if m <= 0:
            continue
        keep = flow[:m] > float(min_flow)
        if keep.any():
            a_parts.append(pos[:m][keep])
            b_parts.append(pos[1:m + 1][keep])
            f_parts.append(flow[:m][keep])
            if river_ids is not None:
                ids = np.asarray(river_ids[k], dtype=np.int64)
                # A segment where a headwater joins its world river takes the river's id.
                r_parts.append(np.where(ids[:m] >= 0, ids[:m], ids[1:m + 1])[keep])
            else:
                r_parts.append(np.full(int(keep.sum()), -1, dtype=np.int64))
    if not a_parts:
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, empty.copy(), np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
    a = np.concatenate(a_parts)
    b = np.concatenate(b_parts)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return a, b, np.concatenate(f_parts), np.concatenate(r_parts)


def flow_ramp(flow: np.ndarray, *, min_flow: float, full_flow: float) -> np.ndarray:
    """``clip(ln(flow / min_flow) / ln(full_flow / min_flow), 0, 1)``."""
    lo = max(float(min_flow), 1e-9)
    span = np.log(max(float(full_flow), lo * (1.0 + 1e-6)) / lo)
    return np.clip(np.log(np.maximum(np.asarray(flow, dtype=np.float64), 1e-12) / lo) / span, 0.0, 1.0)
