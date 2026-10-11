"""River channels on a local map: the ground a river covers at its drawn width.

Local maps draw rivers and streams at their bankfull width
(:func:`worldgen.hydrology.discharge.channel_width_km`), so a wide one covers
whole texels.  Settlements stand on a bank, just clear of the channel
(:func:`onto_banks`); city footprints, farmland and village sites skip the
texels a channel covers (:func:`covered_texels`); and a village's fields keep to
its own bank of a wide river (:func:`crossing_pairs`).

All distances are chords on the unit sphere; segments are texel-scale, so the
point-to-chord distance is the ground distance.  Qt-free, deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree

from region.rivers import RegionRivers
from region.upsample import EARTH_RADIUS_KM, RegionFields
from worldgen.hydrology.discharge import channel_width_km

# Push iterations: a site pushed off one channel may land on another (a confluence).
_PUSH_ROUNDS = 4


@dataclass
class Channels:
    """Every drawn (not under-water) river and stream segment, with its width."""

    a: np.ndarray            # (K, 3) segment start, unit sphere
    b: np.ndarray            # (K, 3) segment end
    width_km: np.ndarray     # (K,) bankfull width

    def __len__(self) -> int:
        return int(self.width_km.size)

    @property
    def half_width(self) -> np.ndarray:
        """(K,) half the width, as a chord."""
        return 0.5 * self.width_km / EARTH_RADIUS_KM

    def at_least(self, width_km: float) -> "Channels":
        keep = self.width_km >= float(width_km)
        return Channels(a=self.a[keep], b=self.b[keep], width_km=self.width_km[keep])


def river_channels(rivers: RegionRivers, discharge_scale: float) -> Channels:
    """The channels of ``rivers``' paths and streams; ``discharge_scale`` is m³/s per unit of flow
    (:func:`worldgen.hydrology.discharge.discharge_per_flow_unit`)."""
    a, b, w = [], [], []
    for positions, flows, in_water in [*rivers.paths, *rivers.streams]:
        pts = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        if len(pts) < 2:
            continue
        dry = ~np.asarray(in_water, dtype=bool)[:len(pts) - 1]
        a.append(pts[:-1][dry])
        b.append(pts[1:][dry])
        w.append(channel_width_km(np.asarray(flows, dtype=np.float64)[:len(pts) - 1][dry] * float(discharge_scale)))
    if not a:
        return Channels(a=np.zeros((0, 3)), b=np.zeros((0, 3)), width_km=np.zeros(0))
    return Channels(a=np.concatenate(a), b=np.concatenate(b), width_km=np.concatenate(w))


def _near_pairs(
    channels: Channels, points: np.ndarray, extra: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(point, segment, edge distance, closest point)`` for every point within
    ``half_width + extra`` of a segment's midpoint-centred reach.  Edge distance
    is the distance to the centre line less the half width (negative inside)."""
    empty = np.zeros(0, dtype=np.int64)
    if len(channels) == 0 or len(points) == 0:
        return empty, empty, np.zeros(0), np.zeros((0, 3))
    mid = 0.5 * (channels.a + channels.b)
    half_len = 0.5 * np.linalg.norm(channels.b - channels.a, axis=1)
    reach = half_len + channels.half_width + float(extra)
    lists = KDTree(points).query_ball_point(mid, reach, workers=-1)
    counts = np.array([len(x) for x in lists], dtype=np.int64)
    if counts.sum() == 0:
        return empty, empty, np.zeros(0), np.zeros((0, 3))
    pt = np.concatenate([np.asarray(x, dtype=np.int64) for x in lists])
    seg = np.repeat(np.arange(len(channels), dtype=np.int64), counts)
    a, ab = channels.a[seg], channels.b[seg] - channels.a[seg]
    u = np.clip(np.einsum("ij,ij->i", points[pt] - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300),
                0.0, 1.0)
    closest = a + u[:, None] * ab
    edge = np.linalg.norm(points[pt] - closest, axis=1) - channels.half_width[seg]
    return pt, seg, edge, closest


def covered_texels(local: RegionFields, channels: Channels) -> np.ndarray:
    """``(M, T, T)`` True where a texel's centre lies under a channel."""
    covered = np.zeros(local.in_hex.shape, dtype=bool)
    if len(channels) == 0:
        return covered
    pos = local.positions.reshape(-1, 3)
    # Only texels within the widest reach of some segment midpoint can be under one
    # (padded: _near_pairs keeps points at exactly its reach).
    mid = 0.5 * (channels.a + channels.b)
    reach = 0.5 * np.linalg.norm(channels.b - channels.a, axis=1) + channels.half_width
    near = np.flatnonzero(np.isfinite(KDTree(mid).query(
        pos, distance_upper_bound=1.001 * float(reach.max()) + 1e-12, workers=-1)[0]))
    pt, _, edge, _ = _near_pairs(channels, pos[near], 0.0)
    covered.reshape(-1)[near[pt[edge < 0.0]]] = True
    return covered


def onto_banks(channels: Channels, positions: np.ndarray, clearance_km: float) -> np.ndarray:
    """``positions`` moved straight out of any channel to ``clearance_km`` past its
    bank, on the side they already stand (one exactly on the centre line goes left
    of the flow).  Points already that clear are unchanged."""
    out = np.asarray(positions, dtype=np.float64).reshape(-1, 3).copy()
    clearance = float(clearance_km) / EARTH_RADIUS_KM
    for _ in range(_PUSH_ROUNDS):
        pt, seg, edge, closest = _near_pairs(channels, out, clearance)
        bad = edge < clearance
        if not bad.any():
            break
        pt, seg, edge, closest = pt[bad], seg[bad], edge[bad], closest[bad]
        # Each point's deepest violation first.
        order = np.lexsort((edge, pt))
        first = order[np.r_[True, np.diff(pt[order]) > 0]]
        for k in first:
            p, c, s = out[pt[k]], closest[k], seg[k]
            away = p - c
            norm = float(np.linalg.norm(away))
            if norm < 1e-12:
                away = np.cross(channels.b[s] - channels.a[s], p)
                norm = float(np.linalg.norm(away))
                if norm < 1e-300:
                    continue
            q = c + away / norm * (channels.half_width[s] + clearance)
            out[pt[k]] = q / np.linalg.norm(q)
    return out


def crossing_pairs(
    channels: Channels, origins: np.ndarray, who: np.ndarray, targets: np.ndarray, reach: np.ndarray,
) -> np.ndarray:
    """``(P,)`` True where the straight line from ``origins[who]`` to ``targets``
    crosses a channel.  ``reach`` ``(N,)``: how far each origin's targets lie."""
    origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    cross = np.zeros(len(who), dtype=bool)
    if len(channels) == 0 or len(who) == 0:
        return cross
    mid = 0.5 * (channels.a + channels.b)
    half_len = float((0.5 * np.linalg.norm(channels.b - channels.a, axis=1)).max())
    near = KDTree(mid).query_ball_point(origins, np.asarray(reach, dtype=np.float64) + half_len, workers=-1)
    order = np.argsort(who, kind="stable")
    bounds = np.searchsorted(who[order], np.arange(len(origins) + 1))
    for o, segs in enumerate(near):
        if not segs:
            continue
        pairs = order[bounds[o]:bounds[o + 1]]
        if pairs.size == 0:
            continue
        x = origins[o]
        # Tangent-plane coordinates at the origin.
        e1 = np.cross(x, [0.0, 0.0, 1.0] if abs(x[2]) < 0.9 else [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(x, e1)
        basis = np.stack([e1, e2], axis=1)                    # (3, 2)
        p = (targets[pairs] - x) @ basis                      # (n, 2); origin at 0
        segs = np.asarray(segs, dtype=np.int64)
        a = (channels.a[segs] - x) @ basis                    # (s, 2)
        b = (channels.b[segs] - x) @ basis

        def orient(u, v, w):
            return (v[..., 0] - u[..., 0]) * (w[..., 1] - u[..., 1]) - (v[..., 1] - u[..., 1]) * (w[..., 0] - u[..., 0])

        zero = np.zeros((1, 1, 2))
        pp, aa, bb = p[:, None, :], a[None, :, :], b[None, :, :]
        hit = ((np.sign(orient(aa, bb, zero)) != np.sign(orient(aa, bb, pp)))
               & (np.sign(orient(zero, pp, aa)) != np.sign(orient(zero, pp, bb))))
        cross[pairs] = hit.any(axis=1)
    return cross
