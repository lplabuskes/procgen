"""Sediment load — how much of a river's flow still carries silt to its mouth.

Lakes trap most silt (Brune: 90 %+ for large lakes), so a river leaving one builds little or no delta
until tributaries reload it.  Load accumulates down the stream tree in flow units, each node adding
its own runoff; across a lake outlet only ``1 − trap_efficiency`` passes.  Silt a full lake's deltas
could not lay down (the size cap) *spills* through the outlet untrapped.
"""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from worldgen.hydrology.types import StreamNetwork


def downstream_order(streams: StreamNetwork) -> np.ndarray:
    """The stream nodes, every one after all the nodes draining into it (Kahn's order)."""
    n = len(streams)
    receiver = np.asarray(streams.receiver, dtype=np.int64)
    has_down = receiver >= 0
    indegree = np.bincount(receiver[has_down], minlength=n)
    ready: list[int] = np.flatnonzero(indegree == 0).tolist()
    out: list[int] = []
    while ready:
        k = ready.pop()
        out.append(k)
        r = int(receiver[k])
        if r < 0:
            continue
        indegree[r] -= 1
        if indegree[r] == 0:
            ready.append(r)
    return np.asarray(out, dtype=np.int64)


def sediment_load(
    streams: StreamNetwork, lake: np.ndarray, trap_efficiency: float,
    spill: Optional[Mapping[int, float]] = None, order: Optional[np.ndarray] = None,
) -> np.ndarray:
    """``(K,)`` sediment load at each stream node, in river flow units (≤ its flow).

    ``lake`` is per node: a bool mask or lake labels (> 0).  ``spill`` (by label) passes the outlet
    untrapped.  ``order`` is :func:`downstream_order`, if already known.
    """
    n = len(streams)
    receiver = np.asarray(streams.receiver, dtype=np.int64)
    flow = np.asarray(streams.flow, dtype=np.float64)
    has_down = receiver >= 0
    inflow = np.zeros(n)
    np.add.at(inflow, receiver[has_down], flow[has_down])
    load = np.maximum(flow - inflow, 0.0)       # each node's own runoff, to start
    label = np.asarray(lake).astype(np.int64)
    outlet = has_down & (label > 0) & (label[np.maximum(receiver, 0)] != label)
    trap = float(trap_efficiency)
    left = dict(spill or {})
    for k in (downstream_order(streams) if order is None else order):
        r = int(receiver[k])
        if r < 0:
            continue
        carried = float(load[k])
        if outlet[k]:
            # What the lake could not hold passes whole; the rest is trapped as usual.
            b = int(label[k])
            over = min(left.get(b, 0.0), carried)
            left[b] = left.get(b, 0.0) - over
            carried = over + (1.0 - trap) * (carried - over)
        load[r] += carried
    return np.minimum(load, flow)


def mouth_node(streams: StreamNetwork, flat: int, point: np.ndarray, radius_rad: float) -> Optional[int]:
    """The stream node a river reaches its mouth on: the biggest on the mouth
    texel ``flat``, else the biggest within ``radius_rad`` of ``point`` (None if none)."""
    on = np.flatnonzero(np.asarray(streams.nodes) == int(flat))
    if on.size == 0:
        gap = np.linalg.norm(np.asarray(streams.positions, dtype=np.float64) - np.asarray(point)[None, :], axis=1)
        on = np.flatnonzero(gap <= radius_rad)
    if on.size == 0:
        return None
    return int(on[np.argmax(np.asarray(streams.flow)[on])])
