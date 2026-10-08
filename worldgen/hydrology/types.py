"""River data structures and sphere-surface smoothing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class RiverNode:
    position: np.ndarray          # 3D point on globe surface, shape (3,)
    elevation: float
    face_id: int
    uv: Tuple[float, float]       # UV coordinate within the face in [0,1]^2


@dataclass
class RiverSegment:
    start: RiverNode
    end: RiverNode
    flow_rate: float              # accumulated upstream count at segment start
    tributary_count: int = 0
    in_lake: bool = False         # True when this segment lies inside a lake mask


@dataclass
class StreamNetwork:
    """The rivers' single-receiver accumulation tree, cut at a far lower stream threshold.

    Holds every channel texel plus the texels they drain into (sea, endorheic
    lakes), as flat ids ``face·H·W + row·W + col``.  ``receiver`` indexes these
    arrays (-1 at an outlet or where the tree leaves the network).
    """

    nodes: np.ndarray        # (K,) int64 flat texel id
    positions: np.ndarray    # (K, 3) float32 texel centre, unit sphere
    receiver: np.ndarray     # (K,) int64 index into these arrays, or -1
    flow: np.ndarray         # (K,) float32 tree accumulation (river flow units)
    channel: np.ndarray      # (K,) bool: at or above the threshold, not sea
    stop: np.ndarray         # (K,) bool: sea or endorheic lake — a path ends here
    resolution: int          # face texture side H (= W)

    def __len__(self) -> int:
        return int(self.nodes.shape[0])

    def pack(self) -> dict:
        return {
            "stream_nodes": self.nodes.astype(np.int64),
            "stream_positions": self.positions.astype(np.float32),
            "stream_receiver": self.receiver.astype(np.int64),
            "stream_flow": self.flow.astype(np.float32),
            "stream_channel": self.channel.astype(np.bool_),
            "stream_stop": self.stop.astype(np.bool_),
            "stream_resolution": np.array(self.resolution, dtype=np.int64),
        }

    @classmethod
    def unpack(cls, arrays) -> "StreamNetwork":
        return cls(
            nodes=np.asarray(arrays["stream_nodes"], dtype=np.int64),
            positions=np.asarray(arrays["stream_positions"], dtype=np.float32).reshape(-1, 3),
            receiver=np.asarray(arrays["stream_receiver"], dtype=np.int64),
            flow=np.asarray(arrays["stream_flow"], dtype=np.float32),
            channel=np.asarray(arrays["stream_channel"], dtype=bool),
            stop=np.asarray(arrays["stream_stop"], dtype=bool),
            resolution=int(arrays["stream_resolution"]),
        )


@dataclass
class River:
    segments: List[RiverSegment] = field(default_factory=list)
    source: Optional[RiverNode] = None
    mouth: Optional[RiverNode] = None
    # Flat texel ids (``face·H·W + row·W + col``) of the accumulation-tree nodes it
    # runs through — the ``StreamNetwork`` node ids, so region / local rivers re-cut
    # from that network know which world river they are. A set, not per vertex.
    texels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def as_polyline(self) -> List[np.ndarray]:
        if not self.segments:
            return []
        pts = [self.segments[0].start.position]
        pts += [s.end.position for s in self.segments]
        return pts

    def flow_rates(self) -> List[float]:
        """Returns one flow value per segment (parallel to as_polyline edges)."""
        return [s.flow_rate for s in self.segments]


def flatten_rivers(rivers: Optional[List["River"]]) -> tuple[list, list]:
    """``(paths, texels)``: ``(positions, flows, in_lake)`` polyline tuples (the ``"rivers"``
    feature) and each one's ``River.texels`` (``"river_texels"``), rivers under two points dropped."""
    paths, texels = [], []
    for river in rivers or []:
        polyline = river.as_polyline()
        if len(polyline) < 2:
            continue
        paths.append((
            np.stack(polyline).astype(np.float32),
            np.asarray(river.flow_rates(), dtype=np.float32),
            np.array([s.in_lake for s in river.segments], dtype=bool),
        ))
        texels.append(np.asarray(river.texels, dtype=np.int64))
    return paths, texels


def laplacian_smooth_sphere(
    polyline: List[np.ndarray],
    iterations: int = 8,
    alpha: float = 0.5,
) -> List[np.ndarray]:
    """Laplacian smoothing on the unit sphere, re-projecting each iteration; endpoints pinned."""
    if len(polyline) < 3:
        return [p.astype(np.float32, copy=True) for p in polyline]

    pts = np.stack(polyline).astype(np.float64)
    for _ in range(iterations):
        mids = 0.5 * (pts[:-2] + pts[2:])
        pts[1:-1] += alpha * (mids - pts[1:-1])
        norms = np.linalg.norm(pts[1:-1], axis=1, keepdims=True)
        np.divide(pts[1:-1], norms, out=pts[1:-1], where=norms > 0)
    return [pts[i].astype(np.float32) for i in range(len(pts))]
