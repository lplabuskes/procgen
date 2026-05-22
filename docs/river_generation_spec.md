# River Generation System — Implementation Spec

**For:** Claude Code  
**Language:** Python  
**Purpose:** Procedurally generate flow-weighted river polylines on a globe mesh with per-face elevation textures.

---

## ⚠️ Integration Assumptions — Verify Before Implementing

The following interfaces are **assumed** based on a generic globe mesh architecture.
**Inspect the actual codebase and reconcile before writing any code.**

| Assumption | What to verify |
|---|---|
| `mesh.faces` is iterable, yielding face objects | Check the actual mesh class/type |
| Each face has `.uv_to_world(uv: tuple) -> np.ndarray` | May be named differently or take different args |
| Each face has a `.id: int` or equivalent unique key | Could be index-based instead |
| `mesh.adjacent_face_pairs()` yields `(face_a, face_b, shared_edge)` | Adjacency API likely differs; check half-edge or winged-edge structure |
| Elevation textures are `dict[face_id, np.ndarray]` (float32, shape HxW) | Could be a texture atlas, single equirect map, or GPU resource |
| Globe is a unit sphere (positions normalized) | Affects projection math in smoothing step |
| `sea_level` is expressed in the same units as elevation texture values | Critical for sink detection |

---

## File Layout

Create a single new module. Suggested path (adjust to match project conventions):

```
<project_root>/
└── rivers/
    ├── __init__.py          # re-exports RiverGenerator, River, RiverSegment
    ├── sampler.py           # ElevationSampler
    ├── graph.py             # GlobeFlowGraph
    ├── flow.py              # FlowAccumulator
    ├── extractor.py         # RiverExtractor
    ├── smoothing.py         # catmull_rom_smooth
    └── generator.py         # RiverGenerator (top-level facade)
```

Alternatively, collapse into a single `rivers.py` if the project prefers flat structure.

---

## Data Structures

```python
# rivers/types.py  (or inline in __init__.py)
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class RiverNode:
    position: np.ndarray        # 3D point on globe surface, shape (3,)
    elevation: float            # value sampled from elevation texture
    face_id: int                # which mesh face this node belongs to
    uv: tuple[float, float]     # UV coordinate within that face [0,1]²


@dataclass
class RiverSegment:
    start: RiverNode
    end: RiverNode
    flow_rate: float            # accumulated flow (dimensionless; scale externally if needed)
    tributary_count: int = 0


@dataclass
class River:
    segments: list[RiverSegment] = field(default_factory=list)
    source: Optional[RiverNode] = None
    mouth: Optional[RiverNode] = None

    def as_polyline(self) -> list[np.ndarray]:
        """Returns ordered list of 3D positions from source to mouth."""
        if not self.segments:
            return []
        pts = [self.segments[0].start.position]
        pts += [s.end.position for s in self.segments]
        return pts

    def flow_rates(self) -> list[float]:
        """Returns flow_rate per segment, parallel to as_polyline() edges."""
        return [s.flow_rate for s in self.segments]
```

---

## Phase 1 — Elevation Sampling (`sampler.py`)

Bilinear interpolation into a per-face heightmap.

**⚠️ If your elevation data is a texture atlas or equirectangular map rather than
per-face arrays, replace `self.textures[face_id]` lookup with the appropriate
projection/sampling call.**

```python
import numpy as np


class ElevationSampler:
    def __init__(self, mesh, elevation_textures: dict):
        """
        Args:
            mesh: The globe mesh object (interface TBD — see assumptions above).
            elevation_textures: Mapping of face_id -> np.ndarray (float32, shape HxW).
                                Elevation values should be in a consistent unit
                                comparable to the sea_level parameter used later.
        """
        self.mesh = mesh
        self.textures = elevation_textures

    def sample(self, face_id: int, uv: tuple[float, float]) -> float:
        """Bilinear sample at (u,v) in [0,1]² on the given face."""
        tex = self.textures[face_id]
        h, w = tex.shape
        x = uv[0] * (w - 1)
        y = uv[1] * (h - 1)
        x0, y0 = int(x), int(y)
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)
        fx, fy = x - x0, y - y0
        return float(
            tex[y0, x0] * (1 - fx) * (1 - fy) +
            tex[y0, x1] * fx       * (1 - fy) +
            tex[y1, x0] * (1 - fx) * fy       +
            tex[y1, x1] * fx       * fy
        )
```

---

## Phase 2 — Flow Graph (`graph.py`)

Discretizes each mesh face into a uniform grid of `RiverNode`s, then wires up
intra-face 4-connectivity and cross-face boundary edges.

**⚠️ `_stitch_face_boundaries` is a stub. You must implement it using the actual
mesh adjacency API (half-edge, winged-edge, adjacency list, etc.). The contract
is: for each pair of adjacent faces, find boundary grid nodes that are spatially
coincident (or closest) and add bidirectional edges between them.**

```python
from __future__ import annotations
from rivers.types import RiverNode
from rivers.sampler import ElevationSampler


class GlobeFlowGraph:
    def __init__(self, mesh, sampler: ElevationSampler, resolution: int = 64):
        """
        Args:
            resolution: Number of grid samples per face edge.
                        64 is reasonable for prototyping; use 128-256 for quality.
        """
        self.resolution = resolution
        self.sampler = sampler
        # node key: (face_id, i, j)  where i,j in [0, resolution)
        self.nodes: dict[tuple, RiverNode] = {}
        # adjacency: key -> list of neighbor keys
        self.edges: dict[tuple, list[tuple]] = {}
        self._build(mesh)

    def _build(self, mesh):
        res = self.resolution

        # --- Intra-face grid sampling ---
        # ⚠️ VERIFY: how to enumerate faces and convert UV to world position.
        #    Replace `enumerate(mesh.faces)` and `face.uv_to_world(uv)` as needed.
        for face_id, face in enumerate(mesh.faces):
            for i in range(res):
                for j in range(res):
                    uv = (i / (res - 1), j / (res - 1))
                    pos = face.uv_to_world(uv)   # ⚠️ verify method name/signature
                    elev = self.sampler.sample(face_id, uv)
                    key = (face_id, i, j)
                    self.nodes[key] = RiverNode(
                        position=pos,
                        elevation=elev,
                        face_id=face_id,
                        uv=uv,
                    )
                    # 4-connected intra-face neighbors
                    neighbors = []
                    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < res and 0 <= nj < res:
                            neighbors.append((face_id, ni, nj))
                    self.edges[key] = neighbors

        self._stitch_face_boundaries(mesh)

    def _stitch_face_boundaries(self, mesh):
        """
        *** STUB — must be implemented against the actual mesh API. ***

        Goal: for each pair of adjacent mesh faces, connect boundary grid nodes
        across the shared edge with bidirectional edges.

        Pseudocode contract:
            for face_a, face_b, shared_edge in mesh.adjacent_face_pairs():
                for key_a, key_b in self._match_boundary_nodes(face_a, face_b):
                    self.edges[key_a].append(key_b)
                    self.edges[key_b].append(key_a)

        _match_boundary_nodes should pair nodes by proximity: for each boundary
        node on face_a, find the nearest boundary node on face_b and yield both keys.

        Implementation hints:
          - For cube-sphere meshes: boundary nodes share an exact edge; match by
            index along that edge (they will have the same resolution).
          - For icosphere meshes: use spatial proximity (kdtree or sorted angular
            position along the shared great-circle arc).
          - Threshold for "close enough": roughly 2x the intra-face edge spacing.
        """
        raise NotImplementedError(
            "_stitch_face_boundaries must be implemented using the project's "
            "mesh adjacency API. See docstring for contract and hints."
        )
```

---

## Phase 3 — Flow Accumulation (`flow.py`)

Uses the **Priority-Flood** algorithm to route flow from every land node to the
nearest ocean sink, handling depressions without explicit pit-filling.

```python
from __future__ import annotations
import heapq
from typing import Optional
from rivers.graph import GlobeFlowGraph


class FlowAccumulator:
    def __init__(self, graph: GlobeFlowGraph):
        self.graph = graph
        # flow_dir[key] = parent_key this node drains into; None means ocean sink
        self.flow_dir: dict[tuple, Optional[tuple]] = {}
        # flow_accum[key] = total upstream area (each node contributes 1.0)
        self.flow_accum: dict[tuple, float] = {}

    def compute(self, sea_level: float = 0.0) -> None:
        """
        Args:
            sea_level: Elevation threshold below which nodes are ocean sinks.
                       Must be in the same units as ElevationSampler output.
        """
        nodes = self.graph.nodes
        edges = self.graph.edges

        visited: set[tuple] = set()
        heap: list[tuple] = []   # (effective_elevation, node_key)

        # Seed ocean/sink nodes
        for key, node in nodes.items():
            if node.elevation <= sea_level:
                heapq.heappush(heap, (node.elevation, key))
                self.flow_dir[key] = None
                visited.add(key)

        # Priority-Flood: process lowest nodes first; upstream nodes drain into
        # the node that first reached them (their effective parent).
        while heap:
            elev, key = heapq.heappop(heap)
            for nb_key in edges.get(key, []):
                if nb_key not in visited:
                    nb_node = nodes[nb_key]
                    self.flow_dir[nb_key] = key
                    visited.add(nb_key)
                    # Effective elevation is max(current, neighbor) to prevent
                    # water flowing "up" through depressions in traversal order.
                    eff_elev = max(elev, nb_node.elevation)
                    heapq.heappush(heap, (eff_elev, nb_key))

        # Accumulate flow: process high-to-low, each node adds its count to parent
        for key in nodes:
            self.flow_accum[key] = 1.0

        sorted_keys = sorted(nodes.keys(), key=lambda k: -nodes[k].elevation)
        for key in sorted_keys:
            parent = self.flow_dir.get(key)
            if parent is not None:
                self.flow_accum[parent] = (
                    self.flow_accum.get(parent, 1.0) +
                    self.flow_accum.get(key, 1.0)
                )
```

---

## Phase 4 — River Extraction (`extractor.py`)

Traces rivers downstream from headwater seeds, emitting `RiverSegment`s.

```python
from __future__ import annotations
from typing import Optional
from rivers.types import River, RiverSegment
from rivers.graph import GlobeFlowGraph
from rivers.flow import FlowAccumulator


class RiverExtractor:
    def __init__(
        self,
        graph: GlobeFlowGraph,
        accumulator: FlowAccumulator,
        min_flow_threshold: float = 50.0,
    ):
        """
        Args:
            min_flow_threshold: Minimum flow_accum value for a node to be part of
                                a river. Tune this to control river density.
                                Typical range: 20 (many small streams) – 500 (major rivers only).
        """
        self.graph = graph
        self.acc = accumulator
        self.threshold = min_flow_threshold

    def extract_all(self) -> list[River]:
        """Returns all rivers with segments above the flow threshold."""
        is_above = {
            k for k, v in self.acc.flow_accum.items()
            if v >= self.threshold
        }

        # Headwaters: above threshold, but no upstream neighbor is also above threshold
        headwaters = []
        for key in is_above:
            upstream_above = [
                nb for nb in self._upstream_neighbors(key)
                if nb in is_above
            ]
            if not upstream_above:
                headwaters.append(key)

        rivers = []
        for hw_key in headwaters:
            river = self._trace_river(hw_key)
            if river is not None:
                rivers.append(river)
        return rivers

    def _upstream_neighbors(self, key: tuple) -> list[tuple]:
        return [
            nb for nb in self.graph.edges.get(key, [])
            if self.acc.flow_dir.get(nb) == key
        ]

    def _trace_river(self, start_key: tuple) -> Optional[River]:
        nodes = self.graph.nodes
        river = River(source=nodes[start_key])
        current = start_key

        while True:
            next_key = self.acc.flow_dir.get(current)
            if next_key is None:
                river.mouth = nodes[current]
                break

            seg = RiverSegment(
                start=nodes[current],
                end=nodes[next_key],
                flow_rate=self.acc.flow_accum[current],
            )
            river.segments.append(seg)
            current = next_key

        # Discard trivially short traces
        return river if len(river.segments) > 2 else None
```

---

## Phase 5 — Smoothing (`smoothing.py`)

Catmull-Rom spline interpolation, re-projected onto the unit sphere after each
subdivision step.

**⚠️ If your globe is not a unit sphere (e.g. scaled to planet radius), replace
`pt / np.linalg.norm(pt)` with the appropriate surface projection for your geometry.**

```python
from __future__ import annotations
import numpy as np


def catmull_rom_smooth(
    polyline: list[np.ndarray],
    subdivisions: int = 4,
) -> list[np.ndarray]:
    """
    Smooth a polyline of 3D points using Catmull-Rom splines, then re-project
    each interpolated point onto the unit sphere.

    Args:
        polyline:     Ordered list of np.ndarray positions, shape (3,) each.
        subdivisions: Interpolated points inserted per segment.
                      4 is a good default; 8 for high-quality output.

    Returns:
        Smoothed list of np.ndarray positions on the unit sphere.
    """
    pts = polyline
    if len(pts) < 4:
        return pts

    # Phantom endpoints to give the spline full tangents at start/end
    padded = (
        [pts[0] + (pts[0] - pts[1])]
        + pts
        + [pts[-1] + (pts[-1] - pts[-2])]
    )

    result: list[np.ndarray] = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i-1], padded[i], padded[i+1], padded[i+2]
        for s in range(subdivisions):
            t = s / subdivisions
            pt = 0.5 * (
                (2 * p1) +
                (-p0 + p2) * t +
                (2*p0 - 5*p1 + 4*p2 - p3) * t**2 +
                (-p0 + 3*p1 - 3*p2 + p3) * t**3
            )
            # Re-project onto sphere surface
            norm = np.linalg.norm(pt)
            result.append(pt / norm if norm > 0 else pt)

    # Append final point
    last = pts[-1]
    norm = np.linalg.norm(last)
    result.append(last / norm if norm > 0 else last)
    return result
```

---

## Top-Level Facade (`generator.py`)

```python
from __future__ import annotations
import numpy as np
from rivers.types import River
from rivers.sampler import ElevationSampler
from rivers.graph import GlobeFlowGraph
from rivers.flow import FlowAccumulator
from rivers.extractor import RiverExtractor
from rivers.smoothing import catmull_rom_smooth


class RiverGenerator:
    """
    End-to-end river generation pipeline.

    Usage:
        gen = RiverGenerator(mesh, elevation_textures)
        rivers = gen.generate()
        # Each River has .segments (list[RiverSegment]) and .as_polyline()
        # Each RiverSegment has .flow_rate (float), .start, .end (RiverNode)
    """

    def __init__(
        self,
        mesh,
        elevation_textures: dict,
        resolution: int = 64,
        min_flow: float = 50.0,
        sea_level: float = 0.0,
        smooth_subdivisions: int = 4,
    ):
        """
        Args:
            mesh:                The globe mesh object (see integration assumptions).
            elevation_textures:  face_id -> np.ndarray heightmap (see integration assumptions).
            resolution:          Grid samples per face edge (64–256).
            min_flow:            Flow accumulation threshold for river inclusion.
            sea_level:           Elevation value treated as ocean/sink.
            smooth_subdivisions: Catmull-Rom interpolation steps per segment.
        """
        self.smooth_subdivisions = smooth_subdivisions

        self.sampler = ElevationSampler(mesh, elevation_textures)
        self.graph = GlobeFlowGraph(mesh, self.sampler, resolution)
        self.accumulator = FlowAccumulator(self.graph)
        self.accumulator.compute(sea_level)
        self.extractor = RiverExtractor(self.graph, self.accumulator, min_flow)

    def generate(self) -> list[River]:
        """
        Run the full pipeline and return all rivers.

        Returns:
            List of River objects. Each River exposes:
              .as_polyline()  -> list[np.ndarray]  (3D positions, source to mouth)
              .flow_rates()   -> list[float]        (one per segment/edge)
              .segments       -> list[RiverSegment] (full detail)
              .source         -> RiverNode
              .mouth          -> RiverNode
        """
        rivers = self.extractor.extract_all()

        for river in rivers:
            raw = river.as_polyline()
            smoothed = catmull_rom_smooth(raw, self.smooth_subdivisions)

            # Write smoothed positions back into segments
            for i, seg in enumerate(river.segments):
                if i < len(smoothed):
                    seg.start.position = smoothed[i]
                if i + 1 < len(smoothed):
                    seg.end.position = smoothed[i + 1]

        return rivers
```

---

## Suggested Configuration Presets

```python
# Many fine rivers (dense network, slower)
RiverGenerator(mesh, textures, resolution=128, min_flow=20.0)

# Major rivers only (fast, good for LOD0)
RiverGenerator(mesh, textures, resolution=64,  min_flow=200.0)

# High-quality smoothing
RiverGenerator(mesh, textures, resolution=64,  min_flow=50.0, smooth_subdivisions=8)
```

---

## Implementation Order for Claude Code

Implement in this order to enable incremental testing at each step:

1. **`types.py`** — no dependencies, trivially testable
2. **`sampler.py`** — test with a synthetic 2D array and known UV coords
3. **`graph.py`** — implement intra-face grid only first; stub `_stitch_face_boundaries` with `pass` (not `raise`) to allow early end-to-end runs on single-face meshes
4. **`flow.py`** — test with a small hand-crafted graph (5×5 grid, manual elevation values)
5. **`extractor.py`** — verify headwater detection and trace length filtering
6. **`smoothing.py`** — test standalone with a known zigzag polyline
7. **`generator.py`** — wire up and run end-to-end
8. **`_stitch_face_boundaries`** — implement last, once the mesh adjacency API is understood

---

## Known Unknowns (Resolve During Implementation)

- **Cross-face stitching** is the highest-risk integration point. If the mesh uses a
  half-edge structure, `mesh.half_edges` or `face.adjacent_faces()` may give you
  what you need. If it uses an indexed face set, build an edge-to-faces lookup first.

- **Memory**: At `resolution=128`, a 6-face cube-sphere has ~98k nodes. At 256, ~393k.
  If this is too large, consider a sparse representation where only nodes above a
  flow threshold are kept after accumulation.

- **Performance**: The priority-flood heap processes every node. For large resolutions,
  consider running accumulation in a separate thread or process, or caching results
  to disk between runs.

- **Rainfall weighting**: Currently all nodes contribute `1.0` to flow. For realism,
  weight by latitude (more rain near equator) or a noise-based precipitation map by
  multiplying the initial `flow_accum[key]` value before accumulation.
