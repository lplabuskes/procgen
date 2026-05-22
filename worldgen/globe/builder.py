from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from typing import Callable, Optional

from worldgen.geology.elevation_builder import ElevationBuilder, ElevationParams
from worldgen.globe.globe import Globe


@dataclass
class GlobeParams:
    """Top-level parameters for the full globe-generation pipeline.

    Stage-specific params are nested. New stages (climate, hydrology, etc.)
    should add their own params dataclass as a new field here.
    """
    elevation: ElevationParams = field(default_factory=ElevationParams)


class GlobeBuilder:
    """Orchestrates every stage of globe generation.

    Each stage's builder is stored as an attribute (e.g. `elevation_builder`)
    so callers can read intermediate results without re-running work.
    `build()` returns a `Globe` wrapping the resulting mesh.
    """

    def __init__(self, params: GlobeParams | None = None) -> None:
        self.params: GlobeParams = params or GlobeParams()
        self.elevation_builder: ElevationBuilder = ElevationBuilder(self.params.elevation)
        self.globe: Optional[Globe] = None

    def build(self, progress: Optional[Callable[[str], None]] = None) -> Globe:
        mesh = self.elevation_builder.build(progress=progress)
        globe = Globe(mesh=mesh)
        eb = self.elevation_builder
        if eb.rivers:
            paths_with_flow = []
            for river in eb.rivers:
                polyline = river.as_polyline()
                if len(polyline) < 2:
                    continue
                positions = np.stack(polyline).astype(np.float32)  # (M, 3)
                flows = river.flow_rates()                          # length M-1
                paths_with_flow.append((positions, flows))
            if paths_with_flow:
                globe.add_feature("rivers", paths_with_flow)
        self.globe = globe
        return globe
