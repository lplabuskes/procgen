from __future__ import annotations

from dataclasses import dataclass, field
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
        if self.elevation_builder.rivers is not None:
            globe.add_feature("rivers", self.elevation_builder.rivers)
        self.globe = globe
        return globe
