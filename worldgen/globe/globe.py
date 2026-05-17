from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

from worldgen.mesh.mymesh import PolygonMesh


class Globe:
    """Container for a generated world.

    Holds the surface `mesh` (rasterized per-face data) plus a generic
    `features` registry for vector-style overlays (rivers, coastlines, etc.)
    that don't fit naturally into the per-face/per-pixel structure.
    """

    def __init__(self, mesh: PolygonMesh) -> None:
        self.mesh: PolygonMesh = mesh
        self._features: Dict[str, Any] = {}

    def add_feature(self, name: str, data: Any) -> None:
        self._features[name] = data

    def get_feature(self, name: str, default: Any = None) -> Any:
        return self._features.get(name, default)

    def has_feature(self, name: str) -> bool:
        return name in self._features

    def remove_feature(self, name: str) -> None:
        self._features.pop(name, None)

    def feature_names(self) -> Iterator[str]:
        return iter(self._features.keys())
