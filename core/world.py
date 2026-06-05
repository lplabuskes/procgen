"""World — the generic container a Pipeline reads and writes.

Holds the surface ``mesh`` (rasterized per-face data: textures + metadata) plus a
generic feature registry for vector/point/graph entities (rivers, minerals,
coastlines, and future cities/roads/realms) that don't fit the per-face raster.

This is the promoted form of the old ``worldgen.globe.Globe`` (which now aliases
``World``). The mesh is ``Optional`` because the mesh-producing stage (elevation)
sets it during ``Pipeline.build``; ``BuildContext.mesh`` guards access before then.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.mesh import PolygonMesh


class World:
    """Container for a generated world: a mesh plus a named feature registry."""

    def __init__(self, mesh: "Optional[PolygonMesh]" = None) -> None:
        self.mesh: "Optional[PolygonMesh]" = mesh
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
