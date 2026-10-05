"""World — the generic container a Pipeline reads and writes.

Holds the surface ``mesh`` (per-face textures + metadata) plus a feature registry
for vector/point/graph entities that don't fit the per-face raster.  The
constructor loads a mesh from the shared cache (building it on first run);
:meth:`World.from_mesh` wraps an existing one.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterator, Optional, TYPE_CHECKING

from core import mesh_cache
from core.mesh_cache import MeshParams

if TYPE_CHECKING:
    from core.mesh import PolygonMesh


class World:
    """Container for a generated world: a mesh plus a named feature registry."""

    def __init__(
        self,
        mesh_params: Optional[MeshParams] = None,
        *,
        seed: Optional[int] = None,
        progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        params = mesh_params if mesh_params is not None else MeshParams()
        if not mesh_cache.is_populated():
            if progress is not None:
                progress("Building mesh cache (first run only)…")
            mesh_cache.build(params, progress=progress)
        self.mesh: "PolygonMesh" = mesh_cache.load_random(seed=seed)
        # For per-world-deterministic RNG downstream.
        self.seed: Optional[int] = seed
        self._features: Dict[str, Any] = {}
        # Set by ElevationBuilder after ocean data is available.
        self.antimeridian: float = math.pi

    @classmethod
    def from_mesh(cls, mesh: "PolygonMesh") -> "World":
        """Wrap an already-built mesh without touching the cache."""
        self = cls.__new__(cls)
        self.mesh = mesh
        self.seed = None
        self._features = {}
        self.antimeridian = math.pi
        return self

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
