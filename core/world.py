"""World — the generic container a Pipeline reads and writes.

Holds the surface ``mesh`` (rasterized per-face data: textures + metadata) plus a
generic feature registry for vector/point/graph entities (rivers, minerals,
coastlines, and future cities/roads/realms) that don't fit the per-face raster.

This is the promoted form of the old ``worldgen.globe.Globe``. The World owns its
mesh: the constructor loads one from the shared cache (building the cache on first
run), so ``mesh`` is always present. Use :meth:`World.from_mesh` to wrap an
already-built mesh (e.g. when loading a saved world).
"""
from __future__ import annotations

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
        # Retained so downstream stages can derive per-world-deterministic RNG
        # (e.g. polity scalars) — same seed ⇒ same world ⇒ same generation.
        self.seed: Optional[int] = seed
        self._features: Dict[str, Any] = {}

    @classmethod
    def from_mesh(cls, mesh: "PolygonMesh") -> "World":
        """Wrap an already-built mesh without touching the cache."""
        self = cls.__new__(cls)
        self.mesh = mesh
        self.seed = None
        self._features = {}
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
