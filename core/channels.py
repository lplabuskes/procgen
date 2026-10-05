"""Typed channel descriptors — the inter-stage data contract.

- ``MetaChannel``   — per-face scalars in ``mesh.face_metadata``.
- ``TextureChannel``— ``(N, H, W, C)`` arrays in ``mesh.face_textures``.
- ``FeatureChannel``— vector/point/graph entities in the ``World`` feature registry.

Stages export the channels they produce as module-level constants and consumers
import them, making dependencies explicit edges rather than string literals.
Mesh channels operate on a ``PolygonMesh``; ``FeatureChannel`` on a ``World``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.mesh import PolygonMesh
    from core.world import World


@dataclass(frozen=True)
class MetaChannel:
    """A named per-face scalar in ``mesh.face_metadata``; ``read`` returns ``(N,)``.

    Presence is tested on face 0: metadata keys are written uniformly across faces.
    """
    name: str
    dtype: Any = np.float64
    default: Any = 0.0

    def present(self, mesh: "PolygonMesh") -> bool:
        meta = mesh.face_metadata
        return len(meta) > 0 and self.name in meta[0]

    def read(self, mesh: "PolygonMesh") -> np.ndarray:
        meta = mesh.face_metadata
        if self.present(mesh):
            return np.array(
                [m.get(self.name, self.default) for m in meta], dtype=self.dtype
            )
        return np.full(len(meta), self.default, dtype=self.dtype)

    def write(self, mesh: "PolygonMesh", values: Any) -> None:
        mesh.apply_metadata_channel(self.name, np.asarray(values).tolist())


@dataclass(frozen=True)
class TextureChannel:
    """A named ``(N, H, W, C)`` per-face texture stored in ``mesh.face_textures``."""
    name: str
    n_channels: int = 1

    def present(self, mesh: "PolygonMesh") -> bool:
        return self.name in mesh.face_textures

    def read(self, mesh: "PolygonMesh") -> np.ndarray:
        return mesh.face_textures[self.name]

    def write(self, mesh: "PolygonMesh", tex: np.ndarray) -> None:
        mesh.set_face_textures(self.name, tex, n_channels=self.n_channels)


@dataclass(frozen=True)
class FeatureChannel:
    """A named entity (any Python object) in the ``World`` feature registry; ``read`` gives ``None`` if absent."""
    name: str

    def present(self, world: "World") -> bool:
        return world.has_feature(self.name)

    def read(self, world: "World") -> Any:
        return world.get_feature(self.name)

    def write(self, world: "World", data: Any) -> None:
        world.add_feature(self.name, data)


# Union used by Stage.requires / Stage.provides declarations.
Channel = Union[MetaChannel, TextureChannel, FeatureChannel]
