"""Typed channel descriptors — the inter-stage data contract.

Stages read and write shared world state through three descriptor kinds instead
of magic strings:

- ``MetaChannel``   — per-face scalars in ``mesh.face_metadata``.
- ``TextureChannel``— ``(N, H, W, C)`` arrays in ``mesh.face_textures``.
- ``FeatureChannel``— vector/point/graph entities in the ``World`` feature registry.

Each stage exports the channels it produces (and that another stage consumes) as
module-level constants; downstream stages import those constants, so the
dependency is an explicit, type-checked edge rather than a string literal.

``MetaChannel`` and ``TextureChannel`` operate on a ``PolygonMesh`` — that is
where their data lives, and both pipeline ``build`` methods (via ``ctx.mesh``)
and deeper helper modules already hold a mesh. ``FeatureChannel`` operates on a
``World`` because the feature registry lives there.
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
    """A named per-face scalar stored in ``mesh.face_metadata``.

    ``read`` returns an ``(N,)`` array (one value per face); ``write`` bulk-sets
    the key across all faces. Presence is tested on face 0, matching the
    established convention that metadata keys are written uniformly across faces.
    """
    name: str
    dtype: Any = np.float64
    default: Union[float, bool, int] = 0.0

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

    def write(self, mesh: "PolygonMesh", values: np.ndarray) -> None:
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
    """A named vector/point/graph entity in the ``World`` feature registry.

    Features are arbitrary Python objects (lists of ``River`` objects,
    ``(positions, kinds, abundances)`` tuples, dataclasses, …) that don't fit the
    per-face raster — so unlike the mesh channels there is no array dtype to
    declare. ``read`` returns the stored object (or ``None`` if absent).
    """
    name: str

    def present(self, world: "World") -> bool:
        return world.has_feature(self.name)

    def read(self, world: "World") -> Any:
        return world.get_feature(self.name)

    def write(self, world: "World", data: Any) -> None:
        world.add_feature(self.name, data)


# Union used by Stage.requires / Stage.provides declarations.
Channel = Union[MetaChannel, TextureChannel, FeatureChannel]
