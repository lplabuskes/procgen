"""Typed channel descriptors — the inter-stage data contract.

Stages read and write the shared world through ``MetaChannel`` (per-face scalars
in ``mesh.face_metadata``) and ``TextureChannel`` (``(N, H, W, C)`` arrays in
``mesh.face_textures``) instead of magic strings. Each stage exports the channels
it produces as module-level constants; downstream stages import those constants,
so the dependency is an explicit, type-checked edge rather than a string literal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
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

    def present(self, world: "World") -> bool:
        meta = world.mesh.face_metadata
        return len(meta) > 0 and self.name in meta[0]

    def read(self, world: "World") -> np.ndarray:
        meta = world.mesh.face_metadata
        if self.present(world):
            return np.array(
                [m.get(self.name, self.default) for m in meta], dtype=self.dtype
            )
        return np.full(len(meta), self.default, dtype=self.dtype)

    def write(self, world: "World", values: np.ndarray) -> None:
        world.mesh.apply_metadata_channel(self.name, np.asarray(values).tolist())


@dataclass(frozen=True)
class TextureChannel:
    """A named ``(N, H, W, C)`` per-face texture stored in ``mesh.face_textures``."""
    name: str
    n_channels: int = 1

    def present(self, world: "World") -> bool:
        return self.name in world.mesh.face_textures

    def read(self, world: "World") -> np.ndarray:
        return world.mesh.face_textures[self.name]

    def write(self, world: "World", tex: np.ndarray) -> None:
        world.mesh.set_face_textures(self.name, tex, n_channels=self.n_channels)


# Union used by Stage.requires / Stage.provides declarations.
Channel = Union[MetaChannel, TextureChannel]
