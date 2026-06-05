"""Pipeline vocabulary — Stage, Pipeline, and BuildContext.

A ``Pipeline`` is an ordered list of ``Stage`` objects run against a shared
``World`` via a ``BuildContext``. Orchestration, display, and persistence all
become ``for stage in pipeline: stage.<hook>(...)`` — so adding a generation
stage is one new class appended to the list, with no edits to central
god-functions.

``BuildContext`` carries the world, a progress callback, and a deterministic
per-stage RNG fan-out: ``ctx.rng("climate")`` returns a reproducible stream
derived from the context seed and the stage name (order-independent), so stages
stop managing their own seed bookkeeping.
"""
from __future__ import annotations

import hashlib
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Mapping, Tuple, TYPE_CHECKING

import numpy as np

from core.world import World

if TYPE_CHECKING:
    from core.mesh import PolygonMesh
    from core.channels import Channel
    from core.display import DisplayRegistry


ProgressFn = Callable[[str], None]


def _noop_progress(_msg: str) -> None:
    return None


@dataclass
class BuildContext:
    """Shared state threaded through every stage's ``build``.

    Args:
        world:    The container stages read/write. Its ``mesh`` is ``None`` until
                  the mesh-producing stage runs; access via ``ctx.mesh`` after.
        seed:     Base seed for the per-stage RNG fan-out. ``None`` → a random
                  base is drawn once at construction (reproducible within this
                  context, random across runs).
        progress: Optional status callback.
    """
    world: World
    seed: "int | None" = None
    progress: ProgressFn = _noop_progress
    _base: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._base = self.seed if self.seed is not None else secrets.randbits(63)

    @property
    def mesh(self) -> "PolygonMesh":
        if self.world.mesh is None:
            raise RuntimeError(
                "World mesh not available; a mesh-producing stage must run first"
            )
        return self.world.mesh

    def report(self, msg: str) -> None:
        self.progress(msg)

    def rng(self, name: str) -> np.random.Generator:
        """Deterministic per-name RNG: stable for a given (seed, name) pair,
        independent of call order across stages."""
        h = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")
        return np.random.default_rng(np.random.SeedSequence([self._base, h]))


class Stage(ABC):
    """One unit of world generation.

    Subclasses must implement ``build``; the display and persistence hooks
    default to no-ops so a stage overrides only what it needs. ``requires`` /
    ``provides`` may declare the channels a stage consumes/produces for optional
    ordering validation (deferred).
    """

    name: str = "stage"
    requires: "Tuple[Channel, ...]" = ()
    provides: "Tuple[Channel, ...]" = ()

    @abstractmethod
    def build(self, ctx: BuildContext) -> None:
        ...

    def register_display(
        self, world: World, reg: "DisplayRegistry", *, debug: bool = False
    ) -> None:
        return None

    def serialize(self) -> dict[str, np.ndarray]:
        return {}

    def load(self, arrays: Mapping[str, np.ndarray], ctx: BuildContext) -> None:
        return None


class Pipeline:
    """An ordered collection of stages run against a shared ``BuildContext``."""

    def __init__(self, stages: List[Stage]) -> None:
        self.stages: List[Stage] = list(stages)

    def __iter__(self) -> Iterator[Stage]:
        return iter(self.stages)

    def __len__(self) -> int:
        return len(self.stages)

    def build(self, ctx: BuildContext) -> World:
        for stage in self.stages:
            stage.build(ctx)
        return ctx.world
