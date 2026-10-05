"""Pipeline vocabulary — Stage, Pipeline, and BuildContext.

A ``Pipeline`` runs an ordered list of ``Stage`` objects against a shared
``World`` via a ``BuildContext``; build, display and persistence are each a walk
over the stages.  ``ctx.rng(name)`` gives a stream derived from the seed and the
name, independent of call order.
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
        world:    The container stages read/write (its mesh via ``ctx.mesh``).
        seed:     Base seed for the per-stage RNG fan-out. ``None`` → the
                  world's seed, else a random base drawn once here.
        progress: Optional status callback.
    """
    world: World
    seed: "int | None" = None
    progress: ProgressFn = _noop_progress
    _base: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.seed is None:
            self.seed = self.world.seed
        self._base = self.seed if self.seed is not None else secrets.randbits(63)

    @property
    def mesh(self) -> "PolygonMesh":
        return self.world.mesh

    def report(self, msg: str) -> None:
        self.progress(msg)

    def rng(self, name: str) -> np.random.Generator:
        """Deterministic RNG per (seed, name), independent of call order."""
        h = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")
        return np.random.default_rng(np.random.SeedSequence([self._base, h]))


class Stage(ABC):
    """One unit of world generation; display and persistence hooks default to no-ops.

    ``requires`` / ``provides`` declare channels consumed/produced (validation deferred).
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


class Pipeline(Stage):
    """An ordered collection of stages; itself a ``Stage`` whose hooks delegate
    to its children, so pipelines nest."""

    name: str = "pipeline"

    def __init__(self, stages: List[Stage]) -> None:
        self.stages: List[Stage] = list(stages)
        self.world: "World | None" = None

    def __iter__(self) -> Iterator[Stage]:
        return iter(self.stages)

    def __len__(self) -> int:
        return len(self.stages)

    def build(self, ctx: BuildContext) -> None:
        for stage in self.stages:
            stage.build(ctx)
        self.world = ctx.world

    def register_display(
        self, world: World, reg: "DisplayRegistry", *, debug: bool = False
    ) -> None:
        for stage in self.stages:
            stage.register_display(world, reg, debug=debug)

    def serialize(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for stage in self.stages:
            arrays.update(stage.serialize())
        return arrays

    def load(self, arrays: Mapping[str, np.ndarray], ctx: BuildContext) -> None:
        for stage in self.stages:
            stage.load(arrays, ctx)
        self.world = ctx.world

    def apply_display(self, window, *, debug: bool = False) -> None:
        """Build one ``DisplayRegistry`` from every (nested) stage and wire it into ``window``."""
        from viz.mapview.registry import apply_registry
        from core.display import DisplayRegistry

        if self.world is None:
            return
        reg = DisplayRegistry()
        self.register_display(self.world, reg, debug=debug)
        apply_registry(reg, window, self.world)
