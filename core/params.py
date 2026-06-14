"""Shared utilities for declaring UI-driven params dataclasses.

Usage::

    from core.params import field, paramclass, param_field as _f

    @paramclass
    class FooParams:
        x: float = _f(1.0, label="X value", min=0.0, max=10.0)
        sub: SubParams = field(default_factory=SubParams, metadata={"label": "Sub"})
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

paramclass = dataclass


def param_field(default, *, label: str, min: float | int, max: float | int,
                step: float | None = None, hidden: bool = False) -> Any:
    """Declare a leaf param field with UI dialog metadata."""
    meta: dict = {"label": label, "min": min, "max": max}
    if step is not None:
        meta["step"] = step
    if hidden:
        meta["hidden"] = True
    return field(default=default, metadata=meta)


__all__ = ["field", "param_field", "paramclass"]
