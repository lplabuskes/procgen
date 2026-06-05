"""Back-compat shim — ``Globe`` is now ``core.world.World``.

The generic world container was promoted into ``core`` so non-worldgen layers
(realmgen, future city/name generators) can depend on it without importing
worldgen. ``Globe`` remains as an alias so existing imports keep working; new
code should import ``World`` from ``core.world``.
"""
from __future__ import annotations

from core.world import World

# Existing code imports ``Globe`` from here; keep the name bound to the one
# canonical implementation in core.
Globe = World

__all__ = ["Globe", "World"]
