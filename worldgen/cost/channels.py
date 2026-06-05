"""Channels produced by the cost stage."""
from __future__ import annotations

import numpy as np

from core.channels import MetaChannel
from worldgen.geology.channels import IS_OCEAN

# Sailable-water union (ocean ∪ lake). Falls back to is_ocean for worlds saved
# before the cost stage wrote is_water.
IS_WATER = MetaChannel("is_water", dtype=np.bool_, fallback=IS_OCEAN, default=False)

# Water face pinched between opposite land neighbours.
IS_STRAIT = MetaChannel("is_strait", dtype=np.bool_, default=False)
