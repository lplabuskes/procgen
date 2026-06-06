"""Channels produced by the cost stage."""
from __future__ import annotations

import numpy as np

from core.channels import FeatureChannel, MetaChannel

IS_WATER = MetaChannel("is_water", dtype=np.bool_, default=False)

# Water face pinched between opposite land neighbours.
IS_STRAIT = MetaChannel("is_strait", dtype=np.bool_, default=False)

# ── Features ─────────────────────────────────────────────────────────────────
# The full ``CostSurfaces`` dataclass (per-face cost + river flow field)
# consumed by realmgen.
COST_SURFACES = FeatureChannel("cost_surfaces")
