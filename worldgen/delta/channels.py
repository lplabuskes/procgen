"""Channels produced by the delta stage."""
from __future__ import annotations

import numpy as np

from core.channels import FeatureChannel, MetaChannel, TextureChannel

# Per-texel delta plain, (N, H, W, 2): channel 0 is 1 on the deposited plain
# (built land and flattened lowland alike), channel 1 the marsh on its seaward
# part, in [0, 1].  The region and local tiers copy it down (bilinear, not
# conserved) and site distributaries on it.
DELTA = TextureChannel("delta", n_channels=2)

# Per-face share of the face's texels on the delta plain / in its marsh.
DELTA_FRACTION = MetaChannel("delta_fraction", dtype=np.float64)
MARSH_FRACTION = MetaChannel("marsh_fraction", dtype=np.float64)

# ``List[DeltaFan]`` — one record per delta (apex, new mouth, axis, radius, flow).
DELTA_FANS = FeatureChannel("delta_fans")
# (M, 3) the delta rivers' mouths at their new fronts — the debug overlay, and
# the markers the region / local tiers snap onto their re-routed mouths.
DELTA_MOUTHS = FeatureChannel("deltas")
