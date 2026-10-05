"""Channels produced by the delta stage."""
from __future__ import annotations

import numpy as np

from core.channels import FeatureChannel, MetaChannel, TextureChannel

# Per-texel delta plain, (N, H, W, 1): 1 on the deposited plain (built land and
# flattened lowland alike).  Its marshes are the wetland stage's.  The region and
# local tiers copy it down (bilinear, not conserved) and site distributaries on it.
DELTA = TextureChannel("delta", n_channels=1)

# Per-face share of the face's texels on the delta plain.
DELTA_FRACTION = MetaChannel("delta_fraction", dtype=np.float64)

# ``List[DeltaFan]`` — one record per delta (apex, new mouth, axis, radius, flow).
DELTA_FANS = FeatureChannel("delta_fans")
# (M, 3) the delta rivers' mouths at their new fronts — the debug overlay, and
# the markers the region / local tiers snap onto their re-routed mouths.
DELTA_MOUTHS = FeatureChannel("deltas")
