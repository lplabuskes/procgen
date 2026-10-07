"""Channels produced by the wetland stage."""
from __future__ import annotations

import numpy as np

from core.channels import MetaChannel, TextureChannel
from core.codec import TextureCodec

# Wetness score in [0, 1] per texel, (N, H, W, 1).  The globe marks the Wetland biome
# above ``WetlandParams.threshold``; the region and local tiers refine it and mark
# their own, lower, thresholds (smaller wetlands: riparian marsh, valley-floor fens).
WETLAND = TextureChannel("wetland", n_channels=1, codec=TextureCodec(step=1e-3))

# Per-face share of the face's in-polygon texels in the Wetland biome.  The face's
# ``biome`` stays its dominant *climate* biome; consumers read wetland from here.
WETLAND_FRACTION = MetaChannel("wetland_fraction", dtype=np.float64)
