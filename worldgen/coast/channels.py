"""Channels produced by the coast stage."""
from __future__ import annotations

import numpy as np

from core.channels import MetaChannel, TextureChannel
from core.codec import TextureCodec

# Per-face coast smoothness in [0, 1]: 1 = long smooth beaches and deltas,
# 0 = crenellated, rugged, fjord-like shores.  For hover and the world-map scheme.
COAST_SMOOTHNESS_FACE = MetaChannel("coast_smoothness", dtype=np.float64, default=0.5)

# The same field rasterised continuously across face edges, (N, H, W, 1).  The
# region and local tiers read it to style the noise along their shorelines.
COAST_SMOOTHNESS = TextureChannel("coast_smoothness", n_channels=1, codec=TextureCodec(step=1 / 255))
