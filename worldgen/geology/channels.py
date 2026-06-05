"""Channels produced by the elevation/geology stage."""
from __future__ import annotations

import numpy as np

from core.channels import MetaChannel, TextureChannel

# Per-face ocean classification (written in elevation_builder).
IS_OCEAN = MetaChannel("is_ocean", dtype=np.bool_, default=False)

# Single-channel elevation in km, (N, H, W, 1).
ELEVATION = TextureChannel("elevation", n_channels=1)
