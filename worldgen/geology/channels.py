"""Channels produced by the elevation/geology stage."""
from __future__ import annotations

import numpy as np

from core.channels import MetaChannel, TextureChannel
from core.codec import TextureCodec

# ── Per-face metadata ────────────────────────────────────────────────────────
# Consumed by resources, cost, realmgen.
IS_OCEAN = MetaChannel("is_ocean", dtype=np.bool_, default=False)
# Per-face tectonic+erosion elevation in km (pre-noise, not the texture mean);
# consumed by the resources placement models.
ELEVATION_FACE = MetaChannel("elevation", dtype=np.float64)
# Plate classification ("oceanic"/"continental") and boundary classification
# ("interior"/"convergent"/"divergent"/"transform") — consumed by minerals.
PLATE_TYPE = MetaChannel("plate_type", dtype=object, default="oceanic")
BOUNDARY_TYPE = MetaChannel("boundary_type", dtype=object, default="interior")
# Per-face volcanic activity [0, 1] — consumed by resources (fertility/minerals).
VOLCANIC_ACTIVITY = MetaChannel("volcanic_activity", dtype=np.float64)

# ── Per-face textures ────────────────────────────────────────────────────────
# Single-channel elevation in km, (N, H, W, 1).
# Saved to 0.1 m: region/local lake levels and drainage refine it.
ELEVATION = TextureChannel("elevation", n_channels=1, codec=TextureCodec(step=1e-4))
# Single-channel ocean fraction in [0, 1], (N, H, W, 1).
OCEAN_MASK = TextureChannel("ocean_mask", n_channels=1)
