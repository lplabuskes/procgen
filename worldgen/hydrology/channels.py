"""Channels produced by the hydrology stage."""
from __future__ import annotations

from core.channels import TextureChannel

# 1.0 where a pixel is inside the post-shrinking lake footprint, else 0.0.
LAKE_MASK = TextureChannel("lake_mask", n_channels=1)
