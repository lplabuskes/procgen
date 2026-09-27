"""Channels produced by the hydrology stage."""
from __future__ import annotations

from core.channels import FeatureChannel, TextureChannel

# 1.0 where a pixel is inside the post-shrinking lake footprint, else 0.0.
LAKE_MASK = TextureChannel("lake_mask", n_channels=1)

# ── Features ─────────────────────────────────────────────────────────────────
# The rich ``List[River]`` objects consumed by resources and cost.
RIVER_OBJECTS = FeatureChannel("river_objects")
# Flattened ``(positions, flows, in_lake)`` polyline tuples consumed by realmgen
# (and used for rendering/serialization).
RIVERS = FeatureChannel("rivers")
# The ``StreamNetwork``: the rivers' accumulation tree cut at the (much lower)
# stream threshold — town siting and the region tier's tributaries.
STREAMS = FeatureChannel("streams")
