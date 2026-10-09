"""Channels produced by the hydrology stage."""
from __future__ import annotations

from core.channels import FeatureChannel, TextureChannel
from core.codec import TextureCodec

# 1.0 where a pixel is inside the post-shrinking lake footprint, else 0.0.
LAKE_MASK = TextureChannel("lake_mask", n_channels=1)
# Depression-filled surface (km): the lake level over lakes, elevation elsewhere.
# Saved to 0.1 m as a difference from elevation (zero off the lakes).
FILLED_HEIGHT = TextureChannel("filled_height", n_channels=1,
                               codec=TextureCodec(step=1e-4, relative_to="elevation"))

# Per texel, the flat texel id (face·H·W + row·W + col) of the first stream-network
# channel texel its downhill path reaches (itself on a channel), -1 where it reaches
# the sea or an endorheic sink first: the drainage basin it belongs to. Exact integers.
DRAINAGE_LANDING = TextureChannel("drainage_landing", n_channels=1, codec=TextureCodec(step=1.0))

# ── Features ─────────────────────────────────────────────────────────────────
# The rich ``List[River]`` objects consumed by resources and cost.
RIVER_OBJECTS = FeatureChannel("river_objects")
# Flattened ``(positions, flows, in_lake)`` polyline tuples (realmgen, rendering).
RIVERS = FeatureChannel("rivers")
# Per ``"rivers"`` entry, the flat texel ids of the tree nodes it runs through
# (``River.texels``): the link from ``StreamNetwork`` nodes to world rivers.
RIVER_TEXELS = FeatureChannel("river_texels")
# The ``StreamNetwork``: the rivers' accumulation tree cut at the (much lower)
# stream threshold — town siting and the region tier's tributaries.
STREAMS = FeatureChannel("streams")
