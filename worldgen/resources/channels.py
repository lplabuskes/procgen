"""Channels produced by the resources stage."""
from __future__ import annotations

import numpy as np

from core.channels import FeatureChannel, MetaChannel, TextureChannel
from core.codec import TextureCodec

FERTILITY_MEAN = MetaChannel("fertility_mean", dtype=np.float64)

# ── Per-face textures (0–1, saved to 0.001) ────────────────────────────────────
FERTILITY = TextureChannel("fertility", n_channels=1, codec=TextureCodec(step=1e-3))
FOREST_DENSITY = TextureChannel("forest_density", n_channels=1, codec=TextureCodec(step=1e-3))

# ── Features ─────────────────────────────────────────────────────────────────
# ``(positions (M,3), kinds (M,), abundances (M,))`` deposit tuples consumed by
# realmgen.
MINERALS = FeatureChannel("minerals")
RARE_BIO = FeatureChannel("rare_bio")
# ``(positions (K,3), deposit (K,), share (K,))``: the workings each ``minerals``
# deposit is split into inside its face (``share`` sums to 1 per deposit).
MINERAL_SITES = FeatureChannel("mineral_sites")
