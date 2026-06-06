"""Channels produced by the resources stage."""
from __future__ import annotations

import numpy as np

from core.channels import FeatureChannel, MetaChannel

FERTILITY_MEAN = MetaChannel("fertility_mean", dtype=np.float64)

# ── Features ─────────────────────────────────────────────────────────────────
# ``(positions (M,3), kinds (M,), abundances (M,))`` deposit tuples consumed by
# realmgen.
MINERALS = FeatureChannel("minerals")
RARE_BIO = FeatureChannel("rare_bio")
