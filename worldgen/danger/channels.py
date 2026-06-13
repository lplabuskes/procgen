"""Channels produced by the danger stage."""
from __future__ import annotations

import numpy as np

from core.channels import FeatureChannel, MetaChannel

# Per-face danger scalar in [0, 1], stored in face metadata for hover/display.
DANGER = MetaChannel("danger", dtype=np.float32, default=0.0)

# The same danger array stored as a world feature for efficient numpy access by realmgen.
DANGER_ARRAY = FeatureChannel("danger")
