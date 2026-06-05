"""Channels produced by the realm (human-geography) stage."""
from __future__ import annotations

import numpy as np

from core.channels import MetaChannel

SUITABILITY = MetaChannel("suitability", dtype=np.float64)
SITUATIONAL_VALUE = MetaChannel("situational_value", dtype=np.float64)
