"""Channels produced by the realm (human-geography) stage."""
from __future__ import annotations

import numpy as np

from core.channels import MetaChannel

SUITABILITY = MetaChannel("suitability", dtype=np.float64)
SITUATIONAL_VALUE = MetaChannel("situational_value", dtype=np.float64)
TRADE_FLOW = MetaChannel("trade_flow", dtype=np.float64)
# Stage 4 — Population & Calibration
POPULATION = MetaChannel("population", dtype=np.float64)
HINTERLAND_ID = MetaChannel("hinterland_id", dtype=np.int32)  # settlement index or -1
