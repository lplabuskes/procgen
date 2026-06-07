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
# Stage 5 — Polities
POLITY_ID = MetaChannel("polity_id", dtype=np.int32, default=-1)  # polity index or -1
BORDER_TENSION = MetaChannel("border_tension", dtype=np.float64)  # contested-border heat
