"""Channels produced by the realm (human-geography) stage."""
from __future__ import annotations

import numpy as np

from core.channels import MetaChannel

SUITABILITY = MetaChannel("suitability", dtype=np.float64)
SITUATIONAL_VALUE = MetaChannel("situational_value", dtype=np.float64)
TRADE_FLOW = MetaChannel("trade_flow", dtype=np.float64)
# Stage 4 — Population & Calibration
POPULATION = MetaChannel("population", dtype=np.float64)  # urban headcount (people) at settlement faces
RURAL_POPULATION = MetaChannel("rural_population", dtype=np.float64)  # per-face rural people (land only)
HINTERLAND_ID = MetaChannel("hinterland_id", dtype=np.int32)  # settlement index or -1
# Towns — carved out of rural_population, which the town stage reduces to match
TOWN_POPULATION = MetaChannel("town_population", dtype=np.float64)  # people in towns per face
# The part of a land face's rural population (before towns) fed by its own shallows
FISHING_POPULATION = MetaChannel("fishing_population", dtype=np.float64)
# Miners per face (people at its mineral deposits), counted in rural_population
# until the town stage gathers them into mining towns.
MINING_POPULATION = MetaChannel("mining_population", dtype=np.float64)
# Stage 5 — Polities
POLITY_ID = MetaChannel("polity_id", dtype=np.int32, default=-1)  # directly-governing polity index or -1
SOVEREIGN_ID = MetaChannel("sovereign_id", dtype=np.int32, default=-1)  # root polity of the face's tree or -1
CONTROL_CLASS = MetaChannel("control_class", dtype=np.int32, default=-1)  # ControlClass at settlement faces, else -1
BORDER_TENSION = MetaChannel("border_tension", dtype=np.float64)  # contested-border heat
AUTHORITY = MetaChannel("authority", dtype=np.float64)  # winning authority value per face (control strength)
