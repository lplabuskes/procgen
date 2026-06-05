"""Channels produced by the climate stage."""
from __future__ import annotations

import numpy as np

from core.channels import MetaChannel

TEMPERATURE_WARMEST = MetaChannel("temperature_warmest", dtype=np.float64)
TEMPERATURE_COLDEST = MetaChannel("temperature_coldest", dtype=np.float64)
PRECIPITATION_ANNUAL = MetaChannel("precipitation_annual", dtype=np.float64)
DISTANCE_TO_COAST = MetaChannel("distance_to_coast", dtype=np.float64)
