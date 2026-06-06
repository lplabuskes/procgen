"""Channels produced by the climate stage."""
from __future__ import annotations

import numpy as np

from core.channels import MetaChannel, TextureChannel

# ── Per-face metadata ────────────────────────────────────────────────────────
TEMPERATURE_ANNUAL = MetaChannel("temperature_annual", dtype=np.float64)
TEMPERATURE_WARMEST = MetaChannel("temperature_warmest", dtype=np.float64)
TEMPERATURE_COLDEST = MetaChannel("temperature_coldest", dtype=np.float64)
PRECIPITATION_ANNUAL = MetaChannel("precipitation_annual", dtype=np.float64)
DISTANCE_TO_COAST = MetaChannel("distance_to_coast", dtype=np.float64)
# Dominant biome index per face (climate.classification ids) — consumed by
# resources and cost.
BIOME = MetaChannel("biome", dtype=np.int32, default=0)

# ── Per-face textures ────────────────────────────────────────────────────────
# Per-season fields, (N, H, W, K=4): temperature °C and precipitation mm/season —
# consumed by hydrology and resources.
TEMPERATURE = TextureChannel("temperature", n_channels=4)
PRECIPITATION = TextureChannel("precipitation", n_channels=4)
