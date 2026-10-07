"""Channels produced by the climate stage."""
from __future__ import annotations

import numpy as np

from core.channels import MetaChannel, TextureChannel
from core.codec import TextureCodec

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
# consumed by hydrology and resources.  Saved to 0.05 °C (≈ 8 m of lapse rate) and
# 0.1% of precipitation (log-coded).
TEMPERATURE = TextureChannel("temperature", n_channels=4, codec=TextureCodec(step=0.05))
PRECIPITATION = TextureChannel("precipitation", n_channels=4, codec=TextureCodec(step=0.002, log=True, log_offset=1.0))
# Per-season moisture (N, H, W, K=4); display only after the build, saved to 0.25% (log-coded).
HUMIDITY = TextureChannel("humidity", n_channels=4, codec=TextureCodec(step=0.005, log=True, log_offset=1e-3))
