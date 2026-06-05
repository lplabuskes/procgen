"""Channels produced by the resources stage."""
from __future__ import annotations

import numpy as np

from core.channels import MetaChannel

FERTILITY_MEAN = MetaChannel("fertility_mean", dtype=np.float64)
