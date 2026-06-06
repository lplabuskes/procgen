"""Channels produced by the cost stage."""
from __future__ import annotations

import numpy as np

from core.channels import MetaChannel

IS_WATER = MetaChannel("is_water", dtype=np.bool_, default=False)

# Water face pinched between opposite land neighbours.
IS_STRAIT = MetaChannel("is_strait", dtype=np.bool_, default=False)
