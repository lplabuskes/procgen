"""Ocean currents — placeholder for the future stage.

Reserved seam: when this lands, `compute_currents` will return per-face
temperature and precipitation deltas that get fed back into
`temperature.surface_temperatures` and `precipitation.surface_precipitation`
via their `coastal_current_temp_delta` and `current_wet_delta` arguments.
"""
from __future__ import annotations


def compute_currents(*args, **kwargs):
    raise NotImplementedError("Ocean currents stage is not implemented yet.")
