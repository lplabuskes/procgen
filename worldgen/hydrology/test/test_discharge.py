"""River flow → discharge → channel width."""
import numpy as np
import pytest

from worldgen.hydrology.discharge import channel_width_km


def test_width_follows_hydraulic_geometry():
    # An Amazon (~200k m³/s) is a few km wide, a Rhine (~2k m³/s) a few hundred m.
    assert 2.0 < float(channel_width_km(200_000.0)) < 5.0
    assert 0.2 < float(channel_width_km(2_000.0)) < 0.5


def test_width_is_monotonic_and_zero_at_no_flow():
    w = channel_width_km(np.array([0.0, 10.0, 1e3, 1e5]))
    assert w[0] == pytest.approx(0.0)
    assert np.all(np.diff(w) > 0)
