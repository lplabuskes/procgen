"""Unit tests for realmgen.refine.barriers — packed river segments and the flow ramp."""
from __future__ import annotations

import numpy as np
import pytest

from realmgen.refine.barriers import flow_ramp, pack_river_segments


def _equator(lon0=-0.2, lon1=0.2, n=21):
    lon = np.linspace(lon0, lon1, n)
    return np.column_stack([np.cos(lon), np.sin(lon), np.zeros(n)])


def test_pack_drops_lake_and_low_flow_segments():
    pos = _equator(n=4)
    river = (pos, np.array([500.0, 2000.0, 3000.0]), np.array([False, True, False]))
    seg = pack_river_segments([river], min_flow=1000.0)
    assert seg.shape == (1, 7)
    assert seg[0, 6] == 3000.0
    assert np.allclose(seg[0, :3], pos[2]) and np.allclose(seg[0, 3:6], pos[3])


def test_flow_ramp_is_log_between_min_and_full():
    r = flow_ramp(np.array([500.0, 1000.0, np.sqrt(1000.0 * 8000.0), 8000.0, 50000.0]),
                  min_flow=1000.0, full_flow=8000.0)
    assert r.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])


def test_pack_carries_world_river_ids():
    pos = _equator(n=4)
    river = (pos, np.array([2000.0, 2000.0, 2000.0]), np.array([False, True, False]))
    seg = pack_river_segments([river], river_ids=[np.array([-1, 5, 5, 5])])
    assert seg.shape == (2, 8)
    assert seg[:, 7].tolist() == [5.0, 5.0]       # a headwater segment joining river 5 takes its id
