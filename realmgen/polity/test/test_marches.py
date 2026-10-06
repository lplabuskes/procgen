"""Unit tests for realmgen.polity.marches — marcher autonomy."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from realmgen.polity.marches import frontier_exposure, march_autonomy
from realmgen.polity.test.conftest import _line_graph


class TestFrontierExposure:
    def test_no_border_is_zero(self):
        out = frontier_exposure(_line_graph(5), np.zeros(5), reach=2.0)
        assert (out == 0.0).all()

    def test_decays_from_the_border(self):
        t = np.array([0.0, 0.0, 0.0, 0.0, 0.8])
        out = frontier_exposure(_line_graph(5, cost=1.0), t, reach=2.0)
        np.testing.assert_allclose(out, np.exp(-np.array([4.0, 3.0, 2.0, 1.0, 0.0]) / 2.0))

    def test_takes_the_strongest_border(self):
        # A weak border next door loses to a hot one two faces away.
        t = np.array([1.0, 0.0, 0.0, 0.1])
        out = frontier_exposure(_line_graph(4, cost=1.0), t, reach=4.0)
        assert out[2] == pytest.approx(np.exp(-2.0 / 4.0))

    def test_full_tension_border_is_one(self):
        out = frontier_exposure(_line_graph(3), np.array([1.0, 0.0, 0.0]), reach=1.0)
        assert out[0] == pytest.approx(1.0)

    def test_tension_normalised_by_quantile(self):
        # Border tensions 0.1 and 0.2: the q=1 reference is 0.2, so the hot face hits 1.
        t = np.array([0.1, 0.0, 0.0, 0.2])
        out = frontier_exposure(_line_graph(4), t, reach=1.0, tension_quantile=1.0)
        assert out[3] == pytest.approx(1.0)
        assert out[0] == pytest.approx(0.5)

    def test_unreachable_faces_are_zero(self):
        g = csr_matrix(([1.0, 1.0], ([0, 1], [1, 0])), shape=(4, 4))  # faces 2, 3 cut off
        out = frontier_exposure(g, np.array([0.5, 0.0, 0.0, 0.0]), reach=1.0)
        assert out[2] == 0.0 and out[3] == 0.0


class TestMarchAutonomy:
    def test_off_or_interior_unchanged(self):
        assert march_autonomy(0.3, 0.9, 0.0) == pytest.approx(0.3)
        assert march_autonomy(0.3, 0.0, 0.8) == pytest.approx(0.3)

    def test_raises_toward_one(self):
        assert march_autonomy(0.2, 1.0, 0.5) == pytest.approx(0.6)
        assert march_autonomy(0.2, 0.5, 0.5) < march_autonomy(0.2, 1.0, 0.5)

    def test_stays_in_unit_interval(self):
        assert march_autonomy(0.9, 1.0, 1.0) == pytest.approx(1.0)
        assert 0.0 <= march_autonomy(0.0, 2.0, 2.0) <= 1.0
