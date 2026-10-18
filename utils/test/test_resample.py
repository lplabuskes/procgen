import numpy as np
import pytest

from utils.resample import anchor_means, settle_owners


def test_anchor_means_ignores_zero_weight_samples():
    values = np.array([[1.0], [3.0], [100.0]])
    anchors = np.array([0, 0, 0])
    weights = np.array([1.0, 1.0, 0.0])
    means, counts = anchor_means(values, anchors, weights, 2)
    assert means[0, 0] == pytest.approx(2.0)
    assert counts[0] == 2.0
    assert counts[1] == 0.0


def test_settle_owners_resolves_chains_and_cycles():
    # 0 self; 1→0 chain; 2→3→4→2 cycle with 5→2 feeding it; 6↔7 two-cycle.
    owner = np.array([0, 0, 3, 4, 2, 2, 7, 6])
    settled = settle_owners(owner)
    assert np.array_equal(settled[settled], settled)
    assert np.array_equal(settled, [0, 0, 2, 2, 2, 2, 6, 6])
