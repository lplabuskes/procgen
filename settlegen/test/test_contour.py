"""Marching squares and the frame-closed regions below a level."""
import math

import numpy as np
from shapely.geometry import box

from settlegen.contour import contour_segments, regions_below

T, SIDE = 64, 1000.0


def _grid() -> np.ndarray:
    c = ((np.arange(T) + 0.5) / T - 0.5) * SIDE
    xx, yy = np.meshgrid(c, c[::-1])
    return np.stack([xx, yy], axis=-1)


def test_a_disc_contours_to_a_circle():
    xy = _grid()
    r = np.hypot(xy[..., 0], xy[..., 1])
    segs = contour_segments(r, xy, level=300.0)
    ends = segs.reshape(-1, 2)
    assert np.allclose(np.hypot(ends[:, 0], ends[:, 1]), 300.0, atol=2.0)
    regions = regions_below(r, xy, box(-SIDE / 2, -SIDE / 2, SIDE / 2, SIDE / 2), level=300.0)
    assert len(regions.geoms) == 1
    assert regions.area == np.clip(regions.area, 0.98 * math.pi * 300 ** 2, 1.01 * math.pi * 300 ** 2)


def test_regions_cut_by_the_frame_close_along_its_edge():
    xy = _grid()
    frame = box(-SIDE / 2, -SIDE / 2, SIDE / 2, SIDE / 2)
    # Below the level west of x = 100: a band closed by the frame's edge.
    west = regions_below(xy[..., 0] - 100.0, xy, frame)
    assert abs(west.area - 600.0 * SIDE) < 0.01 * 600.0 * SIDE
    assert west.is_valid and west.within(frame.buffer(1e-6))
    # Wholly above / below.
    assert regions_below(np.ones((T, T)), xy, frame).is_empty
    assert abs(regions_below(-np.ones((T, T)), xy, frame).area - frame.area) < 1e-6


def test_saddles_and_holes():
    xy = _grid()
    r = np.hypot(xy[..., 0], xy[..., 1])
    ring = np.abs(r - 250.0) - 60.0                        # below in an annulus
    regions = regions_below(ring, xy, box(-SIDE / 2, -SIDE / 2, SIDE / 2, SIDE / 2))
    assert len(regions.geoms) == 1 and len(regions.geoms[0].interiors) == 1
    expected = math.pi * (310.0 ** 2 - 190.0 ** 2)
    assert abs(regions.area - expected) < 0.02 * expected
