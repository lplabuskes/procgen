"""Regional streams: the region's own drainage network below the world's."""
import numpy as np
import pytest

from region.corridor import raster_frame
from region.drainage import accumulate, priority_flood
from region.rivers import build_region_rivers
from region.streams import RegionStreamParams, _river_nodes, build_region_streams
from region.upsample import RegionUpsampleParams, build_region_fields

REGION_ID = 40   # coast, a lake and a peak (see conftest)
# The synthetic region's raster nodes are ~1,000 km² each; thresholds of a few
# nodes give it a network to test.  Its smallest world stream is ~129,000 km², so ~2,000 km².
PARAMS = RegionStreamParams(min_catchment_pct=1.55)


@pytest.fixture(scope="module")
def fields(synthetic_globe):
    return build_region_fields(synthetic_globe, REGION_ID, RegionUpsampleParams(n_cols=6, tex_size=16))


@pytest.fixture(scope="module")
def region_rivers(fields, synthetic_globe):
    return build_region_rivers(fields, synthetic_globe)


@pytest.fixture(scope="module")
def streams(fields, synthetic_globe, region_rivers):
    return build_region_streams(fields, synthetic_globe, region_rivers, PARAMS)


def test_flood_drains_every_node_to_an_outlet():
    """A slope with a pit: the pit is filled over, and every node's parent chain
    reaches the outlet column, upstream before downstream in the flood order."""
    n = 6
    elevation = np.tile(np.arange(n, dtype=np.float64), (n, 1))   # rises with the column
    elevation[3, 3] = -1.0                                         # a pit
    outlet = np.zeros((n, n), dtype=bool)
    outlet[:, 0] = True
    parent, order, _ = priority_flood(elevation.ravel(), outlet.ravel(), n)
    assert len(order) == n * n
    rank = np.empty(n * n, dtype=np.int64)
    rank[order] = np.arange(n * n)
    for node in range(n * n):
        steps = 0
        while parent[node] >= 0:
            assert rank[parent[node]] < rank[node]
            node = int(parent[node])
            steps += 1
            assert steps <= n * n
        assert outlet.ravel()[node]
    acc = accumulate(parent, order, np.where(outlet.ravel(), 0.0, 1.0))
    assert acc[outlet.ravel()].sum() == pytest.approx(n * (n - 1))


def test_the_region_has_streams(streams):
    assert len(streams) > 5
    for positions, flows, in_water in streams:
        assert len(positions) == len(flows) + 1 == len(in_water) + 1
        assert not in_water.any()


def test_flow_grows_downstream_and_stays_below_the_worlds_network(streams, synthetic_globe):
    world = synthetic_globe.get_feature("streams")
    cap = float(world.flow[world.channel].min())
    for _, flows, _ in streams:
        assert np.all(np.diff(flows) >= 0.0)
        assert flows.min() > 0.0 and flows.max() <= cap + 1e-3


def test_streams_never_run_along_a_river(streams, fields, region_rivers):
    """A stream drains *into* the region's rivers: only its last node may be on one."""
    lo, step, n = raster_frame(fields, PARAMS.raster_texels, PARAMS.margin)
    river = _river_nodes(fields, region_rivers, lo, step, n)
    assert river.any()
    for positions, _, _ in streams:
        s, t, _ = fields.geometry.ecef_to_local(np.asarray(positions, dtype=np.float64))
        col = np.rint((s - lo) / step).astype(np.int64)
        row = np.rint((t - lo) / step).astype(np.int64)
        assert not river[row[:-1] * n + col[:-1]].any()


def test_a_higher_threshold_gives_fewer_streams(fields, synthetic_globe, region_rivers, streams):
    coarser = build_region_streams(
        fields, synthetic_globe, region_rivers, RegionStreamParams(min_catchment_pct=3.1))
    assert len(coarser) < len(streams)


def test_streams_are_deterministic(fields, synthetic_globe, region_rivers, streams):
    again = build_region_streams(fields, synthetic_globe, region_rivers, PARAMS)
    assert len(again) == len(streams)
    for (a, _, _), (b, _, _) in zip(streams, again):
        np.testing.assert_array_equal(a, b)
