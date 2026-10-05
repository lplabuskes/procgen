"""Regional rivers: the world's stream network re-routed over the refined relief."""
import numpy as np
import pytest

from region.corridor import Raster
from region.rivers import (
    RegionRiverParams, _hide_reentries, _route_run, _start_on_drawn, _water_factor, build_region_rivers,
)
from region.upsample import RegionUpsampleParams, build_region_fields

REGION_ID = 40   # coast, a lake and a peak (see conftest)


@pytest.fixture(scope="module")
def fields(synthetic_globe):
    return build_region_fields(synthetic_globe, REGION_ID, RegionUpsampleParams(n_cols=6, tex_size=16))


@pytest.fixture(scope="module")
def region_rivers(fields, synthetic_globe):
    return build_region_rivers(fields, synthetic_globe)


def _global_texel_chord(globe) -> float:
    gw = globe.mesh.face_textures["elevation"].shape[2]
    return float(np.sqrt(np.mean(globe.mesh.face_areas))) / (gw - 1)


def _chord_to(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Chord distance from each point to the nearest of ``targets``."""
    return np.linalg.norm(points[:, None, :] - targets[None, :, :], axis=-1).min(axis=1)


def _stream_texels(globe) -> np.ndarray:
    streams = globe.get_feature("streams")
    return np.asarray(streams.positions[streams.channel | streams.stop], dtype=np.float64)


def test_the_stream_network_is_drawn(region_rivers):
    assert len(region_rivers.paths) > 5
    for positions, flows, in_water in region_rivers.paths:
        assert len(positions) == len(flows) + 1 == len(in_water) + 1


def test_paths_stay_in_their_corridor(region_rivers, synthetic_globe):
    """A path follows the refined valleys but never leaves the band the global
    grid leaves open around the world's stream texels."""
    corridor = RegionRiverParams().corridor_global_texels * _global_texel_chord(synthetic_globe)
    texels = _stream_texels(synthetic_globe)
    for positions, _, _ in region_rivers.paths:
        d = _chord_to(np.asarray(positions, dtype=np.float64), texels)
        # Between stream texels the centreline runs up to half a texel from
        # either; the mouth may reach a stem up to two corridors away.
        assert np.median(d) < 1.5 * corridor
        assert d.max() < 3.0 * corridor


def test_paths_are_refined_not_copied(region_rivers, synthetic_globe):
    texels = _stream_texels(synthetic_globe)
    pts = np.concatenate([np.asarray(p[0], dtype=np.float64) for p in region_rivers.paths])
    assert np.median(_chord_to(pts, texels)) > 1e-4


def test_flow_is_the_networks_and_grows_downstream(region_rivers, synthetic_globe):
    streams = synthetic_globe.get_feature("streams")
    lo, hi = float(streams.flow[streams.channel].min()), float(streams.flow.max())
    for _, flows, _ in region_rivers.paths:
        assert np.all(np.diff(flows) >= 0.0)
        assert flows.min() >= lo - 1e-3 and flows.max() <= hi + 1e-3


def test_confluences_join_two_paths(region_rivers, fields):
    assert len(region_rivers.confluences) > 0
    spacing = fields.grid.texel_spacing_rad(fields.tex_size)
    for c in region_rivers.confluences:
        touching = [
            p for p in region_rivers.paths
            if _chord_to(np.asarray(p[0], dtype=np.float64), c[None, :].astype(np.float64)).min() < 2.0 * spacing
        ]
        assert len(touching) >= 2


def test_streams_reach_the_refined_sea(region_rivers, fields):
    ocean = fields.positions[fields.textures["ocean_mask"][..., 0] > 0.5]
    mouths = np.stack([np.asarray(p[0][-1], dtype=np.float64) for p in region_rivers.paths])
    spacing = fields.grid.texel_spacing_rad(fields.tex_size)
    assert (_chord_to(mouths, ocean) < 2.0 * spacing).any()


def test_generation_is_deterministic(fields, synthetic_globe, region_rivers):
    again = build_region_rivers(fields, synthetic_globe)
    assert len(again.paths) == len(region_rivers.paths)
    for a, b in zip(again.paths, region_rivers.paths):
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


def test_no_stream_network_no_rivers(fields, synthetic_globe):
    streams = synthetic_globe.get_feature("streams")
    synthetic_globe.remove_feature("streams")
    try:
        result = build_region_rivers(fields, synthetic_globe)
    finally:
        synthetic_globe.add_feature("streams", streams)
    assert result.paths == [] and len(result.confluences) == 0


def test_mouths_are_where_paths_reach_the_refined_sea(region_rivers, fields):
    assert len(region_rivers.mouths) > 0
    assert len(region_rivers.mouths) == len(region_rivers.mouth_flows)
    ends = np.stack([np.asarray(p[0][-1], dtype=np.float64) for p in region_rivers.paths])
    assert _chord_to(region_rivers.mouths, ends).max() < 1e-9
    ocean = fields.positions[fields.textures["ocean_mask"][..., 0] > 0.5]
    spacing = fields.grid.texel_spacing_rad(fields.tex_size)
    assert _chord_to(region_rivers.mouths, ocean).max() < 2.0 * spacing


def test_deltas_snap_to_the_biggest_mouth_in_reach(region_rivers):
    from region.rivers import snap_deltas
    mouths, flows = region_rivers.mouths, region_rivers.mouth_flows
    big = int(np.argmax(flows))
    near = mouths[big] + np.array([1e-4, 0.0, 0.0])
    far = -mouths[big]
    moved = snap_deltas(np.stack([near, far]), region_rivers, reach=1e-3)
    assert np.allclose(moved[0], mouths[big])
    assert np.allclose(moved[1], far)


def test_leaving_and_reentering_one_lake_stays_hidden():
    # Lake 1, a shore crossing, lake 1 again, land, lake 2, land: only the
    # crossing back into the same lake is hidden; the run to lake 2 and the
    # outflow are real.
    bodies = np.array([0, 1, 1, 0, 0, 1, 0, 0, 2, 0])
    hidden = _hide_reentries(bodies > 0, bodies)
    assert hidden.tolist() == [False, True, True, True, True, True, False, False, True, False]


def _bay_raster(n: int = 21) -> Raster:
    """Unit-step raster, ground falling eastward into a bay (cols ≥ 12, rows 5–15)."""
    rows, cols = np.divmod(np.arange(n * n), n)
    ocean = (cols >= 12) & (rows >= 5) & (rows <= 15)
    return Raster(lo=0.0, step=1.0, n=n, sampled=np.ones(n * n, dtype=bool), positions=np.zeros((n * n, 3)),
                  values={}, anomaly=-0.1 * cols, ocean=ocean, lake=np.zeros(n * n, dtype=bool))


def test_a_reroute_keeps_to_land_where_its_parent_does():
    raster = _bay_raster()
    s, t = np.full(2, 10.0), np.array([0.0, 20.0])
    no_sinks = np.zeros(raster.n ** 2, dtype=bool)
    params = RegionRiverParams()
    free = _route_run(raster, s, t, 3.0, no_sinks, False, params)
    assert free is not None and raster.water[free].any()      # valley-seeking hugs the bay
    dry = _route_run(raster, s, t, 3.0, no_sinks, False, params, _water_factor(raster, s, t, np.zeros(1, bool), 3.0))
    assert dry is not None and not raster.water[dry].any()
    # Within the corridor of a stretch where the parent itself is under water, water is free.
    wet = _water_factor(raster, s, t, np.ones(1, dtype=bool), 3.0)
    cols = np.arange(raster.n ** 2) % raster.n
    assert np.all(wet[np.abs(cols - 10) <= 3] == 1.0) and np.all(wet[raster.water & (cols > 13)] > 1.0)


def test_a_branch_starts_on_the_nearest_drawn_node():
    raster = _bay_raster()
    on_path = np.zeros(raster.n ** 2, dtype=np.int64)
    on_path[7 * raster.n + 4] = 1                              # row 7, col 4
    s, t = _start_on_drawn(raster, on_path, np.array([5.0, 9.0]), np.array([8.0, 2.0]), 3.0)
    assert (s[0], t[0]) == (4.0, 7.0) and (s[1], t[1]) == (9.0, 2.0)
    far, _ = _start_on_drawn(raster, on_path, np.array([15.0, 9.0]), np.array([8.0, 2.0]), 3.0)
    assert far[0] == 15.0
