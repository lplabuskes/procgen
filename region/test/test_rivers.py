"""Regional rivers: the world's stream network re-routed over the refined relief."""
import numpy as np
import pytest

from region.rivers import RegionRiverParams, build_region_rivers
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
