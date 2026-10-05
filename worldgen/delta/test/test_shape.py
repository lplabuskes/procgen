"""Delta shape: wave power against river power, and the footprints it gives on a
synthetic coast (see test_deposit for the globe)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from worldgen.delta.builder import DeltaParams
from worldgen.delta.deposit import build_deltas
from worldgen.delta.shape import ARCUATE, BIRDFOOT, CUSPATE, WaveField, make_shape, wave_dominance
from worldgen.delta.test.test_deposit import EAST, UP, _river, _textures

# A big river, so the footprint spans enough texels to have a shape.
FLOW = 200_000.0
FORCED = {
    BIRDFOOT: dict(birdfoot_below=1.1, cuspate_above=1.2),
    ARCUATE: dict(birdfoot_below=-0.2, cuspate_above=1.2),
    CUSPATE: dict(birdfoot_below=-0.2, cuspate_above=-0.1),
}


def test_dominance_falls_with_flow_and_rises_with_waves():
    p = DeltaParams()
    assert wave_dominance(1.0, 10_000.0, p) < wave_dominance(1.0, 1_000.0, p)
    assert wave_dominance(2.0, 5_000.0, p) > wave_dominance(0.5, 5_000.0, p)
    assert 0.0 <= wave_dominance(0.0, 5_000.0, p) <= wave_dominance(1e6, 5_000.0, p) <= 1.0


def test_shape_weights_follow_the_thresholds():
    p = DeltaParams()
    head, axis = EAST, -UP
    rng = np.random.default_rng(0)
    assert make_shape(0.0, head, axis, 5000.0, 100.0, p, rng).name == BIRDFOOT
    assert make_shape(0.5, head, axis, 5000.0, 100.0, p, rng).name == ARCUATE
    assert make_shape(1.0, head, axis, 5000.0, 100.0, p, rng).name == CUSPATE
    bird = make_shape(0.0, head, axis, 5000.0, 100.0, p, rng)
    assert len(bird.finger_dirs) >= 2
    # Fingers lie in the tangent plane at the head, seaward.
    assert np.allclose(bird.finger_dirs @ head, 0.0, atol=1e-9)
    assert np.all(bird.finger_dirs @ axis > 0.0)


@pytest.fixture(scope="module")
def waves(mesh):
    """Sea south of the equator; ``winds`` blowing north (onshore) or south (offshore)."""
    c = np.asarray(mesh.face_centroids, dtype=np.float64)
    sea = c @ UP < 0.0
    north = UP[None, :] - c * (c @ UP)[:, None]
    north /= np.maximum(np.linalg.norm(north, axis=1, keepdims=True), 1e-12)
    spacing = float(np.sqrt(np.mean(mesh.face_areas)))
    return {
        "onshore": WaveField(c, sea, 10.0 * north[:, None, :], spacing),
        "offshore": WaveField(c, sea, -10.0 * north[:, None, :], spacing),
        "calm": WaveField(c, sea, None, spacing),
        "enclosed": WaveField(c, sea & (c @ UP > -0.1), 10.0 * north[:, None, :], spacing),
    }


def test_onshore_wind_over_open_sea_makes_waves(waves):
    p = DeltaParams()
    power = {k: w.power(EAST, -UP, p) for k, w in waves.items()}
    assert power["onshore"] > power["offshore"] > 0.0      # offshore: swell only
    assert np.isclose(power["offshore"], power["calm"])
    # A narrow sea (short fetch) is calmer than the open ocean.
    assert power["enclosed"] < 0.5 * power["onshore"]


def _footprint_of(mesh, index, shape):
    elev, ocean, lake, filled = _textures(mesh)
    ocean0 = ocean.copy()
    river = _river(index, ocean, FLOW)
    params = dataclasses.replace(DeltaParams(), **FORCED[shape])
    result = build_deltas(mesh, [river], elev, ocean, lake, filled, None, params, seed=1)
    fan = result.fans[0]
    plain = result.delta[..., 0].reshape(-1) > 0
    built = plain & (ocean0.reshape(-1) > 0)
    ids = np.flatnonzero(built & np.isin(np.arange(plain.size), index.inside_ids))
    rel = index.positions[ids] - np.asarray(fan.head, dtype=np.float64)[None, :]
    forward = rel @ -UP * 6371.0
    lateral = np.abs(rel @ np.cross(EAST, UP)) * 6371.0
    return fan, forward, lateral


@pytest.fixture(scope="module")
def shapes(mesh, index):
    return {s: _footprint_of(mesh, index, s) for s in FORCED}


def test_each_forced_shape_is_reported(shapes):
    for name, (fan, _, _) in shapes.items():
        assert fan.shape == name
    assert len(shapes[BIRDFOOT][0].fingers) >= 2
    assert len(shapes[ARCUATE][0].fingers) == 0 and len(shapes[CUSPATE][0].fingers) == 0


def test_birdfoot_reaches_furthest_and_cuspate_spreads_along_shore(shapes):
    reach = {s: float(f.max()) for s, (_, f, _) in shapes.items()}
    spread = {s: float(np.percentile(lat, 90) / max(np.percentile(f, 90), 1e-9))
              for s, (_, f, lat) in shapes.items()}
    assert reach[BIRDFOOT] > reach[ARCUATE] > reach[CUSPATE]
    assert spread[CUSPATE] > spread[ARCUATE]
