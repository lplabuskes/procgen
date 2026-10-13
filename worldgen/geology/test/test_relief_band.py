"""The relief band: the region's detail averaged over each global texel, and the preview that sizes it."""
import dataclasses

import numpy as np
import pytest

from core.mesh import Mesh
from core.pipeline import BuildContext
from core.world import World
from worldgen.geology.channels import ELEVATION, RELIEF_BAND
from worldgen.geology.elevation_builder import ElevationBuilder, ElevationParams
from worldgen.geology.preview import preview_ruggedness, preview_temperature
from worldgen.geology.relief_band import (
    ReliefBandParams,
    _footprint_mean,
    _upsample_axis,
    compute_relief_band,
    trend_elevation,
)

TEX = 8


@pytest.fixture(scope="module")
def mesh():
    return Mesh.icosphere(3).mesh_dual()


def _const(mesh, value: float) -> np.ndarray:
    return np.full((len(mesh.faces), TEX, TEX, 1), value, dtype=np.float32)


def test_upsample_axis_keeps_texels_and_interpolates_between():
    tex = np.array([[0.0, 3.0, 6.0]])
    fine = _upsample_axis(tex, 3, axis=1)
    np.testing.assert_allclose(fine, [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])


def test_footprint_mean_averages_each_texels_sub_points():
    fine = np.arange(7 * 7, dtype=np.float64).reshape(1, 7, 7)
    mean = _footprint_mean(fine, 3, 3)
    assert mean[0, 1, 1] == pytest.approx(fine[0, 2:5, 2:5].mean())   # centre: the full 3 × 3
    assert mean[0, 0, 0] == pytest.approx(fine[0, 0:2, 0:2].mean())   # corner: the sub-points that exist
    np.testing.assert_allclose(_footprint_mean(np.ones((2, 7, 7)), 3, 3), 1.0)


def test_band_is_off_when_disabled_and_grows_with_ruggedness(mesh):
    jag = _const(mesh, 0.5)
    off = compute_relief_band(mesh, _const(mesh, 0.8), jag, 7, ReliefBandParams(enabled=False))
    assert off.shape == (len(mesh.faces), TEX, TEX, 1) and not off.any()
    smooth = compute_relief_band(mesh, _const(mesh, 0.1), jag, 7, ReliefBandParams())
    rough = compute_relief_band(mesh, _const(mesh, 0.9), jag, 7, ReliefBandParams())
    assert rough.std() > 3.0 * smooth.std() > 0.0                  # (the coarse mesh hits the octave cap)
    assert abs(float(rough.mean())) < 0.2 * float(rough.std())     # zero-mean noise, averaged
    again = compute_relief_band(mesh, _const(mesh, 0.9), jag, 7, ReliefBandParams())
    np.testing.assert_array_equal(rough, again)                     # deterministic per world seed


def test_footprint_mean_is_smoother_than_point_samples(mesh):
    rug, jag = _const(mesh, 0.7), _const(mesh, 0.5)
    point = compute_relief_band(mesh, rug, jag, 7, ReliefBandParams(subsamples=1))
    box = compute_relief_band(mesh, rug, jag, 7, ReliefBandParams(subsamples=3))
    assert box.std() < point.std()


def test_trend_elevation_takes_the_band_off(mesh):
    n = len(mesh.faces)
    elev = np.full((n, TEX, TEX, 1), 0.5, dtype=np.float32)
    band = np.zeros_like(elev)
    band[:, 0, 0, 0] = 0.2
    ELEVATION.write(mesh, elev + band)
    try:
        assert np.allclose(trend_elevation(mesh), elev + band)       # no band yet: the elevation itself
        RELIEF_BAND.write(mesh, band)
        assert np.allclose(trend_elevation(mesh), elev)
    finally:
        mesh.face_textures.pop("relief_band", None)
        mesh.face_textures.pop("elevation", None)


def test_preview_roughens_cold_and_high_ground(mesh):
    n = len(mesh.faces)
    lat = np.arcsin(np.clip(mesh.face_centroids[:, 2] / np.linalg.norm(mesh.face_centroids, axis=1), -1, 1))
    t_low = preview_temperature(mesh, np.zeros(n))
    t_high = preview_temperature(mesh, np.full(n, 2.0))
    assert np.all(t_high < t_low)                                   # the lapse rate
    assert t_low[np.argmax(np.abs(lat))] < t_low[np.argmin(np.abs(lat))]
    seed = np.full(n, 0.3)
    rho = preview_ruggedness(mesh, seed, np.zeros(n))
    polar, equatorial = np.abs(lat) > np.radians(70), np.abs(lat) < np.radians(20)
    assert rho[polar].mean() > 0.4 and rho[equatorial].max() == pytest.approx(0.3, abs=0.02)


def test_elevation_stage_adds_the_band():
    params = ElevationParams()
    params = dataclasses.replace(params, elevation=dataclasses.replace(params.elevation, elevation_tex_size=TEX))
    banded, plain = World(seed=3), World(seed=3)
    ElevationBuilder(params, seed=3).build(BuildContext(world=banded, seed=3))
    off = dataclasses.replace(params, relief_band=ReliefBandParams(enabled=False))
    ElevationBuilder(off, seed=3).build(BuildContext(world=plain, seed=3))
    assert RELIEF_BAND.present(banded.mesh) and not RELIEF_BAND.present(plain.mesh)
    band = RELIEF_BAND.read(banded.mesh)
    assert band.std() > 0.005
    # The band is what was added: the trend surface is the unbanded world, up to seam smoothing.
    diff = trend_elevation(banded.mesh) - np.asarray(ELEVATION.read(plain.mesh), dtype=np.float64)
    assert np.median(np.abs(diff)) < 1e-3
