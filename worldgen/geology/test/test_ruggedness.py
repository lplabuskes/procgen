"""The ruggedness seed: each tectonic cause pushes the way it should, and the field stays in [0, 1]."""
import dataclasses

import numpy as np
import pytest

from core.mesh import Mesh
from core.pipeline import BuildContext
from core.world import World
from worldgen.geology.channels import RUGGEDNESS, RUGGEDNESS_FACE
from worldgen.geology.elevation_builder import ElevationBuilder, ElevationParams
from worldgen.geology.ruggedness import (
    RuggednessParams,
    boundary_strengths,
    ruggedness_seed,
    ruggedness_texture,
)


@pytest.fixture(scope="module")
def mesh():
    return Mesh.icosphere(3).mesh_dual()


def _interior(n: int) -> list[dict]:
    return [{"boundary_type": "interior", "normal_avg": 0.0, "neighbor_plate_types": set()} for _ in range(n)]


def _quiet(**overrides: float) -> RuggednessParams:
    """Params with every term off but the ones named."""
    quiet = RuggednessParams(noise_weight=0.0, orogenic_weight=0.0, rift_weight=0.0, volcanic_weight=0.0,
                             uplift_weight=0.0, interior_weight=0.0, smoothing_passes=0)
    return dataclasses.replace(quiet, **overrides)


def test_boundary_strengths_collision_beats_arc():
    data = [
        {"boundary_type": "convergent", "normal_avg": -2.0, "neighbor_plate_types": {"continental"}},
        {"boundary_type": "convergent", "normal_avg": -2.0, "neighbor_plate_types": {"oceanic"}},
        {"boundary_type": "divergent", "normal_avg": 1.0, "neighbor_plate_types": {"continental"}},
        {"boundary_type": "divergent", "normal_avg": 1.0, "neighbor_plate_types": {"oceanic"}},
    ]
    types = ["continental", "continental", "continental", "oceanic"]
    orogenic, rift = boundary_strengths(types, data, rate_norm=2.0, arc_factor=0.5)
    np.testing.assert_allclose(orogenic, [1.0, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(rift, [0.0, 0.0, 0.5, 0.0])   # oceanic ridges aren't rifts


def test_orogen_raises_and_decays(mesh):
    n = len(mesh.faces)
    data = _interior(n)
    data[0] = {"boundary_type": "convergent", "normal_avg": -1.0, "neighbor_plate_types": {"continental"}}
    params = _quiet(base=0.2, orogenic_weight=0.5)
    rho, terms = ruggedness_seed(mesh, params, 0, plate_types=["continental"] * n, boundary_data=data,
                                 rate_norm=1.0)
    near = mesh.face_adjacency[0][0]
    assert rho[0] == pytest.approx(0.7)
    assert 0.2 < rho[near] < rho[0]
    assert rho.min() == pytest.approx(0.2, abs=0.005)    # far faces barely feel it


def test_interior_smooths_away_from_boundaries(mesh):
    n = len(mesh.faces)
    data = _interior(n)
    data[0] = {"boundary_type": "transform", "normal_avg": 0.0, "neighbor_plate_types": {"continental"}}
    params = _quiet(base=0.5, interior_weight=0.3)
    rho, _ = ruggedness_seed(mesh, params, 0, plate_types=["continental"] * n, boundary_data=data,
                             rate_norm=1.0)
    assert rho[0] == pytest.approx(0.5)
    assert rho.min() < 0.5 - 0.25                      # far interior loses most of the weight


def test_volcanism_and_clip(mesh):
    n = len(mesh.faces)
    volc = np.zeros(n)
    volc[3] = 1.0
    params = _quiet(base=0.9, volcanic_weight=0.5)
    rho, _ = ruggedness_seed(mesh, params, 0, plate_types=["oceanic"] * n, boundary_data=_interior(n),
                             rate_norm=1.0, volcanism=volc)
    assert rho[3] == 1.0
    assert rho.min() >= 0.0 and rho.max() <= 1.0


def test_base_shifts_uniformly(mesh):
    n = len(mesh.faces)
    types, data = ["continental"] * n, _interior(n)
    lo, _ = ruggedness_seed(mesh, RuggednessParams(base=0.3), 7, plate_types=types, boundary_data=data, rate_norm=1.0)
    hi, _ = ruggedness_seed(mesh, RuggednessParams(base=0.4), 7, plate_types=types, boundary_data=data, rate_norm=1.0)
    inside = (lo > 0.0) & (hi < 1.0)
    np.testing.assert_allclose((hi - lo)[inside], 0.1, atol=1e-9)


def test_texture_shape_range_and_face_means(mesh):
    n = len(mesh.faces)
    rng = np.random.default_rng(0)
    values = rng.uniform(0.0, 1.0, n)
    tex = ruggedness_texture(mesh, values, 8)
    assert tex.shape == (n, 8, 8, 1) and tex.dtype == np.float32
    assert tex.min() >= 0.0 and tex.max() <= 1.0


def test_elevation_stage_writes_ruggedness():
    world = World(seed=3)
    params = ElevationParams()
    params = dataclasses.replace(params, elevation=dataclasses.replace(params.elevation, elevation_tex_size=8))
    builder = ElevationBuilder(params, seed=3)
    builder.build(BuildContext(world=world, seed=3))
    mesh = world.mesh
    face = np.asarray(RUGGEDNESS_FACE.read(mesh), dtype=np.float64)
    tex = RUGGEDNESS.read(mesh)
    assert face.shape == (len(mesh.faces),) and tex.shape == (len(mesh.faces), 8, 8, 1)
    assert 0.0 <= face.min() and face.max() <= 1.0
    assert face.std() > 0.05                            # tectonics and noise give it structure


def test_face_relief_is_log_linear_and_falls_back_to_the_span(mesh):
    from worldgen.geology.ruggedness import (
        FACE_RELIEF_RUGGED_KM, FACE_RELIEF_SMOOTH_KM, face_relief, face_relief_km,
    )
    relief = face_relief_km(np.array([0.0, 0.5, 1.0, 2.0]))
    assert relief[0] == pytest.approx(FACE_RELIEF_SMOOTH_KM)
    assert relief[1] == pytest.approx(np.sqrt(FACE_RELIEF_SMOOTH_KM * FACE_RELIEF_RUGGED_KM), rel=1e-5)
    assert relief[2] == pytest.approx(FACE_RELIEF_RUGGED_KM) and relief[3] == relief[2]   # clipped
    n = len(mesh.faces)
    elev = np.zeros((n, 4, 4, 1), dtype=np.float32)
    elev[:, 0, 0, 0] = 0.3
    mesh.set_face_textures("elevation", elev, n_channels=1)
    try:
        assert np.allclose(face_relief(mesh), 0.3)          # no ruggedness: the elevation span
        RUGGEDNESS_FACE.write(mesh, np.ones(n))
        assert np.allclose(face_relief(mesh), FACE_RELIEF_RUGGED_KM)
    finally:
        mesh.face_textures.pop("elevation", None)
        for meta in mesh.face_metadata:
            meta.pop("ruggedness", None)
