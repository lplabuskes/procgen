"""The jaggedness seed: young orogens sharpen, plate interiors round, the field stays in [0, 1]."""
import dataclasses

import numpy as np
import pytest

from core.mesh import Mesh
from core.pipeline import BuildContext
from core.world import World
from worldgen.geology.channels import JAGGEDNESS, JAGGEDNESS_FACE
from worldgen.geology.elevation_builder import ElevationBuilder, ElevationParams
from worldgen.geology.jaggedness import JaggednessParams, jaggedness_seed


@pytest.fixture(scope="module")
def mesh():
    return Mesh.icosphere(3).mesh_dual()


def _interior(n: int) -> list[dict]:
    return [{"boundary_type": "interior", "normal_avg": 0.0, "neighbor_plate_types": set()} for _ in range(n)]


def _quiet(**overrides: float) -> JaggednessParams:
    """Params at neutral base with every term off but the ones named."""
    quiet = JaggednessParams(base=0.5, young_weight=0.0, old_weight=0.0, smoothing_passes=0)
    return dataclasses.replace(quiet, **overrides)


def test_without_terms_it_is_the_base(mesh):
    n = len(mesh.faces)
    j, _ = jaggedness_seed(mesh, _quiet(), plate_types=["continental"] * n, boundary_data=_interior(n),
                           rate_norm=1.0)
    np.testing.assert_allclose(j, 0.5)


def test_young_orogen_is_jagged_and_decays(mesh):
    n = len(mesh.faces)
    data = _interior(n)
    data[0] = {"boundary_type": "convergent", "normal_avg": -1.0, "neighbor_plate_types": {"continental"}}
    j, _ = jaggedness_seed(mesh, _quiet(young_weight=0.4), plate_types=["continental"] * n, boundary_data=data,
                           rate_norm=1.0)
    near = mesh.face_adjacency[0][0]
    assert j[0] == pytest.approx(0.9)
    assert 0.5 < j[near] < j[0]
    assert j.min() == pytest.approx(0.5, abs=0.005)


def test_plate_interiors_roll(mesh):
    n = len(mesh.faces)
    data = _interior(n)
    data[0] = {"boundary_type": "transform", "normal_avg": 0.0, "neighbor_plate_types": {"continental"}}
    j, _ = jaggedness_seed(mesh, _quiet(old_weight=0.3), plate_types=["continental"] * n, boundary_data=data,
                           rate_norm=1.0)
    assert j[0] == pytest.approx(0.5)
    assert j.min() < 0.5 - 0.25
    assert j.min() >= 0.0 and j.max() <= 1.0


def test_elevation_stage_writes_jaggedness():
    world = World(seed=3)
    params = ElevationParams()
    params = dataclasses.replace(params, elevation=dataclasses.replace(params.elevation, elevation_tex_size=8))
    ElevationBuilder(params, seed=3).build(BuildContext(world=world, seed=3))
    mesh = world.mesh
    face = np.asarray(JAGGEDNESS_FACE.read(mesh), dtype=np.float64)
    tex = JAGGEDNESS.read(mesh)
    assert face.shape == (len(mesh.faces),) and tex.shape == (len(mesh.faces), 8, 8, 1)
    assert 0.0 <= face.min() and face.max() <= 1.0
    assert face.std() > 0.03                            # boundaries and interiors give it structure
