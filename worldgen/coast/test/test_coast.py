"""The coast-smoothness field: each physical cause pushes the way it should."""
from types import SimpleNamespace

import numpy as np
import pytest

from core.mesh import Mesh
from worldgen.coast.builder import (
    CoastParams,
    _decayed_max,
    coast_smoothness,
    river_mouth_strength,
)


@pytest.fixture(scope="module")
def mesh():
    return Mesh.icosphere(3).mesh_dual()


def _quiet(**overrides) -> CoastParams:
    """No noise and no diffusion, so each term's effect is read off directly."""
    base: dict = dict(noise_weight=0.0, smoothing_passes=0)
    base.update(overrides)
    return CoastParams(**base)


def test_range_and_determinism(mesh):
    a, _ = coast_smoothness(mesh, CoastParams(), 3)
    b, _ = coast_smoothness(mesh, CoastParams(), 3)
    c, _ = coast_smoothness(mesh, CoastParams(), 4)
    assert a.shape == (len(mesh.faces),)
    assert np.all((a >= 0.0) & (a <= 1.0))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_without_inputs_it_is_the_base(mesh):
    smooth, terms = coast_smoothness(mesh, _quiet(), 0)
    assert np.allclose(smooth, 1.0 - CoastParams().base_ruggedness)
    assert set(terms) == {"noise"}


def test_convergent_margins_are_rugged_and_fade_with_distance(mesh):
    n = len(mesh.faces)
    convergent = np.zeros(n, dtype=bool)
    convergent[0] = True
    smooth, _ = coast_smoothness(mesh, _quiet(), 0, convergent=convergent)
    neighbour = mesh.face_adjacency[0][0]
    base = 1.0 - CoastParams().base_ruggedness
    assert smooth[0] < smooth[neighbour] < base
    assert np.isclose(smooth.max(), base, atol=0.01)   # far away: all but untouched


def test_volcanism_and_cold_are_rugged(mesh):
    n = len(mesh.faces)
    volcanic = np.zeros(n)
    volcanic[5] = 1.0
    smooth, _ = coast_smoothness(mesh, _quiet(), 0, volcanic=volcanic)
    assert smooth[5] < smooth[6]

    p = _quiet()
    temperature = np.full(n, p.glacial_warm_c + 10.0)
    temperature[7] = p.glacial_cold_c - 5.0
    smooth, _ = coast_smoothness(mesh, p, 0, temperature_annual=temperature)
    assert np.isclose(smooth[7], 1.0 - p.base_ruggedness - p.glacial_weight)
    assert np.isclose(smooth[8], 1.0 - p.base_ruggedness)


def test_river_mouths_are_smooth(mesh):
    n = len(mesh.faces)
    mouths = np.zeros(n)
    mouths[9] = 1.0
    smooth, _ = coast_smoothness(mesh, _quiet(), 0, mouth_strength=mouths)
    neighbour = mesh.face_adjacency[9][0]
    assert smooth[9] > smooth[neighbour] > 1.0 - CoastParams().base_ruggedness


def test_diffusion_spreads_a_spike(mesh):
    n = len(mesh.faces)
    convergent = np.zeros(n, dtype=bool)
    convergent[0] = True
    sharp, _ = coast_smoothness(mesh, _quiet(tectonic_decay_hops=0.1), 0, convergent=convergent)
    soft, _ = coast_smoothness(
        mesh, _quiet(tectonic_decay_hops=0.1, smoothing_passes=3), 0, convergent=convergent
    )
    assert soft[0] > sharp[0]
    assert soft[mesh.face_adjacency[0][0]] < sharp[mesh.face_adjacency[0][0]]


def test_decayed_max_decays_per_hop():
    adjacency = [[1], [0, 2], [1, 3], [2]]               # a path
    out = _decayed_max(adjacency, np.array([1.0, 0.0, 0.0, 0.0]), decay_hops=1.0)
    assert np.allclose(out, np.exp(-np.arange(4.0)))


def _river(face: int, flow: float):
    mouth = SimpleNamespace(face_id=face)
    return SimpleNamespace(mouth=mouth, segments=[SimpleNamespace(flow_rate=flow)])


def test_river_mouth_strength_scales_with_sqrt_flow():
    rivers = [_river(2, 100.0), _river(4, 25.0), SimpleNamespace(mouth=None, segments=[])]
    strength = river_mouth_strength(rivers, 6)
    assert np.allclose(strength, [0.0, 0.0, 1.0, 0.0, 0.5, 0.0])
    assert not river_mouth_strength([], 3).any()


def test_stage_writes_a_continuous_texture_and_load_restores_the_face_values():
    from core.pipeline import BuildContext
    from core.world import World
    from worldgen.coast.builder import CoastBuilder

    dual = Mesh.icosphere(3).mesh_dual()
    dual.set_face_textures("elevation", np.zeros((len(dual.faces), 8, 8, 1), dtype=np.float32), n_channels=1)
    world = World.from_mesh(dual)
    world.seed = 11
    stage = CoastBuilder()
    stage.build(BuildContext(world=world))

    tex = dual.face_textures["coast_smoothness"]
    assert tex.shape == (len(dual.faces), 8, 8, 1)
    assert np.all((tex >= 0.0) & (tex <= 1.0))
    built = np.array([m["coast_smoothness"] for m in dual.face_metadata])

    for meta in dual.face_metadata:
        meta.pop("coast_smoothness")
    stage.load({}, BuildContext(world=world))
    loaded = np.array([m["coast_smoothness"] for m in dual.face_metadata])
    # Loaded values are the texture's face means: close to, not equal to, the built ones.
    assert np.abs(loaded - built).max() < 0.1
