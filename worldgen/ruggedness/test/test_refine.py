"""The ruggedness refinement: cold roughens, river mouths and wetlands smooth, the seed carries through."""
from types import SimpleNamespace

import dataclasses

import numpy as np
import pytest

from core.mesh import Mesh
from worldgen.geology.ruggedness import decayed_max
from worldgen.ruggedness.builder import (
    RuggednessRefineParams,
    refine_ruggedness,
    river_mouth_strength,
)

SEED = 0.4


@pytest.fixture(scope="module")
def mesh():
    return Mesh.icosphere(3).mesh_dual()


def _seed(mesh) -> np.ndarray:
    return np.full(len(mesh.faces), SEED)


def _quiet(**overrides) -> RuggednessRefineParams:
    """No diffusion, so each term's effect is read off directly."""
    return dataclasses.replace(RuggednessRefineParams(smoothing_passes=0), **overrides)


def test_without_inputs_it_is_the_seed(mesh):
    rho, terms = refine_ruggedness(mesh, _seed(mesh), _quiet())
    assert np.allclose(rho, SEED)
    assert terms == {}


def test_cold_is_rugged(mesh):
    n = len(mesh.faces)
    p = _quiet()
    temperature = np.full(n, p.glacial_warm_c + 10.0)
    temperature[7] = p.glacial_cold_c - 5.0
    rho, _ = refine_ruggedness(mesh, _seed(mesh), p, temperature_annual=temperature)
    assert np.isclose(rho[7], SEED + p.glacial_weight)
    assert np.isclose(rho[8], SEED)


def test_river_mouths_are_smooth_and_fade(mesh):
    n = len(mesh.faces)
    mouths = np.zeros(n)
    mouths[9] = 1.0
    rho, _ = refine_ruggedness(mesh, _seed(mesh), _quiet(), mouth_strength=mouths)
    neighbour = mesh.face_adjacency[9][0]
    assert rho[9] < rho[neighbour] < SEED


def test_wetlands_are_smooth(mesh):
    n = len(mesh.faces)
    wet = np.zeros(n)
    wet[4] = 0.5
    p = _quiet()
    rho, _ = refine_ruggedness(mesh, _seed(mesh), p, wetland_fraction=wet)
    assert np.isclose(rho[4], SEED - 0.5 * p.wetland_weight)
    assert np.isclose(rho[5], SEED)


def test_clipped_and_diffused(mesh):
    n = len(mesh.faces)
    wet = np.zeros(n)
    wet[0] = 1.0
    seed = np.full(n, 0.1)
    sharp, _ = refine_ruggedness(mesh, seed, _quiet(wetland_weight=1.0), wetland_fraction=wet)
    soft, _ = refine_ruggedness(mesh, seed, _quiet(wetland_weight=1.0, smoothing_passes=3), wetland_fraction=wet)
    assert sharp[0] == 0.0 and sharp.min() >= 0.0
    near = mesh.face_adjacency[0][0]
    assert soft[near] < sharp[near]                    # the dip spreads to neighbours


def test_decayed_max_decays_per_hop():
    adjacency = [[1], [0, 2], [1, 3], [2]]               # a path
    out = decayed_max(adjacency, np.array([1.0, 0.0, 0.0, 0.0]), decay_hops=1.0)
    assert np.allclose(out, np.exp(-np.arange(4.0)))


def _river(face: int, flow: float):
    mouth = SimpleNamespace(face_id=face)
    return SimpleNamespace(mouth=mouth, segments=[SimpleNamespace(flow_rate=flow)])


def test_river_mouth_strength_scales_with_sqrt_flow():
    rivers = [_river(2, 100.0), _river(4, 25.0), SimpleNamespace(mouth=None, segments=[])]
    strength = river_mouth_strength(rivers, 6)
    assert np.allclose(strength, [0.0, 0.0, 1.0, 0.0, 0.5, 0.0])
    assert not river_mouth_strength([], 3).any()


def test_stage_rewrites_the_field_and_load_restores_the_face_values():
    from core.pipeline import BuildContext
    from core.world import World
    from worldgen.geology.channels import RUGGEDNESS_FACE
    from worldgen.ruggedness.builder import RuggednessBuilder

    dual = Mesh.icosphere(3).mesh_dual()
    n = len(dual.faces)
    dual.set_face_textures("elevation", np.zeros((n, 8, 8, 1), dtype=np.float32), n_channels=1)
    RUGGEDNESS_FACE.write(dual, np.full(n, SEED))
    wet = np.zeros(n)
    wet[:10] = 1.0
    dual.apply_metadata_channel("wetland_fraction", wet.tolist())
    world = World.from_mesh(dual)
    stage = RuggednessBuilder()
    stage.build(BuildContext(world=world))

    tex = dual.face_textures["ruggedness"]
    assert tex.shape == (n, 8, 8, 1)
    assert np.all((tex >= 0.0) & (tex <= 1.0))
    built = np.asarray(RUGGEDNESS_FACE.read(dual), dtype=np.float64)
    assert built[:10].mean() < SEED - 0.1

    for meta in dual.face_metadata:
        meta.pop("ruggedness")
    stage.load({}, BuildContext(world=world))
    loaded = np.asarray(RUGGEDNESS_FACE.read(dual), dtype=np.float64)
    # Loaded values are the texture's face means: close to, not equal to, the built ones.
    assert np.abs(loaded - built).max() < 0.1


def test_stage_needs_the_seed():
    from core.pipeline import BuildContext
    from core.world import World
    from worldgen.ruggedness.builder import RuggednessBuilder

    dual = Mesh.icosphere(2).mesh_dual()
    dual.set_face_textures("elevation", np.zeros((len(dual.faces), 4, 4, 1), dtype=np.float32), n_channels=1)
    with pytest.raises(ValueError):
        RuggednessBuilder().build(BuildContext(world=World.from_mesh(dual)))


def _network(dual, nodes, flows, tex):
    from worldgen.hydrology.types import StreamNetwork
    pos = np.asarray(dual.face_texture_positions(tex, tex), dtype=np.float32).reshape(-1, 3)[nodes]
    k = len(nodes)
    return StreamNetwork(
        nodes=np.asarray(nodes, dtype=np.int64), positions=pos, receiver=np.full(k, -1, dtype=np.int64),
        flow=np.asarray(flows, dtype=np.float32), channel=np.ones(k, dtype=bool), stop=np.zeros(k, dtype=bool),
        resolution=tex,
    )


def test_floodplain_follows_big_gentle_rivers():
    from worldgen.ruggedness.builder import floodplain_strength
    tex = 8
    dual = Mesh.icosphere(3).mesh_dual()
    n = len(dual.faces)
    dual.set_face_textures("elevation", np.zeros((n, tex, tex, 1), dtype=np.float32), n_channels=1)
    # This mesh's texels are ~100 km: widen the plains to reach past the river's own texel.
    p = RuggednessRefineParams(floodplain_full_flow=10_000.0, floodplain_width_km=60.0)
    centre = 5 * tex * tex + 4 * tex + 4                     # face 5, mid-texture
    big = floodplain_strength(dual, _network(dual, [centre], [10_000.0], tex), p).reshape(-1)
    small = floodplain_strength(dual, _network(dual, [centre], [600.0], tex), p).reshape(-1)
    below = floodplain_strength(dual, _network(dual, [centre], [100.0], tex), p).reshape(-1)
    assert big[centre] == pytest.approx(1.0, abs=1e-4)       # full flow, flat ground
    assert 0.0 < small[centre] < 0.5                         # sqrt(600 / 10k)
    assert not below.any()                                   # under the river threshold
    assert big[(centre + 3 * n * tex * tex // 2) % (n * tex * tex)] == 0.0   # far away
    assert (big > 0.05).sum() > (small > 0.05).sum()         # bigger rivers, wider plains


def test_floodplain_skips_steep_reaches():
    from worldgen.ruggedness.builder import floodplain_strength
    tex = 8
    dual = Mesh.icosphere(3).mesh_dual()
    n = len(dual.faces)
    ramp = np.broadcast_to(np.linspace(0.0, 30.0, tex, dtype=np.float32)[None, None, :, None], (n, tex, tex, 1))
    dual.set_face_textures("elevation", np.ascontiguousarray(ramp), n_channels=1)
    centre = 5 * tex * tex + 4 * tex + 4
    steep = floodplain_strength(dual, _network(dual, [centre], [20_000.0], tex), RuggednessRefineParams())
    assert steep.reshape(-1)[centre] < 0.05                  # a gorge, not a floodplain


def test_jaggedness_refine_aridity_and_glacial(mesh):
    from worldgen.ruggedness.builder import JaggednessRefineParams, refine_jaggedness

    n = len(mesh.faces)
    p = JaggednessRefineParams(smoothing_passes=0)
    seed = np.full(n, 0.5)
    rugged = np.full(n, 0.5)
    rugged[3], rugged[4] = 1.0, 0.0
    aridity = np.full(n, 2.0)
    aridity[2] = p.aridity_arid / 2.0
    temperature = np.full(n, p.glacial_warm_c + 10.0)
    temperature[[3, 4]] = p.glacial_cold_c - 5.0

    j, terms = refine_jaggedness(mesh, seed, p, ruggedness=rugged)
    np.testing.assert_allclose(j, 0.5)
    assert terms == {}
    j, _ = refine_jaggedness(mesh, seed, p, ruggedness=rugged, temperature_annual=temperature, aridity_index=aridity)
    assert j[2] == pytest.approx(0.5 + p.aridity_weight)       # deserts are jagged
    assert j[3] == pytest.approx(0.5 + p.glacial_weight)       # alpine
    assert j[4] == pytest.approx(0.5 - p.glacial_weight)       # ice-scoured lowland
    assert j[5] == pytest.approx(0.5)


def test_stage_refines_jaggedness_and_load_restores_it():
    from core.pipeline import BuildContext
    from core.world import World
    from worldgen.geology.channels import JAGGEDNESS_FACE, RUGGEDNESS_FACE
    from worldgen.ruggedness.builder import RuggednessBuilder

    dual = Mesh.icosphere(3).mesh_dual()
    n = len(dual.faces)
    dual.set_face_textures("elevation", np.zeros((n, 8, 8, 1), dtype=np.float32), n_channels=1)
    RUGGEDNESS_FACE.write(dual, np.full(n, SEED))
    JAGGEDNESS_FACE.write(dual, np.full(n, 0.5))
    temperature = np.full(n, 15.0)
    precipitation = np.full(n, 3000.0)
    precipitation[:10] = 50.0
    dual.apply_metadata_channel("temperature_annual", temperature.tolist())
    dual.apply_metadata_channel("precipitation_annual", precipitation.tolist())
    world = World.from_mesh(dual)
    stage = RuggednessBuilder()
    stage.build(BuildContext(world=world))

    tex = dual.face_textures["jaggedness"]
    assert tex.shape == (n, 8, 8, 1)
    built = np.asarray(JAGGEDNESS_FACE.read(dual), dtype=np.float64)
    assert built[:10].mean() > 0.6 and np.median(built) == pytest.approx(0.5, abs=0.02)

    for meta in dual.face_metadata:
        meta.pop("jaggedness")
    stage.load({}, BuildContext(world=world))
    assert np.abs(np.asarray(JAGGEDNESS_FACE.read(dual), dtype=np.float64) - built).max() < 0.1
