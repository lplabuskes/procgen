"""Town stage: sizes, rural density, siting, and the stage's people accounting."""
from __future__ import annotations

import numpy as np
import pytest

from core.mesh import Mesh
from core.pipeline import BuildContext
from core.world import World
from realmgen.channels import HINTERLAND_ID, MINING_POPULATION, RURAL_POPULATION, TOWN_POPULATION
from realmgen.population.calibration import KM2_PER_STERADIAN
from realmgen.refine.sites import SiteField
from realmgen.test.conftest import _settlement
from realmgen.towns.builder import (
    MINING_DENSITY_TEXTURE, RURAL_DENSITY_TEXTURE, TownParams, TownStage, roll_town_walls,
)
from realmgen.towns.placement import (
    compute_mining_density, compute_rural_density, draw_town_sizes, place_towns, shore_texels, texel_owners,
)
from realmgen.towns.types import TOWN_MINING, Towns
from worldgen.cost.surfaces import EARTH_RADIUS_KM

TEX = 8


@pytest.fixture(scope="module")
def dual():
    return Mesh.icosphere(2).mesh_dual()


def _fields(dual):
    """Ocean where z < -0.3, smoothly varying fertility elsewhere."""
    pos = dual.face_texture_positions(TEX, TEX)
    ocean = (pos[..., 2] < -0.3).astype(np.float32)[..., None]
    fert = (0.5 + 0.5 * np.sin(4.0 * pos[..., 0]) * np.cos(3.0 * pos[..., 1])).astype(np.float32)[..., None]
    return pos, ocean, fert


def _owned(dual, pos):
    owner = texel_owners(dual.face_centroids, pos)
    return owner, owner == np.arange(len(dual.faces))[:, None, None]


# ── sizes ────────────────────────────────────────────────────────────────────


def test_town_sizes_bounded_and_spend_the_budget():
    rng = np.random.default_rng(0)
    sizes = draw_town_sizes(50_000.0, rng, town_min=1_000.0, town_max=7_000.0, alpha=1.0, cap=1e9)
    assert np.all(sizes >= 1_000.0 - 1e-9) and np.all(sizes <= 7_000.0 + 1e-9)
    assert np.all(np.diff(sizes) <= 0)                      # largest first
    assert 49_000.0 <= sizes.sum() <= 51_000.0              # budget ± one minimum town


def test_town_sizes_small_budget_is_unbiased():
    total = sum(
        draw_town_sizes(300.0, np.random.default_rng(k), town_min=1_000.0, town_max=7_000.0,
                        alpha=1.0, cap=1e9).sum()
        for k in range(4000)
    )
    assert total / 4000 == pytest.approx(300.0, rel=0.1)


def test_town_sizes_never_exceed_the_cap():
    for k in range(200):
        sizes = draw_town_sizes(900.0, np.random.default_rng(k), town_min=1_000.0, town_max=7_000.0,
                                alpha=1.0, cap=950.0)
        assert sizes.sum() <= 950.0


# ── rural density ────────────────────────────────────────────────────────────


def test_rural_density_conserves_each_face(dual):
    pos, ocean, fert = _fields(dual)
    owner, owned = _owned(dual, pos)
    n = len(dual.faces)
    rural = np.linspace(0.0, 5e5, n)
    areas = np.asarray(dual.face_areas) * KM2_PER_STERADIAN
    land = ocean[..., 0] < 0.5
    density, _ = compute_rural_density(rural, areas, fert[..., 0].astype(np.float64), land, owner, 2.0)

    texel_area = areas / owned.sum(axis=(1, 2))
    held = (density * owned).sum(axis=(1, 2)) * texel_area
    has_land = (land & owned).any(axis=(1, 2))
    np.testing.assert_allclose(held[has_land], rural[has_land], rtol=1e-9)
    assert np.all(density[~land] == 0.0)


def test_rural_density_spreads_evenly_over_barren_land(dual):
    pos, ocean, _ = _fields(dual)
    owner, owned = _owned(dual, pos)
    n = len(dual.faces)
    land = ocean[..., 0] < 0.5
    density, _ = compute_rural_density(
        np.full(n, 1e4), np.ones(n), np.zeros(land.shape), land, owner, 2.0)
    f = int(np.argmax((land & owned).sum(axis=(1, 2))))
    vals = density[f][land[f] & owned[f]]
    assert vals.min() > 0 and np.allclose(vals, vals[0])


def test_rural_density_puts_the_fishers_on_the_shore(dual):
    pos, ocean, fert = _fields(dual)
    owner, owned = _owned(dual, pos)
    n = len(dual.faces)
    land = ocean[..., 0] < 0.5
    areas = np.asarray(dual.face_areas) * KM2_PER_STERADIAN
    rural = np.full(n, 1e5)
    fishing = np.full(n, 4e4)
    density, fish = compute_rural_density(
        rural, areas, fert[..., 0].astype(np.float64), land, owner, 2.0, fishing)

    texel_area = areas / owned.sum(axis=(1, 2))
    shore = shore_texels(land)
    coastal = (land & owned).any(axis=(1, 2)) & (shore & owned).any(axis=(1, 2))
    np.testing.assert_allclose(((density * owned).sum(axis=(1, 2)) * texel_area)[coastal], rural[coastal], rtol=1e-9)
    np.testing.assert_allclose(((fish * owned).sum(axis=(1, 2)) * texel_area)[coastal], fishing[coastal], rtol=1e-9)
    owned_coastal = owned & coastal[:, None, None]
    assert np.all(fish[owned_coastal & ~shore] == 0.0)
    assert np.all(density >= fish - 1e-12)


# ── siting ───────────────────────────────────────────────────────────────────


def _place(dual, requests, *, spacing_km=20.0, city=None, params=None, streams=None):
    pos, ocean, fert = _fields(dual)
    owner, _ = _owned(dual, pos)
    n = len(dual.faces)
    land = ocean[..., 0] < 0.5
    density, _ = compute_rural_density(
        np.full(n, 1e5), np.full(n, 1e5), fert[..., 0].astype(np.float64), land, owner, 2.0)
    sites = SiteField(dual, ocean, None, fert, None, positions=pos, streams=streams)
    cp = np.zeros((0, 3)) if city is None else np.asarray(city[0])[None, :]
    return place_towns(
        requests, sites=sites, density=density, adjacency=dual.face_adjacency,
        city_positions=cp, city_faces=np.array([] if city is None else [city[1]], dtype=np.int32),
        city_radii=np.array([] if city is None else [city[2]]),
        hinterland=np.full(n, -1, dtype=np.int32), params=params or TownParams(),
        town_spacing=spacing_km / EARTH_RADIUS_KM, mining_radius=0.01, route_radius=0.01,
        mineral_points=None, route_points=None, seed=7,
    ), sites


def test_towns_keep_their_spacing_and_their_face(dual):
    faces = [f for f in range(len(dual.faces)) if dual.face_centroids[f][2] > 0.3][:6]
    requests = [(f, np.full(4, 1_000.0)) for f in faces]
    (towns, unplaced, _), sites = _place(dual, requests, spacing_km=200.0)
    assert len(towns) + unplaced.sum() / 1_000.0 == 4 * len(faces)
    gaps = np.linalg.norm(towns.positions[:, None] - towns.positions[None], axis=2)
    np.fill_diagonal(gaps, np.inf)
    assert gaps.min() >= 200.0 / EARTH_RADIUS_KM - 1e-9
    assert np.all(sites.owner(towns.positions) == towns.face_id)


def test_town_with_no_room_stays_rural(dual):
    f = next(f for f in range(len(dual.faces)) if dual.face_centroids[f][2] > 0.5)
    # A city whose exclusion disc covers the whole neighbourhood.
    (towns, unplaced, _), _ = _place(
        dual, [(f, np.array([2_000.0]))], city=(dual.face_centroids[f], f, 1.0))
    assert len(towns) == 0 and unplaced[f] == 2_000.0


def _column_streams(dual):
    """A stream network whose channels are texel column 3 of every face."""
    from worldgen.hydrology.types import StreamNetwork
    n = len(dual.faces)
    nodes = (np.arange(n)[:, None] * TEX * TEX + np.arange(TEX)[None, :] * TEX + 3).ravel()
    k = nodes.size
    pos = dual.face_texture_positions(TEX, TEX).reshape(-1, 3)[nodes]
    return StreamNetwork(
        nodes=nodes.astype(np.int64), positions=pos.astype(np.float32),
        receiver=np.full(k, -1, dtype=np.int64), flow=np.full(k, 100.0, dtype=np.float32),
        channel=np.ones(k, dtype=bool), stop=np.zeros(k, dtype=bool), resolution=TEX,
    )


def test_streams_draw_river_towns(dual):
    """Stream texels earn the stream bonus and make river towns; without the
    network (city siting's case) nothing changes."""
    from realmgen.towns.types import TOWN_RIVER
    faces = [f for f in range(len(dual.faces)) if dual.face_centroids[f][2] > 0.3][:8]
    requests = [(f, np.array([1_000.0])) for f in faces]
    strong = TownParams(stream_bonus=1e6, route_bonus=0.0, mining_bonus=0.0, port_bonus=0.0)
    (towns, _, _), sites = _place(dual, requests, params=strong, streams=_column_streams(dual))
    assert len(towns) == len(faces) and np.all(towns.kind == TOWN_RIVER)
    (plain, _, _), plain_sites = _place(dual, requests, params=strong)
    assert not np.any(plain.kind == TOWN_RIVER)
    assert not plain_sites.face(faces[0]).stream.any() and sites.face(faces[0]).stream.any()


def test_siting_is_deterministic(dual):
    faces = [f for f in range(len(dual.faces)) if dual.face_centroids[f][2] > 0.3][:4]
    requests = [(f, np.array([3_000.0, 1_000.0])) for f in faces]
    (a, _, _), _ = _place(dual, requests)
    (b, _, _), _ = _place(dual, requests)
    np.testing.assert_array_equal(a.positions, b.positions)


def test_towns_pack_round_trip():
    towns = Towns(
        positions=np.eye(3), population=np.array([1e3, 2e3, 5e3]),
        face_id=np.array([0, 1, 2], dtype=np.int32), parent=np.array([-1, 0, 1], dtype=np.int32),
        kind=np.array([0, 1, 4], dtype=np.int8),
    )
    back = Towns.unpack(towns.pack())
    np.testing.assert_allclose(back.positions, towns.positions, atol=1e-7)
    np.testing.assert_array_equal(back.population, towns.population)
    np.testing.assert_array_equal(back.kind, towns.kind)
    np.testing.assert_array_equal(back.parent, towns.parent)
    np.testing.assert_array_equal(back.has_walls, towns.has_walls)
    np.testing.assert_array_equal(back.wall_strength, towns.wall_strength)


# ── walls ────────────────────────────────────────────────────────────────────


def _many_towns(k: int, pop: float) -> Towns:
    return Towns(
        positions=np.tile([1.0, 0.0, 0.0], (k, 1)), population=np.full(k, pop),
        face_id=np.arange(k, dtype=np.int32) % 50, parent=np.full(k, -1, dtype=np.int32),
        kind=np.zeros(k, dtype=np.int8),
    )


def test_town_walls_rarer_than_cities_likelier_big_and_in_danger():
    p = TownParams()
    calm, wild = np.zeros(50), np.ones(50)
    small, big = _many_towns(2000, p.town_min), _many_towns(2000, 7000.0)
    rate = {
        (name, d): roll_town_walls(t, danger, 3, p, 7000.0)[0].mean()
        for name, t in (("small", small), ("big", big)) for d, danger in (("calm", calm), ("wild", wild))
    }
    assert rate["small", "calm"] == pytest.approx(p.wall_base_prob, abs=0.02)
    assert rate["small", "calm"] < rate["big", "calm"] < rate["big", "wild"]
    assert rate["small", "calm"] < rate["small", "wild"]
    # Even the likeliest town is walled less often than the least likely city (base 0.3).
    assert rate["big", "wild"] < 0.4
    has, strength = roll_town_walls(big, wild, 3, p, 7000.0)
    assert np.all((strength > 0) == has)
    assert np.all((strength[has] >= p.wall_min_strength) & (strength[has] <= p.wall_max_strength))
    np.testing.assert_array_equal(has, roll_town_walls(big, wild, 3, p, 7000.0)[0])   # deterministic


# ── stage ────────────────────────────────────────────────────────────────────


def _world(dual):
    _, ocean, fert = _fields(dual)
    world = World.from_mesh(dual)
    world.seed = 3
    mesh = world.mesh
    n = len(mesh.faces)
    mesh.set_face_textures("ocean_mask", ocean, n_channels=1)
    mesh.set_face_textures("fertility", fert, n_channels=1)
    is_water = (ocean[..., 0] > 0.5).mean(axis=(1, 2)) > 0.5
    rural = np.where(is_water, 0.0, 2e5)
    RURAL_POPULATION.write(mesh, rural)
    TOWN_POPULATION.write(mesh, np.zeros(n))
    HINTERLAND_ID.write(mesh, [-1] * n)
    city_face = int(np.flatnonzero(~is_water)[0])
    world.add_feature("settlements", [
        _settlement(face_id=city_face, pop=50_000.0, position=np.asarray(mesh.face_centroids[city_face]))
    ])
    return world, rural


def test_stage_carves_towns_out_of_the_countryside(dual):
    world, rural0 = _world(dual)
    stage = TownStage(TownParams(town_fraction=0.2))
    ctx = BuildContext(world=world, seed=3)
    stage.build(ctx)
    mesh = world.mesh
    towns = world.get_feature("towns")
    assert len(towns) > 0
    rural, in_towns = RURAL_POPULATION.read(mesh), TOWN_POPULATION.read(mesh)
    np.testing.assert_allclose(rural + in_towns, rural0, rtol=1e-12)
    np.testing.assert_allclose(
        np.bincount(towns.face_id, weights=towns.population, minlength=len(rural)), in_towns)
    assert mesh.face_textures[RURAL_DENSITY_TEXTURE].shape == (len(rural), TEX, TEX, 1)

    # Idempotent: a second run starts from the same countryside.
    stage.build(ctx)
    np.testing.assert_allclose(RURAL_POPULATION.read(mesh) + TOWN_POPULATION.read(mesh), rural0, rtol=1e-12)
    np.testing.assert_array_equal(world.get_feature("towns").positions, towns.positions)


def test_mining_density_puts_each_deposits_miners_on_its_texel(dual):
    pos, ocean, _ = _fields(dual)
    owner, owned = _owned(dual, pos)
    n = len(dual.faces)
    land = ocean[..., 0] < 0.5
    areas = np.full(n, 1e5)
    f = next(f for f in range(n) if (owned[f] & land[f]).any())
    texel = np.argwhere(owned[f] & land[f])[0]
    point = pos[f, texel[0], texel[1]] / np.linalg.norm(pos[f, texel[0], texel[1]])
    density, unsited = compute_mining_density(
        point[None], np.array([f]), np.array([3_000.0]), pos, areas, land, owner)
    assert np.count_nonzero(density) == 1 and density[f, texel[0], texel[1]] > 0
    texel_area = areas[f] / owned[f].sum()
    assert density.sum() * texel_area == pytest.approx(3_000.0)
    assert unsited.sum() == 0.0


def test_stage_gathers_miners_into_mining_towns_by_their_deposit(dual):
    world, rural0 = _world(dual)
    mesh = world.mesh
    pos = dual.face_texture_positions(TEX, TEX)
    owner, owned = _owned(dual, pos)
    land = mesh.face_textures["ocean_mask"][..., 0] < 0.5
    city = world.get_feature("settlements")[0].face_id
    f = next(f for f in range(len(rural0))
             if rural0[f] > 0 and f != city and f not in dual.face_adjacency[city] and (owned[f] & land[f]).sum() > 8)
    texel = np.argwhere(owned[f] & land[f])[0]
    deposit = pos[f, texel[0], texel[1]] / np.linalg.norm(pos[f, texel[0], texel[1]])
    miners = 20_000.0
    world.add_feature("minerals", (deposit[None], np.zeros(1, dtype=np.int8), np.ones(1, dtype=np.float32)))
    world.add_feature("mine_workers", np.array([miners]))
    rural0 = rural0.copy()
    rural0[f] += miners
    RURAL_POPULATION.write(mesh, rural0)
    mining = np.zeros(len(rural0))
    mining[f] = miners
    MINING_POPULATION.write(mesh, mining)

    radius_km = 600.0
    stage = TownStage(TownParams(town_fraction=0.0, mining_town_fraction=0.6,
                                 mining_radius_km=radius_km, site_jitter=0.0))
    stage.build(BuildContext(world=world, seed=3))
    towns = world.get_feature("towns")
    # Only the miners make towns here, all of the mining kind and by the deposit.
    assert len(towns) > 0 and np.all(towns.face_id == f) and np.all(towns.kind == TOWN_MINING)
    chord = np.linalg.norm(towns.positions - deposit, axis=1)
    assert chord.max() < 2.0 * radius_km / EARTH_RADIUS_KM
    in_towns = TOWN_POPULATION.read(mesh)
    np.testing.assert_allclose(RURAL_POPULATION.read(mesh) + in_towns, rural0, rtol=1e-12)
    # The miners left in camps stay on the deposit's texel, conserved per face.
    mine_tex = mesh.face_textures[MINING_DENSITY_TEXTURE][..., 0]
    texel_area = (dual.face_areas[f] * KM2_PER_STERADIAN) / owned[f].sum()
    left = float(mine_tex[f].sum()) * texel_area
    assert left == pytest.approx(miners - in_towns[f], rel=1e-5)
    assert mine_tex[f, texel[0], texel[1]] * texel_area == pytest.approx(left, rel=1e-5)
